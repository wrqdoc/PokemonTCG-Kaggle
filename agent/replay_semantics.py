"""Position-robust semantic decoding for CABT replay decisions.

Recorded actions are option indices because that is the simulator protocol.  We
use the indices only to locate the recorded options.  Model/alignment identities
are then reconstructed from option type, zones, card IDs, card serials, targets
and effect parameters; the raw option ordinal is never part of the semantic key.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Iterator


AREA_NAMES = {
    1: "DECK", 2: "HAND", 3: "DISCARD", 4: "ACTIVE", 5: "BENCH",
    6: "PRIZE", 7: "STADIUM", 8: "ENERGY", 9: "TOOL",
    10: "PRE_EVOLUTION", 11: "PLAYER", 12: "LOOKING",
}
SELECT_TYPE_NAMES = {
    0: "MAIN", 1: "CARD", 2: "ATTACHED_CARD", 3: "CARD_OR_ATTACHED_CARD",
    4: "ENERGY", 5: "SKILL", 6: "ATTACK", 7: "EVOLVE", 8: "COUNT",
    9: "YES_NO", 10: "SPECIAL_CONDITION",
}
SELECT_CONTEXT_NAMES = {
    0: "MAIN", 1: "SETUP_ACTIVE_POKEMON", 2: "SETUP_BENCH_POKEMON",
    3: "SWITCH", 4: "TO_ACTIVE", 5: "TO_BENCH", 6: "TO_FIELD",
    7: "TO_HAND", 8: "DISCARD", 9: "TO_DECK", 10: "TO_DECK_BOTTOM",
    11: "TO_PRIZE", 12: "NOT_MOVE", 13: "DAMAGE_COUNTER",
    14: "DAMAGE_COUNTER_ANY", 15: "DAMAGE", 16: "REMOVE_DAMAGE_COUNTER",
    17: "HEAL", 18: "EVOLVES_FROM", 19: "EVOLVES_TO", 20: "DEVOLVE",
    21: "ATTACH_FROM", 22: "ATTACH_TO", 23: "DETACH_FROM", 24: "LOOK",
    25: "EFFECT_TARGET", 26: "DISCARD_ENERGY_CARD", 27: "DISCARD_TOOL_CARD",
    28: "SWITCH_ENERGY_CARD", 29: "DISCARD_CARD_OR_ATTACHED_CARD",
    30: "DISCARD_ENERGY", 31: "TO_HAND_ENERGY", 32: "TO_DECK_ENERGY",
    33: "SWITCH_ENERGY", 34: "SKILL_ORDER", 35: "ATTACK",
    36: "DISABLE_ATTACK", 37: "EVOLVE", 38: "DRAW_COUNT",
    39: "DAMAGE_COUNTER_COUNT", 40: "REMOVE_DAMAGE_COUNTER_COUNT",
    41: "IS_FIRST", 42: "MULLIGAN", 43: "ACTIVATE", 44: "FIRST_EFFECT",
    45: "MORE_DEVOLVE", 46: "COIN_HEAD", 47: "AFFECT_SPECIAL_CONDITION",
    48: "RECOVER_SPECIAL_CONDITION",
}
OPTION_TYPE_NAMES = {
    0: "NUMBER", 1: "YES", 2: "NO", 3: "CARD", 4: "TOOL_CARD",
    5: "ENERGY_CARD", 6: "ENERGY", 7: "PLAY", 8: "ATTACH",
    9: "EVOLVE", 10: "ABILITY", 11: "DISCARD", 12: "RETREAT",
    13: "ATTACK", 14: "END", 15: "SKILL", 16: "SPECIAL_CONDITION",
}
SOURCE_REQUIRED = {3, 4, 5, 6, 7, 8, 9, 10, 11}
TARGET_REQUIRED = {8, 9}


def as_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def player_role(player_index: int | None, me: int) -> str:
    if player_index is None:
        return "NONE"
    return "SELF" if player_index == me else "OPPONENT" if player_index == 1 - me else f"PLAYER_{player_index}"


def card_identity(card: Any) -> dict[str, Any] | None:
    if not isinstance(card, dict):
        return None
    if card.get("_hidden"):
        return {"hidden": True}
    card_id = int(card.get("id") or card.get("cardId") or 0)
    serial = int(card.get("serial") or 0)
    if not card_id and not serial:
        return None
    return {"card_id": card_id, "serial": serial}


def player_state(observation: dict, player_index: int) -> dict:
    players = ((observation.get("current") or {}).get("players") or [])
    return players[player_index] if 0 <= player_index < len(players) and isinstance(players[player_index], dict) else {}


def zone(observation: dict, area: int | None, player_index: int) -> list:
    if area is None:
        return []
    current, select = observation.get("current") or {}, observation.get("select") or {}
    state = player_state(observation, player_index)
    zones = {
        1: select.get("deck"), 2: state.get("hand"), 3: state.get("discard"),
        4: state.get("active"), 5: state.get("bench"), 6: state.get("prize"),
        7: current.get("stadium"), 12: current.get("looking"),
    }
    return as_list(zones.get(area))


def zone_item(observation: dict, area: Any, index: Any, player_index: int) -> dict | None:
    try:
        area_i, index_i = int(area), int(index)
    except (TypeError, ValueError):
        return None
    values = zone(observation, area_i, player_index)
    if not 0 <= index_i < len(values):
        return None
    value = values[index_i]
    # Prize cards are intentionally represented by null.  Every such choice is
    # one observable equivalence class; leaking its list position would create
    # a meaningless option-index label.
    if value is None and area_i == 6:
        return {"_hidden": True}
    return value if isinstance(value, dict) else None


def source_entity(observation: dict, option: dict) -> tuple[dict | None, int | None, int | None]:
    current = observation.get("current") or {}
    me = int(current.get("yourIndex") or 0)
    option_type = int(option.get("type") or 0)
    owner = int(option.get("playerIndex", me))
    area = int(option.get("area") or 0) or None
    if option_type in (7, 8, 9):
        owner, area = me, 2
    base = zone_item(observation, area, option.get("index"), owner)
    if option_type == 4 and base is not None:
        attached = as_list(base.get("tools"))
        idx = option.get("toolIndex")
        base = attached[int(idx)] if isinstance(idx, int) and 0 <= idx < len(attached) else None
    elif option_type in (5, 6) and base is not None:
        attached = as_list(base.get("energyCards"))
        idx = option.get("energyIndex")
        base = attached[int(idx)] if isinstance(idx, int) and 0 <= idx < len(attached) else None
    return base, area, owner


def target_entity(observation: dict, option: dict) -> tuple[dict | None, int | None, int | None]:
    me = int((observation.get("current") or {}).get("yourIndex") or 0)
    owner = int(option.get("inPlayPlayerIndex", me))
    area = int(option.get("inPlayArea") or 0) or None
    return zone_item(observation, area, option.get("inPlayIndex"), owner), area, owner


def semantic_option(observation: dict, option: dict) -> dict[str, Any]:
    current = observation.get("current") or {}
    me = int(current.get("yourIndex") or 0)
    option_type = int(option.get("type") or 0)
    source, source_area, source_owner = source_entity(observation, option)
    target, target_area, target_owner = target_entity(observation, option)
    if option_type in (12, 13):
        source_area, source_owner = 4, me
        source = zone_item(observation, 4, 0, me)
    if option_type == 15 and source is None:
        source = {"id": option.get("cardId"), "serial": option.get("serial")}
    result = {
        "option_type": option_type,
        "option_name": OPTION_TYPE_NAMES.get(option_type, f"OPTION_{option_type}"),
        "source_area": AREA_NAMES.get(source_area, None),
        "source_player": player_role(source_owner, me),
        "source": card_identity(source),
        "target_area": AREA_NAMES.get(target_area, None),
        "target_player": player_role(target_owner, me) if target_area is not None else "NONE",
        "target": card_identity(target),
        "number": option.get("number"),
        "count": option.get("count"),
        "attack_id": option.get("attackId"),
        "skill_card_id": option.get("cardId"),
        "skill_serial": option.get("serial"),
        "special_condition_type": option.get("specialConditionType"),
    }
    return result


def semantic_key(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def valid_recorded_action(select: dict, action: Any) -> bool:
    if not isinstance(action, list):
        return False
    options = select.get("option") or []
    minimum, maximum = int(select.get("minCount") or 0), int(select.get("maxCount") or 0)
    return (minimum <= len(action) <= maximum and len(action) == len(set(action))
            and all(isinstance(index, int) and 0 <= index < len(options) for index in action))


def iter_decisions(replay: dict) -> Iterator[dict[str, Any]]:
    steps = replay.get("steps") or []
    for step_index in range(max(0, len(steps) - 1)):
        current_step, next_step = steps[step_index], steps[step_index + 1]
        if not isinstance(current_step, list) or not isinstance(next_step, list):
            continue
        for seat in range(min(len(current_step), len(next_step))):
            entry, following = current_step[seat], next_step[seat]
            if not isinstance(entry, dict) or not isinstance(following, dict):
                continue
            observation = entry.get("observation")
            if entry.get("status") != "ACTIVE" or not isinstance(observation, dict):
                continue
            select, action = observation.get("select"), following.get("action")
            if not isinstance(select, dict) or not valid_recorded_action(select, action):
                continue
            options = select.get("option") or []
            semantics = [semantic_option(observation, option) for option in options]
            keys = [semantic_key(value) for value in semantics]
            selected = [semantics[index] for index in action]
            selected_keys = [keys[index] for index in action]
            counts = Counter(keys)
            yield {
                "step_index": step_index,
                "seat": seat,
                "observation": observation,
                "select": select,
                "action": list(action),
                "semantics": semantics,
                "semantic_keys": keys,
                "selected_semantics": selected,
                "selected_semantic_keys": selected_keys,
                "semantic_collision_count": sum(count - 1 for count in counts.values() if count > 1),
                "selected_semantics_unique": all(counts[key] == 1 for key in selected_keys),
            }


def unresolved_references(decision: dict[str, Any]) -> list[dict[str, Any]]:
    problems = []
    for index, semantic in enumerate(decision["semantics"]):
        option_type = semantic["option_type"]
        if option_type in SOURCE_REQUIRED and semantic["source"] is None:
            problems.append({"option_index": index, "kind": "source", "option_type": option_type})
        if option_type in TARGET_REQUIRED and semantic["target"] is None:
            problems.append({"option_index": index, "kind": "target", "option_type": option_type})
    return problems
