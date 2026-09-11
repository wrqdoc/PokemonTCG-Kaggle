"""Stream CABT replays into compact, versioned grouped-BC NumPy shards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from replay_semantics import (
    AREA_NAMES, iter_decisions, semantic_key, source_entity, target_entity,
)


SCHEMA_VERSION = "global-bc-grouped-shard-v1"
MAX_ENTITIES = 96
GLOBAL_NUM_NAMES = [
    "turn", "turn_action_count", "own_turn", "went_first", "supporter_played",
    "stadium_played", "energy_attached", "retreated", "remaining_overage",
    "observation_step", "own_deck_count", "own_hand_count", "own_discard_count",
    "own_prize_count", "own_bench_count", "own_active_count", "own_total_damage",
    "own_total_energy", "opp_deck_count", "opp_hand_count", "opp_discard_count",
    "opp_prize_count", "opp_bench_count", "opp_active_count", "opp_total_damage",
    "opp_total_energy", "select_min", "select_max", "remain_damage",
    "remain_energy", "select_deck_visible", "looking_visible", "legal_count",
    "selection_ordinal", "selected_so_far", "remaining_required", "remaining_allowed",
]
ENTITY_NUM_NAMES = [
    "hp", "max_hp", "damage_counter", "energy_units", "energy_cards",
    "tools", "appear_this_turn", "evolution_depth",
]
ACTION_NUM_NAMES = [
    "is_stop", "number", "count", "source_hp", "source_max_hp", "source_damage",
    "source_energy", "target_hp", "target_max_hp", "target_damage", "target_energy",
]
DATE_CODE = {"7-17": 0, "7-18": 1, "7-19": 2, "7-20": 3, "7-21": 4,
             "7-11": 5, "7-12": 6, "7-13": 7, "7-14": 8, "7-15": 9, "7-16": 10,
             "7-06": 11, "7-07": 12, "7-08": 13, "7-09": 14, "7-10": 15,
             "7-01": 16, "7-02": 17, "7-03": 18, "7-04": 19, "7-05": 20,
             "7-22": 21, "7-23": 22, "7-24": 23,
             "7-25": 24, "7-26": 25, "7-27": 26, "7-28": 27,
             "7-29": 28, "7-30": 29, "7-31": 30, "8-1": 31, "8-2": 32}
AREA_CODE = {name: value for value, name in AREA_NAMES.items()}


def safe_float(value: Any, scale: float = 1.0) -> float:
    try:
        return float(value or 0) / scale
    except (TypeError, ValueError):
        return 0.0


def as_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def player(observation: dict, seat: int) -> dict:
    players = ((observation.get("current") or {}).get("players") or [])
    return players[seat] if 0 <= seat < len(players) and isinstance(players[seat], dict) else {}


def in_play(state: dict) -> list[dict]:
    return [card for card in as_list(state.get("active")) + as_list(state.get("bench")) if isinstance(card, dict)]


def damage(card: dict) -> float:
    if card.get("damageCounter") is not None:
        return safe_float(card.get("damageCounter"), 40.0)
    return max(0.0, safe_float(card.get("maxHp"), 400.0) - safe_float(card.get("hp"), 400.0))


def pokemon_energy_count(card: dict) -> int:
    return max(len(as_list(card.get("energies"))), len(as_list(card.get("energyCards"))))


def global_features(decision: dict, selection_ordinal: int, selected_so_far: int) -> list[float]:
    observation, select = decision["observation"], decision["select"]
    current = observation.get("current") or {}
    me = int(current.get("yourIndex") or decision["seat"])
    mine, theirs = player(observation, me), player(observation, 1 - me)
    mine_play, their_play = in_play(mine), in_play(theirs)
    first_player = int(current.get("firstPlayer", -1))
    turn = int(current.get("turn") or 0)
    own_turn = sum(1 for value in range(1, turn + 1) if (value - 1) % 2 == me) if turn > 0 else 0
    action_count = len(decision["action"])
    minimum, maximum = int(select.get("minCount") or 0), int(select.get("maxCount") or 0)
    values = [
        safe_float(turn, 200), safe_float(current.get("turnActionCount"), 100), safe_float(own_turn, 100),
        float(me == first_player), float(bool(current.get("supporterPlayed"))),
        float(bool(current.get("stadiumPlayed"))), float(bool(current.get("energyAttached"))),
        float(bool(current.get("retreated"))), safe_float(observation.get("remainingOverageTime"), 600),
        safe_float(observation.get("step"), 5000), safe_float(mine.get("deckCount"), 60),
        safe_float(mine.get("handCount", len(as_list(mine.get("hand")))), 30),
        safe_float(len(as_list(mine.get("discard"))), 60), safe_float(len(as_list(mine.get("prize"))), 6),
        safe_float(len(as_list(mine.get("bench"))), 5), safe_float(len(as_list(mine.get("active"))), 1),
        sum(damage(card) for card in mine_play) / 6, sum(pokemon_energy_count(card) for card in mine_play) / 30,
        safe_float(theirs.get("deckCount"), 60), safe_float(theirs.get("handCount"), 30),
        safe_float(len(as_list(theirs.get("discard"))), 60), safe_float(len(as_list(theirs.get("prize"))), 6),
        safe_float(len(as_list(theirs.get("bench"))), 5), safe_float(len(as_list(theirs.get("active"))), 1),
        sum(damage(card) for card in their_play) / 6, sum(pokemon_energy_count(card) for card in their_play) / 30,
        safe_float(minimum, 10), safe_float(maximum, 10), safe_float(select.get("remainDamageCounter"), 40),
        safe_float(select.get("remainEnergyCost"), 10), safe_float(len(as_list(select.get("deck"))), 60),
        safe_float(len(as_list(current.get("looking"))), 60), safe_float(len(decision["semantics"]), 128),
        safe_float(selection_ordinal, 10), safe_float(selected_so_far, 10),
        safe_float(max(0, minimum - selected_so_far), 10), safe_float(max(0, maximum - selected_so_far), 10),
    ]
    assert len(values) == len(GLOBAL_NUM_NAMES)
    return values


def card_numeric(card: dict) -> list[float]:
    return [
        safe_float(card.get("hp"), 400), safe_float(card.get("maxHp"), 400), damage(card),
        safe_float(len(as_list(card.get("energies"))), 10),
        safe_float(len(as_list(card.get("energyCards"))), 10),
        safe_float(len(as_list(card.get("tools"))), 5), float(bool(card.get("appearThisTurn"))),
        safe_float(len(as_list(card.get("preEvolution"))) + len(as_list(card.get("evolutionCards"))), 3),
    ]


def state_entities(observation: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    current, select = observation.get("current") or {}, observation.get("select") or {}
    me = int(current.get("yourIndex") or 0)
    rows: list[tuple[int, int, int, list[float], int]] = []

    def add(card: Any, zone_code: int, owner_code: int, priority: int) -> None:
        if not isinstance(card, dict):
            return
        card_id = int(card.get("id") or card.get("cardId") or 0)
        if not card_id:
            return
        rows.append((card_id, zone_code, owner_code, card_numeric(card), priority))

    def add_side(state: dict, owner_code: int) -> None:
        for card in as_list(state.get("active")):
            add(card, 4, owner_code, 0)
            if isinstance(card, dict):
                for attached in as_list(card.get("energyCards")): add(attached, 8, owner_code, 1)
                for attached in as_list(card.get("tools")): add(attached, 9, owner_code, 1)
        for card in as_list(state.get("bench")):
            add(card, 5, owner_code, 0)
            if isinstance(card, dict):
                for attached in as_list(card.get("energyCards")): add(attached, 8, owner_code, 1)
                for attached in as_list(card.get("tools")): add(attached, 9, owner_code, 1)
        for card in as_list(state.get("discard")): add(card, 3, owner_code, 4)

    mine, theirs = player(observation, me), player(observation, 1 - me)
    for card in as_list(mine.get("hand")): add(card, 2, 0, 2)
    add_side(mine, 0)
    add_side(theirs, 1)
    for card in as_list(current.get("stadium")): add(card, 7, 2, 1)
    if isinstance(current.get("stadium"), dict): add(current.get("stadium"), 7, 2, 1)
    for card in as_list(select.get("deck")): add(card, 13, 0, 3)
    for card in as_list(current.get("looking")): add(card, 12, 0, 3)
    rows.sort(key=lambda value: value[4])
    truncated = max(0, len(rows) - MAX_ENTITIES)
    rows = rows[:MAX_ENTITIES]
    card_ids = np.zeros(MAX_ENTITIES, dtype=np.uint16)
    zones = np.zeros(MAX_ENTITIES, dtype=np.uint8)
    owners = np.zeros(MAX_ENTITIES, dtype=np.uint8)
    numeric = np.zeros((MAX_ENTITIES, len(ENTITY_NUM_NAMES)), dtype=np.float16)
    for index, (card_id, zone_code, owner_code, values, _) in enumerate(rows):
        card_ids[index], zones[index], owners[index], numeric[index] = card_id, zone_code, owner_code + 1, values
    return card_ids, zones, owners, numeric, truncated


def raw_action_numeric(card: dict | None) -> tuple[float, float, float, float]:
    if not isinstance(card, dict):
        return 0.0, 0.0, 0.0, 0.0
    return (safe_float(card.get("hp"), 400), safe_float(card.get("maxHp"), 400), damage(card),
            safe_float(pokemon_energy_count(card), 10))


def action_features(observation: dict, option: dict | None) -> tuple[list[int], list[float]]:
    if option is None:
        # Categorical IDs are option_type+1; 18 is the synthetic STOP token,
        # distinct from the simulator's SPECIAL_CONDITION type 16 -> ID 17.
        return [18, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1.0] + [0.0] * (len(ACTION_NUM_NAMES) - 1)
    option_type = int(option.get("type") or 0)
    source, source_area, source_owner = source_entity(observation, option)
    target, target_area, target_owner = target_entity(observation, option)
    me = int((observation.get("current") or {}).get("yourIndex") or 0)
    if option_type in (12, 13):
        source = as_list(player(observation, me).get("active"))
        source = source[0] if source and isinstance(source[0], dict) else None
        source_area, source_owner = 4, me
    source_id = int((source or {}).get("id") or option.get("cardId") or 0)
    target_id = int((target or {}).get("id") or 0)
    source_role = 1 if source_owner == me else 2 if source_owner == 1 - me else 0
    target_role = 1 if target_owner == me else 2 if target_owner == 1 - me else 0
    categorical = [
        option_type + 1, source_id, target_id, int(source_area or 0), int(target_area or 0),
        source_role, target_role, int(option.get("attackId") or 0) % 4096,
        int(option.get("cardId") or 0), int(option.get("specialConditionType", -1)) + 1,
    ]
    src = raw_action_numeric(source)
    dst = raw_action_numeric(target)
    numeric = [0.0, safe_float(option.get("number"), 20), safe_float(option.get("count"), 20),
               *src, *dst]
    assert len(numeric) == len(ACTION_NUM_NAMES)
    return categorical, numeric


def submitted_deck(replay: dict, seat: int) -> list[int]:
    for step in replay.get("steps") or []:
        if isinstance(step, list) and seat < len(step) and isinstance(step[seat], dict):
            action = step[seat].get("action")
            if isinstance(action, list) and len(action) == 60 and all(isinstance(value, int) for value in action):
                return sorted(action)
    return []


class ShardBuilder:
    def __init__(self) -> None:
        self.data: dict[str, list] = {key: [] for key in (
            "global_num", "context", "select_type", "deck_cards", "entity_card", "entity_zone",
            "entity_owner", "entity_num", "action_cat", "action_num", "target", "option_index",
            "episode_id", "date", "seat", "winner", "went_first", "score_bin", "score",
            "is_alakazam", "legal_count", "original_action_count", "selection_ordinal",
            "target_option_type", "deck_hash",
        )}
        self.offsets = [0]
        self.stats = Counter()

    def add_group(self, decision: dict, meta: dict, deck: list[int], candidates: list[int | None],
                  target_keys: set[str], selection_ordinal: int) -> None:
        observation, select = decision["observation"], decision["select"]
        global_num = global_features(decision, selection_ordinal, selection_ordinal)
        entity_card, entity_zone, entity_owner, entity_num, truncated = state_entities(observation)
        action_cat, action_num, target, option_indices = [], [], [], []
        options = select.get("option") or []
        for candidate in candidates:
            option = None if candidate is None else options[candidate]
            cat, num = action_features(observation, option)
            action_cat.append(cat); action_num.append(num); option_indices.append(-1 if candidate is None else candidate)
            key = "__STOP__" if candidate is None else decision["semantic_keys"][candidate]
            target.append(int(key in target_keys))
        if not any(target):
            raise ValueError("Grouped target mask is empty")
        deck_array = np.zeros(60, dtype=np.uint16)
        deck_array[:min(60, len(deck))] = deck[:60]
        seat = decision["seat"]
        target_types = [action_cat[index][0] - 1 for index, value in enumerate(target) if value]
        deck_hash = int.from_bytes(hashlib.sha256(bytes(np.asarray(deck_array, dtype=np.uint16))).digest()[:8], "little")
        values = {
            "global_num": np.asarray(global_num, dtype=np.float16), "context": int(select.get("context") or 0),
            "select_type": int(select.get("type") or 0), "deck_cards": deck_array,
            "entity_card": entity_card, "entity_zone": entity_zone, "entity_owner": entity_owner,
            "entity_num": entity_num, "episode_id": int(meta["episode_id"]),
            "date": DATE_CODE[meta["source_date"]], "seat": seat,
            "winner": int(seat == int(meta["winner_seat"])),
            "went_first": int(seat == int(meta["first_player"])),
            "score_bin": int(meta["score_bin"]), "score": float(meta["avg_score"]),
            "is_alakazam": int(743 in deck), "legal_count": len(candidates),
            "original_action_count": len(decision["action"]), "selection_ordinal": selection_ordinal,
            "target_option_type": min(target_types) if target_types else 17, "deck_hash": deck_hash,
        }
        for key, value in values.items(): self.data[key].append(value)
        self.data["action_cat"].extend(action_cat); self.data["action_num"].extend(action_num)
        self.data["target"].extend(target); self.data["option_index"].extend(option_indices)
        self.offsets.append(len(self.data["target"]))
        self.stats.update(groups=1, actions=len(candidates), entity_truncations=truncated,
                          equivalence_targets=max(0, sum(target) - 1))

    def add_decision(self, decision: dict, meta: dict, deck: list[int]) -> None:
        remaining: list[int] = list(range(len(decision["semantics"])))
        minimum = int(decision["select"].get("minCount") or 0)
        for ordinal, selected_index in enumerate(decision["action"]):
            target_key = decision["semantic_keys"][selected_index]
            # Once minCount is satisfied, stopping is a legal competing action
            # even if the expert continues selecting more cards.
            candidates: list[int | None] = list(remaining)
            if ordinal >= minimum:
                candidates.append(None)
            self.add_group(decision, meta, deck, candidates, {target_key}, ordinal)
            remaining.remove(selected_index)
        maximum = int(decision["select"].get("maxCount") or 0)
        if len(decision["action"]) < maximum:
            self.add_group(decision, meta, deck, list(remaining) + [None], {"__STOP__"}, len(decision["action"]))

    def save(self, path: Path, extra: dict[str, Any]) -> dict[str, Any]:
        arrays = {
            "offsets": np.asarray(self.offsets, dtype=np.int64),
            "action_cat": np.asarray(self.data["action_cat"], dtype=np.uint16),
            "action_num": np.asarray(self.data["action_num"], dtype=np.float16),
            "target": np.asarray(self.data["target"], dtype=np.uint8),
            "option_index": np.asarray(self.data["option_index"], dtype=np.int16),
        }
        dtypes = {
            "global_num": np.float16, "context": np.uint8, "select_type": np.uint8,
            "deck_cards": np.uint16, "entity_card": np.uint16, "entity_zone": np.uint8,
            "entity_owner": np.uint8, "entity_num": np.float16, "episode_id": np.int64,
            "date": np.uint8, "seat": np.uint8, "winner": np.uint8, "went_first": np.uint8,
            "score_bin": np.uint8, "score": np.float16, "is_alakazam": np.uint8,
            "legal_count": np.uint16, "original_action_count": np.uint8,
            "selection_ordinal": np.uint8, "target_option_type": np.uint8, "deck_hash": np.uint64,
        }
        for key, dtype in dtypes.items(): arrays[key] = np.asarray(self.data[key], dtype=dtype)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        metadata = {"schema_version": SCHEMA_VERSION, "path": str(path.resolve()),
                    "size_bytes": path.stat().st_size, **dict(self.stats), **extra}
        path.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return metadata


def load_manifest(path: Path, phases: set[str]) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    result = []
    for row in rows:
        if row["phase"] not in phases:
            continue
        for key in ("score_bin", "length_bin", "steps", "winner_seat", "first_player"):
            row[key] = int(row[key])
        row["avg_score"] = float(row["avg_score"])
        result.append(row)
    return result


def enrich_runtime_metadata(meta: dict[str, Any], replay: dict[str, Any]) -> None:
    """Fill replay-derived fields for scale manifests without a separate raw scan."""
    if int(meta.get("winner_seat", -1)) < 0:
        rewards = replay.get("rewards") or []
        numeric = [(seat, reward) for seat, reward in enumerate(rewards[:2])
                   if isinstance(reward, (int, float))]
        meta["winner_seat"] = max(numeric, key=lambda item: item[1])[0] if numeric else -1
    if int(meta.get("first_player", -1)) < 0:
        for step in replay.get("steps") or []:
            for entry in step if isinstance(step, list) else []:
                observation = entry.get("observation") if isinstance(entry, dict) else None
                current = observation.get("current") if isinstance(observation, dict) else None
                value = current.get("firstPlayer") if isinstance(current, dict) else None
                if value in (0, 1):
                    meta["first_player"] = int(value)
                    break
            if int(meta.get("first_player", -1)) >= 0:
                break
    if int(meta.get("steps", -1)) < 0:
        meta["steps"] = len(replay.get("steps") or [])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", action="append", required=True)
    parser.add_argument("--episodes-per-shard", type=int, default=50)
    parser.add_argument("--limit-episodes", type=int)
    args = parser.parse_args()
    rows = load_manifest(args.manifest, set(args.phase))
    if args.limit_episodes: rows = rows[:args.limit_episodes]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_metadata, corpus_stats = [], Counter()
    start = time.time()
    builder, shard_rows, shard_index = ShardBuilder(), [], 0
    for episode_index, meta in enumerate(rows, start=1):
        path = Path(meta["json_path"])
        with path.open(encoding="utf-8") as source: replay = json.load(source)
        enrich_runtime_metadata(meta, replay)
        decks = [submitted_deck(replay, seat) for seat in (0, 1)]
        decisions = list(iter_decisions(replay))
        before = builder.stats["groups"]
        for decision in decisions:
            builder.add_decision(decision, meta, decks[decision["seat"]])
        builder.stats.update(episodes=1, source_bytes=path.stat().st_size,
                             replay_decisions=len(decisions), unlabeled_roots=max(0, 1 if replay.get("statuses") and "TIMEOUT" in replay["statuses"] else 0))
        shard_rows.append(meta["episode_id"])
        if len(shard_rows) == args.episodes_per_shard or episode_index == len(rows):
            output = args.output_dir / f"shard_{shard_index:04d}.npz"
            metadata = builder.save(output, {"episode_ids": shard_rows, "phase_set": sorted(set(args.phase))})
            shard_metadata.append(metadata); corpus_stats.update(builder.stats)
            print(f"saved {output.name}: episodes={len(shard_rows)} groups={metadata['groups']} actions={metadata['actions']}", flush=True)
            builder, shard_rows, shard_index = ShardBuilder(), [], shard_index + 1
    total_shard_bytes = sum(item["size_bytes"] for item in shard_metadata)
    corpus = {
        "schema_version": SCHEMA_VERSION, "manifest": str(args.manifest.resolve()),
        "phases": sorted(set(args.phase)), "episodes": len(rows), "shards": shard_metadata,
        "stats": dict(corpus_stats), "global_num_names": GLOBAL_NUM_NAMES,
        "entity_num_names": ENTITY_NUM_NAMES, "action_num_names": ACTION_NUM_NAMES,
        "max_entities": MAX_ENTITIES, "total_shard_bytes": total_shard_bytes,
        "raw_source_bytes": corpus_stats["source_bytes"],
        "compression_ratio_shard_over_raw": total_shard_bytes / max(1, corpus_stats["source_bytes"]),
        "shard_bytes_per_episode": total_shard_bytes / max(1, len(rows)),
        "elapsed_seconds": time.time() - start,
    }
    corpus_path = args.output_dir / "corpus_manifest.json"
    corpus_path.write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: corpus[key] for key in ("episodes", "stats", "total_shard_bytes", "raw_source_bytes",
                                                   "compression_ratio_shard_over_raw", "shard_bytes_per_episode",
                                                   "elapsed_seconds")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
