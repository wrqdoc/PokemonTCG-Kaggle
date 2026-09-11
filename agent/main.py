"""Pure BC + a narrow, exact, noise-robust turn search.

Relationship to the two searches that were already tried and rejected offline:

* ``main_forward_search_v52`` ranked actions by a hand-written state score.
* The 2026-08-03 experiment ranked them by the value net.
  Both were beaten by BC alone by 15-34 pp on expert agreement.

This one deliberately does NOT rank actions by any evaluator. It only ever
overrides BC when the simulator itself proves a strictly better outcome under an
objective the game defines: **prizes taken this turn**. Three design rules, each
one earned from a specific failure:

1. **BC is the default.** The prior is strong (67-72% top-1 on held-out human
   MAIN roots); nothing overrides it without proof.
2. **Robust, not max.** A line counts only if it beats BC's line under EVERY
   determinization, not the luckiest one. Taking the max over sampled rollouts
   manufactured a +19.7% mirage in the offline probe -- 73% of it was pure
   max-of-noise (reports/PRIZE_LINE_SEARCH_PROBE_20260803.md). Min-over-D is
   immune to that by construction: a line that only works when you draw the
   right card fails in some determinization and is discarded, while "I already
   have the energy and the attacker, attacking now KOs" holds in all of them.
3. **Hard time budgets.** The host gives 600 s TOTAL per game on 1.6 vCPUs with
   no per-turn increment. Pure BC measures at 3.1 s median / 6.9 s worst per
   game, so there is room -- but a pathological turn must never be able to eat
   the pool. Both a per-turn and a per-game budget degrade to plain BC.

Any exception anywhere degrades to plain BC.
"""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

import runtime_policy
from runtime_policy import ATTACK_OPTION_TYPE, END_OPTION_TYPE, GlobalBCPolicy

_root = Path(runtime_policy.__file__).parent


def _read_deck(path: Path) -> list[int]:
    with path.open(encoding="utf-8") as handle:
        return [int(line.strip()) for line in handle if line.strip()]


MY_DECK = _read_deck(_root / "deck.csv")
_policy = GlobalBCPolicy(_root / "model.pt", MY_DECK, device="cpu",
                         latency_budget_ms=2000.0)

MAIN_CONTEXT = 0
FILLER_ENERGY = 3
BASIC_FALLBACK = 305
TOP_K = 4                    # opening options re-examined per root
NODE_BUDGET = 140            # simulator steps per determinization per root
DETERMINIZATIONS = 2         # a line must win under all of them
FOLLOWUP_CAP = 40            # max simulator steps when finishing a turn
GAME_SEARCH_BUDGET_S = 180.0  # << 600 s pool, leaves ~3x margin at 1.6 vCPU
TURN_SEARCH_BUDGET_S = 4.0

_SEARCH_ENABLED = True
_GAME_TIME = 0.0
_TURN = None
_TURN_TIME = 0.0

try:  # the native simulator ships with this package; BC-only builds omit it
    from cg.api import (
        search_begin, search_end, search_release, search_step, to_observation_class,
    )
except Exception:  # noqa: BLE001
    _SEARCH_ENABLED = False


# --------------------------------------------------------------------------


def _normalize(value):
    """asdict() materialises optional fields as None; replay JSON omits them.

    replay_semantics reads ``option.get("playerIndex", me)`` and a present-but-None
    key turns that into int(None). Drop None-valued KEYS, keep None list ELEMENTS
    (those are the simulator's face-down markers).
    """
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [None if v is None else _normalize(v) for v in value]
    return value


def _visible_ids(state: dict) -> list[int]:
    ids: list[int] = []
    for card in state.get("hand") or []:
        ids.append(int(card["id"]))
    for card in state.get("discard") or []:
        ids.append(int(card["id"]))
    for pokemon in (state.get("active") or []) + (state.get("bench") or []):
        if not pokemon:
            continue
        ids.append(int(pokemon["id"]))
        for key in ("preEvolution", "energyCards", "tools"):
            ids.extend(int(c["id"]) for c in pokemon.get(key) or [])
    for card in state.get("prize") or []:
        if card:
            ids.append(int(card["id"]))
    return ids


def _hidden_guess(full_deck: list[int], state: dict, include_hand: bool, rotate: int):
    remaining = Counter(full_deck)
    for card_id in _visible_ids(state):
        if remaining[card_id] > 0:
            remaining[card_id] -= 1
    pool = list(remaining.elements())
    if pool and rotate:
        offset = rotate % len(pool)
        pool = pool[offset:] + pool[:offset]
    hand_n = int(state.get("handCount") or 0) if include_hand else 0
    prize_n = len(state.get("prize") or [])
    deck_n = int(state.get("deckCount") or 0)
    pool.extend([FILLER_ENERGY] * max(0, hand_n + prize_n + deck_n - len(pool)))
    return (pool[hand_n + prize_n:hand_n + prize_n + deck_n],
            pool[hand_n:hand_n + prize_n],
            pool[:hand_n])


def _legal_fallback(observation: dict) -> list[int]:
    select = observation.get("select") or {}
    options = select.get("option") or []
    if not options:
        return []
    minimum = int(select.get("minCount") or 1)
    maximum = int(select.get("maxCount") or minimum) or minimum
    take = max(1, min(max(1, minimum), max(1, maximum), len(options)))
    types = [int(option.get("type") or 0) for option in options]
    if ATTACK_OPTION_TYPE in types:
        preferred = [i for i, t in enumerate(types) if t != END_OPTION_TYPE]
        if len(preferred) >= take:
            return preferred[:take]
    return list(range(take))


def _bc_action(observation: dict) -> list[int]:
    result = _policy.decide(observation)
    # Latency is never a reason to discard a computed action -- see runtime_policy.
    if result.action and not result.ood and not result.disaster_veto:
        return list(result.action)
    return _legal_fallback(observation)


def _prizes_left(observation: dict, seat: int) -> int:
    players = (observation.get("current") or {}).get("players") or []
    return len(players[seat].get("prize") or []) if seat < len(players) else 6


def _turn_over(observation: dict, root_turn: int) -> bool:
    current = observation.get("current") or {}
    if observation.get("select") is None or int(current.get("result", -1)) >= 0:
        return True
    return int(current.get("turn") or 0) != root_turn


def _finish_turn(state, root_turn: int, seat: int, start_prizes: int) -> int:
    """Let BC play out the rest of the turn; return prizes taken."""
    created: list[int] = []
    observation = _normalize(asdict(state.observation))
    try:
        for _ in range(FOLLOWUP_CAP):
            if _turn_over(observation, root_turn):
                break
            action = _bc_action(observation)
            if not action:
                break
            state = search_step(state.searchId, action)
            created.append(state.searchId)
            observation = _normalize(asdict(state.observation))
        return start_prizes - _prizes_left(observation, seat)
    finally:
        for search_id in reversed(created):
            search_release(search_id)


def _candidates(observation: dict) -> list[int]:
    """BC's top-k option indices, ties broken toward the lowest index."""
    select = observation.get("select") or {}
    options = select.get("option") or []
    decision = _policy._decision(observation)
    batch = _policy._batch(decision, list(range(len(options))), 0)
    import torch
    with torch.inference_mode():
        scores = _policy.model(batch)[0].float().cpu().numpy()
    return [int(i) for i in np.argsort(-scores, kind="stable")[:TOP_K]]


class _TurnDFS:
    """Bounded DFS over MAIN action sequences for the rest of this turn.

    Re-ranking only BC's top-k FIRST actions (and letting BC finish) fires zero
    times -- measured 0/121 roots under every aggregation rule, because a single
    swapped opening followed by BC's own play converges back to BC's own line.
    The offline probe only ever found candidate lines when it branched at EVERY
    MAIN node and evaluated ~181 complete turns per root. So the runtime has to
    branch too, not just re-order the opening.

    Returns, per opening option index, the best prize count reachable under this
    determinization.
    """

    def __init__(self, seat: int, root_turn: int, start_prizes: int, budget: int) -> None:
        self.seat, self.root_turn = seat, root_turn
        self.start_prizes, self.budget = start_prizes, budget
        self.nodes = 0
        self.best: dict[int, int] = {}

    def _advance(self, state):
        created: list[int] = []
        observation = _normalize(asdict(state.observation))
        while not _turn_over(observation, self.root_turn):
            if int((observation.get("select") or {}).get("context") or 0) == MAIN_CONTEXT:
                break
            if self.nodes >= self.budget:
                break
            action = _bc_action(observation)
            if not action:
                break
            state = search_step(state.searchId, action)
            self.nodes += 1
            created.append(state.searchId)
            observation = _normalize(asdict(state.observation))
        return state, observation, created

    def _record(self, opening: int, state, observation: dict) -> None:
        # Every leaf must be a COMPLETED turn. Scoring a truncated line scores it
        # below BC's own line, which is impossible since BC's line is in the space.
        if not _turn_over(observation, self.root_turn):
            taken = _finish_turn(state, self.root_turn, self.seat, self.start_prizes)
        else:
            taken = self.start_prizes - _prizes_left(observation, self.seat)
        if taken > self.best.get(opening, -1):
            self.best[opening] = taken

    def run(self, state, opening: int) -> None:
        state, observation, created = self._advance(state)
        try:
            if _turn_over(observation, self.root_turn) or self.nodes >= self.budget:
                self._record(opening, state, observation)
                return
            options = (observation.get("select") or {}).get("option") or []
            for index in range(len(options)):
                if self.nodes >= self.budget:
                    self._record(opening, state, observation)
                    break
                child_ids: list[int] = []
                try:
                    child = search_step(state.searchId, [index])
                    self.nodes += 1
                    child_ids.append(child.searchId)
                    self.run(child, opening)
                except Exception:  # noqa: BLE001
                    pass
                finally:
                    for search_id in reversed(child_ids):
                        search_release(search_id)
        finally:
            for search_id in reversed(created):
                search_release(search_id)


def _should_search(observation: dict, root_turn: int) -> bool:
    """Cost control: a full DFS at every MAIN root does not fit 600 s on 1.6 vCPU.

    Search where prizes can plausibly move: the turn's first decision (the whole
    turn is still ahead) or any root where an attack is already legal.
    """
    if _TURN != root_turn:
        return True
    options = (observation.get("select") or {}).get("option") or []
    return any(int(o.get("type") or 0) == ATTACK_OPTION_TYPE for o in options)


def _search_override(observation: dict) -> list[int] | None:
    select = observation.get("select") or {}
    if int(select.get("context") or 0) != MAIN_CONTEXT:
        return None
    options = select.get("option") or []
    if len(options) < 2:
        return None
    current = observation.get("current") or {}
    players = current.get("players") or []
    if len(players) != 2:
        return None
    me = int(current.get("yourIndex") or 0)
    root_turn = int(current.get("turn") or 0)
    start_prizes = _prizes_left(observation, me)
    if start_prizes == 0:
        return None

    if not _should_search(observation, root_turn):
        return None
    candidates = _candidates(observation)
    if len(candidates) < 2:
        return None
    scores = {index: None for index in candidates}

    for draw in range(DETERMINIZATIONS):
        my_deck, my_prize, _ = _hidden_guess(MY_DECK, players[me], False, draw * 7 + 1)
        # We do not know the opponent's list; our own deck is the only prior a
        # runtime agent has. Offline this cost ~1 pp vs an oracle opponent deck.
        op_deck, op_prize, op_hand = _hidden_guess(MY_DECK, players[1 - me], True, draw * 11 + 3)
        active = players[1 - me].get("active") or []
        op_active = [BASIC_FALLBACK] if (active and active[0] is None) else []
        root = search_begin(to_observation_class(observation), my_deck, my_prize,
                            op_deck, op_prize, op_hand, op_active)
        try:
            dfs = _TurnDFS(me, root_turn, start_prizes, NODE_BUDGET)
            for index in candidates:
                child_ids: list[int] = []
                try:
                    child = search_step(root.searchId, [index])
                    child_ids.append(child.searchId)
                    dfs.run(child, index)
                except Exception:  # noqa: BLE001
                    dfs.best.setdefault(index, -1)
                finally:
                    for search_id in reversed(child_ids):
                        search_release(search_id)
            for index in candidates:
                taken = dfs.best.get(index, -1)
                previous = scores[index]
                # MIN over determinizations -- the robustness rule. See module docstring.
                scores[index] = taken if previous is None else min(previous, taken)
        finally:
            search_release(root.searchId)
            search_end()

    baseline = scores[candidates[0]]
    best = max(candidates, key=lambda i: (scores[i], -candidates.index(i)))
    if scores[best] > baseline and best != candidates[0]:
        return [best]
    return None


def agent(observation: dict) -> list[int]:
    global _GAME_TIME, _TURN, _TURN_TIME
    if observation.get("select") is None:
        _GAME_TIME, _TURN, _TURN_TIME = 0.0, None, 0.0
        return list(MY_DECK)

    bc_action = _bc_action(observation)
    if not _SEARCH_ENABLED:
        return bc_action

    turn = int((observation.get("current") or {}).get("turn") or 0)
    if turn != _TURN:
        _TURN, _TURN_TIME = turn, 0.0
    if _GAME_TIME >= GAME_SEARCH_BUDGET_S or _TURN_TIME >= TURN_SEARCH_BUDGET_S:
        return bc_action

    started = time.perf_counter()
    try:
        override = _search_override(observation)
    except Exception:  # noqa: BLE001
        override = None
    elapsed = time.perf_counter() - started
    _GAME_TIME += elapsed
    _TURN_TIME += elapsed
    return override if override else bc_action


__all__ = ["agent", "MY_DECK"]
