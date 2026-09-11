"""Safe runtime for Full BC and margin-gated hybrid policies.

The learned scorer never sees an option ordinal.  It scores only the currently
legal semantic actions and handles multi-card SELECT roots autoregressively.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch

from encode_bc_shards import (
    MAX_ENTITIES, action_features, global_features, state_entities,
)
from replay_semantics import semantic_key, semantic_option
from train_grouped_bc import CARD_VOCAB, TRAIN_ENTITY_LIMIT, scorer_from_config


Fallback = Callable[[dict], list[int]]
ATTACK_OPTION_TYPE = 13
END_OPTION_TYPE = 14


@dataclass
class DecisionDiagnostics:
    action: list[int]
    margins: list[float]
    probabilities: list[float]
    min_margin: float
    elapsed_ms: float
    context: int
    legal_count: int
    used_fallback: bool = False
    fallback_reason: str | None = None
    ood: bool = False
    disaster_veto: bool = False
    slow: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class GlobalBCPolicy:
    def __init__(self, checkpoint: Path | str, deck: Iterable[int], device: str = "cpu",
                 latency_budget_ms: float = 2000.0) -> None:
        payload = torch.load(Path(checkpoint), map_location=device, weights_only=False)
        config = payload["config"]
        # Must honour the architecture the checkpoint was saved with: building a
        # plain scorer for a cross-option checkpoint would drop the block and
        # silently ship a different policy than the one that was validated.
        self.model = scorer_from_config(config).to(device)
        self.model.load_state_dict(payload["model"])
        self.model.eval()
        self.device = torch.device(device)
        self.entity_limit = int(config.get("entity_limit", TRAIN_ENTITY_LIMIT))
        if not 1 <= self.entity_limit <= MAX_ENTITIES:
            raise ValueError(
                f"checkpoint entity_limit={self.entity_limit} outside [1, {MAX_ENTITIES}]"
            )
        self.deck = np.zeros(60, dtype=np.int64)
        self.set_deck(deck)
        # Soft: only sets DecisionDiagnostics.slow for auditing. It must never
        # gate whether the computed action is used -- see decide().
        self.latency_budget_ms = latency_budget_ms

    def set_deck(self, deck: Iterable[int]) -> None:
        self.deck.fill(0)
        values = list(deck)[:60]
        self.deck[:len(values)] = values

    @staticmethod
    def _decision(observation: dict) -> dict:
        select = observation.get("select") or {}
        options = select.get("option") or []
        semantics = [semantic_option(observation, option) for option in options]
        current = observation.get("current") or {}
        return {
            "observation": observation, "select": select,
            "seat": int(current.get("yourIndex") or 0), "action": [],
            "semantics": semantics,
            "semantic_keys": [semantic_key(value) for value in semantics],
        }

    @staticmethod
    def _ood(decision: dict) -> bool:
        context = int(decision["select"].get("context") or 0)
        select_type = int(decision["select"].get("type") or 0)
        if not 0 <= context < 64 or not 0 <= select_type < 16:
            return True
        for option in decision["select"].get("option") or []:
            cat, _ = action_features(decision["observation"], option)
            if any(value < 0 for value in cat) or cat[0] >= 20:
                return True
            if any(cat[index] >= CARD_VOCAB for index in (1, 2, 8)):
                return True
        return False

    def _batch(self, decision: dict, candidates: list[int | None], ordinal: int) -> dict[str, torch.Tensor]:
        observation = decision["observation"]
        entity_card, entity_zone, entity_owner, entity_num, _ = state_entities(observation)
        cats, nums = zip(*(action_features(observation, None if value is None else
                                           decision["select"]["option"][value])
                           for value in candidates))
        return {
            "global_num": torch.tensor([global_features(decision, ordinal, ordinal)], dtype=torch.float32,
                                       device=self.device),
            "context": torch.tensor([int(decision["select"].get("context") or 0)], dtype=torch.long,
                                    device=self.device),
            "select_type": torch.tensor([int(decision["select"].get("type") or 0)], dtype=torch.long,
                                        device=self.device),
            "deck_cards": torch.tensor(self.deck[None], dtype=torch.long, device=self.device),
            "entity_card": torch.tensor(entity_card[None, :self.entity_limit], dtype=torch.long,
                                        device=self.device),
            "entity_zone": torch.tensor(entity_zone[None, :self.entity_limit], dtype=torch.long,
                                        device=self.device),
            "entity_owner": torch.tensor(entity_owner[None, :self.entity_limit], dtype=torch.long,
                                         device=self.device),
            "entity_num": torch.tensor(entity_num[None, :self.entity_limit], dtype=torch.float32,
                                        device=self.device),
            "action_cat": torch.tensor([cats], dtype=torch.long, device=self.device),
            "action_num": torch.tensor([nums], dtype=torch.float32, device=self.device),
            "legal_mask": torch.ones((1, len(candidates)), dtype=torch.bool, device=self.device),
        }

    def _score(self, decision: dict, candidates: list[int | None], ordinal: int) -> tuple[int, float, float]:
        batch = self._batch(decision, candidates, ordinal)
        with torch.inference_mode():
            probabilities = self.model(batch).softmax(-1)[0]
        order = torch.argsort(probabilities, descending=True)
        best = int(order[0])
        second = float(probabilities[order[1]]) if len(candidates) > 1 else 0.0
        return best, float(probabilities[best] - second), float(probabilities[best])

    def decide(self, observation: dict) -> DecisionDiagnostics:
        started = time.perf_counter()
        decision = self._decision(observation)
        options = decision["select"].get("option") or []
        context = int(decision["select"].get("context") or 0)
        if not options or self._ood(decision):
            elapsed = (time.perf_counter() - started) * 1000
            return DecisionDiagnostics([], [], [], 0.0, elapsed, context, len(options),
                                       ood=True, fallback_reason="empty_or_ood")
        minimum = int(decision["select"].get("minCount") or 0)
        maximum = int(decision["select"].get("maxCount") or 0)
        if maximum <= 0:
            maximum = max(minimum, 1)
        remaining: list[int] = list(range(len(options)))
        selected: list[int] = []
        margins: list[float] = []
        probabilities: list[float] = []
        while remaining and len(selected) < maximum:
            candidates: list[int | None] = list(remaining)
            if len(selected) >= minimum:
                candidates.append(None)
            best, margin, probability = self._score(decision, candidates, len(selected))
            choice = candidates[best]
            margins.append(margin); probabilities.append(probability)
            if choice is None:
                break
            selected.append(choice); remaining.remove(choice)
        elapsed = (time.perf_counter() - started) * 1000
        types = [int(options[index].get("type") or 0) for index in selected]
        legal_types = {int(option.get("type") or 0) for option in options}
        disaster = bool(types and set(types) == {END_OPTION_TYPE} and ATTACK_OPTION_TYPE in legal_types)
        # A slow decision is REPORTED, never discarded. Throwing the action away
        # after computing it cannot refund the time already spent -- it only
        # replaces a model move with "first legal option". On 170 online replays
        # this fired 144 times out of 15,829 decisions and 139 of those took
        # option 0; among setup-Active roots the slow decisions picked option 0
        # 26/27 times vs 30/47 for fast ones (Fisher p=0.00156), i.e. it was
        # silently randomising opening play, most likely on first-inference
        # warm-up. The host budget is 600 s TOTAL per game and we measure 3.1 s
        # median, so there was never anything to protect.
        return DecisionDiagnostics(
            selected, margins, probabilities, min(margins, default=0.0), elapsed,
            context, len(options), fallback_reason=None,
            disaster_veto=disaster, slow=elapsed > self.latency_budget_ms,
        )


class SafeFullBC:
    def __init__(self, policy: GlobalBCPolicy, fallback: Fallback) -> None:
        self.policy, self.fallback = policy, fallback

    def __call__(self, observation: dict) -> list[int]:
        result = self.policy.decide(observation)
        if result.ood or result.disaster_veto or result.fallback_reason:
            return self.fallback(observation)
        return result.action


class GatedHybrid:
    def __init__(self, policy: GlobalBCPolicy, fallback: Fallback, margin_threshold: float,
                 allowed_contexts: Iterable[int]) -> None:
        self.policy, self.fallback = policy, fallback
        self.margin_threshold = float(margin_threshold)
        self.allowed_contexts = set(map(int, allowed_contexts))
        self.last_diagnostics: DecisionDiagnostics | None = None

    def __call__(self, observation: dict) -> list[int]:
        fallback_action = self.fallback(observation)
        result = self.policy.decide(observation)
        reason = None
        if result.ood: reason = "ood"
        elif result.disaster_veto: reason = "hard_guard"
        elif result.fallback_reason: reason = result.fallback_reason
        elif result.context not in self.allowed_contexts: reason = "context_not_covered"
        elif result.min_margin < self.margin_threshold: reason = "low_margin"
        if reason is not None:
            result.used_fallback = True; result.fallback_reason = reason
            result.action = list(fallback_action)
        self.last_diagnostics = result
        return result.action
