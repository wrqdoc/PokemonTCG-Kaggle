"""Train/evaluate a dynamic legal-action scorer on grouped BC shards."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn


CARD_VOCAB = 4096
ATTACK_VOCAB = 4096
# Backward-compatible default used by legacy value/helpfulness components.
# The grouped-BC trainer and policy runtime use the checkpoint-specific value.
TRAIN_ENTITY_LIMIT = 32
MAX_SHARD_ENTITIES = 96
EPISODE_SAMPLERS = ("legacy", "phase_target_v1")
# global_num stores turn / 200 and own deck count / 60 (encode_bc_shards.py).
# Turn 10 is approximately the fifth turn for a seat and matches the frozen
# entity-limit evaluation's late-game boundary.
LATE_TURN_SCALED = 9.5 / 200.0
DECK_BANDS = (
    ("deck_7_10", 6.5 / 60.0, 10.5 / 60.0),
    ("deck_4_6", 3.5 / 60.0, 6.5 / 60.0),
    ("deck_0_3", -0.5 / 60.0, 3.5 / 60.0),
)


def scorer_from_config(config: dict) -> "DynamicActionScorer":
    """Build a scorer with the architecture the checkpoint was actually saved with.

    Checkpoints predating cross-option attention have no `cross_option_heads`
    key, so the default of 0 reproduces the original architecture exactly.
    """
    return DynamicActionScorer(
        config["global_dim"], config["entity_num_dim"], config["action_num_dim"],
        config["hidden"], cross_option_heads=int(config.get("cross_option_heads", 0)),
        cross_option_layers=int(config.get("cross_option_layers", 1)),
        cross_option_ff=bool(config.get("cross_option_ff", False)),
    )


class DynamicActionScorer(nn.Module):
    """Scores each legal option for one decision group.

    `cross_option_heads` > 0 adds attention ACROSS the candidate options. Without
    it every option is embedded and scored in isolation -- the model never sees
    the other options it is choosing between, so it can only learn a static
    per-option value. The 2026-08-05 holdout decomposition is the shape that
    predicts: equivalence-class accuracy falls monotonically with how many
    options must be compared (forced 100% -> 2-3 87.8% -> 4-8 72.7% -> 9+ 66.5%),
    and 93% of the error mass sits at >=4 options. "Is this the best available?"
    is inherently comparative; independent scoring cannot express it.

    The block is a residual gated by a ZERO-INITIALISED scalar, so a freshly
    constructed model is numerically identical to the old architecture. That is
    what makes `--resume` from a pre-attention checkpoint safe: training starts
    from exactly the previous policy and only opens the comparative path as the
    gate moves off zero.
    """

    def __init__(self, global_dim: int, entity_num_dim: int, action_num_dim: int,
                 hidden: int = 192, card_dim: int = 64,
                 cross_option_heads: int = 0, cross_option_layers: int = 1,
                 cross_option_ff: bool = False) -> None:
        super().__init__()
        self.card = nn.Embedding(CARD_VOCAB, card_dim, padding_idx=0)
        self.zone = nn.Embedding(16, 16, padding_idx=0)
        self.owner = nn.Embedding(4, 8, padding_idx=0)
        self.context = nn.Embedding(64, 24)
        self.select_type = nn.Embedding(16, 12)
        self.option_type = nn.Embedding(20, 24, padding_idx=0)
        self.attack = nn.Embedding(ATTACK_VOCAB, 24, padding_idx=0)
        self.special = nn.Embedding(16, 8, padding_idx=0)
        self.entity_mlp = nn.Sequential(
            nn.Linear(card_dim + 16 + 8 + entity_num_dim, hidden), nn.GELU(),
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(global_dim + 24 + 12 + card_dim + hidden * 2, hidden * 2),
            nn.GELU(), nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, hidden), nn.GELU(),
        )
        action_input = 24 + card_dim * 3 + 16 * 2 + 8 * 2 + 24 + 8 + action_num_dim
        self.action_mlp = nn.Sequential(
            nn.Linear(action_input, hidden * 2), nn.GELU(), nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden), nn.GELU(),
        )
        self.state_proj = nn.Linear(hidden, hidden)
        self.action_proj = nn.Linear(hidden, hidden)
        self.interaction = nn.Sequential(
            nn.Linear(hidden * 4, hidden), nn.GELU(), nn.Linear(hidden, 1),
        )
        self.cross_option_heads = int(cross_option_heads)
        if self.cross_option_heads:
            if hidden % self.cross_option_heads:
                raise ValueError(
                    f"hidden={hidden} not divisible by cross_option_heads={self.cross_option_heads}"
                )
            # One gated sub-layer per entry: attention, then (optionally) a
            # feed-forward. The 2026-08-05 single-layer, attention-only block was
            # only half a transformer block -- it could route information between
            # options but had no per-option capacity to transform what it routed.
            # Every gate is zero-initialised, so ANY depth still starts as an
            # exact no-op and resumes byte-identically from a plain checkpoint.
            self.cross_norm = nn.ModuleList()
            self.cross_attn = nn.ModuleList()
            self.cross_gate = nn.ParameterList()
            self.cross_ff_norm = nn.ModuleList()
            self.cross_ff = nn.ModuleList()
            self.cross_ff_gate = nn.ParameterList()
            for _ in range(max(1, int(cross_option_layers))):
                self.cross_norm.append(nn.LayerNorm(hidden))
                self.cross_attn.append(nn.MultiheadAttention(
                    hidden, self.cross_option_heads, batch_first=True))
                self.cross_gate.append(nn.Parameter(torch.zeros(1)))
                if cross_option_ff:
                    self.cross_ff_norm.append(nn.LayerNorm(hidden))
                    self.cross_ff.append(nn.Sequential(
                        nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, hidden)))
                    self.cross_ff_gate.append(nn.Parameter(torch.zeros(1)))
            self._register_load_state_dict_pre_hook(self._upgrade_legacy_cross_keys)

    @staticmethod
    def _upgrade_legacy_cross_keys(state_dict, prefix, *_args) -> None:
        """Map the 2026-08-05 single-layer key names onto the ModuleList layout.

        That version stored `cross_norm.weight` / `cross_attn.in_proj_weight` /
        `cross_gate`; depth support turned those into `cross_norm.0.weight` etc.
        `xopt803` and `xopt805` -- one of which is the shipped champion -- were
        trained with the old names, so without this hook loading them would fail
        (or, worse, silently drop the block if anyone reached for strict=False).
        """
        legacy = {
            f"{prefix}cross_gate": f"{prefix}cross_gate.0",
            f"{prefix}cross_norm.weight": f"{prefix}cross_norm.0.weight",
            f"{prefix}cross_norm.bias": f"{prefix}cross_norm.0.bias",
            f"{prefix}cross_attn.in_proj_weight": f"{prefix}cross_attn.0.in_proj_weight",
            f"{prefix}cross_attn.in_proj_bias": f"{prefix}cross_attn.0.in_proj_bias",
            f"{prefix}cross_attn.out_proj.weight": f"{prefix}cross_attn.0.out_proj.weight",
            f"{prefix}cross_attn.out_proj.bias": f"{prefix}cross_attn.0.out_proj.bias",
        }
        for old, new in legacy.items():
            if old in state_dict and new not in state_dict:
                state_dict[new] = state_dict.pop(old)

    def encode_state(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        entity = torch.cat([
            self.card(batch["entity_card"].clamp_max(CARD_VOCAB - 1)),
            self.zone(batch["entity_zone"]), self.owner(batch["entity_owner"]),
            batch["entity_num"],
        ], dim=-1)
        entity = self.entity_mlp(entity)
        mask = batch["entity_card"].ne(0).unsqueeze(-1)
        mean_pool = (entity * mask).sum(1) / mask.sum(1).clamp_min(1)
        max_pool = entity.masked_fill(~mask, -1e4).max(1).values
        max_pool = torch.where(mask.any(1), max_pool, torch.zeros_like(max_pool))
        deck_mask = batch["deck_cards"].ne(0).unsqueeze(-1)
        deck = self.card(batch["deck_cards"].clamp_max(CARD_VOCAB - 1))
        deck = (deck * deck_mask).sum(1) / deck_mask.sum(1).clamp_min(1)
        features = torch.cat([
            batch["global_num"], self.context(batch["context"]),
            self.select_type(batch["select_type"]), deck, mean_pool, max_pool,
        ], dim=-1)
        return self.global_mlp(features)

    def encode_action(self, cat: torch.Tensor, num: torch.Tensor) -> torch.Tensor:
        return self.action_mlp(torch.cat([
            self.option_type(cat[..., 0]), self.card(cat[..., 1].clamp_max(CARD_VOCAB - 1)),
            self.card(cat[..., 2].clamp_max(CARD_VOCAB - 1)), self.zone(cat[..., 3]),
            self.zone(cat[..., 4]), self.owner(cat[..., 5]), self.owner(cat[..., 6]),
            self.attack(cat[..., 7].clamp_max(ATTACK_VOCAB - 1)),
            self.card(cat[..., 8].clamp_max(CARD_VOCAB - 1)), self.special(cat[..., 9]), num,
        ], dim=-1))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        state = self.encode_state(batch)
        action = self.encode_action(batch["action_cat"], batch["action_num"])
        if self.cross_option_heads:
            # Let each option see the others it is competing against. Padding
            # slots are excluded as keys; every group has >=1 legal option, so
            # no row is fully masked (which would produce NaN).
            pad = ~batch["legal_mask"]
            for index in range(len(self.cross_attn)):
                normed = self.cross_norm[index](action)
                attended, _ = self.cross_attn[index](
                    normed, normed, normed, key_padding_mask=pad, need_weights=False)
                action = action + self.cross_gate[index] * attended
                if index < len(self.cross_ff):
                    action = action + self.cross_ff_gate[index] * self.cross_ff[index](
                        self.cross_ff_norm[index](action))
        state_expanded = state.unsqueeze(1).expand_as(action)
        s, a = self.state_proj(state_expanded), self.action_proj(action)
        dot = (s * a).sum(-1, keepdim=True) / math.sqrt(s.shape[-1])
        residual = self.interaction(torch.cat([s, a, s * a, torch.abs(s - a)], dim=-1))
        logits = (dot + residual).squeeze(-1)
        return logits.masked_fill(~batch["legal_mask"], -1e9)


def shard_paths(corpus_dirs: Path | list[Path]) -> list[Path]:
    if isinstance(corpus_dirs, Path):
        corpus_dirs = [corpus_dirs]
    # Preserve caller corpus order so an N->M learning curve reuses the exact
    # capped N-episode RNG prefix before sampling the appended episodes.
    return [path for corpus_dir in corpus_dirs for path in sorted(corpus_dir.glob("shard_*.npz"))]


def materialize_shard(data: Any) -> dict[str, np.ndarray]:
    """Load every compressed NPZ member exactly once for this shard.

    ``numpy.lib.npyio.NpzFile.__getitem__`` decompresses the requested member
    on every access.  The sampler and ragged collator access the same members
    many times per group, which otherwise leaves the GPU waiting on repeated
    ZIP decompression.  A shard is small enough to keep decompressed in memory,
    and the returned arrays are byte-identical to direct NPZ reads.
    """
    if isinstance(data, dict):
        return data
    files = getattr(data, "files", None)
    if files is None:
        raise TypeError("shard source must be a dict or expose an NPZ files list")
    return {key: data[key] for key in files}


# There is no precomputed flag for these archetypes (only is_alakazam, from when
# that was the piloted deck), so membership is derived from deck_cards at load
# time. Equivalent to how is_alakazam is built at encode time.
GRIMMSNARL_EX_CARD = 648
LOPUNNY_EX_CARD = 849
OGERPON_EX_CARD = 96
DRAGAPULT_EX_CARD = 121


# Opponent-conditioned oversampling. The ladder we actually play is NOT the corpus:
# our replays put Cinderace at 10.5% and Mega Lucario ex at 9.7% of opponents, while
# the training corpus has them at 2.5% and 1.4%. Those are exactly the matchups we
# sit at ~50% in, while Alakazam (19.9% train / 21.5% ladder) we win 17-1. This lets
# a run repeat under-represented matchups so training frequency matches the field we
# face. The map is (episode_id:seat) -> opponent archetype, built by a precompute
# pass because shards store only the deciding seat's deck.
_OPPONENT_MAP: dict[str, str] | None = None


def load_opponent_map(path: Path) -> None:
    global _OPPONENT_MAP
    _OPPONENT_MAP = json.loads(Path(path).read_text(encoding="utf-8"))


def oversample_indices(data: Any, indices: np.ndarray,
                       factors: dict[str, int]) -> np.ndarray:
    """Repeat groups whose OPPONENT archetype is under-represented."""
    if not factors or _OPPONENT_MAP is None:
        return indices
    episodes, seats = data["episode_id"], data["seat"]
    extra = []
    for index in indices:
        opponent = _OPPONENT_MAP.get(f"{int(episodes[index])}:{int(seats[index])}")
        repeat = factors.get(opponent, 1) if opponent else 1
        if repeat > 1:
            extra.extend([int(index)] * (repeat - 1))
    if not extra:
        return indices
    return np.concatenate([indices, np.asarray(extra, dtype=indices.dtype)])


def group_indices(data: Any, mode: str, min_score: float = 0.0) -> np.ndarray:
    keep = np.ones(len(data["context"]), dtype=bool)
    # `score` is the episode-average rating of BOTH seats, so a threshold keeps
    # games where both players are strong. The corpus spans 1039-1345 with median
    # 1100; training on all of it imitates the median player, which is roughly
    # where our agent already sits. Raising the floor imitates the top of the
    # field instead -- the most direct answer to "what do stronger players do".
    if min_score > 0:
        keep &= data["score"].astype(np.float32) >= min_score
    if mode == "winners": keep &= data["winner"].astype(bool)
    elif mode == "alakazam": keep &= data["is_alakazam"].astype(bool)
    elif mode == "winner_alakazam": keep &= data["winner"].astype(bool) & data["is_alakazam"].astype(bool)
    elif mode == "grimmsnarl": keep &= (data["deck_cards"] == GRIMMSNARL_EX_CARD).any(axis=1)
    elif mode == "winner_grimmsnarl":
        # `score` above is the episode-average rating of BOTH seats, so neither it
        # nor --min-score identifies a strong ACTOR: a high-average game still
        # contains the weaker side's and the loser's moves, and plain "grimmsnarl"
        # imitates winner and loser alike. `winner` IS per-seat, so this is the
        # closest actor-quality filter the shards actually support.
        keep &= data["winner"].astype(bool) & (data["deck_cards"] == GRIMMSNARL_EX_CARD).any(axis=1)
    elif mode == "lopunny": keep &= (data["deck_cards"] == LOPUNNY_EX_CARD).any(axis=1)
    elif mode == "winner_lopunny":
        keep &= data["winner"].astype(bool) & (data["deck_cards"] == LOPUNNY_EX_CARD).any(axis=1)
    elif mode == "ogerpon": keep &= (data["deck_cards"] == OGERPON_EX_CARD).any(axis=1)
    elif mode == "winner_ogerpon":
        keep &= data["winner"].astype(bool) & (data["deck_cards"] == OGERPON_EX_CARD).any(axis=1)
    elif mode == "dragapult": keep &= (data["deck_cards"] == DRAGAPULT_EX_CARD).any(axis=1)
    elif mode == "winner_dragapult":
        keep &= data["winner"].astype(bool) & (data["deck_cards"] == DRAGAPULT_EX_CARD).any(axis=1)
    elif mode != "all": raise ValueError(mode)
    return np.flatnonzero(keep)


def cap_groups_per_episode(data: Any, indices: np.ndarray, maximum: int,
                           rng: np.random.Generator) -> np.ndarray:
    if maximum <= 0:
        return indices
    chosen = []
    episodes = data["episode_id"][indices]
    for episode in np.unique(episodes):
        episode_indices = indices[episodes == episode]
        by_context: dict[int, list[int]] = defaultdict(list)
        for index in episode_indices:
            by_context[int(data["context"][index])].append(int(index))
        local = []
        # Guarantee breadth before filling with high-impact MAIN/action roots.
        shuffled_by_context = {}
        for context in sorted(by_context):
            values = np.asarray(by_context[context]); rng.shuffle(values)
            shuffled_by_context[context] = values
            local.extend(map(int, values[:1]))
        for context in sorted(shuffled_by_context):
            local.extend(map(int, shuffled_by_context[context][1:3]))
        local_set = set(local)
        remaining = [int(index) for index in episode_indices if int(index) not in local_set]
        remaining.sort(key=lambda index: (
            int(data["context"][index] != 0),
            int(data["target_option_type"][index] not in (7, 8, 9, 10, 11, 12, 13, 14)),
            int(data["original_action_count"][index] <= 1),
            rng.random(),
        ))
        local.extend(remaining[:max(0, maximum - len(local))])
        chosen.extend(local[:maximum])
    result = np.asarray(chosen, dtype=np.int64); rng.shuffle(result)
    return result


def cap_groups_phase_target_v1(data: Any, indices: np.ndarray, maximum: int,
                               rng: np.random.Generator) -> np.ndarray:
    """Cap episodes while reserving bounded phase/target tail anchors.

    Each tail family contributes only a small number of distinct rows: at most
    one ATTACK, pre-first-ATTACK, and late anchor per seat, plus one anchor from
    each mutually exclusive deck band (7-10, 4-6, 0-3).  Context and target-type
    breadth are then restored before the legacy MAIN/action-root priority fills
    the remaining capacity.  No row can satisfy more than one anchor slot.
    """
    if maximum <= 0:
        return indices
    episode_ids = data["episode_id"]
    seats_array = data["seat"]
    contexts = data["context"]
    target_types = data["target_option_type"]
    action_counts = data["original_action_count"]
    global_num = data["global_num"]
    chosen: list[int] = []
    episodes = episode_ids[indices]
    for episode in np.unique(episodes):
        episode_indices = np.asarray(indices[episodes == episode], dtype=np.int64)
        # One fixed random rank per row avoids repeatedly sampling a long game's
        # dense tail while retaining seed-controlled stochasticity.
        random_rank = {
            int(index): float(rng.random()) for index in episode_indices
        }
        local: list[int] = []
        local_set: set[int] = set()

        def add(candidates: Iterable[int], limit: int = 1) -> None:
            if len(local) >= maximum:
                return
            available = sorted(
                (int(index) for index in candidates if int(index) not in local_set),
                key=lambda index: random_rank[index],
            )
            for index in available[:min(limit, maximum - len(local))]:
                local.append(index)
                local_set.add(index)

        seats = sorted(int(value) for value in np.unique(seats_array[episode_indices]))
        # ATTACK gets one independent anchor per represented seat.
        for seat in seats:
            add(index for index in episode_indices
                if int(seats_array[index]) == seat
                and int(target_types[index]) == 13)

        # Preserve one decision immediately before each seat's first ATTACK.
        # Shards are emitted chronologically, so the global row index provides a
        # stable within-seat ordering, including setup choices on that same turn.
        for seat in seats:
            seat_indices = [int(index) for index in episode_indices
                            if int(seats_array[index]) == seat]
            attacks = [index for index in seat_indices
                       if int(target_types[index]) == 13]
            if attacks:
                before = [index for index in seat_indices if index < min(attacks)]
                if before:
                    latest_turn = max(float(global_num[index, 0]) for index in before)
                    add(index for index in before
                        if float(global_num[index, 0]) == latest_turn)

        # Late games receive one anchor per seat rather than a quota proportional
        # to episode length.
        for seat in seats:
            add(index for index in episode_indices
                if int(seats_array[index]) == seat
                and float(global_num[index, 0]) >= LATE_TURN_SCALED)

        # The nested <=10/<=6/<=3 thresholds are represented by disjoint bands;
        # a deck<=3 row therefore cannot consume all three quota slots.
        for _, lower, upper in DECK_BANDS:
            add(index for index in episode_indices
                if lower < float(global_num[index, 10]) <= upper)

        by_context: dict[int, list[int]] = defaultdict(list)
        by_target: dict[int, list[int]] = defaultdict(list)
        for index in episode_indices:
            by_context[int(contexts[index])].append(int(index))
            by_target[int(target_types[index])].append(int(index))

        # Preserve the legacy breadth guarantee whenever the cap makes it
        # feasible, counting tail anchors that already represent a context.
        represented_contexts = {int(contexts[index]) for index in local}
        for context in sorted(by_context):
            if context not in represented_contexts:
                add(by_context[context])

        # Add bounded target breadth (one row per target type, including ATTACK)
        # without constructing an unbounded context x target Cartesian quota.
        represented_targets = {int(target_types[index]) for index in local}
        for target_type in sorted(by_target):
            if target_type not in represented_targets:
                add(by_target[target_type])

        # Match legacy's second/third sample per context before priority fill.
        for context in sorted(by_context):
            already = sum(int(contexts[index]) == context for index in local)
            add(by_context[context], max(0, 3 - already))

        remaining = [int(index) for index in episode_indices if int(index) not in local_set]
        remaining.sort(key=lambda index: (
            int(contexts[index] != 0),
            int(target_types[index] not in (7, 8, 9, 10, 11, 12, 13, 14)),
            int(action_counts[index] <= 1),
            random_rank[index],
        ))
        local.extend(remaining[:max(0, maximum - len(local))])
        chosen.extend(local[:maximum])
    result = np.asarray(chosen, dtype=np.int64); rng.shuffle(result)
    return result


def sample_groups_per_episode(data: Any, indices: np.ndarray, maximum: int,
                              rng: np.random.Generator, sampler: str = "legacy") -> np.ndarray:
    if sampler == "legacy":
        return cap_groups_per_episode(data, indices, maximum, rng)
    if sampler == "phase_target_v1":
        return cap_groups_phase_target_v1(data, indices, maximum, rng)
    raise ValueError(f"unknown episode sampler: {sampler}")


def collate(data: Any, indices: np.ndarray, device: torch.device,
            entity_limit: int = TRAIN_ENTITY_LIMIT) -> dict[str, torch.Tensor]:
    if not 1 <= entity_limit <= data["entity_card"].shape[1]:
        raise ValueError(
            f"entity_limit={entity_limit} outside shard width "
            f"{data['entity_card'].shape[1]}"
        )
    offsets = data["offsets"]
    lengths = offsets[indices + 1] - offsets[indices]
    max_actions = int(lengths.max())
    batch_size = len(indices)
    action_cat = np.zeros((batch_size, max_actions, data["action_cat"].shape[1]), dtype=np.int64)
    action_num = np.zeros((batch_size, max_actions, data["action_num"].shape[1]), dtype=np.float32)
    target = np.zeros((batch_size, max_actions), dtype=np.float32)
    legal = np.zeros((batch_size, max_actions), dtype=bool)
    option_index = np.full((batch_size, max_actions), -2, dtype=np.int16)
    for row, index in enumerate(indices):
        start, end = int(offsets[index]), int(offsets[index + 1])
        size = end - start
        action_cat[row, :size] = data["action_cat"][start:end]
        action_num[row, :size] = data["action_num"][start:end]
        target[row, :size] = data["target"][start:end]
        legal[row, :size] = True
        option_index[row, :size] = data["option_index"][start:end]
    batch = {
        "global_num": torch.as_tensor(data["global_num"][indices].astype(np.float32), device=device),
        "context": torch.as_tensor(data["context"][indices].astype(np.int64), device=device),
        "select_type": torch.as_tensor(data["select_type"][indices].astype(np.int64), device=device),
        "deck_cards": torch.as_tensor(data["deck_cards"][indices].astype(np.int64), device=device),
        "entity_card": torch.as_tensor(data["entity_card"][indices, :entity_limit].astype(np.int64), device=device),
        "entity_zone": torch.as_tensor(data["entity_zone"][indices, :entity_limit].astype(np.int64), device=device),
        "entity_owner": torch.as_tensor(data["entity_owner"][indices, :entity_limit].astype(np.int64), device=device),
        "entity_num": torch.as_tensor(data["entity_num"][indices, :entity_limit].astype(np.float32), device=device),
        "action_cat": torch.as_tensor(action_cat, device=device),
        "action_num": torch.as_tensor(action_num, device=device),
        "target": torch.as_tensor(target, device=device),
        "legal_mask": torch.as_tensor(legal, device=device),
        "option_index": torch.as_tensor(option_index, device=device),
    }
    for key in ("episode_id", "date", "seat", "winner", "went_first", "score_bin", "is_alakazam",
                "legal_count", "original_action_count", "selection_ordinal", "target_option_type", "deck_hash"):
        batch[key] = torch.as_tensor(data[key][indices].astype(np.int64), device=device)
    return batch


def grouped_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    distribution = target / target.sum(-1, keepdim=True).clamp_min(1)
    return -(distribution * torch.log_softmax(logits, dim=-1)).sum(-1).mean()


def batches(indices: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(indices), batch_size): yield indices[start:start + batch_size]


def length_bucketed_batches(data: Any, indices: np.ndarray, batch_size: int,
                            rng: np.random.Generator | None = None) -> list[np.ndarray]:
    lengths = data["offsets"][indices + 1] - data["offsets"][indices]
    ordered = indices[np.argsort(lengths, kind="stable")]
    result = [ordered[start:start + batch_size] for start in range(0, len(ordered), batch_size)]
    if rng is not None:
        rng.shuffle(result)
    return result


def write_progress(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    # Telemetry must never terminate a multi-hour training run. On Windows a
    # reader, indexer, or antivirus process can briefly hold the destination
    # open and make os.replace raise WinError 5. Retry transient sharing
    # failures, then skip this progress sample while leaving training intact.
    for attempt in range(20):
        try:
            temporary.write_text(serialized, encoding="utf-8")
            temporary.replace(path)
            return
        except PermissionError:
            time.sleep(0.05 * (attempt + 1))
    print(json.dumps({
        "warning": "progress_write_skipped",
        "path": str(path),
        "status": payload.get("status"),
    }), flush=True)


def save_recovery_checkpoint(
    path: Path | None,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    rng: np.random.Generator,
    ordered_paths: list[Path],
    completed_shards: int,
    total_loss: float,
    groups: int,
    epoch: int,
    mode: str,
    batch_size: int,
    max_groups_per_episode: int,
    entity_limit: int,
    episode_sampler: str = "legacy",
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "schema_version": (
            "grouped-bc-mid-epoch-recovery-v2"
            if episode_sampler == "legacy"
            else "grouped-bc-mid-epoch-recovery-v3"
        ),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "rng_state": rng.bit_generator.state,
        "ordered_paths": [str(item.resolve()) for item in ordered_paths],
        "completed_shards": completed_shards,
        "total_loss": total_loss,
        "groups": groups,
        "epoch": epoch,
        "mode": mode,
        "batch_size": batch_size,
        "max_groups_per_episode": max_groups_per_episode,
        "entity_limit": entity_limit,
        "saved_unix": time.time(),
    }
    if episode_sampler != "legacy":
        payload["episode_sampler"] = episode_sampler
    try:
        torch.save(payload, temporary)
    except OSError as exc:
        print(json.dumps({
            "warning": "recovery_checkpoint_save_skipped",
            "path": str(path),
            "error": f"{type(exc).__name__}: {exc}",
        }), flush=True)
        return
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            time.sleep(0.05 * (attempt + 1))
    print(json.dumps({
        "warning": "recovery_checkpoint_replace_skipped",
        "path": str(path),
    }), flush=True)


def recovery_entity_limit(recovery: dict[str, Any]) -> int:
    schema = recovery.get("schema_version")
    if schema == "grouped-bc-mid-epoch-recovery-v1":
        return TRAIN_ENTITY_LIMIT
    if schema in ("grouped-bc-mid-epoch-recovery-v2", "grouped-bc-mid-epoch-recovery-v3"):
        if "entity_limit" not in recovery:
            raise ValueError("v2 recovery checkpoint is missing entity_limit")
        return int(recovery["entity_limit"])
    raise ValueError("unsupported recovery checkpoint schema")


def recovery_episode_sampler(recovery: dict[str, Any]) -> str:
    schema = recovery.get("schema_version")
    if schema in (
        "grouped-bc-mid-epoch-recovery-v1",
        "grouped-bc-mid-epoch-recovery-v2",
    ):
        return "legacy"
    if schema == "grouped-bc-mid-epoch-recovery-v3":
        sampler = str(recovery.get("episode_sampler", ""))
        if sampler not in EPISODE_SAMPLERS:
            raise ValueError(f"invalid v3 recovery episode sampler: {sampler}")
        return sampler
    raise ValueError("unsupported recovery checkpoint schema")


def train_epoch(model: nn.Module, paths: list[Path], optimizer: torch.optim.Optimizer,
                scaler: torch.amp.GradScaler, device: torch.device, mode: str,
                batch_size: int, rng: np.random.Generator,
                max_groups_per_episode: int, progress_path: Path | None = None,
                epoch: int = 1, recovery_path: Path | None = None,
                 recovery_every_shards: int = 0,
                 recovery: dict[str, Any] | None = None,
                 entity_limit: int = TRAIN_ENTITY_LIMIT,
                 episode_sampler: str = "legacy",
                 oversample_factors: dict[str, int] | None = None,
                 min_score: float = 0.0) -> dict[str, float]:
    model.train()
    if recovery is None:
        total_loss = 0.0
        groups = 0
        ordered = list(paths)
        rng.shuffle(ordered)
        completed_shards = 0
    else:
        ordered = [Path(item) for item in recovery["ordered_paths"]]
        if set(map(str, ordered)) != set(str(path.resolve()) for path in paths):
            raise ValueError("recovery shard set does not match requested training corpus")
        completed_shards = int(recovery["completed_shards"])
        total_loss = float(recovery["total_loss"])
        groups = int(recovery["groups"])
        rng.bit_generator.state = recovery["rng_state"]
    started = time.perf_counter()
    for shard_index, path in enumerate(
        ordered[completed_shards:], start=completed_shards + 1
    ):
        with np.load(path, allow_pickle=False) as compressed:
            data = materialize_shard(compressed)
            indices = group_indices(data, mode, min_score)
            indices = sample_groups_per_episode(
                data, indices, max_groups_per_episode, rng, episode_sampler
            )
            # after capping, so the cap still bounds each episode and the repeats
            # multiply only the groups that survived it
            indices = oversample_indices(data, indices, oversample_factors or {})
            for selected in length_bucketed_batches(data, indices, batch_size, rng):
                batch = collate(data, selected, device, entity_limit); optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    logits = model(batch); loss = grouped_loss(logits, batch["target"])
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(optimizer); scaler.update()
                total_loss += float(loss.detach()) * len(selected); groups += len(selected)
        elapsed = time.perf_counter() - started
        session_shards = shard_index - completed_shards
        rate = session_shards / max(elapsed, 1e-9)
        write_progress(progress_path, {
            "status": "training",
            "epoch": epoch,
            "entity_limit": entity_limit,
            "shards_completed": shard_index,
            "shards_total": len(ordered),
            "recovered_shards": completed_shards,
            "session_shards": session_shards,
            "groups_completed": groups,
            "loss_so_far": total_loss / max(1, groups),
            "elapsed_seconds": elapsed,
            "eta_seconds": (len(ordered) - shard_index) / max(rate, 1e-9),
            "updated_unix": time.time(),
        })
        if (
            recovery_path is not None
            and recovery_every_shards > 0
            and (
                shard_index % recovery_every_shards == 0
                or shard_index == len(ordered)
            )
        ):
            save_recovery_checkpoint(
                recovery_path, model, optimizer, scaler, rng, ordered,
                shard_index, total_loss, groups, epoch, mode, batch_size,
                max_groups_per_episode, entity_limit, episode_sampler,
            )
    return {"loss": total_loss / max(1, groups), "groups": groups}


def metric_row(logits: torch.Tensor, batch: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    target = batch["target"].bool(); legal = batch["legal_mask"]
    # stable=True is load-bearing, not cosmetic. action_features() does not encode an
    # entity serial, so two copies of the same card are byte-identical to the model and
    # get exactly equal logits, while the target (built from serial-bearing semantic
    # keys) marks only the copy the human played. ~26% of multi-candidate groups have a
    # tied top-2. Without a stable sort the winner among ties is unspecified, and the
    # 2026-08-01 audit measured a 7.1 pp swing in top-1 (61.86% unstable vs 68.97%
    # lowest-index) on one checkpoint -- larger than every model improvement this
    # project has ever reported. Stable + descending breaks ties toward the lowest
    # index, which is reproducible across devices and torch builds.
    order = logits.argsort(dim=-1, descending=True, stable=True)
    top1 = order[:, 0]
    correct = target.gather(1, top1[:, None]).squeeze(1)
    # Equivalence-class hit: credit the pick if it is indistinguishable (in the features
    # the model actually receives) from some target action. This is the honest number --
    # choosing between two identical copies has no effect in game, so `correct` alone
    # understates the policy. Report both; never mix them across reports.
    cat, num = batch["action_cat"], batch["action_num"]
    picked_cat = cat.gather(1, top1[:, None, None].expand(-1, 1, cat.shape[-1]))
    picked_num = num.gather(1, top1[:, None, None].expand(-1, 1, num.shape[-1]))
    same_action = (cat == picked_cat).all(-1) & (num == picked_num).all(-1)
    correct_class = (same_action & target & legal).any(1)
    top2_indices = order[:, :min(2, order.shape[1])]
    top2 = target.gather(1, top2_indices).any(1)
    ranked_target = target.gather(1, order)
    ranks = ranked_target.float().argmax(1) + 1
    ce = -(target.float() / target.sum(1, keepdim=True).clamp_min(1) * torch.log_softmax(logits, -1)).sum(1)
    probabilities = torch.softmax(logits, dim=-1)
    top_values = probabilities.topk(min(2, probabilities.shape[1]), dim=1).values
    margin = top_values[:, 0] - (top_values[:, 1] if top_values.shape[1] > 1 else torch.zeros_like(top_values[:, 0]))
    # A forced one-legal-action root is not a meaningful high-confidence model
    # decision and must never dominate a gated-hybrid coverage curve.
    margin = torch.where(batch["legal_count"] > 1, margin, torch.zeros_like(margin))
    chosen_type = batch["action_cat"].gather(1, top1[:, None, None].expand(-1, 1, batch["action_cat"].shape[-1]))[:, 0, 0] - 1
    legal_types = batch["action_cat"][..., 0] - 1
    disaster = chosen_type.eq(14) & (legal_types.eq(13) & legal).any(1)
    result = {"correct": correct, "correct_class": correct_class, "top2": top2,
              "mrr": 1.0 / ranks.float(), "ce": ce,
              "margin": margin, "disaster": disaster, "chosen_type": chosen_type}
    for key in ("context", "date", "seat", "winner", "went_first", "score_bin", "is_alakazam",
                "legal_count", "original_action_count", "selection_ordinal", "target_option_type", "deck_hash"):
        result[key] = batch[key]
    return {key: value.detach().cpu().numpy() for key, value in result.items()}


def summarize(mask: np.ndarray, values: dict[str, np.ndarray]) -> dict[str, Any]:
    count = int(mask.sum())
    if not count: return {"count": 0}
    top1 = float(values["correct"][mask].mean())
    top1_class = float(values["correct_class"][mask].mean())
    return {"count": count, "top1": top1,
            # top1_equivalence_class credits picks that are indistinguishable from a
            # target in the model's own feature space (duplicate copies of a card).
            # The gap is the share of the metric that is a serial-labelling artifact,
            # not policy quality -- see reports/TARGET_EQUIVALENCE_AND_METRIC_AUDIT_20260801.md
            "top1_equivalence_class": top1_class,
            "top1_serial_artifact_gap": top1_class - top1,
            "top2": float(values["top2"][mask].mean()), "mrr": float(values["mrr"][mask].mean()),
            "grouped_ce": float(values["ce"][mask].mean()), "margin_mean": float(values["margin"][mask].mean()),
            "illegal_actions": 0, "disaster_actions": int(values["disaster"][mask].sum())}


@torch.no_grad()
def evaluate(model: nn.Module, paths: list[Path], device: torch.device, mode: str,
             batch_size: int, fixed_thresholds: dict[str, float] | None = None,
             entity_limit: int = TRAIN_ENTITY_LIMIT) -> dict[str, Any]:
    model.eval(); collected: dict[str, list[np.ndarray]] = defaultdict(list); start = time.perf_counter()
    for path in paths:
        with np.load(path, allow_pickle=False) as compressed:
            data = materialize_shard(compressed)
            indices = group_indices(data, mode)
            for selected in length_bucketed_batches(data, indices, batch_size):
                batch = collate(data, selected, device, entity_limit)
                output = metric_row(model(batch), batch)
                for key, value in output.items(): collected[key].append(value)
    values = {key: np.concatenate(parts) for key, parts in collected.items()}
    n = len(values["correct"]); report = {"overall": summarize(np.ones(n, dtype=bool), values)}
    groups = {}
    definitions = {
        "context": values["context"], "date": values["date"], "seat": values["seat"],
        "winner": values["winner"], "went_first": values["went_first"],
        "score_bin": values["score_bin"], "alakazam": values["is_alakazam"],
        "target_option_type": values["target_option_type"],
    }
    for group_name, array in definitions.items():
        groups[group_name] = {str(value): summarize(array == value, values) for value in np.unique(array)}
    groups["candidate_complexity"] = {
        "one_legal": summarize(values["legal_count"] == 1, values),
        "multi_legal": summarize(values["legal_count"] > 1, values),
        "multi_select_sequence": summarize((values["original_action_count"] > 1) | (values["selection_ordinal"] > 0), values),
    }
    report["groups"] = groups
    eligible = np.flatnonzero(values["legal_count"] > 1)
    order = eligible[np.argsort(-values["margin"][eligible])]
    curves = {}
    for fraction in (0.05, 0.10, 0.20, 0.30):
        selected = order[:max(1, int(len(eligible) * fraction))]; mask = np.zeros(n, dtype=bool); mask[selected] = True
        item = summarize(mask, values)
        item["eligible_multi_candidate_groups"] = int(len(eligible))
        item["coverage_of_all_groups"] = float(len(selected) / max(1, n))
        item["threshold_margin"] = float(values["margin"][selected].min()) if len(selected) else None
        item["context_counts"] = {str(value): int((values["context"][mask] == value).sum())
                                  for value in np.unique(values["context"][mask])}
        curves[f"top_{int(fraction * 100)}pct"] = item
    report["confidence_coverage"] = curves
    if fixed_thresholds:
        fixed = {}
        for label, threshold in fixed_thresholds.items():
            mask = (values["legal_count"] > 1) & (values["margin"] >= threshold)
            item = summarize(mask, values)
            item["threshold_margin"] = float(threshold)
            item["eligible_multi_candidate_groups"] = int(len(eligible))
            item["coverage_of_all_groups"] = float(mask.sum() / max(1, n))
            item["coverage_of_multi_candidate_groups"] = float(mask.sum() / max(1, len(eligible)))
            item["context_counts"] = {
                str(value): int((values["context"][mask] == value).sum())
                for value in np.unique(values["context"][mask])
            }
            fixed[label] = item
        report["fixed_thresholds"] = fixed
    report["latency"] = {"total_seconds": time.perf_counter() - start,
                         "microseconds_per_group": (time.perf_counter() - start) * 1e6 / max(1, n)}
    return report


def save_checkpoint(path: Path, model: nn.Module, config: dict, epoch: int, validation: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": config, "epoch": epoch, "validation": validation}, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-corpus", type=Path, nargs="+", required=True)
    parser.add_argument("--validation-corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("all", "winners", "alakazam", "winner_alakazam",
                 "grimmsnarl", "winner_grimmsnarl", "lopunny",
                 "winner_lopunny", "ogerpon", "winner_ogerpon",
                 "dragapult", "winner_dragapult"),
        default="all",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--cross-option-layers", type=int, default=1,
                        help="stacked cross-option sub-layers (each gated, zero-init)")
    parser.add_argument("--cross-option-ff", action="store_true",
                        help="add a gated feed-forward after each attention sub-layer, "
                             "completing the transformer block")
    parser.add_argument("--cross-option-heads", type=int, default=0,
                        help="attention heads ACROSS candidate options (0 = off, the "
                             "original architecture). Residual is zero-initialised, so "
                             "--resume from a pre-attention checkpoint is safe.")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-groups-per-episode", type=int, default=0)
    parser.add_argument(
        "--episode-sampler", choices=EPISODE_SAMPLERS, default="legacy",
        help="Per-episode group selection rule; legacy preserves prior cap behavior.",
    )
    parser.add_argument("--entity-limit", type=int, default=TRAIN_ENTITY_LIMIT,
                        help="Number of ordered state entities to consume from each 96-wide shard.")
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--min-score", type=float, default=0.0,
                        help="Keep only groups whose episode-average player rating is "
                             ">= this. Imitates the top of the field instead of its median.")
    parser.add_argument("--opponent-map", type=Path, default=None,
                        help="JSON of 'episode:seat' -> opponent archetype (precomputed).")
    parser.add_argument("--oversample-opponent", action="append", default=[],
                        help="ARCHETYPE=FACTOR; repeat groups facing that opponent. "
                             "Use to match the ladder's matchup mix rather than the corpus's.")
    parser.add_argument("--progress-file", type=Path,
                        help="Atomically updated JSON progress file (does not affect training).")
    parser.add_argument("--recovery-checkpoint", type=Path,
                        help="Mid-epoch recovery checkpoint used for automatic restarts.")
    parser.add_argument("--recovery-every-shards", type=int, default=0,
                        help="Save recovery state after this many completed shards.")
    parser.add_argument("--resume", type=Path,
                        help="Initialize from a compatible global BC checkpoint (for example Alakazam fine-tuning).")
    args = parser.parse_args()
    if not 1 <= args.entity_limit <= MAX_SHARD_ENTITIES:
        parser.error(f"--entity-limit must be in [1, {MAX_SHARD_ENTITIES}]")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    oversample_factors = {}
    for spec in args.oversample_opponent:
        name, _, factor = spec.rpartition("=")
        oversample_factors[name] = int(factor)
    if oversample_factors:
        if not args.opponent_map:
            raise SystemExit("--oversample-opponent requires --opponent-map")
        load_opponent_map(args.opponent_map)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_shards = shard_paths(args.train_corpus)
    if not train_shards:
        raise ValueError(f"no train shards found in: {args.train_corpus}")
    with np.load(train_shards[0], allow_pickle=False) as sample:
        config = {"global_dim": sample["global_num"].shape[1], "entity_num_dim": sample["entity_num"].shape[2],
                  "action_num_dim": sample["action_num"].shape[1], "hidden": args.hidden, "mode": args.mode,
                  "seed": args.seed, "option_index_feature": False, "entity_limit": args.entity_limit,
                  "length_bucketed_batches": True, "max_groups_per_episode": args.max_groups_per_episode,
                  "episode_sampler": args.episode_sampler,
                  "cross_option_heads": args.cross_option_heads,
                  "cross_option_layers": args.cross_option_layers,
                  "cross_option_ff": args.cross_option_ff,
                  "device": str(device), "train_corpora": [str(path) for path in args.train_corpus]}
    model = scorer_from_config(config).to(device)
    if args.resume:
        resume = torch.load(args.resume, map_location=device, weights_only=False)
        resume_config = resume["config"]
        expected = (config["global_dim"], config["entity_num_dim"], config["action_num_dim"],
                    config["hidden"], config["entity_limit"])
        actual = (resume_config["global_dim"], resume_config["entity_num_dim"],
                  resume_config["action_num_dim"], resume_config["hidden"],
                  int(resume_config.get("entity_limit", TRAIN_ENTITY_LIMIT)))
        if actual != expected:
            raise ValueError(f"incompatible resume checkpoint dimensions: {actual} != {expected}")
        # Adding the zero-init cross-option block on top of a pre-attention
        # checkpoint is the one tolerated mismatch: those keys are absent from
        # the old state_dict and the block is a no-op at init. Every other
        # missing or unexpected key stays a hard error -- silently dropping
        # weights would look like a normal (but crippled) training run.
        missing, unexpected = model.load_state_dict(resume["model"], strict=False)
        tolerated = [key for key in missing if key.startswith("cross_")]
        surprises = [key for key in missing if not key.startswith("cross_")]
        if surprises or unexpected:
            raise ValueError(
                f"incompatible resume checkpoint weights: missing={surprises} "
                f"unexpected={list(unexpected)}"
            )
        if tolerated:
            print(f"resume: adding zero-init cross-option block ({len(tolerated)} tensors); "
                  "model starts numerically identical to the source checkpoint")
        config["initialized_from"] = str(args.resume.resolve())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    train_paths, val_paths = train_shards, shard_paths(args.validation_corpus)
    args.output_dir.mkdir(parents=True, exist_ok=True); history = []; rng = np.random.default_rng(args.seed)
    recovery = None
    if args.recovery_checkpoint and args.recovery_checkpoint.exists():
        recovery = torch.load(
            args.recovery_checkpoint, map_location=device, weights_only=False
        )
        recovery_limit = recovery_entity_limit(recovery)
        recovery_sampler = recovery_episode_sampler(recovery)
        expected_recovery = (
            1, args.mode, args.batch_size, args.max_groups_per_episode,
            args.entity_limit, args.episode_sampler,
        )
        actual_recovery = (
            int(recovery["epoch"]), recovery["mode"], int(recovery["batch_size"]),
            int(recovery["max_groups_per_episode"]),
            recovery_limit, recovery_sampler,
        )
        if actual_recovery != expected_recovery:
            raise ValueError(
                f"recovery settings mismatch: {actual_recovery} != {expected_recovery}"
            )
        model.load_state_dict(recovery["model"])
        optimizer.load_state_dict(recovery["optimizer"])
        scaler.load_state_dict(recovery["scaler"])
    best, best_epoch = float("inf"), -1
    write_progress(args.progress_file, {
        "status": "starting", "epochs": args.epochs,
        "train_shards": len(train_paths), "validation_shards": len(val_paths),
        "entity_limit": args.entity_limit, "episode_sampler": args.episode_sampler,
        "recovered_shards": int(recovery["completed_shards"]) if recovery else 0,
        "updated_unix": time.time(),
    })
    for epoch in range(1, args.epochs + 1):
        train = train_epoch(model, train_paths, optimizer, scaler, device, args.mode, args.batch_size, rng,
                            args.max_groups_per_episode, args.progress_file, epoch,
                            args.recovery_checkpoint, args.recovery_every_shards,
                            recovery if recovery and int(recovery["epoch"]) == epoch else None,
                            args.entity_limit, args.episode_sampler,
                            oversample_factors, args.min_score)
        recovery = None
        save_checkpoint(args.output_dir / f"epoch_{epoch:02d}_prevalidation.pt", model, config, epoch, {})
        if args.recovery_checkpoint:
            for stale in (
                args.recovery_checkpoint,
                args.recovery_checkpoint.with_suffix(
                    args.recovery_checkpoint.suffix + ".tmp"
                ),
            ):
                try:
                    stale.unlink(missing_ok=True)
                except OSError:
                    pass
        write_progress(args.progress_file, {
            "status": "validating", "epoch": epoch, "train": train,
            "validation_shards": len(val_paths), "updated_unix": time.time(),
        })
        print(json.dumps({"epoch": epoch, "train_complete": train, "checkpoint": "prevalidation"}), flush=True)
        validation = evaluate(
            model, val_paths, device, "all", args.batch_size,
            entity_limit=args.entity_limit,
        )
        record = {"epoch": epoch, "train": train, "validation": validation}; history.append(record)
        print(json.dumps({"epoch": epoch, "train": train, "validation": validation["overall"],
                          "confidence_coverage": validation["confidence_coverage"]}, ensure_ascii=False), flush=True)
        if validation["overall"]["grouped_ce"] < best:
            best, best_epoch = validation["overall"]["grouped_ce"], epoch
            save_checkpoint(args.output_dir / "best.pt", model, config, epoch, validation)
        save_checkpoint(args.output_dir / "last.pt", model, config, epoch, validation)
    result = {"config": config, "parameters": sum(p.numel() for p in model.parameters()),
              "best_epoch": best_epoch, "history": history}
    (args.output_dir / "training_report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_progress(args.progress_file, {
        "status": "completed", "best_epoch": best_epoch,
        "training_report": str((args.output_dir / "training_report.json").resolve()),
        "validation": history[-1]["validation"]["overall"] if history else None,
        "updated_unix": time.time(),
    })


if __name__ == "__main__":
    main()
