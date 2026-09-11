# Pokemon TCG AI — DFS + Cross-Option BC

This repository preserves the `bc_prize_search_dfs_grimmsnarl_deckv3_xopt810_v1`
agent developed for the Pokemon TCG AI Battle Challenge.

The agent combines:

- a behavior-cloning policy trained from high-level replay decisions;
- one layer of eight-head attention across the currently legal actions;
- a narrow deterministic search that may override BC only when every sampled
  determinization proves a strictly better same-turn prize outcome;
- hard per-turn and per-game time budgets, with automatic fallback to pure BC.

## Repository layout

```text
agent/                       Exact runnable package contents
  main.py                    Competition entry point
  model.pt                   Trained policy checkpoint
  deck.csv                   Grimmsnarl deck list
  runtime_policy.py          Policy inference
  train_grouped_bc.py        Grouped behavior-cloning trainer
  encode_bc_shards.py        Replay-to-training-shard encoder
  replay_semantics.py        Action/state semantics
  cg/                        Native simulator bindings used by DFS
docs/                        Selected experiment and evaluation reports
release/                     Exact submitted-format tar.gz archive
SHA256SUMS.txt               Integrity checksum for the archive
```

## Requirements

- Python 3.10 or newer
- NumPy
- PyTorch
- 64-bit Windows or Linux for the bundled native simulator

Install the Python dependencies in a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux/macOS shell:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The competition engine imports `agent/main.py` and calls its `agent(observation)`
function. The exact ready-to-upload competition archive is available under
`release/`.

## Model and search design

`model.pt` contains the trained cross-option BC model. The policy ranks legal
actions jointly rather than scoring each option in isolation. BC remains the
default decision maker.

DFS is deliberately narrow. It does not replace the learned policy with a
hand-written evaluator. It inspects a small number of top BC actions and only
overrides the BC choice when a better immediate prize outcome is robust across
all tested determinizations. Exceptions, unsupported simulator states, or time
budget exhaustion fall back to BC.

## Evaluation context

In the final five-agent round robin, DFS+xopt scored 206–194 overall. Its direct
100-game comparison with the pure xopt reference was 50–50. Combining that test
with an earlier independent 400-game batch produced 263–237 (52.60%), with a
confidence interval that still crossed 50%.

See `docs/` for the search probe, final round-robin result, and override
telemetry.

## Reproducibility scope

This repository includes the final checkpoint, inference implementation,
training and encoding code, deck, simulator bindings, and exact runtime archive.
The large raw replay corpus and encoded training shards are intentionally not
included, so the historical training run cannot be reproduced byte-for-byte
from raw data using this repository alone.

## Archive integrity

```text
9e09467ab49f5398d90d60dcd50269f50bc20e0e21cdd7a00c4600731f94128e
```

## Third-party components

The `agent/cg/` directory contains native competition simulator components.
Their rights remain with their respective owners. Confirm the competition's
redistribution terms before changing this repository from private to public.
