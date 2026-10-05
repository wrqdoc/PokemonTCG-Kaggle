# Pokémon TCG AI — Cross-Option Behavior Cloning with Conservative DFS

A competition agent developed for the **Pokémon TCG AI Battle Challenge**.

The final agent is primarily a **behavior-cloned neural policy** trained from high-level replay decisions. It jointly reasons over the current legal action set using cross-option attention, and uses a small simulator-based DFS as a conservative tactical verifier.

> **The neural policy is the main decision maker. DFS is not a replacement for the policy.**
> Search only overrides the BC action when simulation can robustly demonstrate a strictly better same-turn Prize outcome.

The submitted system finished **457th in the competition**.

---

## Overview

The agent follows this decision pipeline:

```text
Expert / high-level replay data
            │
            ▼
 Semantic replay parsing
            │
            ▼
 State + legal-action encoding
            │
            ▼
 Grouped Behavior Cloning
            │
            ▼
 Cross-option neural policy
            │
            ▼
       Default BC action
            │
            │ selected MAIN decisions
            ▼
   Simulator-based DFS probe
            │
            ▼
 Multiple determinizations
            │
            ▼
Does another action guarantee
more Prizes this turn?
       │             │
      No            Yes
       │             │
       ▼             ▼
  Keep BC       Override BC
```

The design intentionally separates **strategic imitation** from **tactical verification**:

* the neural policy handles general gameplay;
* legal actions are compared jointly rather than independently;
* sequential multi-selection decisions are handled autoregressively;
* DFS is restricted to situations where the simulator can establish an immediate tactical improvement;
* unsupported simulator states, exceptions, or time-budget exhaustion automatically fall back to the neural policy.

---

## Key Components

| Component                      | Final agent | Purpose                                                 |
| ------------------------------ | ----------: | ------------------------------------------------------- |
| Behavior cloning               |           ✅ | Main gameplay policy                                    |
| Semantic state/action encoding |           ✅ | Represent cards, zones, targets and action context      |
| Cross-option attention         |           ✅ | Compare currently legal actions jointly                 |
| Autoregressive multi-select    |           ✅ | Handle sequential card-selection decisions              |
| Same-turn DFS                  |           ✅ | Detect tactically superior immediate Prize lines        |
| Hidden-state determinization   |           ✅ | Test search candidates under sampled hidden information |
| Learned value-function search  |           ❌ | Experimented with, but not used in the final agent      |
| Reinforcement learning / PPO   |           ❌ | Not used                                                |

---

# 1. Behavior-Cloning Policy

The core of the agent is a grouped behavior-cloning policy.

Instead of treating an action as a fixed class:

```text
state → action_id
```

the model receives both the current game state and the **current legal action set**:

```text
state
+
legal_action_1
legal_action_2
...
legal_action_N
        │
        ▼
joint action scoring
        │
        ▼
P(action | state, legal actions)
```

This matters because Pokémon TCG has a large, state-dependent action space. The identity and meaning of an action depend on cards, zones, targets, attacks and the current interaction context.

The problem is therefore better expressed as:

> Given this state and these particular legal actions, which option should be preferred?

rather than:

> Is this action generally good?

---

## Cross-Option Attention

Each legal action is first embedded from its semantic attributes.

Depending on the action, these features may include information such as:

```text
action type
source card
target card
source zone
target zone
ownership
selection type
attack identity
card identity
HP / damage / energy context
```

The model also builds a representation of the current game state from card and global-state features.

The legal-action embeddings are then passed through a **cross-option multi-head attention layer**.

```text
                  ┌──────── Action A
                  │
State ────────────┼──────── Action B
                  │
                  ├──────── Action C
                  │
                  └──────── Action D
                           │
                           ▼
                 Cross-option attention
                           │
                           ▼
                  One logit per action
```

This lets the model reason about an option **relative to the alternatives currently available**.

The final checkpoint uses one layer of **8-head cross-option attention**.

---

# 2. Grouped Behavior Cloning

Training examples are constructed from replay decisions.

For a state \(s\) with legal-action set \(A_s\), the policy predicts:

$$
\pi_\theta(a \mid s, A_s)
$$

and is trained against the action selected in the replay.

The objective is standard grouped cross-entropy:

$$
\mathcal{L}_{BC}
=
-\log
\pi_\theta(a_{\text{expert}} \mid s,A_s)
$$

The important part is not a specialized loss function, but the representation of the **entire legal decision set as one training example**.

This keeps training aligned with inference:

```text
game state
+
actual legal actions
        ↓
joint policy evaluation
        ↓
selected action
```

---

# 3. Replay-to-Training Pipeline

The repository includes the code required to convert replay information into grouped BC training examples.

The pipeline is roughly:

```text
Competition replays
        │
        ▼
replay_semantics.py
        │
        ▼
semantic state/action reconstruction
        │
        ▼
encode_bc_shards.py
        │
        ▼
encoded grouped training shards
        │
        ▼
train_grouped_bc.py
        │
        ▼
model.pt
```

Training records preserve semantic information about the state and decision rather than relying only on opaque action IDs.

The encoding pipeline supports filtering or constructing datasets from subsets of replay data, allowing experiments with different replay-quality or deck distributions.

---

# 4. Autoregressive Multi-Select Decisions

Some Pokémon TCG interactions require selecting multiple cards sequentially.

For example:

```text
choose card A
      ↓
choose card B
      ↓
choose card C
      ↓
STOP
```

The agent does not attempt to predict the complete combination as a single action.

Instead, it models the process autoregressively:

$$
P(a_1 \mid s)
$$

$$
P(a_2 \mid s,a_1)
$$

$$
P(a_3 \mid s,a_1,a_2)
$$

and, once the minimum required number of selections has been reached, a synthetic **STOP** option can be introduced.

This allows the same variable-action policy architecture to handle both ordinary actions and multi-card selections.

---

# 5. Conservative DFS

Behavior cloning remains the default policy at inference time.

DFS is used only as a narrow tactical verifier.

The search does **not** attempt to solve the complete game and does not use a hand-designed long-horizon board evaluation as the primary decision rule.

Instead, it asks a much smaller question:

> Can another candidate action provably produce more Prizes during the current turn?

---

## Search Procedure

At selected MAIN decisions:

```text
BC policy
   │
   ▼
rank legal actions
   │
   ▼
select a small set of candidates
   │
   ▼
clone simulator state
   │
   ▼
run same-turn DFS
   │
   ▼
measure immediate Prize outcome
```

Only a small number of high-ranking policy actions need to be investigated.

This keeps search bounded and preserves the learned policy as the primary source of strategic judgment.

---

# 6. Hidden Information and Determinization

Pokémon TCG is a partially observable game.

The exact hidden state of the opponent's hand or deck may not be known, so a single simulated future can be misleading.

The DFS therefore evaluates candidates under multiple sampled determinizations:

```text
Candidate action A
     │
     ├── hidden state D1 → Prize result
     ├── hidden state D2 → Prize result
     ├── hidden state D3 → Prize result
     └── ...
```

The search is deliberately conservative.

Conceptually, an action can be evaluated using its worst tested determinization:

$$
Q(a)
=
\min_{d \in D} V(a,d)
$$

where \(V(a,d)\) is the same-turn Prize outcome under determinization \(d\).

A search candidate only overrides the BC action when it demonstrates a **strictly better robust Prize result**.

Conceptually:

$$
Q(a_{\text{search}})
>
Q(a_{\text{BC}})
$$

Otherwise, the BC decision is retained.

---

## Example

Suppose BC prefers:

```text
Attach Energy
→ play Supporter
→ attack
→ take 1 Prize
```

DFS discovers another line:

```text
Rare Candy
→ evolve
→ attack
```

and obtains:

```text
Determination 1: 2 Prizes
Determination 2: 2 Prizes
Determination 3: 2 Prizes
```

while the BC line consistently obtains:

```text
1 Prize
```

The search may override BC.

However, if the candidate produces:

```text
Determination 1: 2 Prizes
Determination 2: 2 Prizes
Determination 3: 0 Prizes
```

the conservative criterion can reject the override.

The goal is not to make DFS responsible for general strategy. Its job is to catch tactical opportunities that can be verified reliably by the simulator.

---

# 7. Why Not Use Search Everywhere?

Earlier experiments explored stronger dependence on forward search and learned or handcrafted evaluation functions.

In practice, inaccurate long-horizon evaluation can easily make search worse than a strong imitation policy.

This is particularly problematic in a partially observable card game:

```text
incorrect hidden-state assumption
             +
imperfect value function
             +
deeper search
             =
confidently wrong decision
```

The final design therefore gives the neural policy authority over general strategic decisions and restricts search to a narrow objective that the simulator can evaluate exactly:

```text
immediate same-turn Prize gain
```

This is the central design principle of the hybrid agent.

---

# 8. Runtime Safety and Fallback

Competition agents operate under strict time constraints.

The implementation therefore uses hard search budgets.

If any of the following occurs:

```text
search timeout
unsupported simulator state
native simulator exception
invalid search branch
insufficient remaining budget
```

the system falls back to the behavior-cloned policy.

The runtime hierarchy is therefore:

```text
BC policy
   ↓
optional DFS verification
   ↓
safe BC fallback
```

rather than:

```text
DFS succeeds or agent fails
```

---

# 9. Evaluation

The hybrid search component was tested against the pure cross-option BC policy.

Reported internal results include:

| Evaluation                              |  Result |
| --------------------------------------- | ------: |
| Final five-agent round robin            | 206–194 |
| DFS+xopt vs pure xopt, 100 games        |   50–50 |
| Combined with an earlier 400-game batch | 263–237 |
| Combined win rate                       |   52.6% |

The combined result favors the DFS-enhanced version numerically, but the reported confidence interval still crosses 50%.

Therefore, these experiments should **not** be interpreted as strong evidence that DFS significantly improves overall playing strength.

A safer interpretation is:

> The behavior-cloned neural policy provides most of the agent's strength. Conservative DFS can detect some tactical overrides, but the available evaluation does not establish a statistically significant overall improvement.

---

# 10. Competition Result

This system was developed for the **Pokémon TCG AI Battle Challenge** and finished **457th** on the competition leaderboard.

The competition result provides the context in which the agent was developed and evaluated, but this repository is primarily intended to preserve the modeling and search approach rather than present the leaderboard position as the main contribution.

---

# Repository Layout

```text
agent/
├── main.py
│   Competition entry point and BC/DFS decision logic
│
├── model.pt
│   Trained cross-option BC checkpoint
│
├── deck.csv
│   Competition deck list
│
├── runtime_policy.py
│   Neural-policy inference implementation
│
├── train_grouped_bc.py
│   Grouped behavior-cloning training code
│
├── encode_bc_shards.py
│   Replay → grouped training-data encoder
│
├── replay_semantics.py
│   Semantic state/action reconstruction
│
└── cg/
    Native competition simulator components used by DFS

docs/
└── Experiment and evaluation reports

release/
└── Exact competition-format submission archive

requirements.txt
SHA256SUMS.txt
README.md
```

---

# Requirements

* Python 3.10+
* NumPy
* PyTorch
* 64-bit Windows or Linux for the bundled native simulator components

Create a virtual environment:

```bash
python -m venv .venv
```

### Linux

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

# Running the Competition Agent

The competition engine imports:

```text
agent/main.py
```

and calls:

```python
agent(observation)
```

The exact competition-format archive used by the repository is preserved under:

```text
release/
```

---

# Training

The BC training pipeline consists of two main stages.

### 1. Encode replay decisions

```text
replays
   ↓
encode_bc_shards.py
   ↓
grouped BC shards
```

### 2. Train the policy

```text
grouped BC shards
   ↓
train_grouped_bc.py
   ↓
model.pt
```

The final checkpoint included in this repository can be used directly for inference.

---

# Reproducibility

The repository includes:

* the final trained checkpoint;
* runtime inference code;
* grouped BC training code;
* replay encoding code;
* semantic state/action processing;
* the competition deck;
* native simulator bindings used by DFS;
* the final submission-format archive.

The large original replay corpus and encoded training shards are **not included**.

As a result, the historical training run cannot be reproduced byte-for-byte from raw data using this repository alone.

However, the code needed to reconstruct the replay-to-training pipeline is included.

---

# What This Project Is — and Is Not

This project **is**:

* a behavior-cloned Pokémon TCG agent;
* a variable legal-action policy;
* a cross-option attention model;
* an autoregressive multi-selection policy;
* a hybrid neural-policy + simulator-search system;
* an experiment in conservative search under hidden information.

This project is **not**:

* a reinforcement-learning agent;
* a PPO implementation;
* a full-game AlphaZero system;
* an MCTS-based primary policy;
* a claim that DFS is stronger than BC in all situations.

The main policy was learned through supervised imitation from replay decisions.

---

# Potential Extensions

The architecture is also a natural starting point for reinforcement learning.

One possible extension is:

```text
Replay data
    ↓
Grouped BC
    ↓
Cross-option policy checkpoint
    ↓
Add value head
    ↓
Self-play
    ↓
PPO / actor-critic training
    ↓
Opponent population
    ↓
Deck-specific or matchup-specific fine-tuning
```

The existing semantic state representation, variable legal-action encoder and autoregressive selection logic address several of the engineering problems that otherwise have to be solved before self-play RL becomes practical.

---

# Third-Party Components

The `agent/cg/` directory contains native simulator components originating from the competition environment.

Rights to those components remain with their respective owners.

Before redistributing or repackaging those files, verify the applicable competition and third-party terms.

---

# Summary

The final agent can be summarized as:

```text
high-level replays
      ↓
semantic encoding
      ↓
grouped behavior cloning
      ↓
cross-option neural policy
      ↓
default action
      ↓
optional conservative DFS
      ↓
robust same-turn Prize check
      ↓
final action
```

The central idea is simple:

> **Use learning for general strategy and exact simulation only where it can safely prove a tactical improvement.**
