# Pokémon TCG AI — Cross-Option Behavior Cloning with Conservative DFS

A competition agent developed for the **Pokémon TCG AI Battle Challenge**.

The agent is primarily a **behavior-cloned neural policy** trained from replay decisions.  
It jointly reasons over the current legal action set using cross-option attention, while a small simulator-based DFS acts as a conservative tactical verifier.

> **The neural policy is the main decision maker. DFS is only an optional tactical override.**
>
> Search replaces the BC action only when simulation can robustly demonstrate a strictly better same-turn Prize outcome.

The submitted agent finished **457th** in the competition.

---

## Overview

The final decision pipeline is:

```text
Replay decisions
      │
      ▼
Semantic state / action encoding
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
Simulator-based DFS
      │
      ▼
Multiple determinizations
      │
      ▼
Can another action robustly
take more Prizes this turn?
      │
   ┌──┴──┐
   │     │
  No    Yes
   │     │
   ▼     ▼
Keep BC Override
```

The design deliberately separates two responsibilities:

- **Behavior cloning** handles general strategy.
- **DFS** checks a small number of immediate tactical alternatives.
- **Cross-option attention** lets legal actions be compared against one another.
- **Autoregressive selection** handles multi-card decisions.
- **Determinization** reduces reliance on a single guess about hidden information.
- **Safe fallback** returns to the BC policy whenever search fails or times out.

---

## Architecture

| Component | Used in final agent? | Purpose |
|---|---:|---|
| Behavior cloning | Yes | Main gameplay policy |
| Semantic state encoding | Yes | Represent the current board and cards |
| Semantic action encoding | Yes | Represent dynamic legal actions |
| Cross-option attention | Yes | Compare legal actions jointly |
| Autoregressive multi-select | Yes | Handle sequential card selections |
| Same-turn DFS | Yes | Detect immediate tactical improvements |
| Hidden-state determinization | Yes | Test candidates under multiple hidden-state hypotheses |
| Learned value search | No | Experimented with, but not used in the final agent |
| PPO / reinforcement learning | No | Not used |

---

## 1. Dynamic Legal-Action Policy

Pokémon TCG does not have a small fixed action space.

The legal actions available at any moment depend on:

- cards in play;
- cards in hand;
- current interaction context;
- valid targets;
- attacks;
- selection constraints;
- game phase.

Instead of predicting a fixed `action_id`, the model receives:

```text
Current state
+
Current legal-action set
        │
        ▼
Joint action scoring
        │
        ▼
Probability over legal actions
```

Conceptually, the policy predicts:

```math
\pi_\theta(a \mid s, A_s)
```

where:

- `s` is the current game state;
- `A_s` is the set of currently legal actions;
- `a` is one candidate action.

The important question is therefore not:

> Is this action generally good?

but:

> Among the actions available right now, which one is best?

---

## 2. Cross-Option Attention

Each legal action is represented using semantic features such as:

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

The game state is encoded separately from board and card information.

The candidate action embeddings are then processed jointly:

```text
                  Action A
                     │
                  Action B
                     │
State ───────► Cross-option Attention ───────► action logits
                     │
                  Action C
                     │
                  Action D
```

This allows the model to reason about actions **relative to the alternatives currently available**, rather than scoring every action independently.

The final policy uses an **8-head cross-option attention layer**.

---

## 3. Grouped Behavior Cloning

Training examples are reconstructed from replay decisions.

For every decision, the training sample contains:

```text
game state
+
all legal actions
+
action selected in the replay
```

The model predicts a distribution over the legal-action set:

```math
\pi_\theta(a \mid s, A_s)
```

and is trained using standard cross-entropy:

```math
\mathcal{L}_{BC}
=
-\log \pi_\theta(a_{\mathrm{expert}} \mid s, A_s)
```

The loss itself is intentionally simple.

The more important design choice is that **the entire legal-action set is treated as one grouped decision**.

Training and inference therefore have the same structure:

```text
State
+
Actual legal actions
        │
        ▼
Joint policy evaluation
        │
        ▼
Selected action
```

---

## 4. Replay-to-Training Pipeline

The repository contains the code required to convert replay decisions into grouped BC training examples.

```text
Competition replays
        │
        ▼
replay_semantics.py
        │
        ▼
Semantic reconstruction
        │
        ▼
encode_bc_shards.py
        │
        ▼
Grouped training shards
        │
        ▼
train_grouped_bc.py
        │
        ▼
model.pt
```

The encoded training data preserves semantic information about both the state and legal actions instead of relying only on opaque action IDs.

The replay pipeline can also be used to construct different training subsets for experiments involving replay quality, decks, or matchup distributions.

---

## 5. Autoregressive Multi-Select

Some game interactions require selecting multiple cards sequentially.

For example:

```text
Choose card A
      │
      ▼
Choose card B
      │
      ▼
Choose card C
      │
      ▼
STOP
```

Rather than treating every possible card combination as a separate action, the policy handles these decisions autoregressively.

Conceptually:

```math
P(a_1 \mid s)
```

```math
P(a_2 \mid s, a_1)
```

```math
P(a_3 \mid s, a_1, a_2)
```

Once the minimum required number of selections has been reached, a synthetic `STOP` option can be introduced.

This lets the same dynamic-action architecture handle both normal actions and variable-length multi-card selections.

---

## 6. Conservative DFS

The behavior-cloned policy remains the default decision maker.

DFS is only used at selected MAIN decisions as a tactical verifier.

It does **not** attempt to solve the entire game.

It asks a much narrower question:

> Can another candidate action reliably take more Prizes during the current turn?

The search procedure is roughly:

```text
BC policy
   │
   ▼
Rank legal actions
   │
   ▼
Select a small candidate set
   │
   ▼
Clone simulator state
   │
   ▼
Run same-turn DFS
   │
   ▼
Compare immediate Prize outcomes
```

Only a limited number of policy candidates are searched.

This keeps the search bounded and leaves general strategy to the learned policy.

---

## 7. Hidden Information

Pokémon TCG is partially observable.

The exact contents or ordering of hidden cards may not be known, meaning a search result based on one assumed hidden state can be misleading.

The DFS therefore evaluates candidate actions under multiple determinizations:

```text
Candidate action
      │
      ├── Determinization 1 → Prize result
      ├── Determinization 2 → Prize result
      ├── Determinization 3 → Prize result
      └── ...
```

The search uses a conservative evaluation.

Conceptually:

```math
Q(a)
=
\min_{d \in D} V(a,d)
```

where `V(a,d)` is the same-turn Prize outcome for action `a` under determinization `d`.

A search action replaces the BC action only when it achieves a strictly better robust result:

```math
Q(a_{\mathrm{search}})
>
Q(a_{\mathrm{BC}})
```

Otherwise, the policy keeps the original BC decision.

---

## Example

Suppose the BC policy prefers:

```text
Attach Energy
→ Play Supporter
→ Attack
→ Take 1 Prize
```

DFS discovers another candidate:

```text
Rare Candy
→ Evolve
→ Attack
```

and simulation produces:

```text
Determinization 1: 2 Prizes
Determinization 2: 2 Prizes
Determinization 3: 2 Prizes
```

while the BC line consistently produces:

```text
1 Prize
```

The DFS candidate may override the BC decision.

However, if the candidate instead produces:

```text
Determinization 1: 2 Prizes
Determinization 2: 2 Prizes
Determinization 3: 0 Prizes
```

the conservative search criterion can reject the override.

The purpose of DFS is therefore not to replace learned strategy, but to catch tactical opportunities that can be verified reliably by the simulator.

---

## 8. Why Not Search Everything?

Several search-heavy approaches were explored during development.

The main problem is that deeper search is only useful when both of the following are sufficiently accurate:

```text
Hidden-state assumptions
+
Leaf-state evaluation
```

In a partially observable card game, errors compound quickly:

```text
Incorrect hidden-state assumption
             +
Imperfect evaluation function
             +
Deeper search
             =
Confidently wrong decision
```

The final agent therefore uses the learned BC policy for general strategic judgment and restricts DFS to an objective the simulator can evaluate directly:

```text
Immediate same-turn Prize gain
```

This is the central idea behind the hybrid design:

> **Use learning for general strategy and exact simulation only where it can safely verify a tactical improvement.**

---

## 9. Runtime Fallback

Competition agents operate under strict time limits.

The DFS implementation therefore has bounded runtime and safe fallback behavior.

If search encounters:

```text
timeout
unsupported simulator state
native simulator exception
invalid branch
insufficient remaining budget
```

the system returns to the BC policy.

The runtime hierarchy is:

```text
BC policy
    │
    ▼
Optional DFS verification
    │
    ▼
BC fallback if necessary
```

The agent never depends on DFS succeeding in order to produce a legal action.

---

## 10. Evaluation

The DFS-enhanced policy was evaluated against the pure cross-option BC policy.

Reported internal results include:

| Evaluation | Result |
|---|---:|
| Five-agent round robin | 206–194 |
| DFS+xopt vs pure xopt, 100 games | 50–50 |
| Combined with an earlier 400-game batch | 263–237 |
| Combined win rate | 52.6% |

The combined result numerically favors the DFS-enhanced agent.

However, the available evaluation is not strong enough to establish a statistically significant overall improvement over pure BC.

A conservative interpretation is:

> **Most of the agent's playing strength comes from the behavior-cloned neural policy. DFS can identify useful tactical overrides, but the experiments do not prove that it consistently improves overall win rate.**

---

## Competition Result

The agent was developed for the **Pokémon TCG AI Battle Challenge** and finished **457th** on the competition leaderboard.

The leaderboard result is included as development context rather than as the main contribution of the repository.

The main technical focus of this project is the combination of:

```text
semantic action representation
+
grouped behavior cloning
+
cross-option attention
+
autoregressive selection
+
conservative simulator search
```

---

## Repository Structure

```text
agent/
├── main.py
│   Competition entry point and BC / DFS decision logic
│
├── model.pt
│   Trained behavior-cloning checkpoint
│
├── deck.csv
│   Competition deck
│
├── runtime_policy.py
│   Neural-policy inference
│
├── train_grouped_bc.py
│   Grouped BC training
│
├── encode_bc_shards.py
│   Replay → grouped training-data encoding
│
├── replay_semantics.py
│   Semantic state / action reconstruction
│
└── cg/
    Native simulator components used by DFS

docs/
    Experiment and evaluation notes

release/
    Competition submission files

requirements.txt
README.md
```

---

## Installation

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

## Training

The BC training pipeline has two main stages.

### 1. Encode replay decisions

```text
Replay data
    │
    ▼
encode_bc_shards.py
    │
    ▼
Grouped BC shards
```

### 2. Train the policy

```text
Grouped BC shards
    │
    ▼
train_grouped_bc.py
    │
    ▼
model.pt
```

The checkpoint included in the repository can be used directly for inference.

---

## Reproducibility

The repository includes the main components required to inspect and extend the agent:

- trained policy checkpoint;
- inference code;
- grouped BC training code;
- replay encoding code;
- semantic state/action representation;
- competition deck;
- DFS integration;
- submission code.

The original large replay corpus and pre-encoded training shards are not included.

Therefore, the historical training run cannot be reproduced byte-for-byte from this repository alone.

The code required to reconstruct the replay-to-training pipeline is included.

---

## What This Project Is

This project is primarily:

```text
A behavior-cloned Pokémon TCG policy
        +
dynamic legal-action modeling
        +
cross-option attention
        +
autoregressive selection
        +
conservative simulator verification
```

It is **not**:

```text
a PPO agent
a reinforcement-learning system
a full-game AlphaZero implementation
an MCTS-first agent
a claim that DFS always improves over BC
```

The main neural policy is trained through supervised imitation of replay decisions.

---

## Possible RL Extension

The current architecture also provides a natural starting point for reinforcement learning.

For example:

```text
Replay data
    │
    ▼
Grouped BC
    │
    ▼
Cross-option policy checkpoint
    │
    ▼
Add value head
    │
    ▼
Self-play
    │
    ▼
PPO / actor-critic
    │
    ▼
Opponent population
    │
    ▼
Deck-specific or matchup-specific fine-tuning
```

The existing project already solves several engineering problems that otherwise need to be addressed before self-play RL:

- semantic state representation;
- dynamic legal-action encoding;
- variable-size action sets;
- sequential multi-select decisions;
- simulator integration.

---

## Third-Party Components

The `agent/cg/` directory contains native simulator components associated with the competition environment.

Rights to those components remain with their respective owners.

Before redistributing or repackaging third-party simulator files, verify the applicable competition rules and licensing terms.

---

## Summary

The final agent can be summarized as:

```text
Replay decisions
      │
      ▼
Semantic encoding
      │
      ▼
Grouped behavior cloning
      │
      ▼
Cross-option policy
      │
      ▼
Default BC decision
      │
      ▼
Optional conservative DFS
      │
      ▼
Robust same-turn Prize check
      │
      ▼
Final action
```

The core design principle is:

> **Use learning for general strategy, and use exact simulation only where it can safely prove an immediate tactical improvement.**
