# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a single-file Python game client (`barbouncer.py`) for an API-based "bouncer" challenge. The game presents a stream of people with binary attributes; your goal is to decide which to admit/reject while satisfying constraints (minimum counts of specific attributes among admitted people).

The core task is implementing the decision strategy in `decide_person()`.

## Key Files

- `barbouncer.py` — the entire project: API client, game loop, decision logic, state tracking

## Commands

```bash
# Install dependencies
pip install requests

# Run the game
python barbouncer.py
```

## Configuration (top of barbouncer.py)

- `BASE_URL` — API endpoint URL
- `PLAYER_ID` — your player UUID
- `SCENARIO` — scenario number (currently 2)

## Architecture

- **`GameClient`** — HTTP client wrapping two API calls: `new_game()` (initializes a game with venue capacity, constraints, and attribute statistics) and `decide_and_next()` (submits a decision and fetches the next person)
- **`GameState`** — mutable dataclass holding game metadata (`game_id`, `capacity` = K, the venue capacity), constraints, attribute statistics, running counts of accepted/rejected/seen attributes, the people lists, and `combination_probabilities` (dict of BL-estimated joint probabilities for all 2ⁿ attribute patterns, populated at game start via `GameSmarts`)
- **`decide_person(person, state) -> bool`** — decision function using the ILP optimal acceptance plan. Strategy:
  1. If **all constraints are already satisfied**, accept everyone unconditionally
  2. Otherwise, construct person's attribute combo as a set, if it is super set of or same as any combo in `optimal_acceptance_plan` that still need more admits (accepted < target), **accept** and update that combo against which accepted. If the person can be accepted under more than one combo by this logic, accept them under combo with least probablity and update its count.
  3. If the combo is at its target or not in the plan, **reject**
  4. Prints progress of unsatisfied constraints on rejection
- **`run_game()`** — main loop that fetches people one at a time, calls `decide_person()`, submits decisions, and handles terminal states (`completed` or `failed`)
- **State helpers** — `update_state_after_decision()`, `print_attribute_statistics()`, `print_game_summary()`, `dump_initial_game_info()` for tracking and debugging
- **`GameSmarts`** - implements the following functionality
  - Find ALL combinations of attributes (all 2^n of them)
  - Using Bahadur Lazarsfeld expansion find relative frequency (probablity) of occurance of ALL of them individually. Use only pairwise interactions and ignore higher order interactions. 
    - Result should include label, list of attributes present and probablity.
  - Functionality to find the optimal number of each combination of attributes by solving Integer Linear Programming with following Formulation
    - #### Formulation:
      - Inputs: 
        - From GameState : min_count(j) is min count of jth attribute, P(i) is occurance frequency/probablity of ith attribute combination, **K is the venue capacity** (parsed from the API as `GameState.capacity`) 
      - Output: 
        - x_i = how many people of attribute combination i to accept for each ith attribute combination 
        - Expected number of people rejected R
      - Variables: x_i = how many people of attribute combination i to accept , q
      - **Empty combo `{}` (no attributes) is included in the ILP** — it can receive allocation in the plan
      - Constraints:                                                                                                                                                                                               
        - q >= x_i/P(i) 2^n constrains,for each i < 2^n 
        - sum(x_i) <= K  (fill the club upto capacity)
        - q - Σ x_i >= 0  (R >= 0)
        - For each jth attribute: sum(x_i over combos containing jth attribute) >= min_count(j)                                                                                                                                    
      - Objective:
        - Expected number of people rejected R = q - sum on 0<=i<2^n  (x_i)
  - K (venue capacity) is parsed from the `/new-game` API response and stored in `GameState.capacity`. The ILP is solved once with this K — no brute-force search over K values.
  - **R ≥ 0 is a solver constraint:** `(x_t / P(t)) - Σ x_i >= 0` is added to the ILP for each bottleneck t. Solutions violating this are infeasible by construction, not post-filtered.
## Game Flow

1. `new_game()` returns initial constraints and population attribute statistics
2. Loop: receive a person -> call `decide_person()` -> submit decision -> track state -> repeat
3. Game ends when constraints are met (completed) or a condition fails
