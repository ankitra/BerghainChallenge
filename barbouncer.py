from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from itertools import combinations

import numpy as np
import requests
from scipy.optimize import milp, LinearConstraint, Bounds


# ============================================================
# Configuration
# ============================================================

BASE_URL = "https://berghain.challenges.listenlabs.ai/"  # <-- Fill this in
PLAYER_ID = "680c376f-b3c0-4979-8ec2-16541e1fb69a"
SCENARIO = 3


# ============================================================
# Data Structures
# ============================================================

@dataclass
class Constraint:
    attribute: str
    min_count: int


@dataclass
class AttributeStatistics:
    relative_frequencies: Dict[str, float]
    correlations: Dict[str, Dict[str, float]]


@dataclass
class Person:
    person_index: int
    attributes: Dict[str, bool]


@dataclass
class GameState:
    game_id: str
    constraints: List[Constraint]
    attribute_statistics: AttributeStatistics
    capacity: int  # venue capacity K

    admitted_count: int = 0
    rejected_count: int = 0

    accepted_people: List[Person] = field(default_factory=list)
    rejected_people: List[Person] = field(default_factory=list)

    # Running counts of accepted attributes
    accepted_attribute_counts: Dict[str, int] = field(default_factory=dict)

    # Running counts of rejected attributes
    rejected_attribute_counts: Dict[str, int] = field(default_factory=dict)

    # Total seen counts regardless of decision
    seen_attribute_counts: Dict[str, int] = field(default_factory=dict)

    # BL combination probabilities (populated at game start)
    combination_probabilities: list = field(default_factory=list)

    # Optimal acceptance plan: combo label -> target count
    optimal_acceptance_plan: Dict[str, int] = field(default_factory=dict)

    # Running count of accepted people per combo label
    accepted_combo_counts: Dict[str, int] = field(default_factory=dict)

    # Running count of all seen people per combo label
    seen_combo_counts: Dict[str, int] = field(default_factory=dict)

    # Mapping: frozenset of present attributes -> combo label
    combo_label_lookup: Dict[frozenset, str] = field(default_factory=dict)

    # Mapping: combo label -> probability (for rarest-combo selection)
    _combo_prob_lookup: Dict[str, float] = field(default_factory=dict)

    # Internal: which plan combo was matched by decide_person (superset matching)
    _matched_plan_combo: Optional[str] = None

    status: str = "running"

# ============================================================
# GameSmarts — Attribute Combination Probability Calculator
# ============================================================

@dataclass
class CombinationResult:
    label: str
    attributes: List[str]
    probability: float


@dataclass
class GameSmarts:
    """
    Uses Bahadur-Lazarsfeld expansion (pairwise only) to estimate
    joint probabilities for all 2^n attribute combinations.
    """

    attribute_statistics: AttributeStatistics

    def __post_init__(self):
        self.attribute_names = sorted(
            self.attribute_statistics.relative_frequencies.keys()
        )
        self.n = len(self.attribute_names)
        print(f"  🤖 GameSmarts initialised with {self.n} attributes: {self.attribute_names}")
        print(f"  📊 Ready to compute BL probabilities for 2^{self.n} = {2**self.n} combinations")

    def _marginal(self, attr: str) -> float:
        return self.attribute_statistics.relative_frequencies[attr]

    def _correlation(self, attr1: str, attr2: str) -> float:
        return self.attribute_statistics.correlations.get(attr1, {}).get(attr2, 0.0)

    def _z_score(self, attr: str, present: bool) -> float:
        """Standardized value z_i = (x_i - p_i) / sqrt(p_i (1-p_i))."""
        p = self._marginal(attr)
        variance = p * (1.0 - p)
        if variance == 0.0:
            return 0.0  # deterministic attribute: no interaction
        if present:
            return (1.0 - p) / (variance ** 0.5)
        else:
            return (-p) / (variance ** 0.5)

    def _pairwise_correction(self, pattern: Dict[str, bool]) -> float:
        """Compute 1 + Σ ρ_ij z_i z_j for the given pattern."""
        correction = 1.0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                attr_i = self.attribute_names[i]
                attr_j = self.attribute_names[j]
                rho = self._correlation(attr_i, attr_j)
                if rho == 0.0:
                    continue
                zi = self._z_score(attr_i, pattern[attr_i])
                zj = self._z_score(attr_j, pattern[attr_j])
                correction += rho * zi * zj
        return correction

    def probability_of(self, pattern: Dict[str, bool]) -> float:
        """
        Joint probability P(X = pattern) using the Bahadur-Lazarsfeld expansion
        truncated to pairwise interactions.
        """
        # Π p_i^{x_i} (1-p_i)^{1-x_i}
        product = 1.0
        for attr, present in pattern.items():
            p = self._marginal(attr)
            product *= p if present else (1.0 - p)

        correction = self._pairwise_correction(pattern)
        return max(0.0, product * correction)

    def _pattern_from_subset(self, true_attrs) -> Dict[str, bool]:
        return {a: a in true_attrs for a in self.attribute_names}

    def all_combinations(self) -> List[Dict[str, bool]]:
        """Generate all 2^n attribute-on/off patterns."""
        print(f"  🔄 Generating all 2^{self.n} attribute-on/off patterns...")
        patterns = []
        for r in range(self.n + 1):
            for combo in combinations(self.attribute_names, r):
                patterns.append(self._pattern_from_subset(combo))
        print(f"  ✅ Generated {len(patterns)} patterns")
        return patterns

    def all_combination_probabilities(self) -> List[CombinationResult]:
        """All 2^n results: label, list of present attributes, BL probability."""
        print(f"  📐 Computing Bahadur-Lazarsfeld pairwise probabilities for all patterns...")
        results = []
        patterns = self.all_combinations()
        for idx, pattern in enumerate(patterns):
            true_attrs = sorted(
                attr for attr, v in pattern.items() if v
            )
            label = "{" + ",".join(true_attrs) + "}"
            prob = self.probability_of(pattern)
            results.append(CombinationResult(
                label=label,
                attributes=true_attrs,
                probability=prob,
            ))
            if (idx + 1) % (2 ** max(1, self.n - 3)) == 0:
                print(f"    ⏳ Progress: {idx + 1}/{len(patterns)} patterns computed...")
        results.sort(key=lambda r: -r.probability)
        print(f"  ✅ BL probability computation complete!")
        print(f"  🏆 Top 3 most likely combos:")
        for r in results[:3]:
            print(f"      {r.label}: {r.probability:.4%}")
        return results

    def _build_combo_attr_matrix(
        self, combos: List[CombinationResult], attr_names: List[str]
    ) -> np.ndarray:
        """Binary matrix A[j, i] = 1 if combo i contains attribute j."""
        M = len(combos)
        n = len(attr_names)
        attr_to_idx = {a: i for i, a in enumerate(attr_names)}
        A = np.zeros((n, M), dtype=np.float64)
        for i, combo in enumerate(combos):
            for attr in combo.attributes:
                if attr in attr_to_idx:
                    A[attr_to_idx[attr], i] = 1.0
        return A

    def solve_optimal_acceptance(
        self,
        combos: List[CombinationResult],
        constraints: List[Constraint],
        K: int,
    ) -> Tuple[Dict[str, int], float]:
        """
        Solve the ILP using the epigraph formulation.

        Variables:  x_i = admits of combo i,  q = max_i(x_i / P(i))
        Objective:  minimise R = q - Σ x_i

        Constraints:
          q >= x_i / P(i)               for each combo i  (epigraph)
          sum(x_i) <= K                 (at most venue capacity)
          sum_{i: j in i} x_i >= min_count(j)   (attribute lower bounds)
          q - Σ x_i >= 0                (R >= 0)

        Returns (plan {label -> count}, best_R).
        If no feasible solution is found, returns ({}, inf).
        """
        M = len(combos)
        P = np.array([c.probability for c in combos], dtype=np.float64)

        attr_names = sorted({c.attribute for c in constraints})
        n = len(attr_names)

        min_counts = np.array([
            next(c.min_count for c in constraints if c.attribute == a)
            for a in attr_names
        ], dtype=np.float64)

        A = self._build_combo_attr_matrix(combos, attr_names)

        # ---------------------------------------------------------------
        # Build single ILP: variables = [x_0 ... x_{M-1}  q]  (M+1 total)
        # ---------------------------------------------------------------
        M_var = M + 1
        q_idx = M  # index of q in the variable vector

        # --- Group 1: q >= x_i / P(i)  =>  -x_i + P(i)*q >= 0  (M rows) ---
        A_q = np.zeros((M, M_var), dtype=np.float64)
        for i in range(M):
            A_q[i, i] = -1.0           # coefficient for x_i
            A_q[i, q_idx] = P[i]     # coefficient for q
        lb_q = np.zeros(M)           # >= 0
        ub_q = np.full(M, np.inf)

        # --- Group 2: 0 <= sum(x_i) <= K  (1 row) ---
        A_sum = np.zeros((1, M_var), dtype=np.float64)
        A_sum[0, :M] = 1.0
        lb_sum = np.array([0.0])
        ub_sum = np.array([float(K)])

        # --- Group 3: attribute constraints A @ x >= min_counts  (n rows) ---
        A_attr = np.zeros((n, M_var), dtype=np.float64)
        A_attr[:, :M] = A            # no q coefficient
        lb_attr = min_counts
        ub_attr = np.full(n, np.inf)

        # --- Group 4: q - Σ x_i >= 0  (R >= 0, 1 row) ---
        A_R = np.zeros((1, M_var), dtype=np.float64)
        A_R[0, :M] = -1.0            # -x_i coefficients
        A_R[0, q_idx] = 1.0            # +q
        lb_R = np.array([0.0])
        ub_R = np.array([np.inf])

        # --- Combine ---
        A_total = np.vstack([A_q, A_sum, A_attr, A_R])
        lb_total = np.concatenate([lb_q, lb_sum, lb_attr, lb_R])
        ub_total = np.concatenate([ub_q, ub_sum, ub_attr, ub_R])

        # --- Objective: minimise q - Σ x_i ---
        c = np.zeros(M_var, dtype=np.float64)
        c[:M] = -1.0
        c[q_idx] = 1.0

        # --- Integrality: x_i are integers, q is continuous ---
        integrality = np.zeros(M_var, dtype=np.int32)
        integrality[:M] = 1  # x_i are integer

        print(f"    📐 ILP Constraints (epigraph formulation):")
        print(f"       q ≥ x_i / P(i) for {M} combos (incl. {{}})")
        print(f"       Σ x_i ≤ {K}  (at most venue capacity)")
        for a_idx, attr in enumerate(attr_names):
            included = [combos[i].label for i in range(M) if A[a_idx, i] > 0.5]
            print(f"       Σ[{', '.join(included)}] ≥ {int(min_counts[a_idx])}  (min {attr})")
        print(f"       q - Σ x_i ≥ 0  (R ≥ 0)")
        print(f"       x_i ≥ 0, integer;  q ≥ 0, continuous")
        print(f"       Variables: {M} x_i + 1 q = {M_var} total")
        print(f"  🎯 Solving single ILP (K={K})...")

        result = milp(
            c=c,
            constraints=LinearConstraint(A_total, lb_total, ub_total),
            bounds=Bounds(0, np.inf),
            integrality=integrality,
        )

        if not result.success:
            print(f"  ❌ No feasible ILP solution found for K={K}!")
            return {}, float("inf")

        x = result.x[:M]
        q_val = result.x[q_idx]
        R = float(q_val - np.sum(x))

        print(f"    🧮 Solution: R={R:.2f}, q={q_val:.2f}")
        print(f"       scipy status={result.status}, fun={result.fun:.4f}, message=\"{result.message}\"")
        print(f"       objective c: Σ(-x_i) + q")

        nonzero = [(combos[i].label, int(x[i])) for i in range(M) if x[i] > 0.5]
        if nonzero:
            parts = [f"{label}={count}" for label, count in nonzero]
            print(f"       x_i: {', '.join(parts)}")

        print(f"  🎉 Optimal for K={K}: R={R:.2f}")
        plan = {combos[i].label: int(round(x[i])) for i in range(M)}
        return plan, float(R)


# ============================================================
# API Client
# ============================================================

class GameClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def new_game(self, scenario: int, player_id: str) -> GameState:
        print(f"    📤 GET {self.base_url}/new-game?scenario={scenario}&playerId={player_id[:8]}...")
        response = requests.get(
            f"{self.base_url}/new-game",
            params={
                "scenario": scenario,
                "playerId": player_id,
            },
        )
        response.raise_for_status()
        print(f"    📥 Response: {response.status_code} OK ({len(response.content)} bytes)")

        data = response.json()

        constraints = [
            Constraint(
                attribute=c["attribute"],
                min_count=c["minCount"],
            )
            for c in data["constraints"]
        ]

        attribute_statistics = AttributeStatistics(
            relative_frequencies=data["attributeStatistics"]["relativeFrequencies"],
            correlations=data["attributeStatistics"]["correlations"],
        )

        return GameState(
            game_id=data["gameId"],
            constraints=constraints,
            attribute_statistics=attribute_statistics,
            capacity=data.get("capacity", 1000),
        )

    def decide_and_next(
        self,
        game_id: str,
        person_index: int,
        accept: Optional[bool] = None,
    ) -> dict:
        accept_str = str(accept).lower() if accept is not None else "None"
        print(f"    📤 GET /decide-and-next person={person_index} accept={accept_str}")
        params = {
            "gameId": game_id,
            "personIndex": person_index,
        }

        # First request may omit accept
        if accept is not None:
            params["accept"] = accept_str

        response = requests.get(
            f"{self.base_url}/decide-and-next",
            params=params,
        )
        response.raise_for_status()
        data = response.json()
        print(f"    📥 Response: {response.status_code} OK — status={data.get('status', '?')}")

        return data


# ============================================================
# Decision Logic (YOU IMPLEMENT THIS)
# ============================================================

def decide_person(person: Person, state: GameState) -> bool:
    """
    Return True to accept, False to reject.

    Uses the optimal acceptance plan from GameSmarts (ILP solution).
    Matches via superset: if person's attributes are a superset of any
    plan combo that still needs admits, accept and count toward that combo.
    """
    present_attrs = frozenset(attr for attr, v in person.attributes.items() if v)
    state._matched_plan_combo = None  # reset any prior match

    # ---------------------------------------------------------------
    # 0. If ALL constraints are already satisfied, accept everyone.
    # ---------------------------------------------------------------
    all_satisfied = all(
        state.accepted_attribute_counts.get(c.attribute, 0) >= c.min_count
        for c in state.constraints
    )
    if all_satisfied:
        print(f"      🎯 ALL constraints satisfied! Accepting everyone from now on.")
        return True

    plan = state.optimal_acceptance_plan
    if not plan:
        print(f"      🤷 No acceptance plan — rejecting")
        return False

    # ---------------------------------------------------------------
    # 1. Superset matching: find ALL plan combos that still need admits
    #    whose attributes are a subset of this person's attributes.
    #    Among them, pick the one with the LOWEST probability (rarest).
    # ---------------------------------------------------------------
    matches = []
    for combo_attrs, label in state.combo_label_lookup.items():
        target = plan.get(label, 0)
        if target == 0:
            continue
        if not combo_attrs.issubset(present_attrs):
            continue
        accepted = state.accepted_combo_counts.get(label, 0)
        if accepted >= target:
            continue
        # Look up probability for this label
        prob = state._combo_prob_lookup.get(label, 0.0)
        matches.append((prob, label, combo_attrs, accepted, target))

    if matches:
        # Sort by probability ascending (rarest first)
        matches.sort(key=lambda m: m[0])
        prob, label, combo_attrs, accepted, target = matches[0]
        if len(matches) > 1:
            others = ", ".join(f"{l}({p:.4%})" for p, l, _, _, _ in matches[1:])
            print(f"      🎯 Multiple matches for {set(present_attrs)}, picked rarest: "
                  f"{label} ({prob:.4%}) over {others}")
        print(f"      ✅ Person has {set(present_attrs)} ⊇ {set(combo_attrs)} "
              f"(combo {label}): accepted {accepted + 1}/{target} — accepting")
        state._matched_plan_combo = label
        return True

    # ---------------------------------------------------------------
    # 2. No matching plan combo found — reject.
    # ---------------------------------------------------------------
    # Show why: check if any still-needed combos are subsets
    still_needed = [(a, l) for l in plan
                    for a, lbl in state.combo_label_lookup.items()
                    if lbl == l and plan.get(l, 0) > state.accepted_combo_counts.get(l, 0)]
    if still_needed:
        needed_sets = [set(a) for a, _ in still_needed]
        print(f"      ⏭️  Person has {set(present_attrs)} — "
              f"still need admits for: {needed_sets} (none are subsets)")
    else:
        remaining = sum(max(0, plan.get(l, 0) - state.accepted_combo_counts.get(l, 0)) for l in plan)
        print(f"      ⏭️  All plan combos at target — rejecting (remaining: {remaining} slots)")

    # Print current constraint progress
    unsatisfied = [c for c in state.constraints
                   if state.accepted_attribute_counts.get(c.attribute, 0) < c.min_count]
    if unsatisfied:
        progress = ", ".join(
            f"{c.attribute}: {state.accepted_attribute_counts.get(c.attribute, 0)}/{c.min_count}"
            for c in unsatisfied
        )
        print(f"      📋 Constraints still needed: {progress}")

    return False


# ============================================================
# State Helpers
# ============================================================
def print_attribute_statistics(state: GameState):
    print("\n" + "═" * 65)
    print("  📊 ATTRIBUTE STATISTICS 📊")
    print("═" * 65)

    all_attributes = sorted(
        set(state.seen_attribute_counts.keys())
        | set(state.accepted_attribute_counts.keys())
        | set(state.rejected_attribute_counts.keys())
    )

    if not all_attributes:
        print("  📭 No attributes recorded yet.")
        return

    print(f"  {'Attribute':20s} {'👀 Seen':>8s} {'✅ Acc':>8s} {'🔴 Rej':>8s} {'📈 Rate':>8s}  {'Bar':>20s}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8}  {'-'*20}")
    for attr in all_attributes:
        accepted = state.accepted_attribute_counts.get(attr, 0)
        rejected = state.rejected_attribute_counts.get(attr, 0)
        seen = state.seen_attribute_counts.get(attr, 0)

        acceptance_rate = (
            accepted / seen if seen > 0 else 0.0
        )

        bar_len = max(1, int(acceptance_rate * 20))
        bar = "🟢" * bar_len + "⚪" * (20 - bar_len)

        print(
            f"  {attr:20s} {seen:>8d} {accepted:>8d} {rejected:>8d} {acceptance_rate:>7.2%}  {bar}"
        )
        
def update_state_after_decision(
    state: GameState,
    person: Person,
    accepted: bool,
):
    # Track all seen attributes and combo
    present = [attr for attr, v in person.attributes.items() if v]
    for attr in present:
        state.seen_attribute_counts[attr] = (
            state.seen_attribute_counts.get(attr, 0) + 1
        )
    present_attrs = frozenset(present)
    seen_label = state.combo_label_lookup.get(present_attrs)
    if seen_label is not None:
        state.seen_combo_counts[seen_label] = (
            state.seen_combo_counts.get(seen_label, 0) + 1
        )

    if accepted:
        state.admitted_count += 1
        state.accepted_people.append(person)

        print(f"      ➕ Admitted! Total accepted: {state.admitted_count}, rejected: {state.rejected_count}")

        for attr in present:
            state.accepted_attribute_counts[attr] = (
                state.accepted_attribute_counts.get(attr, 0) + 1
            )

        # Track per-combo accepted count (use plan-matched combo, if any)
        label = state._matched_plan_combo
        if label is not None:
            state.accepted_combo_counts[label] = (
                state.accepted_combo_counts.get(label, 0) + 1
            )

        # Show constraint progress after this acceptance
        print(f"      📊 Constraint progress:")
        for c in state.constraints:
            cur = state.accepted_attribute_counts.get(c.attribute, 0)
            if cur >= c.min_count:
                print(f"        ✅ {c.attribute}: {cur}/{c.min_count} ✓")
            else:
                pct = cur / c.min_count * 100
                print(f"        ⏳ {c.attribute}: {cur}/{c.min_count} ({pct:.0f}%)")

    else:
        state.rejected_count += 1
        state.rejected_people.append(person)

        print(f"      ➖ Rejected! Total accepted: {state.admitted_count}, rejected: {state.rejected_count}")

        for attr in present:
            state.rejected_attribute_counts[attr] = (
                state.rejected_attribute_counts.get(attr, 0) + 1
            )

    # Running totals
    total = state.admitted_count + state.rejected_count
    if total > 0:
        admit_rate = state.admitted_count / total * 100
        print(f"      📈 Running admit rate: {state.admitted_count}/{total} ({admit_rate:.1f}%)")

def print_game_summary(state: GameState):
    print("\n" + "═" * 65)
    print("  📋 FINAL GAME SUMMARY 📋")
    print("═" * 65)

    status_icon = "✅" if state.status == "completed" else "❌"
    print(f"\n  {status_icon} Status: {state.status.upper()}")

    total = state.admitted_count + state.rejected_count
    admit_rate = state.admitted_count / total * 100 if total > 0 else 0

    print(f"  🚪 Total people processed: {total}")
    print(f"  ✅ Accepted: {state.admitted_count} ({admit_rate:.1f}%)")
    print(f"  ❌ Rejected: {state.rejected_count} ({100 - admit_rate:.1f}%)")

    print(f"\n  📊 Accepted Attribute Counts:")
    print(f"  {'Attribute':20s} {'Count':>8s} {'Target':>8s} {'Status':>10s}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*10}")
    for c in state.constraints:
        current = state.accepted_attribute_counts.get(c.attribute, 0)
        status_icon = "✅" if current >= c.min_count else "❌"
        print(f"  {c.attribute:20s} {current:>8d} {c.min_count:>8d} {status_icon:>10s}")

    for attr, count in sorted(state.accepted_attribute_counts.items()):
        if not any(c.attribute == attr for c in state.constraints):
            print(f"  {attr:20s} {count:>8d} {'(no constraint)':>18s}")

    # Combo arrival stats
    if state.seen_combo_counts:
        print(f"\n  📊 Combo Arrival Stats:")
        print(f"  {'Combo':20s} {'Arrived':>8s} {'Accepted':>8s} {'Accept %':>8s}  {'Bar':>20s}")
        print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}  {'-'*20}")
        for label in sorted(state.seen_combo_counts.keys()):
            arrived = state.seen_combo_counts.get(label, 0)
            accepted = state.accepted_combo_counts.get(label, 0)
            pct = accepted / arrived * 100 if arrived > 0 else 0
            bar_len = max(1, int(pct / 5))
            bar = "🟢" * bar_len + "⚪" * (20 - bar_len)
            print(f"  {label:20s} {arrived:>8d} {accepted:>8d} {pct:>7.1f}%  {bar}")

    print("\n" + "═" * 65)

def dump_initial_game_info(state: GameState):
    print("\n" + "═" * 65)
    print("  📋 INITIAL GAME INFO 📋")
    print("═" * 65)

    print(f"\n  🆔 Game ID: {state.game_id}")
    print(f"  🎬 Scenario: {SCENARIO}")
    print(f"  🏟️ Venue capacity: {state.capacity}")

    print(f"\n  📋 Constraints:")
    for c in state.constraints:
        print(f"    🔲 {c.attribute}: need at least {c.min_count}")

    print(f"\n  📊 Relative Frequencies:")
    for attr, freq in sorted(
        state.attribute_statistics.relative_frequencies.items()
    ):
        bar_len = max(1, int(freq * 30))
        bar = "▓" * bar_len + "░" * (30 - bar_len)
        print(f"    {attr:20s} {freq:>7.4f} |{bar}|")

    print(f"\n  🔗 Correlations:")
    has_correlations = any(
        bool(corrs) for corrs in state.attribute_statistics.correlations.values()
    )
    if has_correlations:
        for attr1, correlations in sorted(
            state.attribute_statistics.correlations.items()
        ):
            if correlations:
                for attr2, value in sorted(correlations.items()):
                    arrow = "🟢" if value > 0 else "🔴"
                    print(f"    {arrow} {attr1:15s} <-> {attr2:15s}: {value:>7.3f}")
    else:
        print("    (none)")

    print("═" * 65 + "\n")

# ============================================================
# Main Game Loop
# ============================================================

def run_game():
    print("=" * 65)
    print("  🚪🎵 BERLINCUBE BOUNCER — LET THE BEATS DECIDE! 🎵🚪")
    print("=" * 65)

    client = GameClient(BASE_URL)
    print(f"\n🌐 Connecting to API at {BASE_URL}")

    # Create game
    print(f"📡 Requesting new game (Scenario {SCENARIO}, Player {PLAYER_ID})...")
    state = client.new_game(
        scenario=SCENARIO,
        player_id=PLAYER_ID,
    )

    print(f"✅ Game created successfully!")
    print(f"🆔 Game ID: {state.game_id}")
    print(f"📋 Number of constraints: {len(state.constraints)}")

    dump_initial_game_info(state)

    # Compute and store Bahadur-Lazarsfeld combination probabilities in state
    print("\n" + "═" * 65)
    print("  🧠 PHASE 1: COMPUTING BL PROBABILITIES 🧠")
    print("═" * 65)
    smarts = GameSmarts(state.attribute_statistics)
    state.combination_probabilities = smarts.all_combination_probabilities()
    print("\n--- Combination Probabilities (BL Pairwise) ---")
    for r in state.combination_probabilities:
        bar_len = max(1, int(r.probability * 50))
        bar = "█" * bar_len + "░" * (50 - bar_len)
        print(f"  {r.label:20s} {r.probability:>6.4%}  |{bar}|")
    print()

    # Build combo label lookup: frozenset of present attributes -> label
    print("🔗 Building combo label lookup table...")
    for r in state.combination_probabilities:
        state.combo_label_lookup[frozenset(r.attributes)] = r.label
        state._combo_prob_lookup[r.label] = r.probability
    print(f"✅ Lookup built for {len(state.combo_label_lookup)} combos")

    # Compute optimal acceptance plan
    print("\n" + "═" * 65)
    print("  🔧 PHASE 2: SOLVING OPTIMAL ACCEPTANCE PLAN 🔧")
    print("═" * 65)
    K = state.capacity
    print(f"🏟️ Venue capacity K={K}")

    best_plan, best_R_global = smarts.solve_optimal_acceptance(
        state.combination_probabilities,
        state.constraints,
        K,
    )

    state.optimal_acceptance_plan = best_plan

    if best_plan:
        print(f"\n🏆 OPTIMAL PLAN: K={K}, expected rejects={best_R_global:.2f}")
        print("📋 Optimal Acceptance Plan:")
        print(f"   {'Combo':20s} {'Target':>8s} {'% of K':>8s}")
        print(f"   {'-'*20} {'-'*8} {'-'*8}")
        for label, count in sorted(best_plan.items()):
            if count > 0:
                pct = count / K * 100
                bar = "█" * max(1, int(pct / 5)) + "░" * max(0, 20 - max(1, int(pct / 5)))
                print(f"   {label:20s} {count:>8d} {pct:>7.0f}%  {bar}")
        print(f"   {'TOTAL':20s} {K:>8d}")
    else:
        print(f"\n⚠️  No feasible plan found for K={K} — will rely on fallback logic!")
    print()

    print("\n" + "═" * 65)
    print("  🎧 PHASE 3: THE MAIN EVENT — LET THEM IN! 🎧")
    print("═" * 65)

    # First fetch does not require accept parameter
    print("📨 Fetching first person...")
    response = client.decide_and_next(
        game_id=state.game_id,
        person_index=0,
    )

    person_count = 0
    while True:
        status = response["status"]
        person_count += 1

        # ----------------------------------------------------
        # Terminal states
        # ----------------------------------------------------
        if status == "completed":
            state.status = "completed"
            state.rejected_count = response["rejectedCount"]

            print(f"\n{'=' * 65}")
            print(f"  🎉🎉🎉 GAME COMPLETED SUCCESSFULLY! 🎉🎉🎉")
            print(f"  ✅ All constraints satisfied in {person_count} rounds!")
            print(f"{'=' * 65}")
            break

        if status == "failed":
            state.status = "failed"

            print(f"\n{'💀' * 16}")
            print(f"  ❌ GAME FAILED ❌")
            print(f"  📝 Reason: {response.get('reason')}")
            print(f"{'💀' * 16}")
            break

        # ----------------------------------------------------
        # Running state
        # ----------------------------------------------------
        state.admitted_count = response["admittedCount"]
        state.rejected_count = response["rejectedCount"]

        next_person_data = response["nextPerson"]

        person = Person(
            person_index=next_person_data["personIndex"],
            attributes=next_person_data["attributes"],
        )

        # Build attribute display with emoji indicators
        attr_strs = []
        for attr, val in person.attributes.items():
            icon = "✅" if val else "❌"
            attr_strs.append(f"{attr}={icon}")

        print(f"\n{'─' * 65}")
        print(f"  👤 Person #{person.person_index} arrives at the door...")
        for attr_str in attr_strs:
            print(f"    {attr_str}")
        print(f"{'─' * 65}")

        # ----------------------------------------------------
        # Decision function call
        # ----------------------------------------------------
        accepted = decide_person(person, state)

        if accepted:
            print(f"  🟢  ==> DECISION: LET THEM IN! 🕺💃")
        else:
            print(f"  🔴  ==> DECISION: SORRY, NOT TONIGHT 🚫")

        # Update local tracking state
        update_state_after_decision(
            state=state,
            person=person,
            accepted=accepted,
        )

        # Request next person
        print(f"  📨 Fetching next person...")
        response = client.decide_and_next(
            game_id=state.game_id,
            person_index=person.person_index,
            accept=accepted,
        )

    print_game_summary(state)


# ============================================================
# Entrypoint
# ============================================================

if __name__ == "__main__":
    run_game()
    
