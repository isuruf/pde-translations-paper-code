module

public import Mathlib.Analysis.InnerProductSpace.PiL2
public import PDETranslations.Multiindex
import Mathlib.Analysis.SpecialFunctions.Pow.Real

@[expose] public section

open scoped BigOperators

namespace PDETranslations

/--
Exact summation formula for the polynomial-weighted geometric series that
occurs in the multipole Taylor tail.  The estimate in the paper follows by
bounding this finite Stirling-number expression when `0 ≤ t < 1`.
-/
theorem tsum_polynomial_geometric (s : ℕ) {t : ℝ} (ht : ‖t‖ < 1) :
    ∑' n : ℕ, (n : ℝ) ^ s * t ^ n =
      ∑ j ∈ Finset.range (s + 1),
        (Nat.stirlingSecond s j : ℝ) * (j.factorial : ℝ) * t ^ j /
          (1 - t) ^ (j + 1) := by
  simpa using (tsum_pow_mul_geometric_of_norm_lt_one s ht)

/--
A reindexed form of the scalar tail estimate used in the multipole
truncation bound.  The original tail starts at `k = p + 1`; here
`k = p + 1 + n`.
-/
theorem tsum_tail_polynomial_geometric_le (s p : ℕ) {t : ℝ}
    (ht0 : 0 ≤ t) (ht1 : t < 1) :
    ∑' n : ℕ, (1 + (p + 1 + n : ℕ) : ℝ) ^ s * t ^ (p + 1 + n) ≤
      (s.factorial : ℝ) * (p + 2 : ℝ) ^ s * t ^ (p + 1) /
        (1 - t) ^ (s + 1) := by
  have ht : ‖t‖ < 1 := by
    simpa [Real.norm_eq_abs, abs_of_nonneg ht0] using ht1
  let A : ℝ := (s.factorial : ℝ) * (p + 2 : ℝ) ^ s * t ^ (p + 1)
  let g : ℕ → ℝ := fun n => A * (((n + s).choose s : ℝ) * t ^ n)
  have hg_summable : Summable g := by
    simpa [g] using (summable_choose_mul_geometric_of_norm_lt_one s ht).mul_left A
  have hpoint : ∀ n : ℕ,
      (1 + (p + 1 + n : ℕ) : ℝ) ^ s * t ^ (p + 1 + n) ≤ g n := by
    intro n
    have hbase : (1 + (p + 1 + n : ℕ) : ℝ) ≤ (p + 2 : ℝ) * (n + 1 : ℝ) := by
      norm_num [Nat.cast_add, Nat.cast_one]
      nlinarith [show (0 : ℝ) ≤ (p : ℝ) * n by positivity]
    have hbase_nonneg : 0 ≤ (1 + (p + 1 + n : ℕ) : ℝ) := by positivity
    have hpow : (1 + (p + 1 + n : ℕ) : ℝ) ^ s ≤
        (p + 2 : ℝ) ^ s * (n + 1 : ℝ) ^ s := by
      rw [← mul_pow]
      exact pow_le_pow_left₀ hbase_nonneg hbase s
    have hfac : 0 < (s.factorial : ℝ) := by positivity
    have hchoose : (n + 1 : ℝ) ^ s ≤ (s.factorial : ℝ) * ((n + s).choose s : ℝ) := by
      rw [mul_comm]
      apply (div_le_iff₀ hfac).mp
      have hsub : n + s + 1 - s = n + 1 := by omega
      simpa [hsub] using (Nat.pow_le_choose (α := ℝ) s (n + s))
    rw [show p + 1 + n = (p + 1) + n by omega, pow_add]
    dsimp [g, A]
    calc
      (1 + (p + 1 + n : ℕ) : ℝ) ^ s * (t ^ (p + 1) * t ^ n)
          ≤ ((p + 2 : ℝ) ^ s * (n + 1 : ℝ) ^ s) * (t ^ (p + 1) * t ^ n) := by
            gcongr
      _ ≤ ((p + 2 : ℝ) ^ s * ((s.factorial : ℝ) * ((n + s).choose s : ℝ))) *
            (t ^ (p + 1) * t ^ n) := by gcongr
      _ = (s.factorial : ℝ) * (p + 2 : ℝ) ^ s * t ^ (p + 1) *
            ((n + s).choose s : ℝ) * t ^ n := by ring
      _ ≤ (s.factorial : ℝ) * (p + 2 : ℝ) ^ s * t ^ (p + 1) *
            (((n + s).choose s : ℝ) * t ^ n) := by
        ring_nf
        exact le_rfl
  have hf_summable : Summable (fun n : ℕ =>
      (1 + (p + 1 + n : ℕ) : ℝ) ^ s * t ^ (p + 1 + n)) :=
    Summable.of_nonneg_of_le (fun n => by positivity) hpoint hg_summable
  calc
    ∑' n : ℕ, (1 + (p + 1 + n : ℕ) : ℝ) ^ s * t ^ (p + 1 + n)
        ≤ ∑' n : ℕ, g n := hf_summable.tsum_le_tsum hpoint hg_summable
    _ = A * (1 / (1 - t) ^ (s + 1)) := by
      rw [show g = fun n => A * (((n + s).choose s : ℝ) * t ^ n) by rfl, tsum_mul_left,
        tsum_choose_mul_geometric_of_norm_lt_one s ht]
    _ = (s.factorial : ℝ) * (p + 2 : ℝ) ^ s * t ^ (p + 1) /
        (1 - t) ^ (s + 1) := by
      dsimp [A]
      rw [div_eq_mul_inv]
      ring

/--
The derivative estimate assumed of a translation-invariant kernel in equation
`(assumption_g)` of the paper.

The explicit real parameter `p'` is uniform in the expansion order `p`, and
`s : ℕ → ℕ` specifies `s_m`; for each `m`, the positive constant `C_{G,m}`
may be chosen uniformly
in `r`, `k`, `z`, and `h`; no smallness of `h` relative to `z` is assumed
(only `z ≠ 0`, since the right-hand side degenerates at `z = 0`).  Spatial vectors use `EuclideanSpace ℝ (Fin d)`,
so `‖·‖` is the Euclidean 2-norm throughout the estimates below.  Only
positive Taylor layers `k > 0` are bounded; the zeroth layer is unrestricted.
-/
noncomputable def SatisfiesDerivativeEstimate
    {d : ℕ} (G : (Fin d → ℝ) → ℂ) (p' : ℝ) (s : ℕ → ℕ) : Prop :=
  ∀ m : ℕ, ∃ C : ℝ, 0 < C ∧
    ∀ r ∈ Multiindex.upTo m, ∀ k : ℕ, ∀ z h : EuclideanSpace ℝ (Fin d),
      z ≠ 0 ∧ k > 0 →
        ‖∑ q ∈ Multiindex.multiindicesOfMagnitude k,
            (Multiindex.deriv (q + r) G z / (Multiindex.factorial q : ℝ)) *
              Multiindex.vectorPower h q‖
          ≤ (C * (1 + k : ℝ) ^ s m / ‖z‖ ^ ((Multiindex.magnitude r : ℝ) + p')) *
              (‖h‖ / ‖z‖) ^ k

/--
`G` agrees with its multi-index Taylor series on every ball about a nonzero
point whose radius is less than that point's norm.  This is the analytic
reconstruction step needed to turn derivative estimates into truncation
estimates.
-/
noncomputable def HasLocalMultiindexTaylorSeries
    {d : ℕ} (G : (Fin d → ℝ) → ℂ) : Prop :=
  ∀ z h : EuclideanSpace ℝ (Fin d), ‖h‖ < ‖z‖ →
    G (z + h) =
      ∑' k : ℕ, ∑ q ∈ Multiindex.multiindicesOfMagnitude k,
        (Multiindex.deriv q G z / (Multiindex.factorial q : ℝ)) *
          Multiindex.vectorPower h q

/--
The order-`p` multipole truncation estimate of equation `(mpole_truncated)`.

For a source `y` in the ball of radius `R` about `c` and a target `x` outside
that ball, the order-`p` Taylor expansion of the translation-invariant kernel
`G (x - y)` about `x - c` has the stated error bound.  The parameters `p'`,
`s`, and `C` correspond respectively to `p'`, `s_0`, and `C_{G,0}` from
`SatisfiesDerivativeEstimate`.
-/
noncomputable def SatisfiesMultipoleTruncationBound
    {d : ℕ} (G : (Fin d → ℝ) → ℂ) (p : ℕ) (p' : ℝ) (s : ℕ) (C : ℝ) : Prop :=
  0 < C ∧ ∀ R : ℝ, ∀ x y c : EuclideanSpace ℝ (Fin d),
    ‖y - c‖ < R → R < ‖x - c‖ →
      ‖G (x - y) -
          ∑ q ∈ Multiindex.upTo p,
            (Multiindex.deriv q G (x - c) / (Multiindex.factorial q : ℝ)) *
              Multiindex.vectorPower (c - y) q‖
        ≤ ((s.factorial : ℝ) * (p + 2 : ℝ) ^ s * C / ‖x - c‖ ^ p') *
            (R / ‖x - c‖) ^ (p + 1) /
              (1 - R / ‖x - c‖) ^ (s + 1)

/--
The order-`p` local truncation estimate of equation `(local_truncated)`.

For a target `x` in the ball of radius `R` about `c` and a source `y` outside
that ball, expand `G (x - y)` about `c - y` with displacement `x - c`.
The parameters `p'`, `s`, and `C` are the same derivative-estimate constants
as in `SatisfiesMultipoleTruncationBound`.
-/
noncomputable def SatisfiesLocalTruncationBound
    {d : ℕ} (G : (Fin d → ℝ) → ℂ) (p : ℕ) (p' : ℝ) (s : ℕ) (C : ℝ) : Prop :=
  0 < C ∧ ∀ R : ℝ, ∀ x y c : EuclideanSpace ℝ (Fin d),
    ‖x - c‖ < R → R < ‖y - c‖ →
      ‖G (x - y) -
          ∑ q ∈ Multiindex.upTo p,
            (Multiindex.deriv q G (c - y) / (Multiindex.factorial q : ℝ)) *
              Multiindex.vectorPower (x - c) q‖
        ≤ ((s.factorial : ℝ) * (p + 2 : ℝ) ^ s * C / ‖y - c‖ ^ p') *
            (R / ‖y - c‖) ^ (p + 1) /
              (1 - R / ‖y - c‖) ^ (s + 1)

/-- Polynomial-weighted geometric series remain summable after shifting the order. -/
theorem summable_tail_polynomial_geometric (s p : ℕ) {t : ℝ}
    (ht0 : 0 ≤ t) (ht1 : t < 1) :
    Summable (fun n : ℕ =>
      (1 + (p + 1 + n : ℕ) : ℝ) ^ s * t ^ (p + 1 + n)) := by
  have ht : ‖t‖ < 1 := by
    simpa [Real.norm_eq_abs, abs_of_nonneg ht0] using ht1
  let A : ℝ := (s.factorial : ℝ) * (p + 2 : ℝ) ^ s * t ^ (p + 1)
  have hg : Summable (fun n : ℕ => A * (((n + s).choose s : ℝ) * t ^ n)) :=
    (summable_choose_mul_geometric_of_norm_lt_one s ht).mul_left A
  refine Summable.of_nonneg_of_le (fun _ => by positivity) (fun n => ?_) hg
  have hbase : (1 + (p + 1 + n : ℕ) : ℝ) ≤ (p + 2 : ℝ) * (n + 1 : ℝ) := by
    norm_num [Nat.cast_add, Nat.cast_one]
    nlinarith [show (0 : ℝ) ≤ (p : ℝ) * n by positivity]
  have hpow : (1 + (p + 1 + n : ℕ) : ℝ) ^ s ≤
      (p + 2 : ℝ) ^ s * (n + 1 : ℝ) ^ s := by
    rw [← mul_pow]
    exact pow_le_pow_left₀ (by positivity) hbase s
  have hchoose : (n + 1 : ℝ) ^ s ≤ (s.factorial : ℝ) * ((n + s).choose s : ℝ) := by
    rw [mul_comm]
    apply (div_le_iff₀ (show (0 : ℝ) < s.factorial by positivity)).mp
    have hsub : n + s + 1 - s = n + 1 := by omega
    simpa [hsub] using (Nat.pow_le_choose (α := ℝ) s (n + s))
  rw [pow_add]
  calc
    (1 + (p + 1 + n : ℕ) : ℝ) ^ s * (t ^ (p + 1) * t ^ n)
        ≤ ((p + 2 : ℝ) ^ s * (n + 1 : ℝ) ^ s) * (t ^ (p + 1) * t ^ n) := by
          gcongr
    _ ≤ ((p + 2 : ℝ) ^ s * ((s.factorial : ℝ) * ((n + s).choose s : ℝ))) *
        (t ^ (p + 1) * t ^ n) := by gcongr
    _ = A * (((n + s).choose s : ℝ) * t ^ n) := by dsimp [A]; ring

/-- Equation `(mpole_truncated)`, for fixed derivative-estimate constants.

Only the zero-multi-index case of the derivative estimate is needed, and
only for positive Taylor layers: the tail starts at `p + 1 > 0`.  The
Taylor-series reconstruction hypothesis identifies the kernel with the series
whose tail is estimated; it is not merely an identity for an exact remainder.
-/
theorem multipole_truncation_bound_of_layer_bound
    {d : ℕ} {G : (Fin d → ℝ) → ℂ} (hTaylor : HasLocalMultiindexTaylorSeries G)
    (p : ℕ) (p' : ℝ) (s : ℕ) (C : ℝ) (hC : 0 < C)
    (hbound : ∀ k : ℕ, ∀ z h : EuclideanSpace ℝ (Fin d), z ≠ 0 ∧ k > 0 →
      ‖∑ q ∈ Multiindex.multiindicesOfMagnitude k,
          (Multiindex.deriv q G z / (Multiindex.factorial q : ℝ)) *
            Multiindex.vectorPower h q‖ ≤
        (C * (1 + k : ℝ) ^ s / ‖z‖ ^ p') * (‖h‖ / ‖z‖) ^ k) :
    SatisfiesMultipoleTruncationBound G p p' s C := by
  refine ⟨hC, ?_⟩
  intro R x y c hy hx
  have hR : 0 < R := lt_of_le_of_lt (norm_nonneg _) hy
  have hz : 0 < ‖x - c‖ := hR.trans hx
  have hh : ‖c - y‖ < R := by simpa only [norm_sub_rev] using hy
  have hlocal : ‖c - y‖ < ‖x - c‖ := hh.trans hx
  let t : ℝ := R / ‖x - c‖
  let A : ℝ := C / ‖x - c‖ ^ p'
  let f : ℕ → ℂ := fun k =>
    ∑ q ∈ Multiindex.multiindicesOfMagnitude k,
      (Multiindex.deriv q G (x - c) / (Multiindex.factorial q : ℝ)) *
        Multiindex.vectorPower (c - y) q
  have ht0 : 0 ≤ t := le_of_lt (div_pos hR hz)
  have ht1 : t < 1 := (div_lt_one hz).mpr hx
  have hA : 0 ≤ A := le_of_lt (div_pos hC (Real.rpow_pos_of_pos hz _))
  have hfbound (k : ℕ) (hk : k > 0) :
        ‖f k‖ ≤ A * ((1 + k : ℝ) ^ s * t ^ k) := by
    have hratio : ‖c - y‖ / ‖x - c‖ ≤ t :=
      (div_le_div_iff_of_pos_right hz).mpr hh.le
    calc
      ‖f k‖ ≤ (C * (1 + k : ℝ) ^ s / ‖x - c‖ ^ p') *
          (‖c - y‖ / ‖x - c‖) ^ k := by
        simpa only [f, Real.norm_eq_abs, WithLp.ofLp_sub] using
                  hbound k (x - c) (c - y) ⟨norm_pos_iff.mp hz, hk⟩
      _ ≤ (C * (1 + k : ℝ) ^ s / ‖x - c‖ ^ p') * t ^ k := by
        gcongr
      _ = A * ((1 + k : ℝ) ^ s * t ^ k) := by dsimp [A]; ring
  have hmajor : Summable (fun n : ℕ =>
      A * ((1 + (p + 1 + n : ℕ) : ℝ) ^ s * t ^ (p + 1 + n))) :=
    (summable_tail_polynomial_geometric s p ht0 ht1).mul_left A
  have htailnorm : Summable (fun n : ℕ => ‖f (n + (p + 1))‖) :=
    Summable.of_nonneg_of_le (fun _ => norm_nonneg _) (fun n => by
      simpa only [Nat.add_comm] using hfbound (n + (p + 1)) (by omega)) hmajor
  have hf : Summable f := (summable_nat_add_iff (p + 1)).mp htailnorm.of_norm
  have hkernel : G (x - y) = ∑' k : ℕ, f k := by
    have heq : x - c + (c - y) = x - y := by abel
    simpa only [f, ← WithLp.ofLp_add, ← WithLp.ofLp_sub, heq] using
          hTaylor (x - c) (c - y) hlocal
  have hremainder : G (x - y) -
      ∑ q ∈ Multiindex.upTo p,
        (Multiindex.deriv q G (x - c) / (Multiindex.factorial q : ℝ)) *
          Multiindex.vectorPower (c - y) q = ∑' n : ℕ, f (n + (p + 1)) := by
    rw [hkernel, Multiindex.sum_upTo_eq_sum_range_multiindicesOfMagnitude]
    have hsplit := hf.sum_add_tsum_nat_add (p + 1)
    dsimp only [f] at hsplit ⊢
    rw [← hsplit]
    abel
  rw [hremainder]
  change ‖(∑' n : ℕ, f (n + (p + 1)) : ℂ)‖ ≤ _
  calc
    ‖∑' n : ℕ, f (n + (p + 1))‖ ≤ ∑' n : ℕ, ‖f (n + (p + 1))‖ :=
      norm_tsum_le_tsum_norm htailnorm
    _ ≤ ∑' n : ℕ,
        A * ((1 + (p + 1 + n : ℕ) : ℝ) ^ s * t ^ (p + 1 + n)) :=
      htailnorm.tsum_le_tsum (fun n => by
        simpa only [Nat.add_comm] using hfbound (n + (p + 1)) (by omega)) hmajor
    _ = A * (∑' n : ℕ,
        (1 + (p + 1 + n : ℕ) : ℝ) ^ s * t ^ (p + 1 + n)) := tsum_mul_left
    _ ≤ A * ((s.factorial : ℝ) * (p + 2 : ℝ) ^ s * t ^ (p + 1) /
        (1 - t) ^ (s + 1)) :=
      mul_le_mul_of_nonneg_left (tsum_tail_polynomial_geometric_le s p ht0 ht1) hA
    _ = ((s.factorial : ℝ) * (p + 2 : ℝ) ^ s * C / ‖x - c‖ ^ p') *
        (R / ‖x - c‖) ^ (p + 1) / (1 - R / ‖x - c‖) ^ (s + 1) := by
      dsimp [A, t]
      ring

/-- The derivative estimate and local Taylor reconstruction imply the paper's
multipole truncation bound, with the same constants for every expansion order. -/
theorem multipole_truncation_bound
    {d : ℕ} {G : (Fin d → ℝ) → ℂ} {p' : ℝ} {s : ℕ → ℕ}
    (hDerivative : SatisfiesDerivativeEstimate G p' s)
    (hTaylor : HasLocalMultiindexTaylorSeries G) :
    ∃ C : ℝ, ∀ p : ℕ,
      SatisfiesMultipoleTruncationBound G p p' (s 0) C := by
  obtain ⟨C, hC, hbound⟩ := hDerivative 0
  refine ⟨C, fun p => ?_⟩
  refine multipole_truncation_bound_of_layer_bound hTaylor p p' (s 0) C hC ?_
  intro k z h hlocal
  have hzero : (0 : Multiindex.Index d) ∈ Multiindex.upTo 0 := by
    simp [Multiindex.mem_upTo, Multiindex.magnitude]
  simpa [Multiindex.magnitude] using hbound 0 hzero k z h hlocal

/-- Local and multipole truncation reuse the same Taylor-tail estimate.

Applying the multipole bound to `(-y, -x, -c)` preserves `x - y` and gives
expansion point `c - y` and displacement `x - c`.  No symmetry of `G` is needed.
-/
theorem local_truncation_bound_of_multipole_bound
    {d : ℕ} {G : (Fin d → ℝ) → ℂ} {p : ℕ} {p' : ℝ} {s : ℕ} {C : ℝ}
    (hMultipole : SatisfiesMultipoleTruncationBound G p p' s C) :
    SatisfiesLocalTruncationBound G p p' s C := by
  obtain ⟨hC, hbound⟩ := hMultipole
  refine ⟨hC, ?_⟩
  intro R x y c hx hy
  have hxy : -y - -x = x - y := by abel
  have hcy : -y - -c = c - y := by abel
  have hxc : -c - -x = x - c := by abel
  have hcx : -x - -c = c - x := by abel
  have hx' : ‖-x - -c‖ < R := by
    simpa only [hcx, norm_sub_rev] using hx
  have hy' : R < ‖-y - -c‖ := by
    simpa only [hcy, norm_sub_rev] using hy
  simpa only [← WithLp.ofLp_sub, hxy, hcy, hxc, norm_sub_rev, abs_norm] using
      hbound R (-y) (-x) (-c) hx' hy'

/-- Equation `(local_truncated)`, for fixed derivative-estimate constants. -/
theorem local_truncation_bound_of_layer_bound
    {d : ℕ} {G : (Fin d → ℝ) → ℂ} (hTaylor : HasLocalMultiindexTaylorSeries G)
    (p : ℕ) (p' : ℝ) (s : ℕ) (C : ℝ) (hC : 0 < C)
    (hbound : ∀ k : ℕ, ∀ z h : EuclideanSpace ℝ (Fin d), z ≠ 0 ∧ k > 0 →
      ‖∑ q ∈ Multiindex.multiindicesOfMagnitude k,
          (Multiindex.deriv q G z / (Multiindex.factorial q : ℝ)) *
            Multiindex.vectorPower h q‖ ≤
        (C * (1 + k : ℝ) ^ s / ‖z‖ ^ p') * (‖h‖ / ‖z‖) ^ k) :
    SatisfiesLocalTruncationBound G p p' s C := by
  exact local_truncation_bound_of_multipole_bound
    (multipole_truncation_bound_of_layer_bound hTaylor p p' s C hC hbound)

/-- The derivative estimate and local Taylor reconstruction imply the paper's
local truncation bound, with the same constants as the multipole bound. -/
theorem local_truncation_bound
    {d : ℕ} {G : (Fin d → ℝ) → ℂ} {p' : ℝ} {s : ℕ → ℕ}
    (hDerivative : SatisfiesDerivativeEstimate G p' s)
    (hTaylor : HasLocalMultiindexTaylorSeries G) :
    ∃ C : ℝ, ∀ p : ℕ,
      SatisfiesLocalTruncationBound G p p' (s 0) C := by
  obtain ⟨C, hbound⟩ := multipole_truncation_bound hDerivative hTaylor
  exact ⟨C, fun p => local_truncation_bound_of_multipole_bound (hbound p)⟩

end PDETranslations
