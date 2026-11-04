module

public import Mathlib.Analysis.InnerProductSpace.PiL2
public import PDETranslations.Ordering
import PDETranslations.DegreeLex
import PDETranslations.Truncation
import Mathlib.Analysis.Complex.Exponential
import Mathlib.Algebra.BigOperators.Ring.Finset

@[expose] public section

open scoped BigOperators

namespace PDETranslations

private theorem l2l_index_lt_iff {d : ℕ} (ν : GradedMonomialOrdering d)
    (i p : ℕ) : i < numberOfMultiindices d p ↔
      Multiindex.magnitude (ν.equiv i) ≤ p := by
  have he := initial_segment_eq_multiindicesUpTo ν p
  have hm : ν.equiv i ∈ (Finset.range (numberOfMultiindices d p)).image ν.equiv ↔
      i < numberOfMultiindices d p := by
    simp only [Finset.mem_image, Finset.mem_range, ν.equiv.injective.eq_iff]
    simp
  rw [he] at hm
  simpa only [multiindicesUpTo, Multiindex.mem_upTo] using hm.symm

/-- The reciprocal multiindex factorials in any graded initial segment sum
    to at most `exp d`, uniformly in the truncation degree `p`.

The initial segment is first reindexed by the multiindices of magnitude at
most `p`, then enlarged to the coordinate cube `0 ≤ α i ≤ p`. Nonnegativity
justifies this enlargement. The cube sum factors into one-dimensional
partial exponential sums, each at most `exp 1`. No restriction on `d` or
`p` is needed (in particular, the empty coordinate product is allowed). -/
theorem l2l_sum_inv_factorial_le_exp {d : ℕ} (ν : GradedMonomialOrdering d)
    (p : ℕ) :
    (∑ i : Fin (numberOfMultiindices d p),
      1 / (Multiindex.factorial (ν.equiv i.val) : ℝ)) ≤ Real.exp (d : ℝ) := by
  classical
  have hreindex :
      (∑ i : Fin (numberOfMultiindices d p),
        1 / (Multiindex.factorial (ν.equiv i.val) : ℝ)) =
      ∑ α ∈ Multiindex.upTo (d := d) p, 1 / (Multiindex.factorial α : ℝ) := by
    apply Finset.sum_bij (fun i _ => ν.equiv i.val)
    · intro i hi
      exact Multiindex.mem_upTo.mpr ((l2l_index_lt_iff ν i.val p).mp i.isLt)
    · intro i hi j hj he
      exact Fin.ext (ν.equiv.injective he)
    · intro α hα
      have hi : ν.equiv.symm α < numberOfMultiindices d p :=
        (l2l_index_lt_iff ν _ p).mpr (by simpa using Multiindex.mem_upTo.mp hα)
      exact ⟨⟨ν.equiv.symm α, hi⟩, Finset.mem_univ _, by simp⟩
    · intro i hi; rfl
  let cube := Finset.univ.pi (fun _ : Fin d => Finset.range (p+1))
  let toIndex := fun g : (i : Fin d) → i ∈ (Finset.univ : Finset (Fin d)) → ℕ =>
    Finsupp.equivFunOnFinite.symm (fun i => g i (by simp))
  have hinj : Function.Injective toIndex := by
    intro g h he
    have he' := congrArg (fun α : Multiindex.Index d => Finsupp.equivFunOnFinite α) he
    funext i hi
    exact congrFun he' i
  have hcube : (∑ α ∈ cube.image toIndex, 1 / (Multiindex.factorial α : ℝ)) =
      ∏ _i : Fin d, ∑ n ∈ Finset.range (p+1), 1 / (n.factorial : ℝ) := by
    rw [Finset.sum_image (fun _ _ _ _ he => hinj he)]
    symm
    convert Finset.prod_sum (Finset.univ : Finset (Fin d))
      (fun _ => Finset.range (p+1)) (fun _ n => 1 / (n.factorial : ℝ)) using 1
    apply Finset.sum_congr rfl
    intro g hg
    simp [toIndex, Multiindex.factorial, Nat.cast_prod, Finset.prod_inv_distrib]
  have hone : (∑ n ∈ Finset.range (p+1), 1 / (n.factorial : ℝ)) ≤ Real.exp 1 := by
    simpa using Real.sum_le_exp_of_nonneg (x := 1) (by norm_num) (p+1)
  calc
    _ = ∑ α ∈ Multiindex.upTo (d := d) p, 1 / (Multiindex.factorial α : ℝ) := hreindex
    _ ≤ ∑ α ∈ cube.image toIndex, 1 / (Multiindex.factorial α : ℝ) := by
      apply Finset.sum_le_sum_of_subset_of_nonneg
      · exact Finset.filter_subset _ _
      · intro α hα hnot
        positivity
    _ = ∏ _i : Fin d, ∑ n ∈ Finset.range (p+1), 1 / (n.factorial : ℝ) := hcube
    _ ≤ ∏ _i : Fin d, Real.exp 1 := by
      apply Finset.prod_le_prod₀
      · intro i hi
        exact Finset.sum_nonneg (fun n hn => by positivity)
      · intro i hi
        exact hone
    _ = Real.exp (d : ℝ) := by
      rw [← Real.exp_sum]
      simp

-- Regroup the natural-index interval into whole layers; do not estimate
-- individual multiindices, since that would lose cancellation within a layer.
private theorem l2l_group_tail {d : ℕ} (ν : GradedMonomialOrdering d)
    (p K J : ℕ) (F : Multiindex.Index d → ℂ) :
    (∑ i ∈ Finset.Ico (numberOfMultiindices d (p-K))
        (numberOfMultiindices d (p-J)), F (ν.equiv i)) =
      ∑ n ∈ (Finset.range (p+1)).filter (fun n => p-K < n ∧ n ≤ p-J),
        ∑ q ∈ Multiindex.multiindicesOfMagnitude n, F q := by
  classical
  let test := fun q : Multiindex.Index d => p-K < Multiindex.magnitude q ∧
    Multiindex.magnitude q ≤ p-J
  have hfirst :
      (∑ i ∈ Finset.Ico (numberOfMultiindices d (p-K))
          (numberOfMultiindices d (p-J)), F (ν.equiv i)) =
        ∑ q ∈ (Multiindex.upTo p).filter test, F q := by
    apply Finset.sum_bij (fun i _ => ν.equiv i)
    · intro i hi
      rcases Finset.mem_Ico.mp hi with ⟨hlo, hhi⟩
      have hdeg := (l2l_index_lt_iff ν i (p-J)).mp hhi
      refine Finset.mem_filter.mpr ⟨Multiindex.mem_upTo.mpr
        (le_trans hdeg (Nat.sub_le _ _)), ?_, hdeg⟩
      have := (l2l_index_lt_iff ν i (p-K)).not.mp (by omega)
      omega
    · intro i hi j hj he
      exact ν.equiv.injective he
    · intro q hq
      rcases Finset.mem_filter.mp hq with ⟨hq, ht⟩
      refine ⟨ν.equiv.symm q, Finset.mem_Ico.mpr ⟨?_, ?_⟩, by simp⟩
      · have hh : ¬ ν.equiv.symm q < numberOfMultiindices d (p-K) := by
          rw [l2l_index_lt_iff ν]
          simpa only [Equiv.apply_symm_apply] using (not_le_of_gt ht.1)
        omega
      · apply (l2l_index_lt_iff ν _ (p-J)).mpr
        simpa only [Equiv.apply_symm_apply] using ht.2
    · intro i hi; rfl
  rw [hfirst, Finset.sum_filter,
    Multiindex.sum_upTo_eq_sum_range_multiindicesOfMagnitude, Finset.sum_filter]
  apply Finset.sum_congr rfl
  intro n hn
  by_cases ht : p-K < n ∧ n ≤ p-J
  · simp only [ht]
    apply Finset.sum_congr rfl
    intro q hq
    simp [Multiindex.mem_multiindicesOfMagnitude.mp hq, ht]
  · simp only [ht, ↓reduceIte]
    apply Finset.sum_eq_zero
    intro q hq
    simp [Multiindex.mem_multiindicesOfMagnitude.mp hq, ht]

-- Keep the real power of ρ intact: t need not be nonnegative. Only the
-- integer-degree denominator and the geometric ratio are compared with r.
private theorem l2l_scalar_bound (p K J n s : ℕ) (t C a R R2 r ρ u w B : ℝ)
    (hJK : J < K) (hn : p-K < n ∧ n ≤ p-J)
    (hC : 0 < C) (hR : 0 ≤ R) (hR2 : 0 ≤ R2) (hR2R : R2 ≤ R)
    (hRr : R < r) (hrρ : r ≤ ρ) (hra : r ≤ a)
    (hu : 0 ≤ u) (huR : u ≤ R) (hw : 0 ≤ w) (hwB : w * a^(K-J) ≤ B) :
    w * R2^K * (C * (1+n : ℝ)^s / ρ^((J : ℝ)+t) * (u/ρ)^n) ≤
      C * (p+1 : ℝ)^s * B * (R/r)^(p+1) / ρ^t := by
  have hr : 0 < r := lt_of_le_of_lt hR hRr
  have hρ : 0 < ρ := lt_of_lt_of_le hr hrρ
  have hpow : (R/r)^(n+K) ≤ (R/r)^(p+1) :=
    pow_le_pow_of_le_one (div_nonneg hR hr.le)
      ((div_le_one hr).mpr hRr.le) (by omega)
  have hpoly : (1+n : ℝ)^s ≤ (p+1 : ℝ)^s := by
    apply pow_le_pow_left₀ (by positivity)
    have : n ≤ p := le_trans hn.2 (Nat.sub_le _ _)
    exact_mod_cast (show 1+n ≤ p+1 by omega)
  have hB : 0 ≤ B := le_trans (mul_nonneg hw (pow_nonneg (le_trans hr.le hra) _)) hwB
  have hwB' : w * r^(K-J) ≤ B :=
    le_trans (mul_le_mul_of_nonneg_left (pow_le_pow_left₀ hr.le hra _) hw) hwB
  have hratio : u/ρ ≤ R/r := by
    calc
      u/ρ ≤ R/ρ := div_le_div_of_nonneg_right huR hρ.le
      _ ≤ R/r := div_le_div_of_nonneg_left hR hr hrρ
  have hid : R^K * (C * (1+n : ℝ)^s / (r^J * ρ^t) * (R/r)^n) =
      C * (1+n : ℝ)^s * r^(K-J) * (R/r)^(n+K) / ρ^t := by
    rw [pow_add]
    simp only [div_pow]
    have hp : r^K = r^(K-J) * r^J := by
      rw [← pow_add]; congr 1; omega
    have hrJ : r^J ≠ 0 := pow_ne_zero _ hr.ne'
    have hrK : r^K ≠ 0 := pow_ne_zero _ hr.ne'
    have hrn : r^n ≠ 0 := pow_ne_zero _ hr.ne'
    have hρt : ρ^t ≠ 0 := (Real.rpow_pos_of_pos hρ t).ne'
    field_simp
    rw [hp]
    ring
  calc
    _ ≤ w * (R^K * (C * (1+n : ℝ)^s / (r^J * ρ^t) * (R/r)^n)) := by
      rw [Real.rpow_add hρ, Real.rpow_natCast]
      have hden : r^J * ρ^t ≤ ρ^J * ρ^t := by gcongr
      have hnum : C * (1+n : ℝ)^s / (ρ^J * ρ^t) ≤
          C * (1+n : ℝ)^s / (r^J * ρ^t) :=
        div_le_div_of_nonneg_left (by positivity) (by positivity) hden
      have hgeom : (u/ρ)^n ≤ (R/r)^n :=
        pow_le_pow_left₀ (div_nonneg hu hρ.le) hratio _
      have hrad : R2^K ≤ R^K := pow_le_pow_left₀ hR2 hR2R _
      simpa only [mul_assoc] using
        mul_le_mul_of_nonneg_left
          (mul_le_mul hrad (mul_le_mul hnum hgeom (by positivity) (by positivity))
            (by positivity) (by positivity)) hw
    _ = C * (1+n : ℝ)^s * (w * r^(K-J)) * (R/r)^(n+K) / ρ^t := by
      rw [hid]; ring
    _ ≤ C * (p+1 : ℝ)^s * B * (R/r)^(p+1) / ρ^t := by
      gcongr

/-- The finite, weighted L2L correction tail with the paper's prefactor `p`.

The natural-index interval is half-open and uses the zero-based graded
ordering: it consists exactly of the full layers `p-K < n ≤ p-J`.
`J < K ≤ p` makes every such layer positive, so no bound on degree zero is
needed. Bounds are assumed on the norm of each *layer sum*, not on the sum
of norms; cancellation within each layer is therefore preserved.

`R2 ≤ R < r ≤ ‖z‖` and `‖h‖ ≤ R` control the geometric ratio by `R/r < 1`.
The separate assumption `r ≤ a` (not `‖z‖ ≤ a`) absorbs the degree gap into
`‖m‖ * a^(K-J) ≤ Kprime`. Positivity of `r`, `a`, and `‖z‖` follows from
these radius assumptions, even when `R = 0`. The exponent `t` is any real
number: its denominator is never compared at different radii.

There are at most `p` positive-degree layers, giving exactly `p`, rather
than `p+1`, in front of the bound. `Kprime` is an explicit nonnegative
uniform bound for the weighted coefficient. -/
theorem l2l_weighted_tail_bound {d : ℕ} (ν : GradedMonomialOrdering d)
    (p K J : ℕ) (hJK : J < K) (hKp : K ≤ p)
    (h z : EuclideanSpace ℝ (Fin d)) (R R2 r a C Kprime : ℝ)
    (m : ℂ) (s : ℕ) (t : ℝ)
    (hR : 0 ≤ R) (hR2 : 0 ≤ R2) (hR2R : R2 ≤ R)
    (hRr : R < r) (hrz : r ≤ ‖z‖) (hra : r ≤ a) (hhR : ‖h‖ ≤ R)
    (hC : 0 < C) (hKprime : 0 ≤ Kprime)
    (hm : ‖m‖ * a^(K-J) ≤ Kprime) (F : Multiindex.Index d → ℂ)
    (hlayer : ∀ n : ℕ, 0 < n →
      ‖∑ q ∈ Multiindex.multiindicesOfMagnitude n, F q‖ ≤
        C * (1+n : ℝ)^s / ‖z‖^((J : ℝ)+t) * (‖h‖/‖z‖)^n) :
    ‖m‖ * R2^K *
      ‖∑ q ∈ Finset.Ico (numberOfMultiindices d (p-K))
        (numberOfMultiindices d (p-J)), F (ν.equiv q)‖ ≤
      (p : ℝ) * C * (p+1 : ℝ)^s * Kprime * (R/r)^(p+1) / ‖z‖^t := by
  classical
  let S := (Finset.range (p+1)).filter (fun n => p-K < n ∧ n ≤ p-J)
  let L := fun n => ∑ q ∈ Multiindex.multiindicesOfMagnitude n, F q
  let D := C * (p+1 : ℝ)^s * Kprime * (R/r)^(p+1) / ‖z‖^t
  have hr : 0 < r := lt_of_le_of_lt hR hRr
  have hz : 0 < ‖z‖ := lt_of_lt_of_le hr hrz
  have hD : 0 ≤ D := by dsimp [D]; positivity
  have hpoint (n : ℕ) (hn : n ∈ S) : ‖m‖ * R2^K * ‖L n‖ ≤ D := by
    have hnn := (Finset.mem_filter.mp hn).2
    calc
      _ ≤ ‖m‖ * R2^K *
          (C * (1+n : ℝ)^s / ‖z‖^((J : ℝ)+t) * (‖h‖/‖z‖)^n) :=
        mul_le_mul_of_nonneg_left (hlayer n (by omega)) (by positivity)
      _ ≤ D := l2l_scalar_bound p K J n s t C a R R2 r ‖z‖ ‖h‖ ‖m‖ Kprime
        hJK hnn hC hR hR2 hR2R hRr hrz hra (norm_nonneg _) hhR (norm_nonneg _) hm
  have hcard : (S.card : ℝ) ≤ (p : ℝ) := by
    have hsub : S ⊆ Finset.Ico 1 (p+1) := by
      intro n hn
      have hnn := (Finset.mem_filter.mp hn).2
      apply Finset.mem_Ico.mpr
      constructor <;> omega
    exact_mod_cast (show S.card ≤ p by
      simpa using le_trans (Finset.card_le_card hsub)
        (le_refl (Finset.Ico 1 (p+1)).card))
  rw [l2l_group_tail ν p K J F]
  change ‖m‖ * R2^K * ‖∑ n ∈ S, L n‖ ≤ _
  calc
    _ ≤ ‖m‖ * R2^K * ∑ n ∈ S, ‖L n‖ :=
      mul_le_mul_of_nonneg_left (norm_sum_le S L) (by positivity)
    _ = ∑ n ∈ S, ‖m‖ * R2^K * ‖L n‖ := by rw [Finset.mul_sum]
    _ ≤ ∑ n ∈ S, D := Finset.sum_le_sum hpoint
    _ = (S.card : ℝ) * D := by simp
    _ ≤ (p : ℝ) * D := mul_le_mul_of_nonneg_right hcard hD
    _ = _ := by dsimp [D]; ring

end PDETranslations
