module

public import Mathlib.Analysis.InnerProductSpace.PiL2
public import PDETranslations.Ordering
import PDETranslations.DegreeLex
import PDETranslations.Truncation

@[expose] public section

open scoped BigOperators

namespace PDETranslations

private theorem m2m_index_lt_iff {d : ℕ} (ν : GradedMonomialOrdering d)
    (i p : ℕ) : i < numberOfMultiindices d p ↔
      Multiindex.magnitude (ν.equiv i) ≤ p := by
  have he := initial_segment_eq_multiindicesUpTo ν p
  have hm : ν.equiv i ∈ (Finset.range (numberOfMultiindices d p)).image ν.equiv ↔
      i < numberOfMultiindices d p := by
    simp only [Finset.mem_image, Finset.mem_range, ν.equiv.injective.eq_iff]
    simp
  rw [he] at hm
  simpa only [multiindicesUpTo, Multiindex.mem_upTo] using hm.symm

private theorem m2m_group_tail {d : ℕ} (ν : GradedMonomialOrdering d)
    (p K J : ℕ) (F : Multiindex.Index d → ℂ) :
    (∑ b ∈ (Finset.univ : Finset (Fin (numberOfMultiindices d p))).filter
        (fun b => numberOfMultiindices d (p-K) ≤ b.val ∧
          b.val < numberOfMultiindices d (p-J)), F (ν.equiv b.val)) =
      ∑ n ∈ (Finset.range (p+1)).filter (fun n => p-K < n ∧ n ≤ p-J),
        ∑ q ∈ Multiindex.multiindicesOfMagnitude n, F q := by
  classical
  let test := fun q : Multiindex.Index d => p-K < Multiindex.magnitude q ∧
    Multiindex.magnitude q ≤ p-J
  have hfirst :
      (∑ b ∈ (Finset.univ : Finset (Fin (numberOfMultiindices d p))).filter
        (fun b => numberOfMultiindices d (p-K) ≤ b.val ∧
          b.val < numberOfMultiindices d (p-J)), F (ν.equiv b.val)) =
      ∑ q ∈ (Multiindex.upTo p).filter test, F q := by
    apply Finset.sum_bij (fun b _ => ν.equiv b.val)
    · intro b hb
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hb
      apply Finset.mem_filter.mpr
      refine ⟨Multiindex.mem_upTo.mpr ((m2m_index_lt_iff ν b.val p).mp b.isLt), ?_⟩
      exact ⟨by have := (m2m_index_lt_iff ν b.val (p-K)).not.mp (by omega); omega,
        (m2m_index_lt_iff ν b.val (p-J)).mp hb.2⟩
    · intro b hb c hc he
      exact Fin.ext (ν.equiv.injective he)
    · intro q hq
      rcases Finset.mem_filter.mp hq with ⟨hq, ht⟩
      have hi : ν.equiv.symm q < numberOfMultiindices d p :=
        (m2m_index_lt_iff ν _ p).mpr (by simpa using Multiindex.mem_upTo.mp hq)
      refine ⟨⟨ν.equiv.symm q, hi⟩, ?_, by simp⟩
      simp only [Finset.mem_filter, Finset.mem_univ, true_and]
      constructor
      · have hh : ¬ ν.equiv.symm q < numberOfMultiindices d (p-K) := by
          rw [m2m_index_lt_iff ν]
          simpa only [Equiv.apply_symm_apply] using (not_le_of_gt ht.1)
        omega
      · apply (m2m_index_lt_iff ν _ (p-J)).mpr
        simpa only [Equiv.apply_symm_apply] using ht.2
    · intro b hb; rfl
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

private theorem m2m_scalar_bound (p K J n s : ℕ) (t C a R R1 ρ u : ℝ)
    (hJK : J < K) (hKp : K ≤ p) (hn : p-K < n ∧ n ≤ p-J)
    (hC : 0 < C) (ha : 0 < a) (hR : 0 ≤ R) (hR1 : 0 ≤ R1)
    (hR1R : R1 ≤ R) (hRρ : R < ρ) (hρa : ρ ≤ a)
    (hu : 0 ≤ u) (huR : u ≤ R) :
    R1^K * (C * (1+n : ℝ)^s / ρ^((J : ℝ)+t) * (u/ρ)^n) ≤
      C * (p+1 : ℝ)^s * max 1 (a^p) * (R/ρ)^(p+1) / ρ^t := by
  have hρ : 0 < ρ := lt_of_le_of_lt hR hRρ
  have hr0 : 0 ≤ R/ρ := div_nonneg hR hρ.le
  have hr1 : R/ρ ≤ 1 := (div_le_one hρ).mpr hRρ.le
  have hpow : (R/ρ)^(n+K) ≤ (R/ρ)^(p+1) :=
    pow_le_pow_of_le_one hr0 hr1 (by omega)
  have hpoly : (1+n : ℝ)^s ≤ (p+1 : ℝ)^s := by
    apply pow_le_pow_left₀ (by positivity)
    have : n ≤ p := le_trans hn.2 (Nat.sub_le _ _)
    exact_mod_cast (show 1+n ≤ p+1 by omega)
  have haPow : ρ^(K-J) ≤ max 1 (a^p) := by
    calc
      ρ^(K-J) ≤ a^(K-J) := pow_le_pow_left₀ hρ.le hρa _
      _ ≤ (max 1 a)^(K-J) := pow_le_pow_left₀ ha.le (le_max_right _ _) _
      _ ≤ (max 1 a)^p := pow_le_pow_right₀ (le_max_left _ _) (by omega)
      _ = max 1 (a^p) := by
        rcases le_total a 1 with ha1 | ha1
        · rw [max_eq_left ha1, one_pow, max_eq_left (pow_le_one₀ ha.le ha1)]
        · rw [max_eq_right ha1, max_eq_right (one_le_pow₀ ha1)]
  have hid : R^K * (C * (1+n : ℝ)^s / ρ^((J : ℝ)+t) * (R/ρ)^n) =
      C * (1+n : ℝ)^s * ρ^(K-J) * (R/ρ)^(n+K) / ρ^t := by
    rw [Real.rpow_add hρ, Real.rpow_natCast, pow_add]
    simp only [div_pow]
    have hp : ρ^K = ρ^(K-J) * ρ^J := by
      rw [← pow_add]; congr 1; omega
    have hzJ : ρ^J ≠ 0 := pow_ne_zero _ hρ.ne'
    have hzK : ρ^K ≠ 0 := pow_ne_zero _ hρ.ne'
    have hzn : ρ^n ≠ 0 := pow_ne_zero _ hρ.ne'
    have hzt : ρ^t ≠ 0 := (Real.rpow_pos_of_pos hρ t).ne'
    field_simp
    rw [hp]
    ring
  calc
    _ ≤ R^K * (C * (1+n : ℝ)^s / ρ^((J : ℝ)+t) * (R/ρ)^n) := by
      gcongr
    _ = C * (1+n : ℝ)^s * ρ^(K-J) * (R/ρ)^(n+K) / ρ^t := hid
    _ ≤ C * (p+1 : ℝ)^s * max 1 (a^p) * (R/ρ)^(p+1) / ρ^t := by
      gcongr

/-- Finite M2M correction tail, grouped by total degree before taking norms.

`F` is the normalized Taylor summand (typically
`fun q => (Multiindex.vectorPower h q / (Multiindex.factorial q : ℝ) : ℂ) *
Multiindex.deriv (q + α) G z`, with `Multiindex.magnitude α = J`).
Only bounds on whole positive-degree layers are required, so cancellation
within a layer is preserved. The explicit constant
`‖m‖ * (p+1) * C * (p+1)^s * max 1 (a^p)` is independent of the radii
and of `h,z`. No hypothesis on the degree-zero layer is needed. -/
theorem m2m_weighted_tail_bound {d : ℕ} (ν : GradedMonomialOrdering d)
    (p K J : ℕ) (hJK : J < K) (hKp : K ≤ p)
    (h z : EuclideanSpace ℝ (Fin d)) (R R1 a C : ℝ) (m : ℂ) (s : ℕ) (t : ℝ)
    (hR : 0 ≤ R) (hR1 : 0 ≤ R1) (hR1R : R1 ≤ R)
    (ha : 0 < a) (hhR : ‖h‖ ≤ R) (hRz : R < ‖z‖) (hza : ‖z‖ ≤ a)
    (hC : 0 < C) (F : Multiindex.Index d → ℂ)
    (hlayer : ∀ n : ℕ, 0 < n →
      ‖∑ q ∈ Multiindex.multiindicesOfMagnitude n, F q‖ ≤
        C * (1+n : ℝ)^s / ‖z‖^((J : ℝ)+t) * (‖h‖/‖z‖)^n) :
    ‖m‖ * R1^K *
      ‖∑ b ∈ (Finset.univ : Finset (Fin (numberOfMultiindices d p))).filter
        (fun b => numberOfMultiindices d (p-K) ≤ b.val ∧
          b.val < numberOfMultiindices d (p-J)), F (ν.equiv b.val)‖ ≤
      (‖m‖ * (p+1 : ℝ) * C * (p+1 : ℝ)^s * max 1 (a^p)) *
        (R/‖z‖)^(p+1) / ‖z‖^t := by
  classical
  let S := (Finset.range (p+1)).filter (fun n => p-K < n ∧ n ≤ p-J)
  let L := fun n => ∑ q ∈ Multiindex.multiindicesOfMagnitude n, F q
  let D := C * (p+1 : ℝ)^s * max 1 (a^p) * (R/‖z‖)^(p+1) / ‖z‖^t
  have hz : 0 < ‖z‖ := lt_of_le_of_lt hR hRz
  have hD : 0 ≤ D := by dsimp [D]; positivity
  have hpoint (n : ℕ) (hn : n ∈ S) : R1^K * ‖L n‖ ≤ D := by
    have hnn := (Finset.mem_filter.mp hn).2
    calc
      _ ≤ R1^K * (C * (1+n : ℝ)^s / ‖z‖^((J : ℝ)+t) * (‖h‖/‖z‖)^n) := by
        exact mul_le_mul_of_nonneg_left (hlayer n (by omega)) (pow_nonneg hR1 _)
      _ ≤ D := m2m_scalar_bound p K J n s t C a R R1 ‖z‖ ‖h‖ hJK hKp hnn
        hC ha hR hR1 hR1R hRz hza (norm_nonneg _) hhR
  have hcard : (S.card : ℝ) ≤ (p+1 : ℝ) := by
    exact_mod_cast (calc S.card ≤ (Finset.range (p+1)).card := Finset.card_filter_le _ _
                        _ = p+1 := Finset.card_range _)
  rw [m2m_group_tail ν p K J F]
  change ‖m‖ * R1^K * ‖∑ n ∈ S, L n‖ ≤ _
  calc
    _ ≤ ‖m‖ * (R1^K * ∑ n ∈ S, ‖L n‖) := by
      have := norm_sum_le S L
      nlinarith [mul_le_mul_of_nonneg_left this (show 0 ≤ ‖m‖ * R1^K by positivity)]
    _ = ‖m‖ * ∑ n ∈ S, R1^K * ‖L n‖ := by rw [Finset.mul_sum]
    _ ≤ ‖m‖ * ∑ n ∈ S, D := by gcongr with n hn; exact hpoint n hn
    _ = ‖m‖ * ((S.card : ℝ) * D) := by simp
    _ ≤ ‖m‖ * ((p+1 : ℝ) * D) := by gcongr
    _ = _ := by dsimp [D]; ring

end PDETranslations
