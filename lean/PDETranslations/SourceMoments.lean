module

public import Mathlib.Analysis.InnerProductSpace.PiL2
public import PDETranslations.Ordering
import Mathlib.Analysis.Complex.Basic
import PDETranslations.Truncation

@[expose] public section

open scoped BigOperators

namespace PDETranslations

variable {d : ℕ} {ι : Type*} [Fintype ι]

/-- The order-`p` uncompressed multipole coefficients of finitely many point
sources about `center`. The whole real normalized monomial is cast to `ℂ`.
Spatial vectors carry the Euclidean 2-norm, not the coordinate Pi norm. -/
noncomputable def finiteSourceMoments (ν : GradedMonomialOrdering d) (p : ℕ)
    (center : EuclideanSpace ℝ (Fin d)) (source : ι → EuclideanSpace ℝ (Fin d))
    (w : ι → ℂ) : Fin (numberOfMultiindices d p) → ℂ :=
  fun q => ∑ i, w i *
    ((Multiindex.vectorPower (center - source i) (ν.equiv q.val) /
      (Multiindex.factorial (ν.equiv q.val) : ℝ) : ℝ) : ℂ)

private theorem norm_vectorPower_le (v : EuclideanSpace ℝ (Fin d))
    (α : Multiindex.Index d) {R : ℝ} (_hR : 0 ≤ R) (hv : ‖v‖ ≤ R) :
    ‖Multiindex.vectorPower v α‖ ≤ R ^ Multiindex.magnitude α := by
  classical
  unfold Multiindex.vectorPower Multiindex.magnitude
  rw [norm_prod, ← Finset.prod_pow_eq_pow_sum]
  apply Finset.prod_le_prod₀
  · intro k _
    exact norm_nonneg _
  · intro k _
    rw [norm_pow]
    exact pow_le_pow_left₀ (norm_nonneg _) ((PiLp.norm_apply_le v k).trans hv) _

private theorem norm_normalizedMonomial_le (v : EuclideanSpace ℝ (Fin d))
    (α : Multiindex.Index d) {R : ℝ} (hR : 0 ≤ R) (hv : ‖v‖ ≤ R) :
    ‖Multiindex.vectorPower v α / (Multiindex.factorial α : ℝ)‖ ≤
      R ^ Multiindex.magnitude α := by
  have hfac : 0 < Multiindex.factorial α := by
    classical
    unfold Multiindex.factorial
    positivity
  have hfac1 : (1 : ℝ) ≤ (Multiindex.factorial α : ℝ) := by
    exact_mod_cast hfac
  rw [norm_div, Real.norm_of_nonneg (by positivity : (0 : ℝ) ≤ Multiindex.factorial α)]
  exact (div_le_self (norm_nonneg _) hfac1).trans (norm_vectorPower_le v α hR hv)

/-- Source-moment growth for sources in the closed Euclidean ball of radius
`R` about `center`. We drop the reciprocal multiindex factorial from the sharper
bound: this loses only a prefactor, not the radius exponent. The resulting
constant is simply the sum of the absolute source strengths. -/
theorem norm_finiteSourceMoments_le (ν : GradedMonomialOrdering d) (p : ℕ)
    (center : EuclideanSpace ℝ (Fin d)) (source : ι → EuclideanSpace ℝ (Fin d))
    (w : ι → ℂ) {R : ℝ} (hR : 0 ≤ R)
    (hsource : ∀ i, ‖source i - center‖ ≤ R)
    (q : Fin (numberOfMultiindices d p)) :
    ‖finiteSourceMoments ν p center source w q‖ ≤
      (∑ i, ‖w i‖) * R ^ Multiindex.magnitude (ν.equiv q.val) := by
  classical
  unfold finiteSourceMoments
  calc
    _ ≤ ∑ i, ‖w i *
        ((Multiindex.vectorPower (center - source i) (ν.equiv q.val) /
          (Multiindex.factorial (ν.equiv q.val) : ℝ) : ℝ) : ℂ)‖ := norm_sum_le _ _
    _ ≤ ∑ i, ‖w i‖ * R ^ Multiindex.magnitude (ν.equiv q.val) := by
      apply Finset.sum_le_sum
      intro i _
      rw [norm_mul, Complex.norm_real]
      apply mul_le_mul_of_nonneg_left _ (norm_nonneg _)
      apply norm_normalizedMonomial_le _ _ hR
      simpa only [norm_sub_rev] using hsource i
    _ = _ := (Finset.sum_mul ..).symm

end PDETranslations
