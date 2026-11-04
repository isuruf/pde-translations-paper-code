module

public import PDETranslations.Biharmonic2DArithmetic
public import PDETranslations.Biharmonic2DDerivatives
public import PDETranslations.Truncation
import Mathlib.Tactic

@[expose] public section

/-!
# Global high-order estimates for the two-dimensional biharmonic kernel

This formalizes `supplementary/biharmonic_kernel_decay.md`, including the actual
mixed-derivative coefficient identity (6), without analytic assumptions.
The supporting modules separate the derivative calculation, complex norm bound,
and scalar factorial arithmetic.

Meaning and proof-route distinctions from the presentation:
* The kernel is totalized at zero (with value zero); every derivative estimate
  explicitly excludes the pole, as does the paper's domain.
* Coordinate functions have the usual sup-norm domain used by `Multiindex.deriv`,
  but lengths in the statements are Euclidean, not sup norms.
* The radial exponent is real subtraction `(|r| : ℝ) - 2`, never truncated
  natural subtraction. Only the polynomial exponent `m - 2` is truncated,
  exactly implementing the stated piecewise choice of `s_m`.
* Rather than differentiating a convergent logarithmic series and extending by
  homogeneity, we compute derivatives of a centered holomorphic potential and
  sum finite binomial identities. This proves (6) directly for all displacements,
  including zero, without a branch-cut assumption on the center.
* The restriction `k + |r| ≥ 3` is essential. The result is NOT the unrestricted
  `SatisfiesDerivativeEstimate` predicate: the omitted low-order terms contain
  logarithms, and do not obey its global `p' = -2` estimate.
-/

open scoped BigOperators

namespace PDETranslations.Biharmonic2DEstimate

private theorem norm_complexCoordinates (x : EuclideanSpace ℝ (Fin 2)) :
    ‖complexCoordinates x‖ = ‖x‖ := by
  apply (sq_eq_sq₀ (norm_nonneg _) (norm_nonneg _)).mp
  rw [← Complex.normSq_eq_norm_sq, normSq_complexCoordinates,
    EuclideanSpace.real_norm_sq_eq, Fin.sum_univ_two]

/-- The coordinate kernel is exactly the Euclidean `‖z‖² log ‖z‖` kernel.
This identity also holds at zero under the totalized logarithm convention. -/
theorem biharmonicKernel_eq_euclidean (z : EuclideanSpace ℝ (Fin 2)) :
    biharmonicKernel z = ‖z‖ ^ 2 * Real.log ‖z‖ := by
  simp only [biharmonicKernel, biharmonicKernelComplex, norm_complexCoordinates]

private theorem radial_factor (k d : ℕ) (hn : 3 ≤ k + d)
    (z h : ℝ) (hz : 0 < z) :
    h ^ k / z ^ (k + d - 2) = (h / z) ^ k / z ^ ((d : ℝ) - 2) := by
  have he : ((k + d - 2 : ℕ) : ℝ) = (k : ℝ) + ((d : ℝ) - 2) := by
    rw [Nat.cast_sub (by omega), Nat.cast_add]
    norm_num
    ring
  have hp : z ^ (k + d - 2) = z ^ k * z ^ ((d : ℝ) - 2) := by
    rw [← Real.rpow_natCast, he, Real.rpow_add hz, Real.rpow_natCast]
  rw [hp, div_pow]
  ring

/-- The constants specified in the text are strictly positive for every budget. -/
theorem biharmonicGrowthConstant_pos (m : ℕ) : 0 < biharmonicGrowthConstant m := by
  unfold biharmonicGrowthConstant
  split_ifs with hm
  · norm_num
  · have hp : (0 : ℝ) < ((m - 1 : ℕ) : ℝ) := by
      exact_mod_cast (show 0 < m - 1 by omega)
    positivity

/-- Equation (2), with the complex coordinates displayed explicitly.
The formula is obtained from the general mixed-derivative identity, rather than
by identifying coefficients of an infinite series. All natural subtractions are
guarded by `k ≥ 3`, and division by `z` is only used away from the origin. -/
theorem biharmonic_zero_order_formula (k : ℕ) (hk : 3 ≤ k)
    (z h : EuclideanSpace ℝ (Fin 2)) (hz : z ≠ 0) :
    (∑ q ∈ Multiindex.multiindicesOfMagnitude k,
      (Multiindex.deriv q biharmonicKernel z / (Multiindex.factorial q : ℝ)) *
        Multiindex.vectorPower h q) =
      ((-1 : ℂ) ^ k *
          (starRingEnd ℂ (complexCoordinates z) * complexCoordinates h ^ k /
            complexCoordinates z ^ (k - 1)) / ((k : ℂ) * ((k - 1 : ℕ) : ℂ)) +
        (-1 : ℂ) ^ (k - 1) *
          (starRingEnd ℂ (complexCoordinates h) * complexCoordinates h ^ (k - 1) /
            complexCoordinates z ^ (k - 2)) /
          (((k - 1 : ℕ) : ℂ) * ((k - 2 : ℕ) : ℂ))).re := by
  have hz' : (z : Fin 2 → ℝ) ≠ 0 := by
    intro he
    apply hz
    ext i
    exact congrFun he i
  have he := sum_deriv_biharmonicKernel_eq_coefficient k (by omega) 0
    (by simpa [Multiindex.magnitude] using hk) z h hz'
  simp only [add_zero, Finsupp.zero_apply] at he
  exact he.trans (biharmonicCoefficient_zero_order k hk _ _)

/-- Equation (7), now for actual mixed derivatives in Euclidean coordinates.
There is no displacement-size hypothesis. The total-order and positive-degree
hypotheses guard the natural subtractions in the scalar factorial expression. -/
theorem biharmonic_derivative_scalar_bound (k : ℕ) (hk : 1 ≤ k)
    (r : Multiindex.Index 2) (hn : 3 ≤ k + Multiindex.magnitude r)
    (z h : EuclideanSpace ℝ (Fin 2)) (hz : z ≠ 0) :
    |∑ q ∈ Multiindex.multiindicesOfMagnitude k,
      (Multiindex.deriv (q + r) biharmonicKernel z / (Multiindex.factorial q : ℝ)) *
        Multiindex.vectorPower h q| ≤
      (2 * ((k + Multiindex.magnitude r - 1 : ℕ) : ℝ) *
        ((k + Multiindex.magnitude r - 3).factorial : ℝ) / (k.factorial : ℝ)) /
        ‖z‖ ^ ((Multiindex.magnitude r : ℝ) - 2) * (‖h‖ / ‖z‖) ^ k := by
  have hz' : (z : Fin 2 → ℝ) ≠ 0 := by
    intro he
    apply hz
    ext i
    exact congrFun he i
  rw [sum_deriv_biharmonicKernel_eq_coefficient k hk r hn _ _ hz']
  have hb := abs_biharmonicCoefficient_le k (r 0) (r 1) hk
    (by simpa [Multiindex.magnitude, Fin.sum_univ_two] using hn)
    (complexCoordinates z) (complexCoordinates h)
    (mt (complexCoordinates_eq_zero_iff _).mp hz')
  simp only [norm_complexCoordinates] at hb
  have hm : r 0 + r 1 = Multiindex.magnitude r := by
    simp [Multiindex.magnitude, Fin.sum_univ_two]
  rw [hm, radial_factor k _ hn _ _ (norm_pos_iff.mpr hz)] at hb
  convert hb using 1
  ring_nf

/-- Equation (4), with exactly the constants and exponents in the text.
This is the high-total-order restriction of the paper's derivative assumption,
not the unrestricted `SatisfiesDerivativeEstimate` predicate. In particular,
`z ≠ 0`, `k ≥ 1`, and `k + |r| ≥ 3` are explicit, and `h` is unrestricted. -/
theorem biharmonic_derivative_estimate (m : ℕ) (r : Multiindex.Index 2)
    (hr : Multiindex.magnitude r ≤ m) (k : ℕ) (hk : 1 ≤ k)
    (hn : 3 ≤ k + Multiindex.magnitude r)
    (z h : EuclideanSpace ℝ (Fin 2)) (hz : z ≠ 0) :
    |∑ q ∈ Multiindex.multiindicesOfMagnitude k,
      (Multiindex.deriv (q + r) biharmonicKernel z / (Multiindex.factorial q : ℝ)) *
        Multiindex.vectorPower h q| ≤
      biharmonicGrowthConstant m * (1 + (k : ℝ)) ^ biharmonicGrowthExponent m /
        ‖z‖ ^ ((Multiindex.magnitude r : ℝ) - 2) * (‖h‖ / ‖z‖) ^ k := by
  exact (biharmonic_derivative_scalar_bound k hk r hn z h hz).trans
    (mul_le_mul_of_nonneg_right
      (div_le_div_of_nonneg_right (biharmonic_scalar_polynomial_bound k _ m hk hr hn)
        (Real.rpow_nonneg (norm_nonneg z) _)) (by positivity))

/-- Equation (3): the sharper zero-extra-derivative estimate, for every `h`.
The negative radial power is a real power, so no natural subtraction occurs. -/
theorem biharmonic_zero_order_sharp_estimate (k : ℕ) (hk : 3 ≤ k)
    (z h : EuclideanSpace ℝ (Fin 2)) (hz : z ≠ 0) :
    |∑ q ∈ Multiindex.multiindicesOfMagnitude k,
      (Multiindex.deriv q biharmonicKernel z / (Multiindex.factorial q : ℝ)) *
        Multiindex.vectorPower h q| ≤
      (2 / ((k : ℝ) * ((k - 2 : ℕ) : ℝ))) /
        ‖z‖ ^ (-2 : ℝ) * (‖h‖ / ‖z‖) ^ k := by
  have hz' : (z : Fin 2 → ℝ) ≠ 0 := by
    intro he
    apply hz
    ext i
    exact congrFun he i
  have he := sum_deriv_biharmonicKernel_eq_coefficient k (by omega) 0
    (by simpa [Multiindex.magnitude] using hk) z h hz'
  simp only [add_zero, Finsupp.zero_apply] at he
  rw [he]
  have hb := abs_biharmonicCoefficient_zero_order_le k hk
    (complexCoordinates z) (complexCoordinates h)
    (mt (complexCoordinates_eq_zero_iff _).mp hz')
  simp only [norm_complexCoordinates] at hb
  rw [show k - 2 = k + 0 - 2 by omega,
    radial_factor k 0 (by simpa using hk) _ _ (norm_pos_iff.mpr hz)] at hb
  simp only [Nat.cast_zero, zero_sub] at hb
  convert hb using 1
  ring_nf

/-- Equation (1), with `C_{G,0} = 4`, `s₀ = 0`, and `p' = -2`.
As in the paper, only degrees at least three are asserted. -/
theorem biharmonic_zero_order_estimate (k : ℕ) (hk : 3 ≤ k)
    (z h : EuclideanSpace ℝ (Fin 2)) (hz : z ≠ 0) :
    |∑ q ∈ Multiindex.multiindicesOfMagnitude k,
      (Multiindex.deriv q biharmonicKernel z / (Multiindex.factorial q : ℝ)) *
        Multiindex.vectorPower h q| ≤
      4 / ‖z‖ ^ (-2 : ℝ) * (‖h‖ / ‖z‖) ^ k := by
  simpa [biharmonicGrowthConstant, biharmonicGrowthExponent, Multiindex.magnitude] using
    biharmonic_derivative_estimate 0 0 (by simp [Multiindex.magnitude]) k
      (by omega) (by simpa [Multiindex.magnitude] using hk) z h hz

end PDETranslations.Biharmonic2DEstimate
