module

public import Mathlib.Basic.Real.Basic
import Mathlib.Algebra.Order.Field.Power
import Mathlib.Tactic

@[expose] public section

/-!
# Shrinking-radius arithmetic for the Laplace Taylor-tail bound

These estimates formalize the Bernoulli argument in `tail-bound.tex` for
`ρ = R / (2 * (k + 1))`. They are independent of the analytic estimates.
-/

namespace PDETranslations.LaplaceTailBound

/-- Bernoulli's inequality gives a uniform lower bound for the shrinking factor. -/
theorem shrinking_factor_pow_ge_half (k : ℕ) :
    (1 / 2 : ℝ) ≤ (1 - 1 / (2 * ((k : ℝ) + 1))) ^ (k + 1) := by
  have hk : 0 < (k : ℝ) + 1 := by positivity
  have hden : 0 < 2 * ((k : ℝ) + 1) := by positivity
  have hfrac : (1 : ℝ) / (2 * ((k : ℝ) + 1)) ≤ 1 / 2 := by
    apply (div_le_div_iff₀ hden (by norm_num)).2
    have : 0 ≤ (k : ℝ) := Nat.cast_nonneg k
    nlinarith
  have hBernoulli := one_add_mul_le_pow
    (show (-2 : ℝ) ≤ -(1 / (2 * ((k : ℝ) + 1))) by linarith) (k + 1)
  have heq : (1 : ℝ) + ((k + 1 : ℕ) : ℝ) *
      -(1 / (2 * ((k : ℝ) + 1))) = 1 / 2 := by
    push_cast
    field_simp
    ring
  rw [heq] at hBernoulli
  simpa only [sub_eq_add_neg] using hBernoulli

/-- The reciprocal shrinking factor costs at most a factor of two, uniformly in `k`. -/
theorem shrinking_factor_le_two (k : ℕ) :
    (1 / (1 - 1 / (2 * ((k : ℝ) + 1)))) ^ (k + 1) ≤ (2 : ℝ) := by
  have h := shrinking_factor_pow_ge_half k
  rw [one_div_pow]
  apply (div_le_iff₀ (by linarith :
    0 < (1 - 1 / (2 * ((k : ℝ) + 1))) ^ (k + 1))).2
  linarith

/-- The chosen radius is positive and strictly smaller than the ambient radius. -/
theorem shrinking_radius_pos_lt (R : ℝ) (hR : 0 < R) (k : ℕ) :
    0 < R / (2 * ((k : ℝ) + 1)) ∧ R / (2 * ((k : ℝ) + 1)) < R := by
  have hden : 0 < 2 * ((k : ℝ) + 1) := by positivity
  constructor
  · exact div_pos hR hden
  · apply (div_lt_iff₀ hden).2
    have : 0 ≤ (k : ℝ) := Nat.cast_nonneg k
    nlinarith

/-- Combining the interior-radius cost and the reciprocal shrinking factor
leaves only polynomial growth in `k` and the exact power `R ^ (k + m + 1)`. -/
theorem shrinking_radius_bound (R : ℝ) (hR : 0 < R) (k m : ℕ) :
    (1 / (R / (2 * ((k : ℝ) + 1))) ^ m) *
        (1 / (R - R / (2 * ((k : ℝ) + 1)))) ^ (k + 1) ≤
      (2 : ℝ) ^ (m + 1) * ((k : ℝ) + 1) ^ m / R ^ (k + m + 1) := by
  have hRne : R ≠ 0 := ne_of_gt hR
  have hk : (k : ℝ) + 1 ≠ 0 := by positivity
  have hsplit : R - R / (2 * ((k : ℝ) + 1)) =
      R * (1 - 1 / (2 * ((k : ℝ) + 1))) := by ring
  have hnormalize :
      (1 / (R / (2 * ((k : ℝ) + 1))) ^ m) *
          (1 / (R - R / (2 * ((k : ℝ) + 1)))) ^ (k + 1) =
        ((2 : ℝ) ^ m * ((k : ℝ) + 1) ^ m / R ^ (k + m + 1)) *
          (1 / (1 - 1 / (2 * ((k : ℝ) + 1)))) ^ (k + 1) := by
    rw [hsplit]
    simp only [div_pow, mul_pow, one_pow, pow_add, pow_one]
    field_simp
  rw [hnormalize]
  calc
    _ ≤ ((2 : ℝ) ^ m * ((k : ℝ) + 1) ^ m / R ^ (k + m + 1)) * 2 :=
      mul_le_mul_of_nonneg_left (shrinking_factor_le_two k) (by positivity)
    _ = _ := by rw [pow_succ]; ring

end PDETranslations.LaplaceTailBound
