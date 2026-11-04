module

public import Mathlib.Analysis.Complex.Basic
public import Mathlib.Data.Nat.Factorial.Basic
import Mathlib.Tactic

@[expose] public section

/-!
# Explicit coefficients for the two-dimensional biharmonic estimate

This file defines the complex expression inside the real part in equation (6) of
`supplementary/biharmonic_kernel_decay.md`, and proves its norm bound (7).
Here `d = r₁ + r₂` and `n = k + d`. Negative powers are written as division
by natural powers. The estimates require `k ≥ 1`, `n ≥ 3`, and `z ≠ 0`,
but impose no restriction on `h` (in particular, `h = 0` is allowed).

These are estimates of an explicit algebraic expression only. No identification
with derivatives of the biharmonic kernel or with Taylor coefficients is claimed.
In particular, the derivative bridge asserted in equation (6) is not proved here.
-/

namespace PDETranslations.Biharmonic2DEstimate

/-- The explicit complex expression inside `Re` in equation (6).
The difference of the derivative orders is taken in `ℝ`, not in `ℕ`.
The definition is total; its intended range is `1 ≤ k` and `3 ≤ k + r₁ + r₂`. -/
noncomputable def biharmonicCoefficientComplex (k r₁ r₂ : ℕ) (z h : ℂ) : ℂ :=
  let n := k + (r₁ + r₂)
  Complex.I ^ r₂ *
    ((-1 : ℂ) ^ n * (((n - 2).factorial : ℝ) / (k.factorial : ℝ) : ℝ) *
        (starRingEnd ℂ z * h ^ k / z ^ (n - 1)) +
      (-1 : ℂ) ^ (n - 1) *
        (((n - 3).factorial : ℝ) / ((k - 1).factorial : ℝ) : ℝ) *
        (starRingEnd ℂ h * h ^ (k - 1) / z ^ (n - 2)) +
      (-1 : ℂ) ^ (n - 1) *
        (((r₁ : ℝ) - (r₂ : ℝ)) * ((n - 3).factorial : ℝ) /
          (k.factorial : ℝ) : ℝ) * (h ^ k / z ^ (n - 2)))

/-- The real part of the explicit expression in equation (6), without any
assertion that it equals a derivative or Taylor coefficient. -/
noncomputable def biharmonicCoefficient (k r₁ r₂ : ℕ) (z h : ℂ) : ℝ :=
  (biharmonicCoefficientComplex k r₁ r₂ z h).re

private theorem monomial_norms (k n : ℕ) (hk : 1 ≤ k) (hn : 3 ≤ n)
    (z h : ℂ) (hz : z ≠ 0) :
    ‖starRingEnd ℂ z * h ^ k / z ^ (n - 1)‖ = ‖h‖ ^ k / ‖z‖ ^ (n - 2) ∧
    ‖starRingEnd ℂ h * h ^ (k - 1) / z ^ (n - 2)‖ =
      ‖h‖ ^ k / ‖z‖ ^ (n - 2) := by
  have hz' : ‖z‖ ≠ 0 := norm_ne_zero_iff.mpr hz
  simp only [norm_div, norm_mul, norm_pow, Complex.norm_conj]
  constructor
  · rw [show n - 1 = (n - 2) + 1 by omega, pow_succ]
    field_simp
  · conv_rhs => rw [show k = (k - 1) + 1 by omega, pow_succ]
    ring

/-- The stronger complex norm version of the estimate following equation (6).
All natural subtractions in the formula are guarded by `hk` and `hn`.
There is no smallness or nonvanishing hypothesis on `h`. -/
theorem norm_biharmonicCoefficientComplex_le (k r₁ r₂ : ℕ)
    (hk : 1 ≤ k) (hn : 3 ≤ k + (r₁ + r₂)) (z h : ℂ) (hz : z ≠ 0) :
    ‖biharmonicCoefficientComplex k r₁ r₂ z h‖ ≤
      (‖h‖ ^ k / ‖z‖ ^ (k + (r₁ + r₂) - 2)) *
        (2 * ((k + (r₁ + r₂) - 1 : ℕ) : ℝ) *
          ((k + (r₁ + r₂) - 3).factorial : ℝ) / (k.factorial : ℝ)) := by
  let n := k + (r₁ + r₂)
  have hn' : 3 ≤ n := hn
  let A : ℝ := ((n - 2).factorial : ℝ) / (k.factorial : ℝ)
  let B : ℝ := ((n - 3).factorial : ℝ) / ((k - 1).factorial : ℝ)
  let C : ℝ := ((n - 3).factorial : ℝ) / (k.factorial : ℝ)
  let H : ℝ := ‖h‖ ^ k / ‖z‖ ^ (n - 2)
  have hA : 0 ≤ A := by dsimp [A]; positivity
  have hB : 0 ≤ B := by dsimp [B]; positivity
  have hC : 0 ≤ C := by dsimp [C]; positivity
  have hH : 0 ≤ H := by dsimp [H]; positivity
  have hdiff : |(r₁ : ℝ) - (r₂ : ℝ)| ≤ (r₁ : ℝ) + (r₂ : ℝ) := by
    apply abs_le.mpr
    constructor <;> linarith [show (0 : ℝ) ≤ r₁ by positivity,
      show (0 : ℝ) ≤ r₂ by positivity]
  obtain ⟨hm₁, hm₂⟩ := monomial_norms k n hk hn' z h hz
  have hbound : ‖biharmonicCoefficientComplex k r₁ r₂ z h‖ ≤
      A * H + B * H + ((r₁ : ℝ) + (r₂ : ℝ)) * C * H := by
    unfold biharmonicCoefficientComplex
    dsimp only
    simp only [mul_div_assoc]
    calc
      _ ≤ ‖(-1 : ℂ) ^ n * (A : ℂ) *
              (starRingEnd ℂ z * h ^ k / z ^ (n - 1))‖ +
          ‖(-1 : ℂ) ^ (n - 1) * (B : ℂ) *
              (starRingEnd ℂ h * h ^ (k - 1) / z ^ (n - 2))‖ +
          ‖(-1 : ℂ) ^ (n - 1) *
              ((((r₁ : ℝ) - (r₂ : ℝ)) * C : ℝ) : ℂ) *
              (h ^ k / z ^ (n - 2))‖ := by
        simp only [mul_div_assoc]
        rw [norm_mul, norm_pow, Complex.norm_I, one_pow, one_mul]
        exact (norm_add_le _ _).trans (add_le_add (norm_add_le _ _) le_rfl)
      _ = A * H + B * H + |(r₁ : ℝ) - (r₂ : ℝ)| * C * H := by
        simp only [norm_mul, norm_pow, norm_neg, norm_one, one_pow, one_mul,
          Complex.norm_real, Real.norm_eq_abs, abs_of_nonneg hA, abs_of_nonneg hB,
          abs_of_nonneg hC, hm₁, hm₂, norm_div]
        rfl
      _ ≤ _ := by gcongr
  have hf₁ : ((n - 2).factorial : ℝ) =
      ((n - 2 : ℕ) : ℝ) * ((n - 3).factorial : ℝ) := by
    rw [show n - 2 = (n - 3) + 1 by omega, Nat.factorial_succ]
    push_cast
    rfl
  have hf₂ : (k.factorial : ℝ) = (k : ℝ) * ((k - 1).factorial : ℝ) := by
    conv_lhs => rw [show k = (k - 1) + 1 by omega, Nat.factorial_succ]
    rw [show k - 1 + 1 = k by omega]
    push_cast
    rfl
  have hsum : ((n - 2 : ℕ) : ℝ) + (k : ℝ) + ((r₁ : ℝ) + (r₂ : ℝ)) =
      2 * ((n - 1 : ℕ) : ℝ) := by
    have h₁ : n - 2 + 2 = n := by omega
    have h₂ : n - 1 + 1 = n := by omega
    have h₃ : n = k + (r₁ + r₂) := rfl
    exact_mod_cast (show n - 2 + k + (r₁ + r₂) = 2 * (n - 1) by omega)
  have hscalar : A + B + ((r₁ : ℝ) + (r₂ : ℝ)) * C =
      2 * ((n - 1 : ℕ) : ℝ) * ((n - 3).factorial : ℝ) / (k.factorial : ℝ) := by
    have hkf : (k.factorial : ℝ) ≠ 0 := by positivity
    have hkf' : ((k - 1).factorial : ℝ) ≠ 0 := by positivity
    dsimp [A, B, C]
    rw [hf₁]
    field_simp
    nlinarith [hf₂, hsum]
  calc
    _ ≤ A * H + B * H + ((r₁ : ℝ) + (r₂ : ℝ)) * C * H := hbound
    _ = H * (A + B + ((r₁ : ℝ) + (r₂ : ℝ)) * C) := by ring
    _ = _ := by rw [hscalar]

/-- The requested absolute-value bound for the real expression in equation (6).
This is the explicit-coefficient estimate (7), not a theorem about derivatives. -/
theorem abs_biharmonicCoefficient_le (k r₁ r₂ : ℕ)
    (hk : 1 ≤ k) (hn : 3 ≤ k + (r₁ + r₂)) (z h : ℂ) (hz : z ≠ 0) :
    |biharmonicCoefficient k r₁ r₂ z h| ≤
      (‖h‖ ^ k / ‖z‖ ^ (k + (r₁ + r₂) - 2)) *
        (2 * ((k + (r₁ + r₂) - 1 : ℕ) : ℝ) *
          ((k + (r₁ + r₂) - 3).factorial : ℝ) / (k.factorial : ℝ)) := by
  exact (Complex.abs_re_le_norm _).trans
    (norm_biharmonicCoefficientComplex_le k r₁ r₂ hk hn z h hz)

private theorem factorial_step_real (j : ℕ) (hj : 1 ≤ j) :
    (j.factorial : ℝ) = (j : ℝ) * ((j - 1).factorial : ℝ) := by
  conv_lhs => rw [show j = (j - 1) + 1 by omega, Nat.factorial_succ]
  rw [show j - 1 + 1 = j by omega]
  push_cast
  rfl

private theorem factorial_ratio_two (j : ℕ) (hj : 2 ≤ j) :
    ((j - 2).factorial : ℝ) / (j.factorial : ℝ) =
      1 / ((j : ℝ) * ((j - 1 : ℕ) : ℝ)) := by
  rw [factorial_step_real j (by omega), factorial_step_real (j - 1) (by omega),
    show j - 1 - 1 = j - 2 by omega]
  have hf : ((j - 2).factorial : ℝ) ≠ 0 := by positivity
  have hj₀ : (j : ℝ) ≠ 0 := by exact_mod_cast (show j ≠ 0 by omega)
  have hj₁ : ((j - 1 : ℕ) : ℝ) ≠ 0 := by
    exact_mod_cast (show j - 1 ≠ 0 by omega)
  field_simp

/-- At zero derivative order the explicit expression (6) reduces to the two
terms displayed in equation (2). This algebraic identity does not identify
these terms with derivatives; it holds even at `z = 0` with Lean's total division. -/
theorem biharmonicCoefficient_zero_order (k : ℕ) (hk : 3 ≤ k) (z h : ℂ) :
    biharmonicCoefficient k 0 0 z h =
      ((-1 : ℂ) ^ k * (starRingEnd ℂ z * h ^ k / z ^ (k - 1)) /
          ((k : ℂ) * ((k - 1 : ℕ) : ℂ)) +
        (-1 : ℂ) ^ (k - 1) *
          (starRingEnd ℂ h * h ^ (k - 1) / z ^ (k - 2)) /
          (((k - 1 : ℕ) : ℂ) * ((k - 2 : ℕ) : ℂ))).re := by
  have hA := factorial_ratio_two k (by omega)
  have hB := factorial_ratio_two (k - 1) (by omega)
  rw [show k - 1 - 2 = k - 3 by omega,
    show k - 1 - 1 = k - 2 by omega] at hB
  simp only [biharmonicCoefficient, biharmonicCoefficientComplex, add_zero,
    pow_zero, Nat.cast_zero, sub_self, zero_mul, zero_div, Complex.ofReal_zero,
    mul_zero, one_mul, hA, hB, Complex.ofReal_div, Complex.ofReal_one,
    Complex.ofReal_mul, Complex.ofReal_natCast]
  congr 1
  ring

/-- The sharp scalar bound of equation (3) for the explicit zero-order
coefficient, obtained by simplifying the factorial factor in (7).
The increment `h` is arbitrary, including zero. -/
theorem abs_biharmonicCoefficient_zero_order_le (k : ℕ) (hk : 3 ≤ k)
    (z h : ℂ) (hz : z ≠ 0) :
    |biharmonicCoefficient k 0 0 z h| ≤
      (‖h‖ ^ k / ‖z‖ ^ (k - 2)) *
        (2 / ((k : ℝ) * ((k - 2 : ℕ) : ℝ))) := by
  have hscalar : 2 * ((k - 1 : ℕ) : ℝ) * ((k - 3).factorial : ℝ) /
      (k.factorial : ℝ) = 2 / ((k : ℝ) * ((k - 2 : ℕ) : ℝ)) := by
    rw [factorial_step_real k (by omega),
      factorial_step_real (k - 1) (by omega),
      show k - 1 - 1 = k - 2 by omega,
      factorial_step_real (k - 2) (by omega),
      show k - 2 - 1 = k - 3 by omega]
    have hf : ((k - 3).factorial : ℝ) ≠ 0 := by positivity
    have hk₀ : (k : ℝ) ≠ 0 := by exact_mod_cast (show k ≠ 0 by omega)
    have hk₁ : ((k - 1 : ℕ) : ℝ) ≠ 0 := by
      exact_mod_cast (show k - 1 ≠ 0 by omega)
    have hk₂ : ((k - 2 : ℕ) : ℝ) ≠ 0 := by
      exact_mod_cast (show k - 2 ≠ 0 by omega)
    field_simp
  simpa only [add_zero, hscalar] using
    abs_biharmonicCoefficient_le k 0 0 (by omega) (by simpa using hk) z h hz

end PDETranslations.Biharmonic2DEstimate
