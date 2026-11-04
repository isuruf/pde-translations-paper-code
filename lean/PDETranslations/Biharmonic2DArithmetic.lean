module

public import Mathlib.Basic.Real.Basic
public import Mathlib.Data.Nat.Factorial.Basic
import Mathlib.Tactic

@[expose] public section

/-!
# Scalar arithmetic for the two-dimensional biharmonic estimate

These bounds concern the scalar factor in equation (7) of
`supplementary/biharmonic_kernel_decay.md`; no kernel analysis is used here.
-/

namespace PDETranslations.Biharmonic2DEstimate

/-- The polynomial exponent in the biharmonic estimate. Natural subtraction
intentionally makes this zero for derivative budgets `m ≤ 2`. -/
def biharmonicGrowthExponent (m : ℕ) : ℕ := m - 2

/-- The constant in the biharmonic estimate. In the second branch `m ≥ 3`,
so both natural subtractions agree with ordinary nonnegative subtraction. -/
def biharmonicGrowthConstant (m : ℕ) : ℝ :=
  if m ≤ 2 then 4 else 2 * ((m - 1 : ℕ) : ℝ) ^ (m - 2)

/-- The low-order numerator bound. The hypothesis `3 ≤ k + d` guards
`k + d - 3`, and `1 ≤ k` guards `k - 1` in the factorial comparison. -/
theorem biharmonic_low_order_numerator_le (k d : ℕ) (hk : 1 ≤ k)
    (hd : d ≤ 2) (hkd : 3 ≤ k + d) :
    2 * (k + d - 1) * (k + d - 3).factorial ≤ 4 * k.factorial := by
  have hf : (k + d - 3).factorial ≤ (k - 1).factorial :=
    Nat.factorial_le (by omega)
  have hlinear : k + d - 1 ≤ 2 * k := by omega
  calc
    _ ≤ 2 * (2 * k) * (k - 1).factorial := Nat.mul_le_mul
      (Nat.mul_le_mul_left 2 hlinear) hf
    _ = 4 * k.factorial := by
      conv_rhs => rw [show k = (k - 1) + 1 by omega, Nat.factorial_succ]
      have : k - 1 + 1 = k := by omega
      rw [this]
      ring

/-- The scalar factor in (7) is at most four for `d ≤ 2`.
The total-order hypothesis guards both subtractions in the numerator. -/
theorem biharmonic_scalar_le_four (k d : ℕ) (hk : 1 ≤ k)
    (hd : d ≤ 2) (hkd : 3 ≤ k + d) :
    2 * ((k + d - 1 : ℕ) : ℝ) * ((k + d - 3).factorial : ℝ) /
      (k.factorial : ℝ) ≤ 4 := by
  apply (div_le_iff₀ (by exact_mod_cast Nat.factorial_pos k)).2
  exact_mod_cast biharmonic_low_order_numerator_le k d hk hd hkd

/-- The `d = 0` scalar factor, in particular, is at most four for `k ≥ 3`.
Here `k ≥ 3` ensures that `k - 1` and `k - 3` are not truncated. -/
theorem biharmonic_zero_order_scalar_le_four (k : ℕ) (hk : 3 ≤ k) :
    2 * ((k - 1 : ℕ) : ℝ) * ((k - 3).factorial : ℝ) /
      (k.factorial : ℝ) ≤ 4 := by
  simpa using biharmonic_scalar_le_four k 0 (by omega) (by omega) (by omega)

/-- Higher-order factorial growth, before division. The assumptions guard
`d - 3`, `d - 2`, `m - 2`, and the total-order subtractions. -/
theorem biharmonic_high_order_numerator_le (k d m : ℕ) (hk : 1 ≤ k)
    (hd : 3 ≤ d) (hdm : d ≤ m) :
    2 * (k + d - 1) * (k + d - 3).factorial ≤
      2 * ((m - 1) * (k + 1)) ^ (m - 2) * k.factorial := by
  let B := (m - 1) * (k + 1)
  have hm : 3 ≤ m := by omega
  have hB : 1 ≤ B := by dsimp [B]; nlinarith [show 2 ≤ m - 1 by omega]
  have hlin : k + d - 1 ≤ B := by
    dsimp [B]
    have : d ≤ m := hdm
    have : m - 1 + 1 = m := by omega
    have hsub : k + d - 1 + 1 = k + d := by omega
    nlinarith [show 2 ≤ m - 1 by omega]
  have hf : (k + d - 3).factorial =
      k.factorial * (k + 1).ascFactorial (d - 3) := by
    rw [Nat.factorial_mul_ascFactorial]
    congr 1
    omega
  have ha : (k + 1).ascFactorial (d - 3) ≤ B ^ (d - 3) := by
    exact (Nat.ascFactorial_le_pow_add k (d - 3)).trans
      (Nat.pow_le_pow_left (by omega) _)
  calc
    _ = 2 * (k + d - 1) * (k.factorial * (k + 1).ascFactorial (d - 3)) := by rw [hf]
    _ ≤ 2 * B * (k.factorial * B ^ (d - 3)) :=
      Nat.mul_le_mul (Nat.mul_le_mul_left 2 hlin) (Nat.mul_le_mul_left _ ha)
    _ = 2 * B ^ (d - 2) * k.factorial := by
      rw [show d - 2 = (d - 3) + 1 by omega, pow_succ]
      ring
    _ ≤ 2 * B ^ (m - 2) * k.factorial := by
      exact Nat.mul_le_mul_right k.factorial (Nat.mul_le_mul_left 2
        (pow_le_pow_right₀ hB (show d - 2 ≤ m - 2 by omega)))

/-- Equation (7)'s scalar factor has the stated polynomial bound, uniformly
for `d ≤ m`. The hypotheses `k ≥ 1` and `k + d ≥ 3` guard the numerator's
natural subtractions; the exponent remains intentionally truncated at zero. -/
theorem biharmonic_scalar_polynomial_bound (k d m : ℕ) (hk : 1 ≤ k)
    (hdm : d ≤ m) (hkd : 3 ≤ k + d) :
    2 * ((k + d - 1 : ℕ) : ℝ) * ((k + d - 3).factorial : ℝ) /
      (k.factorial : ℝ) ≤
      biharmonicGrowthConstant m * (1 + (k : ℝ)) ^ biharmonicGrowthExponent m := by
  by_cases hm : m ≤ 2
  · simpa [biharmonicGrowthConstant, biharmonicGrowthExponent, hm,
      show m - 2 = 0 by omega] using
      biharmonic_scalar_le_four k d hk (by omega) hkd
  · have hm3 : 3 ≤ m := by omega
    have hn : 2 * (k + d - 1) * (k + d - 3).factorial ≤
        2 * ((m - 1) * (k + 1)) ^ (m - 2) * k.factorial := by
      by_cases hd : d ≤ 2
      · have hbase : 2 ≤ (m - 1) * (k + 1) := by
          nlinarith [show 2 ≤ m - 1 by omega]
        have hp : 2 ≤ ((m - 1) * (k + 1)) ^ (m - 2) :=
          hbase.trans (le_self_pow₀ (by omega) (by omega))
        exact (biharmonic_low_order_numerator_le k d hk hd hkd).trans
          (Nat.mul_le_mul_right k.factorial (show 4 ≤
                      2 * ((m - 1) * (k + 1)) ^ (m - 2) by omega))
      · exact biharmonic_high_order_numerator_le k d m hk (by omega) hdm
    apply (div_le_iff₀ (by exact_mod_cast Nat.factorial_pos k)).2
    have hr : (2 : ℝ) * ((k + d - 1 : ℕ) : ℝ) *
        ((k + d - 3).factorial : ℝ) ≤
        2 * (((m - 1 : ℕ) : ℝ) * ((k : ℝ) + 1)) ^ (m - 2) *
          (k.factorial : ℝ) := by exact_mod_cast hn
    simpa [biharmonicGrowthConstant, biharmonicGrowthExponent, hm, mul_pow,
      mul_assoc, add_comm] using hr

end PDETranslations.Biharmonic2DEstimate
