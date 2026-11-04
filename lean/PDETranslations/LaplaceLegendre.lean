module

public import Mathlib.Analysis.Calculus.Deriv.Basic
public import Mathlib.MeasureTheory.Integral.IntervalIntegral.Basic
import Mathlib.Analysis.Complex.RealDeriv
import Mathlib.Analysis.SpecialFunctions.Integrals.Basic
import Mathlib.Analysis.SpecificLimits.Normed

@[expose] public section

/-!
# Integral-defined Legendre coefficients and their sharp bound

The integral definition below gives `|P k u| ≤ 1` on `[-1,1]` without
assuming a generating function or any Taylor estimate. The coefficient API
and geometric majorants can consequently be used independently of the
identification with the Laplace kernel.

The generating-function identity is not proved here. `LaplaceLegendreConnection`
uses an explicit scalar Taylor-coefficient identity to identify these scaled
coefficients with the actual Laplace Taylor terms. None of those identities
is a hypothesis of the bounds in this file.
-/

open scoped BigOperators Topology
open MeasureTheory

namespace PDETranslations.LaplaceLegendre

/-- The complex number in Laplace's integral representation. -/
noncomputable def integrandBase (u θ : ℝ) : ℂ :=
  (u : ℂ) + Complex.I * (Real.sqrt (1 - u ^ 2) : ℂ) * (Real.cos θ : ℂ)

/-- Laplace's integral definition, with real part taken after integration. -/
noncomputable def P (k : ℕ) (u : ℝ) : ℝ :=
  (1 / Real.pi) * (∫ θ in (0 : ℝ)..Real.pi, integrandBase u θ ^ k).re

theorem continuous_integrandBase (u : ℝ) : Continuous (integrandBase u) := by
  unfold integrandBase
  fun_prop

theorem intervalIntegrable_pow (k : ℕ) (u : ℝ) :
    IntervalIntegrable (fun θ => integrandBase u θ ^ k) volume 0 Real.pi :=
  ((continuous_integrandBase u).pow k).intervalIntegrable _ _

/-- The modulus estimate holds for every angle, not only on the integration interval. -/
theorem norm_integrandBase_le_one {u : ℝ} (hu : |u| ≤ 1) (θ : ℝ) :
    ‖integrandBase u θ‖ ≤ 1 := by
  have hu2 : u ^ 2 ≤ 1 := by
    have := abs_le.mp hu
    nlinarith
  have hs : Real.sqrt (1 - u ^ 2) ^ 2 = 1 - u ^ 2 :=
    Real.sq_sqrt (by linarith)
  have hc : Real.cos θ ^ 2 ≤ 1 := by
    have := Real.abs_cos_le_one θ
    have := abs_le.mp this
    nlinarith
  have hn : ‖integrandBase u θ‖ ^ 2 =
      u ^ 2 + (Real.sqrt (1 - u ^ 2) * Real.cos θ) ^ 2 := by
    rw [Complex.sq_norm, Complex.normSq_apply]
    simp only [integrandBase, Complex.add_re, Complex.add_im, Complex.mul_re,
      Complex.mul_im, Complex.ofReal_re, Complex.ofReal_im, Complex.I_re,
      Complex.I_im]
    ring
  have hprod := mul_le_mul_of_nonneg_left hc (sq_nonneg (Real.sqrt (1 - u ^ 2)))
  nlinarith [norm_nonneg (integrandBase u θ)]

theorem norm_pow_le_one (k : ℕ) {u : ℝ} (hu : |u| ≤ 1) (θ : ℝ) :
    ‖integrandBase u θ ^ k‖ ≤ 1 := by
  rw [norm_pow]
  exact pow_le_one₀ (norm_nonneg _) (norm_integrandBase_le_one hu θ)

/-- The unnormalized complex integral has modulus at most `π`. -/
theorem norm_integral_le_pi (k : ℕ) {u : ℝ} (hu : |u| ≤ 1) :
    ‖∫ θ in (0 : ℝ)..Real.pi, integrandBase u θ ^ k‖ ≤ Real.pi := by
  simpa [abs_of_pos Real.pi_pos] using
    intervalIntegral.norm_integral_le_of_norm_le_const
      (a := (0 : ℝ)) (b := Real.pi) (C := (1 : ℝ))
      (fun θ _ => norm_pow_le_one k hu θ)

/-- Sharp, unconditional Legendre bound on the real unit interval. -/
theorem abs_P_le_one (k : ℕ) {u : ℝ} (hu : |u| ≤ 1) : |P k u| ≤ 1 := by
  have h := (Complex.abs_re_le_norm
    (∫ θ in (0 : ℝ)..Real.pi, integrandBase u θ ^ k)).trans
    (norm_integral_le_pi k hu)
  unfold P
  rw [abs_mul, abs_of_pos (by positivity : 0 < 1 / Real.pi)]
  calc
    _ ≤ (1 / Real.pi) * Real.pi := mul_le_mul_of_nonneg_left h (by positivity)
    _ = 1 := by field_simp

@[simp] theorem P_zero (u : ℝ) : P 0 u = 1 := by
  simp [P, intervalIntegral.integral_const, Complex.real_smul, Real.pi_ne_zero]

@[simp] theorem P_one_endpoint (k : ℕ) : P k 1 = 1 := by
  simp [P, integrandBase, intervalIntegral.integral_const, Complex.real_smul,
    Real.pi_ne_zero]

@[simp] theorem P_neg_one_endpoint (k : ℕ) : P k (-1) = (-1) ^ k := by
  have hb (θ : ℝ) : integrandBase (-1) θ = ((-1 : ℝ) : ℂ) := by
    simp [integrandBase]
  unfold P
  simp only [hb, ← Complex.ofReal_pow]
  rw [intervalIntegral.integral_const]
  simp only [sub_zero, Complex.real_smul, Complex.mul_re, Complex.ofReal_re,
    Complex.ofReal_im, mul_zero]
  field_simp

/-- Real part commutes with this integral because the integrand is continuous. -/
theorem P_eq_integral_re (k : ℕ) (u : ℝ) :
    P k u = (1 / Real.pi) *
      ∫ θ in (0 : ℝ)..Real.pi, (integrandBase u θ ^ k).re := by
  unfold P
  congr 1
  exact (Complex.reCLM.intervalIntegral_comp_comm (intervalIntegrable_pow k u)).symm

@[simp] theorem P_one (u : ℝ) : P 1 u = u := by
  rw [P_eq_integral_re]
  simp only [pow_one, integrandBase, Complex.add_re, Complex.mul_re,
    Complex.ofReal_re, Complex.ofReal_im, Complex.I_re, Complex.I_im,
    zero_mul, mul_zero, sub_self, add_zero]
  rw [intervalIntegral.integral_const]
  simp [Real.pi_ne_zero]

/-- Coefficients in the candidate one-variable generating series. -/
noncomputable def coefficient (u : ℝ) (k : ℕ) : ℝ := P k u

theorem abs_coefficient_le_one {u : ℝ} (hu : |u| ≤ 1) (k : ℕ) :
    |coefficient u k| ≤ 1 := abs_P_le_one k hu

/-- The candidate degree-`k` Laplace term, where `R` is distance to the
singularity, `H` is displacement length, and `u` is the angle cosine.
This definition does not assert that the expression is a Taylor term. -/
noncomputable def scaledCoefficient (k : ℕ) (u R H : ℝ) : ℝ :=
  H ^ k / R ^ (k + 1) * P k u

/-- The desired sharp undifferentiated estimate for the candidate terms.
There is no restriction `H < R`. -/
theorem abs_scaledCoefficient_le (k : ℕ) {u R H : ℝ}
    (hu : |u| ≤ 1) (hR : 0 < R) (hH : 0 ≤ H) :
    |scaledCoefficient k u R H| ≤ H ^ k / R ^ (k + 1) := by
  have hn : 0 ≤ H ^ k / R ^ (k + 1) := by positivity
  simpa [scaledCoefficient, abs_mul, abs_of_nonneg hn] using
    mul_le_mul_of_nonneg_left (abs_P_le_one k hu) hn

/-- Each term of the candidate generating series is bounded by a geometric term. -/
theorem abs_series_term_le {u : ℝ} (hu : |u| ≤ 1) (t : ℝ) (k : ℕ) :
    |coefficient u k * t ^ k| ≤ |t| ^ k := by
  rw [abs_mul, abs_pow]
  simpa using mul_le_mul_of_nonneg_right (abs_coefficient_le_one hu k)
    (pow_nonneg (abs_nonneg t) k)

/-- Absolute convergence is proved before using any `tsum` identities. -/
theorem summable_norm_series {u t : ℝ} (hu : |u| ≤ 1) (ht : |t| < 1) :
    Summable (fun k => ‖coefficient u k * t ^ k‖) := by
  apply (summable_geometric_of_lt_one (abs_nonneg t) ht).of_nonneg_of_le
    (fun _ => norm_nonneg _)
  exact fun k => abs_series_term_le hu t k

theorem summable_series {u t : ℝ} (hu : |u| ≤ 1) (ht : |t| < 1) :
    Summable (fun k => coefficient u k * t ^ k) :=
  (summable_norm_series hu ht).of_norm

/-- This series is not yet identified with the classical generating function. -/
noncomputable def generatingSeries (u t : ℝ) : ℝ :=
  ∑' k : ℕ, coefficient u k * t ^ k

/-- Sharp geometric bound for the tail starting at the first omitted index `n`. -/
theorem abs_series_tail_le (n : ℕ) {u t : ℝ}
    (hu : |u| ≤ 1) (ht : |t| < 1) :
    |∑' k : ℕ, coefficient u (n + k) * t ^ (n + k)| ≤
      |t| ^ n / (1 - |t|) := by
  have hg := (hasSum_geometric_of_lt_one (abs_nonneg t) ht).mul_left (|t| ^ n)
  have hb : ∀ k : ℕ, ‖coefficient u (n + k) * t ^ (n + k)‖ ≤
      |t| ^ n * |t| ^ k := by
    intro k
    simpa [pow_add] using abs_series_term_le hu t (n + k)
  simpa [div_eq_mul_inv] using tsum_of_norm_bounded hg hb

/-- The finite-sum remainder estimate for the candidate generating series. -/
theorem abs_generatingSeries_sub_sum_le (n : ℕ) {u t : ℝ}
    (hu : |u| ≤ 1) (ht : |t| < 1) :
    |generatingSeries u t - ∑ k ∈ Finset.range n, coefficient u k * t ^ k| ≤
      |t| ^ n / (1 - |t|) := by
  have heq := (summable_series hu ht).sum_add_tsum_nat_add n
  have htail : generatingSeries u t -
      ∑ k ∈ Finset.range n, coefficient u k * t ^ k =
      ∑' k : ℕ, coefficient u (n + k) * t ^ (n + k) := by
    unfold generatingSeries
    rw [← heq]
    simp [Nat.add_comm]
  rw [htail]
  exact abs_series_tail_le n hu ht

/-- Scaling reduces the candidate Laplace coefficients to the unit generating series. -/
theorem scaledCoefficient_eq (k : ℕ) (u R H : ℝ) :
    scaledCoefficient k u R H = (1 / R) * (coefficient u k * (H / R) ^ k) := by
  simp only [scaledCoefficient, coefficient, div_eq_mul_inv, pow_succ,
    mul_inv_rev, mul_pow, inv_pow, one_mul]
  ring

/-- The candidate Laplace expansion converges absolutely for `H < R`. -/
theorem summable_norm_scaledCoefficients {u R H : ℝ}
    (hu : |u| ≤ 1) (hR : 0 < R) (hH : 0 ≤ H) (hHR : H < R) :
    Summable (fun k => ‖scaledCoefficient k u R H‖) := by
  have ht : |H / R| < 1 := by
    rw [abs_of_nonneg (by positivity : 0 ≤ H / R)]
    exact (div_lt_one hR).mpr hHR
  simpa only [scaledCoefficient_eq, norm_mul] using
    (summable_norm_series hu ht).mul_left ‖1 / R‖

/-- Sharp geometric tail estimate for the candidate Laplace expansion.
The sole missing analytic bridge is identification of these coefficients
with the kernel's actual Taylor terms. -/
theorem abs_scaled_tail_le (n : ℕ) {u R H : ℝ}
    (hu : |u| ≤ 1) (hR : 0 < R) (hH : 0 ≤ H) (hHR : H < R) :
    |∑' k : ℕ, scaledCoefficient (n + k) u R H| ≤
      (H / R) ^ n / (R - H) := by
  have hratio : H / R < 1 := (div_lt_one hR).mpr hHR
  have ht : |H / R| < 1 := by
    rwa [abs_of_nonneg (by positivity : 0 ≤ H / R)]
  have hden : 0 < 1 - H / R := by linarith
  have hdiff : 0 < R - H := by linarith
  simp_rw [scaledCoefficient_eq]
  rw [tsum_mul_left, abs_mul, abs_of_pos (by positivity : 0 < 1 / R)]
  calc
    _ ≤ (1 / R) * (|H / R| ^ n / (1 - |H / R|)) :=
      mul_le_mul_of_nonneg_left (abs_series_tail_le n hu ht) (by positivity)
    _ = (H / R) ^ n / (R - H) := by
      rw [abs_of_nonneg (by positivity : 0 ≤ H / R)]
      field_simp

/-- Finite-sum form of the candidate Laplace remainder bound. This is not a
remainder theorem for the kernel until the generating identity is proved. -/
theorem abs_scaled_sum_sub_sum_le (n : ℕ) {u R H : ℝ}
    (hu : |u| ≤ 1) (hR : 0 < R) (hH : 0 ≤ H) (hHR : H < R) :
    |(∑' k : ℕ, scaledCoefficient k u R H) -
      ∑ k ∈ Finset.range n, scaledCoefficient k u R H| ≤
      (H / R) ^ n / (R - H) := by
  have hs := (summable_norm_scaledCoefficients hu hR hH hHR).of_norm
  have heq := hs.sum_add_tsum_nat_add n
  rw [← heq]
  simpa [Nat.add_comm] using abs_scaled_tail_le n hu hR hH hHR

/-- The imaginary coefficient has square `u² - 1` on the unit interval. -/
theorem imaginaryCoefficient_sq {u : ℝ} (hu : |u| ≤ 1) :
    (Complex.I * (Real.sqrt (1 - u ^ 2) : ℂ)) ^ 2 = (u : ℂ) ^ 2 - 1 := by
  have hu2 : u ^ 2 ≤ 1 := by
    have := abs_le.mp hu
    nlinarith
  have hs := Real.sq_sqrt (show 0 ≤ 1 - u ^ 2 by linarith)
  rw [mul_pow, Complex.I_sq, ← Complex.ofReal_pow, hs]
  push_cast
  ring

/-- A primitive whose endpoint values vanish gives the three-term recurrence. -/
theorem hasDerivAt_recurrencePrimitive (k : ℕ) {u : ℝ} (hu : |u| ≤ 1) (θ : ℝ) :
    HasDerivAt
      (fun θ => (Real.sin θ : ℂ) *
        (Complex.I * (Real.sqrt (1 - u ^ 2) : ℂ)) * integrandBase u θ ^ (k + 1))
      (((k + 2 : ℕ) : ℂ) * integrandBase u θ ^ (k + 2) -
        ((2 * k + 3 : ℕ) : ℂ) * (u : ℂ) * integrandBase u θ ^ (k + 1) +
        ((k + 1 : ℕ) : ℂ) * integrandBase u θ ^ k) θ := by
  let b : ℂ := Complex.I * (Real.sqrt (1 - u ^ 2) : ℂ)
  have hb : b ^ 2 = (u : ℂ) ^ 2 - 1 := imaginaryCoefficient_sq hu
  have hz : HasDerivAt (integrandBase u)
      (b * (-(Real.sin θ : ℂ))) θ := by
    change HasDerivAt (fun x => (u : ℂ) + b * (Real.cos x : ℂ))
      (b * (-(Real.sin θ : ℂ))) θ
    simpa only [Complex.ofReal_neg] using
      ((Real.hasDerivAt_cos θ).ofReal_comp.const_mul b).const_add (u : ℂ)
  have hd := (((Real.hasDerivAt_sin θ).ofReal_comp.mul_const b).mul
    (hz.pow (k + 1)))
  convert hd using 1
  simp only [Nat.add_sub_cancel, Nat.cast_add, Nat.cast_mul, Nat.cast_ofNat,
    Nat.cast_one, Pi.pow_apply]
  have htrig : (Real.sin θ : ℂ) ^ 2 + (Real.cos θ : ℂ) ^ 2 = 1 := by
    exact_mod_cast Real.sin_sq_add_cos_sq θ
  change _ = (Real.cos θ : ℂ) * b * integrandBase u θ ^ (k + 1) +
    (Real.sin θ : ℂ) * b *
      (((k : ℂ) + 1) * integrandBase u θ ^ k * (b * -(Real.sin θ : ℂ)))
  simp only [show k + 2 = k + 1 + 1 by omega, integrandBase]
  dsimp [b] at hb ⊢
  linear_combination
    ((k : ℂ) + 1) *
      ((u : ℂ) + Complex.I * (Real.sqrt (1 - u ^ 2) : ℂ) * (Real.cos θ : ℂ)) ^ k * hb +
    ((k : ℂ) + 1) * (Complex.I * (Real.sqrt (1 - u ^ 2) : ℂ)) ^ 2 *
      ((u : ℂ) + Complex.I * (Real.sqrt (1 - u ^ 2) : ℂ) * (Real.cos θ : ℂ)) ^ k * htrig

end PDETranslations.LaplaceLegendre
