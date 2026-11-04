module

public import Mathlib.Analysis.InnerProductSpace.Harmonic.Basic
public import PDETranslations.ParametricIntegral
import PDETranslations.LaplaceTailArithmetic
import PDETranslations.Multiindex

@[expose] public section

/-!
# Harmonic interior and homogeneous derivative estimates for the Laplace kernel

This file proves Lemma 1 of `tail-bound.tex`, conditional on the classical
three-dimensional Poisson formula. The formula is an explicit hypothesis,
not a global axiom. In particular, differentiation under the integral and the
existence of order-dependent derivative bounds are proved, not assumed.

The sphere measure is normalized to have mass one, so the kernel has no
`1 / (4 * π)` factor: that factor is absorbed into normalized surface area.
The measure and its Poisson representation are supplied together; this file
does not construct surface area measure.

The shrinking-ball argument then gives a derivative estimate, uniform over
orders up to `p`, for harmonic terms satisfying the undifferentiated bound.
The application to the actual Laplace Taylor terms is assembled in
`LaplaceEstimate`, using an explicit scalar Legendre coefficient identity.
-/

open MeasureTheory Set Metric InnerProductSpace
open scoped Topology BigOperators

namespace PDETranslations.LaplaceTailBound

abbrev Space := EuclideanSpace ℝ (Fin 3)
abbrev UnitSphere := sphere (0 : Space) 1

instance : CompactSpace UnitSphere :=
  isCompact_iff_compactSpace.mp (isCompact_sphere (0 : Space) 1)

instance : MeasurableSpace UnitSphere := borel UnitSphere
instance : BorelSpace UnitSphere := ⟨rfl⟩

/-- The classical Poisson formula, in normalized coordinates on an arbitrary
3D ball. This is precisely the ordinary (undifferentiated) formula: for
normalized surface area `μ`, the kernel is `(1 - ‖x‖²) / ‖x - ω‖³`.

Reference: the Poisson integral formula for a ball in three-dimensional
potential theory (e.g. Evans, *Partial Differential Equations*, §2.2.4, Thm. 15).
The explicit radius and center avoid requiring a separate formalization of
harmonicity under affine changes of variables. -/
def PoissonFormula (μ : Measure UnitSphere) : Prop :=
  ∀ (v : Space → ℝ) (z : Space) (ρ : ℝ), 0 < ρ →
    HarmonicOnNhd v (closedBall z ρ) →
    ∀ x ∈ ball (0 : Space) 1,
      v (z + ρ • x) =
        ∫ ω : UnitSphere,
          v (z + ρ • (ω : Space)) •
            ParametricIntegral.poissonKernel x (ω : Space) ∂μ

/-- Invertible scalar dilation of an iterated derivative. Because dilation is
an equivalence, this identity does not require a smoothness hypothesis. -/
theorem iteratedFDeriv_dilation {E : Type*}
    [NormedAddCommGroup E] [NormedSpace ℝ E]
    (ρ : ℝ) (hρ : ρ ≠ 0) (f : E → ℝ)
    (x : E) (n : ℕ) (e : Fin n → E) :
    iteratedFDeriv ℝ n (fun y => f (ρ • y)) x e =
      ρ ^ n * iteratedFDeriv ℝ n f (ρ • x) e := by
  let d : E ≃L[ℝ] E :=
    ContinuousLinearEquiv.smulLeft (R₁ := ℝ) (M₁ := E) (Units.mk0 ρ hρ)
  have hd := d.iteratedFDerivWithin_comp_right
    f uniqueDiffOn_univ (mem_univ (d x)) n
  have heval := congrArg (fun T => T e) hd
  simp only [preimage_univ, iteratedFDerivWithin_univ,
    ContinuousMultilinearMap.compContinuousLinearMap_apply,
    ContinuousLinearEquiv.coe_coe] at heval
  change iteratedFDeriv ℝ n (fun y => f (ρ • y)) x e =
    iteratedFDeriv ℝ n f (ρ • x) (fun i => ρ • e i) at heval
  rw [heval, ContinuousMultilinearMap.map_smul_univ]
  simp

/-- Translation and dilation give the scaling identity used in Lemma 1. -/
theorem iteratedFDeriv_rescale (v : Space → ℝ) (z : Space)
    {ρ : ℝ} (hρ : 0 < ρ) (n : ℕ) (e : Fin n → Space) :
    iteratedFDeriv ℝ n (fun x => v (z + ρ • x)) 0 e =
      ρ ^ n * iteratedFDeriv ℝ n v z e := by
  rw [iteratedFDeriv_dilation ρ (ne_of_gt hρ) (fun y => v (z + y)) 0 n e]
  simp only [smul_zero, iteratedFDeriv_comp_add_left, add_zero]

private theorem sphere_point_mem_closedBall (z : Space) {ρ : ℝ}
    (hρ : 0 < ρ) (ω : UnitSphere) : z + ρ • (ω : Space) ∈ closedBall z ρ := by
  have hw : ‖(ω : Space)‖ = 1 := by simpa only [mem_sphere, dist_zero_right] using ω.property
  simp [mem_closedBall, dist_eq_norm, norm_smul, abs_of_pos hρ, hw]

private theorem continuous_boundary_data {v : Space → ℝ} {z : Space} {ρ : ℝ}
    (hρ : 0 < ρ) (hv : HarmonicOnNhd v (closedBall z ρ)) :
    Continuous (fun ω : UnitSphere => v (z + ρ • (ω : Space))) := by
  apply continuousOn_univ.mp
  have hmap : Continuous (fun ω : UnitSphere => z + ρ • (ω : Space)) :=
    continuous_const.add (continuous_subtype_val.const_smul ρ)
  exact hv.contDiffOn.continuousOn.comp hmap.continuousOn
    (fun ω _ => sphere_point_mem_closedBall z hρ ω)

/-- **Harmonic interior derivative estimate (Lemma 1).** For each order `m`
there is a positive constant, independent of the function, center, radius,
bound, and derivative directions. The result is slightly stronger than the
coordinate version: it holds for any directions of norm at most one.

A supplied uniform bound `M` replaces a supremum, avoiding conditional
completeness bookkeeping. Taking `M` to be the supremum on the closed ball
gives the statement in `tail-bound.tex`. The only mathematical assumption is
`PoissonFormula μ`; kernel regularity and derivative interchange are proved.
-/
theorem exists_harmonic_interior_estimate
    (μ : Measure UnitSphere) [IsProbabilityMeasure μ] (hPoisson : PoissonFormula μ)
    (m : ℕ) :
    ∃ A : ℝ, 0 < A ∧
      ∀ (v : Space → ℝ) (z : Space) (ρ M : ℝ), 0 < ρ →
        HarmonicOnNhd v (closedBall z ρ) →
        (∀ w ∈ closedBall z ρ, |v w| ≤ M) →
        ∀ e : Fin m → Space, (∀ i, ‖e i‖ ≤ 1) →
          |iteratedFDeriv ℝ m v z e| ≤ A / ρ ^ m * M := by
  have hb : Continuous (fun ω : UnitSphere => (ω : Space)) := continuous_subtype_val
  have hnorm : ∀ ω : UnitSphere, ‖(ω : Space)‖ = 1 :=
    fun ω => by simpa only [mem_sphere, dist_zero_right] using ω.property
  obtain ⟨B, hB0, hB⟩ := ParametricIntegral.exists_bound_poissonIntegral
    (μ := μ) hb hnorm 0 (by norm_num) m
  refine ⟨B + 1, by positivity, ?_⟩
  intro v z ρ M hρ hv hM e he
  have hM0 : 0 ≤ M := (abs_nonneg (v z)).trans (hM z (by simp [hρ.le]))
  let g : UnitSphere → ℝ := fun ω => v (z + ρ • (ω : Space))
  let u : Space → ℝ := fun x => v (z + ρ • x)
  have hg : Continuous g := continuous_boundary_data hρ hv
  have hgM : ∀ ω, ‖g ω‖ ≤ M := by
    intro ω
    exact (show |g ω| ≤ M from hM _ (sphere_point_mem_closedBall z hρ ω))
  have hrepr : EqOn u
      (fun y => ∫ ω, g ω • ParametricIntegral.poissonKernel y (ω : Space) ∂μ)
      (ball (0 : Space) 1) := hPoisson v z ρ hρ hv
  have hzero : (0 : Space) ∈ ball (0 : Space) 1 := by simp
  have hlocal := hrepr.eventuallyEq_of_mem (isOpen_ball.mem_nhds hzero)
  have hderiv := (hlocal.iteratedFDeriv ℝ m).eq_of_nhds
  have hbound : ‖iteratedFDeriv ℝ m u 0‖ ≤ B * M := by
    rw [hderiv]
    exact hB g hg M hgM 0 (by simp)
  have hprod : ∏ i : Fin m, ‖e i‖ ≤ 1 := by
    simpa using Finset.prod_le_prod₀ (fun i _ => norm_nonneg (e i)) (fun i _ => he i)
  have happly : |iteratedFDeriv ℝ m u 0 e| ≤ B * M := by
    calc
      |iteratedFDeriv ℝ m u 0 e| = ‖iteratedFDeriv ℝ m u 0 e‖ := rfl
      _ ≤ ‖iteratedFDeriv ℝ m u 0‖ * ∏ i, ‖e i‖ :=
        (iteratedFDeriv ℝ m u 0).le_opNorm e
      _ ≤ ‖iteratedFDeriv ℝ m u 0‖ * 1 :=
        mul_le_mul_of_nonneg_left hprod (norm_nonneg _)
      _ ≤ B * M := by simpa using hbound
  have hscale : iteratedFDeriv ℝ m u 0 e =
      ρ ^ m * iteratedFDeriv ℝ m v z e := iteratedFDeriv_rescale v z hρ m e
  rw [hscale, abs_mul, abs_of_pos (pow_pos hρ m)] at happly
  have hle : |iteratedFDeriv ℝ m v z e| ≤ B / ρ ^ m * M := by
    rw [div_mul_eq_mul_div]
    apply (le_div_iff₀ (pow_pos hρ m)).mpr
    nlinarith [happly]
  exact hle.trans (mul_le_mul_of_nonneg_right
    (div_le_div_of_nonneg_right (by linarith : B ≤ B + 1) (pow_pos hρ m).le) hM0)

/-- The shrinking ball stays away from the singularity, and every point in it
has norm at least the ambient norm minus the radius. -/
theorem norm_lower_bound_on_closedBall (z w : Space) {ρ : ℝ}
    (hw : w ∈ closedBall z ρ) : ‖z‖ - ρ ≤ ‖w‖ := by
  have hd : ‖z - w‖ ≤ ρ := by
    simpa only [mem_closedBall, dist_eq_norm, norm_sub_rev] using hw
  have ht : ‖z‖ ≤ ‖z - w‖ + ‖w‖ := by
    simpa using norm_add_le (z - w) w
  linarith

/-- The analytic core of the homogeneous-term estimate. Any family of harmonic
functions with the undifferentiated Laplace bound has the desired derivative
bound. In the application, `v` is the degree-`k` Taylor term for fixed `h` and
`H = ‖h‖`. No restriction `H < ‖z‖` is needed, including when `H = 0`.

The hypotheses about `v` are deliberately explicit: establishing them for the
actual Taylor terms requires the Legendre argument and harmonicity of kernel
derivatives. -/
theorem exists_homogeneous_derivative_estimate
    (μ : Measure UnitSphere) [IsProbabilityMeasure μ] (hPoisson : PoissonFormula μ)
    (m : ℕ) :
    ∃ C : ℝ, 0 < C ∧
      ∀ (v : Space → ℝ) (k : ℕ) (H : ℝ), 0 ≤ H →
        HarmonicOnNhd v ({0}ᶜ : Set Space) →
        (∀ w : Space, w ≠ 0 → |v w| ≤ H ^ k / ‖w‖ ^ (k + 1)) →
        ∀ z : Space, z ≠ 0 →
          ∀ e : Fin m → Space, (∀ i, ‖e i‖ ≤ 1) →
            |iteratedFDeriv ℝ m v z e| ≤
              C * ((k : ℝ) + 1) ^ m * H ^ k / ‖z‖ ^ (k + m + 1) := by
  obtain ⟨A, hA, hbound⟩ := exists_harmonic_interior_estimate μ hPoisson m
  refine ⟨A * 2 ^ (m + 1), by positivity, ?_⟩
  intro v k H hH hv hterm z hz e he
  have hR : 0 < ‖z‖ := norm_pos_iff.mpr hz
  let ρ : ℝ := ‖z‖ / (2 * ((k : ℝ) + 1))
  obtain ⟨hρ, hρR⟩ := shrinking_radius_pos_lt ‖z‖ hR k
  have hgap : 0 < ‖z‖ - ρ := sub_pos.mpr hρR
  have hlower : ∀ w ∈ closedBall z ρ, ‖z‖ - ρ ≤ ‖w‖ :=
    fun w hw => norm_lower_bound_on_closedBall z w hw
  have hnonzero : ∀ w ∈ closedBall z ρ, w ≠ 0 := by
    intro w hw
    exact norm_pos_iff.mp (hgap.trans_le (hlower w hw))
  have hvball : HarmonicOnNhd v (closedBall z ρ) :=
    hv.mono (fun w hw => by simpa using hnonzero w hw)
  have hM : ∀ w ∈ closedBall z ρ,
      |v w| ≤ H ^ k / (‖z‖ - ρ) ^ (k + 1) := by
    intro w hw
    apply (hterm w (hnonzero w hw)).trans
    exact div_le_div_of_nonneg_left (pow_nonneg hH k)
      (pow_pos hgap (k + 1))
      (pow_le_pow_left₀ hgap.le (hlower w hw) (k + 1))
  have hi := hbound v z ρ (H ^ k / (‖z‖ - ρ) ^ (k + 1)) hρ hvball hM e he
  have harith := mul_le_mul_of_nonneg_left
    (shrinking_radius_bound ‖z‖ hR k m) (show 0 ≤ A * H ^ k by positivity)
  calc
    |iteratedFDeriv ℝ m v z e| ≤
        A / ρ ^ m * (H ^ k / (‖z‖ - ρ) ^ (k + 1)) := hi
    _ = (A * H ^ k) *
        ((1 / ρ) ^ m * (1 / (‖z‖ - ρ)) ^ (k + 1)) := by
      simp only [div_pow, one_pow]
      ring
    _ ≤ (A * H ^ k) *
        (2 ^ (m + 1) * ((k : ℝ) + 1) ^ m / ‖z‖ ^ (k + m + 1)) := by
      simpa only [ρ, one_div_pow] using harith
    _ = _ := by ring

/-- One constant controls all derivative orders up to `p`. This is the
polynomial-in-degree estimate from the proposition, expressed for a harmonic
homogeneous term satisfying its undifferentiated bound. -/
theorem exists_uniform_homogeneous_derivative_estimate
    (μ : Measure UnitSphere) [IsProbabilityMeasure μ] (hPoisson : PoissonFormula μ)
    (p : ℕ) :
    ∃ C : ℝ, 0 < C ∧
      ∀ (m : ℕ), m ≤ p →
        ∀ (v : Space → ℝ) (k : ℕ) (H : ℝ), 0 ≤ H →
          HarmonicOnNhd v ({0}ᶜ : Set Space) →
          (∀ w : Space, w ≠ 0 → |v w| ≤ H ^ k / ‖w‖ ^ (k + 1)) →
          ∀ z : Space, z ≠ 0 →
            ∀ e : Fin m → Space, (∀ i, ‖e i‖ ≤ 1) →
              |iteratedFDeriv ℝ m v z e| ≤
                C * ((k : ℝ) + 1) ^ p * H ^ k / ‖z‖ ^ (k + m + 1) := by
  classical
  choose D hD hbound using exists_homogeneous_derivative_estimate μ hPoisson
  let C : ℝ := 1 + ∑ m ∈ Finset.range (p + 1), D m
  have hsum : 0 ≤ ∑ m ∈ Finset.range (p + 1), D m :=
    Finset.sum_nonneg (fun m _ => (hD m).le)
  have hC : 0 < C := by dsimp [C]; linarith
  refine ⟨C, hC, ?_⟩
  intro m hmp v k H hH hv hterm z hz e he
  have hDC : D m ≤ C := by
    have hle := Finset.single_le_sum (fun i _ => (hD i).le)
      (show m ∈ Finset.range (p + 1) from Finset.mem_range.mpr (by omega))
    dsimp [C]
    linarith
  have hpow : ((k : ℝ) + 1) ^ m ≤ ((k : ℝ) + 1) ^ p :=
    pow_le_pow_right₀ (by linarith [Nat.cast_nonneg (α := ℝ) k] : (1 : ℝ) ≤ (k : ℝ) + 1) hmp
  apply (hbound m v k H hH hv hterm z hz e he).trans
  apply div_le_div_of_nonneg_right _ (by positivity)
  apply mul_le_mul_of_nonneg_right _ (pow_nonneg hH k)
  exact mul_le_mul hDC hpow (by positivity) hC.le

end PDETranslations.LaplaceTailBound
