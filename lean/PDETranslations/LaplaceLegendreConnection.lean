module

public import PDETranslations.LaplaceHarmonic
public import PDETranslations.LaplaceLegendre
import Mathlib.Analysis.Calculus.IteratedDeriv.Lemmas
import Mathlib.Analysis.SpecialFunctions.Sqrt
import PDETranslations.LaplaceTaylorCoefficients

@[expose] public section

/-!
# The Legendre connection for actual Laplace Taylor terms

The only classical Legendre input is `LegendreTaylorFormula`, a scalar
coefficient identity. The vector factorization, derivative identification,
and sharp bound are proved here, not included in that hypothesis.
-/

open scoped Topology ContDiff

namespace PDETranslations.LaplaceLegendreConnection

open LaplaceHarmonic LaplaceLegendre LaplaceTaylorCoefficients

/-- The classical scalar Taylor coefficient identity for the integral-defined `P`.
There is no vector kernel or Taylor-term bound in this assumption.
https://dlmf.nist.gov/14.7#E19
-/
def LegendreTaylorFormula : Prop :=
  ∀ u : ℝ, |u| ≤ 1 → ∀ k : ℕ,
    iteratedDeriv k (fun s : ℝ => 1 / Real.sqrt (1 - 2 * u * s + s ^ 2)) 0 =
      (k.factorial : ℝ) * P k u

/-- The sign agrees with the generating function `1 - 2*u*s + s²`. -/
noncomputable def angleParameter (x h : Space) : ℝ :=
  -inner ℝ x h / (‖x‖ * ‖h‖)

theorem abs_angleParameter_le_one (x h : Space) : |angleParameter x h| ≤ 1 := by
  unfold angleParameter
  rw [abs_div, abs_neg, abs_of_nonneg (mul_nonneg (norm_nonneg _) (norm_nonneg _))]
  exact div_le_one_of_le₀ (abs_real_inner_le_norm x h) (by positivity)

/-- Squared-norm factorization; no smallness condition on the line parameter. -/
theorem norm_line_sq (x h : Space) (hx : x ≠ 0) (hh : h ≠ 0) (t : ℝ) :
    ‖x + t • h‖ ^ 2 = ‖x‖ ^ 2 *
      (1 - 2 * angleParameter x h * (t * (‖h‖ / ‖x‖)) +
        (t * (‖h‖ / ‖x‖)) ^ 2) := by
  have hR : ‖x‖ ≠ 0 := norm_ne_zero_iff.mpr hx
  have hH : ‖h‖ ≠ 0 := norm_ne_zero_iff.mpr hh
  rw [norm_add_sq_real, real_inner_smul_right, norm_smul, Real.norm_eq_abs]
  rw [mul_pow, sq_abs]
  unfold angleParameter
  field_simp
  ring

/-- The actual norm factorization, including line points at the pole. -/
theorem norm_line (x h : Space) (hx : x ≠ 0) (hh : h ≠ 0) (t : ℝ) :
    ‖x + t • h‖ = ‖x‖ * Real.sqrt
      (1 - 2 * angleParameter x h * (t * (‖h‖ / ‖x‖)) +
        (t * (‖h‖ / ‖x‖)) ^ 2) := by
  have hR : 0 < ‖x‖ := norm_pos_iff.mpr hx
  have he := norm_line_sq x h hx hh t
  have hp : 0 ≤ 1 - 2 * angleParameter x h * (t * (‖h‖ / ‖x‖)) +
      (t * (‖h‖ / ‖x‖)) ^ 2 := by
    nlinarith [sq_nonneg ‖x + t • h‖, sq_pos_of_pos hR]
  have hs := Real.sq_sqrt hp
  nlinarith [norm_nonneg (x + t • h), Real.sqrt_nonneg
    (1 - 2 * angleParameter x h * (t * (‖h‖ / ‖x‖)) +
      (t * (‖h‖ / ‖x‖)) ^ 2), mul_nonneg hR.le (Real.sqrt_nonneg
    (1 - 2 * angleParameter x h * (t * (‖h‖ / ‖x‖)) +
      (t * (‖h‖ / ‖x‖)) ^ 2))]

theorem G0_line (x h : Space) (hx : x ≠ 0) (hh : h ≠ 0) :
    (fun t : ℝ => G0 (x + t • h)) =
      fun t => (1 / ‖x‖) * (1 / Real.sqrt
        (1 - 2 * angleParameter x h * (t * (‖h‖ / ‖x‖)) +
          (t * (‖h‖ / ‖x‖)) ^ 2)) := by
  funext t
  rw [G0, norm_line x h hx hh t]
  simp [div_eq_mul_inv, mul_inv_rev, mul_comm]

/-- Local version of the linear-composition rule (global smoothness is unnecessary). -/
theorem iteratedFDeriv_comp_linear_at {E F G : Type*}
    [NormedAddCommGroup E] [NormedSpace ℝ E]
    [NormedAddCommGroup F] [NormedSpace ℝ F]
    [NormedAddCommGroup G] [NormedSpace ℝ G]
    (k : ℕ) (g : G →L[ℝ] E) {f : E → F} (x : G)
    (hf : ContDiffAt ℝ k f (g x)) :
    iteratedFDeriv ℝ k (f ∘ g) x =
      (iteratedFDeriv ℝ k f (g x)).compContinuousLinearMap (fun _ => g) := by
  rcases hf.contDiffOn' le_rfl (by simp) with ⟨s, hs, hx, hf⟩
  have hf' : ContDiffOn ℝ k f s := by simpa using hf
  have ht : IsOpen (g ⁻¹' s) := hs.preimage g.continuous
  have he := g.iteratedFDerivWithin_comp_right hf' hs.uniqueDiffOn ht.uniqueDiffOn hx le_rfl
  rw [iteratedFDerivWithin_of_isOpen k ht hx,
    iteratedFDerivWithin_of_isOpen k hs hx] at he
  exact he

/-- Restriction to an affine line evaluates all derivative slots at its direction. -/
theorem iteratedDeriv_line (k : ℕ) (h x : Space) (hx : x ≠ 0) :
    iteratedDeriv k (fun t : ℝ => G0 (x + t • h)) 0 =
      iteratedFDeriv ℝ k G0 x (fun _ => h) := by
  let g : ℝ →L[ℝ] Space := (ContinuousLinearMap.id ℝ ℝ).smulRight h
  have hf : ContDiffAt ℝ k (fun y : Space => G0 (x + y)) (g 0) := by
    have hG := contDiffOn_G0.contDiffAt
      (isClosed_singleton.isOpen_compl.mem_nhds (show x ∈ ({0}ᶜ : Set Space) by simpa using hx))
    have hG' : ContDiffAt ℝ k G0 (x + (0 : Space)) := by
      simpa using hG.of_le (show (k : ℕ∞ω) ≤ ⊤ from le_top)
    simpa [g, Function.comp_def] using hG'.comp (0 : Space)
      (show ContDiffAt ℝ k (fun y : Space => x + y) 0 from
        contDiffAt_const.add contDiffAt_id)
  rw [iteratedDeriv_eq_iteratedFDeriv]
  change iteratedFDeriv ℝ k ((fun y : Space => G0 (x + y)) ∘ g) 0 _ = _
  rw [iteratedFDeriv_comp_linear_at k g 0 hf]
  simp [ContinuousMultilinearMap.compContinuousLinearMap_apply, g,
    iteratedFDeriv_comp_add_left]

/-- Scaling of scalar iterated derivatives, with only local smoothness at zero. -/
theorem iteratedDeriv_scale (k : ℕ) (f : ℝ → ℝ) (a : ℝ)
    (hf : ContDiffAt ℝ k f 0) :
    iteratedDeriv k (fun t => f (t * a)) 0 = a ^ k * iteratedDeriv k f 0 := by
  let g : ℝ →L[ℝ] ℝ := a • ContinuousLinearMap.id ℝ ℝ
  have hg : (fun t => f (t * a)) = f ∘ g := by
    ext t
    simp [g, mul_comm]
  rw [hg, iteratedDeriv_eq_iteratedFDeriv]
  rw [iteratedFDeriv_comp_linear_at k g 0 (by simpa [g] using hf)]
  simp [ContinuousMultilinearMap.compContinuousLinearMap_apply, g,
    iteratedFDeriv_apply_eq_iteratedDeriv_mul_prod]

/-- Smoothness needed for scalar scaling follows directly from the radicand at zero. -/
theorem contDiffAt_scalarKernel (u : ℝ) (k : ℕ) :
    ContDiffAt ℝ k (fun s : ℝ => 1 / Real.sqrt (1 - 2 * u * s + s ^ 2)) 0 := by
  apply contDiffAt_const.div
  · apply ContDiffAt.sqrt
    · fun_prop
    · norm_num
  · norm_num

/-- Exact identification for a nonzero displacement, conditional only on the
classical scalar coefficient formula. -/
theorem homogeneousTaylorTerm_G0_eq_scaledCoefficient
    (hLegendre : LegendreTaylorFormula) (k : ℕ) (h x : Space)
    (hx : x ≠ 0) (hh : h ≠ 0) :
    homogeneousTaylorTerm k G0 h x =
      scaledCoefficient k (angleParameter x h) ‖x‖ ‖h‖ := by
  rw [homogeneousTaylorTerm_G0_eq_normalized_iteratedFDeriv k h x hx,
    ← iteratedDeriv_line k h x hx, G0_line x h hx hh,
    iteratedDeriv_const_mul_field, iteratedDeriv_scale k _ _
      (contDiffAt_scalarKernel _ k),
    hLegendre _ (abs_angleParameter_le_one x h) k]
  have hk : (k.factorial : ℝ) ≠ 0 := by exact_mod_cast Nat.factorial_ne_zero k
  rw [scaledCoefficient_eq]
  unfold coefficient
  field_simp

/-- The degenerate direction contributes only the zeroth term. -/
theorem homogeneousTaylorTerm_G0_zero_direction (k : ℕ) (x : Space) (hx : x ≠ 0) :
    homogeneousTaylorTerm k G0 0 x = if k = 0 then G0 x else 0 := by
  rw [homogeneousTaylorTerm_G0_eq_normalized_iteratedFDeriv k 0 x hx]
  cases k with
  | zero => simp [iteratedFDeriv_zero_apply]
  | succ k =>
    have he : iteratedFDeriv ℝ (k + 1) G0 x (fun _ => (0 : Space)) = 0 :=
      (iteratedFDeriv ℝ (k + 1) G0 x).map_coord_zero (0 : Fin (k + 1)) rfl
    simp [he]

/-- Sharp bound for every actual homogeneous Taylor term of the Laplace kernel.
No restriction on the size of the displacement is needed. The sole hypothesis
beyond `x ≠ 0` is the explicit scalar `LegendreTaylorFormula`. -/
theorem abs_homogeneousTaylorTerm_G0_le
    (hLegendre : LegendreTaylorFormula) (k : ℕ) (h x : Space) (hx : x ≠ 0) :
    |homogeneousTaylorTerm k G0 h x| ≤ ‖h‖ ^ k / ‖x‖ ^ (k + 1) := by
  by_cases hh : h = 0
  · subst h
    rw [homogeneousTaylorTerm_G0_zero_direction k x hx]
    cases k with
    | zero => simp [G0]
    | succ k => simp
  · rw [homogeneousTaylorTerm_G0_eq_scaledCoefficient hLegendre k h x hx hh]
    exact abs_scaledCoefficient_le k (abs_angleParameter_le_one x h)
      (norm_pos_iff.mpr hx) (norm_nonneg h)

end PDETranslations.LaplaceLegendreConnection
