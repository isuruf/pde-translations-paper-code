module

public import Mathlib.Analysis.Calculus.ContDiff.Defs
public import Mathlib.Analysis.InnerProductSpace.Defs
public import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.Analysis.Calculus.ContDiff.Comp
import Mathlib.Analysis.Calculus.ContDiff.Operations
import Mathlib.Analysis.Calculus.ParametricIntegral
import Mathlib.Analysis.InnerProductSpace.Calculus
import Mathlib.Analysis.Normed.Module.FiniteDimension
import Mathlib.MeasureTheory.Function.LocallyIntegrable
import Mathlib.Topology.Order.Compact

@[expose] public section

/-!
# All-order differentiation of compact-parameter integrals

The derivative interchange is proved from Mathlib's first-order dominated
Bochner differentiation theorem. Joint continuity of the spatial derivatives
supplies the local dominating constants by compactness; no derivative
interchange or PDE estimate is assumed.
-/

open MeasureTheory Set Metric
open scoped Topology

local notation "SmoothOrder" => ((⊤ : ℕ∞) : WithTop ℕ∞)

namespace PDETranslations.ParametricIntegral

section General

variable {E Ω : Type*}
  [NormedAddCommGroup E] [NormedSpace ℝ E] [FiniteDimensional ℝ E]
  [TopologicalSpace Ω] [CompactSpace Ω] [MeasurableSpace Ω] [BorelSpace Ω]
  {μ : Measure Ω} [IsFiniteMeasure μ]
  {U : Set E}

omit [NormedSpace ℝ E] [FiniteDimensional ℝ E] [CompactSpace Ω]
  [MeasurableSpace Ω] [BorelSpace Ω] in
private theorem continuous_section {V : Type*} [TopologicalSpace V]
    {F : E → Ω → V}
    (hF : ContinuousOn (Function.uncurry F) (U ×ˢ univ))
    {x : E} (hx : x ∈ U) : Continuous (F x) := by
  apply continuousOn_univ.mp
  exact hF.comp (continuous_const.prodMk continuous_id).continuousOn
    (by intro ω _; exact ⟨hx, mem_univ ω⟩)

private theorem integrable_continuous {V : Type*} [NormedAddCommGroup V]
    {f : Ω → V} (hf : Continuous f) : Integrable f μ := by
  simpa only [IntegrableOn, Measure.restrict_univ] using
    hf.continuousOn.integrableOn_compact' isCompact_univ MeasurableSet.univ

/-- First-order differentiation on an open spatial domain. Compactness of the
parameter space and of a sufficiently small closed spatial ball gives the
uniform local bound required by dominated Bochner differentiation. -/
theorem hasFDerivAt_integral_of_joint_continuous
    {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]
    {F : E → Ω → V} {F' : E → Ω → E →L[ℝ] V}
    (hU : IsOpen U)
    (hF : ContinuousOn (Function.uncurry F) (U ×ˢ univ))
    (hF' : ContinuousOn (Function.uncurry F') (U ×ˢ univ))
    (hdiff : ∀ x ∈ U, ∀ ω, HasFDerivAt (fun y => F y ω) (F' x ω) x)
    {x : E} (hx : x ∈ U) :
    HasFDerivAt (fun y => ∫ ω, F y ω ∂μ) (∫ ω, F' x ω ∂μ) x := by
  obtain ⟨r, hr, hball⟩ := Metric.mem_nhds_iff.mp (hU.mem_nhds hx)
  have hr2 : 0 < r / 2 := half_pos hr
  have hclosed : closedBall x (r / 2) ⊆ U :=
    (closedBall_subset_ball (half_lt_self hr)).trans hball
  have hcompact : IsCompact (closedBall x (r / 2) ×ˢ (univ : Set Ω)) :=
    (isCompact_closedBall x (r / 2)).prod isCompact_univ
  obtain ⟨C, hC⟩ := hcompact.bddAbove_image
    (hF'.norm.mono (prod_mono hclosed Subset.rfl))
  apply hasFDerivAt_integral_of_dominated_of_fderiv_le
    (s := closedBall x (r / 2)) (bound := fun _ => C)
    (closedBall_mem_nhds x hr2)
  · filter_upwards [hU.mem_nhds hx] with y hy
    exact (integrable_continuous (continuous_section hF hy)).aestronglyMeasurable
  · exact integrable_continuous (continuous_section hF hx)
  · exact (integrable_continuous (continuous_section hF' hx)).aestronglyMeasurable
  · exact ae_of_all _ (fun ω y hy => hC ⟨(y, ω), ⟨hy, mem_univ ω⟩, rfl⟩)
  · exact integrable_const C
  · exact ae_of_all _ (fun ω y hy => hdiff y (hclosed hy) ω)

/-- All spatial derivatives of a compact-parameter integral can be taken under
the integral. The hypotheses concern only the kernel: smooth spatial sections
and joint continuity of each spatial derivative on the open domain. -/
theorem iteratedFDeriv_integral
    {F : E → Ω → ℝ}
    (hU : IsOpen U)
    (hsmooth : ∀ x ∈ U, ∀ ω, ContDiffAt ℝ SmoothOrder (fun y => F y ω) x)
    (hjoint : ∀ n : ℕ, ContinuousOn
      (fun p : E × Ω => iteratedFDeriv ℝ n (fun y => F y p.2) p.1)
      (U ×ˢ univ))
    (n : ℕ) {x : E} (hx : x ∈ U) :
    iteratedFDeriv ℝ n (fun y => ∫ ω, F y ω ∂μ) x =
      ∫ ω, iteratedFDeriv ℝ n (fun y => F y ω) x ∂μ := by
  induction n generalizing x with
  | zero =>
    exact ((continuousMultilinearCurryFin0 ℝ E ℝ).symm.toContinuousLinearEquiv.integral_comp_comm (fun ω => F x ω)).symm
  | succ n ih =>
    -- The operator-space topology does not automatically synthesize this
    -- instance for multilinear-valued operators in this Mathlib version.
    let : ContinuousENorm (E →L[ℝ] E [×n]→L[ℝ] ℝ) := ⟨by fun_prop⟩
    let D : E → Ω → E [×n]→L[ℝ] ℝ :=
      fun y ω => iteratedFDeriv ℝ n (fun z => F z ω) y
    let D' : E → Ω → E →L[ℝ] E [×n]→L[ℝ] ℝ :=
      fun y ω => fderiv ℝ (fun z => D z ω) y

    have hD' : ContinuousOn (Function.uncurry D') (U ×ˢ univ) := by
      change ContinuousOn (fun p : E × Ω =>
        (continuousMultilinearCurryLeftEquiv ℝ (fun _ : Fin (n + 1) => E) ℝ)
          (iteratedFDeriv ℝ (n + 1) (fun z => F z p.2) p.1)) (U ×ˢ univ)
      exact (continuousMultilinearCurryLeftEquiv ℝ
        (fun _ : Fin (n + 1) => E) ℝ).continuous.comp_continuousOn (hjoint (n + 1))
    have hderiv := hasFDerivAt_integral_of_joint_continuous
      (μ := μ) (F := D) (F' := D') hU (hjoint n) hD'
      (fun y hy ω => ((hsmooth y hy ω).iteratedFDeriv_right
        (m := SmoothOrder) (by simp)).differentiableAt (by simp) |>.hasFDerivAt) hx
    have heq : iteratedFDeriv ℝ n (fun y => ∫ ω, F y ω ∂μ) =ᶠ[𝓝 x]
        (fun y => ∫ ω, D y ω ∂μ) := by
      filter_upwards [hU.mem_nhds hx] with y hy using ih hy
    rw [iteratedFDeriv_succ_eq_comp_left, Function.comp_apply,
      heq.fderiv_eq, hderiv.fderiv]
    have hint : Integrable (D' x) μ :=
      integrable_continuous (V := E →L[ℝ] E [×n]→L[ℝ] ℝ)
        (continuous_section hD' hx)
    have hint_succ : Integrable
        (fun ω => iteratedFDeriv ℝ (n + 1) (fun z => F z ω) x) μ :=
      integrable_continuous (V := E [×(n + 1)]→L[ℝ] ℝ)
        (continuous_section (F := fun y ω =>
          iteratedFDeriv ℝ (n + 1) (fun z => F z ω) y) (hjoint (n + 1)) hx)
    ext m
    change (∫ ω, D' x ω ∂μ) (m 0) (Fin.tail m) =
      (∫ ω, iteratedFDeriv ℝ (n + 1) (fun z => F z ω) x ∂μ) m
    rw [ContinuousLinearMap.integral_apply hint,
      ContinuousMultilinearMap.integral_apply (hint.apply_continuousLinearMap (m 0)),
      ContinuousMultilinearMap.integral_apply hint_succ]
    rfl

omit [FiniteDimensional ℝ E] in
/-- Every spatial derivative section is Bochner integrable. This can be used
with `ContinuousMultilinearMap.integral_apply` after derivative interchange. -/
theorem integrable_iteratedFDeriv_section
    {F : E → Ω → ℝ} {n : ℕ}
    (hjoint : ContinuousOn
      (fun p : E × Ω => iteratedFDeriv ℝ n (fun y => F y p.2) p.1)
      (U ×ˢ univ))
    {x : E} (hx : x ∈ U) :
    Integrable (fun ω => iteratedFDeriv ℝ n (fun y => F y ω) x) μ :=
  integrable_continuous (V := E [×n]→L[ℝ] ℝ)
    (continuous_section (F := fun y ω => iteratedFDeriv ℝ n (fun z => F z ω) y)
      hjoint hx)

/-- A bound on the kernel derivative gives the same bound on the derivative
of its integral, multiplied by the total mass of the finite measure. -/
theorem norm_iteratedFDeriv_integral_le
    {F : E → Ω → ℝ}
    (hU : IsOpen U)
    (hsmooth : ∀ x ∈ U, ∀ ω, ContDiffAt ℝ SmoothOrder (fun y => F y ω) x)
    (hjoint : ∀ n : ℕ, ContinuousOn
      (fun p : E × Ω => iteratedFDeriv ℝ n (fun y => F y p.2) p.1)
      (U ×ˢ univ))
    (n : ℕ) {x : E} (hx : x ∈ U) {B : ℝ}
    (hB : ∀ᵐ ω ∂μ, ‖iteratedFDeriv ℝ n (fun y => F y ω) x‖ ≤ B) :
    ‖iteratedFDeriv ℝ n (fun y => ∫ ω, F y ω ∂μ) x‖ ≤ B * μ.real univ := by
  rw [iteratedFDeriv_integral hU hsmooth hjoint n hx]
  exact norm_integral_le_of_norm_le_const hB

/-- Continuous boundary data can be pulled out of every spatial derivative.
Only the unweighted kernel needs the smoothness and joint-continuity proofs. -/
theorem iteratedFDeriv_integral_smul
    {K : E → Ω → ℝ} {g : Ω → ℝ}
    (hU : IsOpen U)
    (hsmooth : ∀ x ∈ U, ∀ ω, ContDiffAt ℝ SmoothOrder (fun y => K y ω) x)
    (hjoint : ∀ n : ℕ, ContinuousOn
      (fun p : E × Ω => iteratedFDeriv ℝ n (fun y => K y p.2) p.1)
      (U ×ˢ univ))
    (hg : Continuous g) (n : ℕ) {x : E} (hx : x ∈ U) :
    iteratedFDeriv ℝ n (fun y => ∫ ω, g ω • K y ω ∂μ) x =
      ∫ ω, g ω • iteratedFDeriv ℝ n (fun y => K y ω) x ∂μ := by
  have hweighted : ∀ n : ℕ, ContinuousOn
      (fun p : E × Ω => iteratedFDeriv ℝ n (fun y => g p.2 • K y p.2) p.1)
      (U ×ˢ univ) := by
    intro n
    apply ((hg.comp continuous_snd).continuousOn.smul (hjoint n)).congr
    intro p hp
    exact (iteratedFDeriv_const_smul_apply'
      ((hsmooth p.1 hp.1 p.2).of_le (by simp)) :
      iteratedFDeriv ℝ n (fun y => g p.2 • K y p.2) p.1 =
        g p.2 • iteratedFDeriv ℝ n (fun y => K y p.2) p.1)
  rw [iteratedFDeriv_integral hU
    (fun y hy ω => (hsmooth y hy ω).const_smul (g ω)) hweighted n hx]
  apply integral_congr_ae
  exact ae_of_all _ (fun ω => iteratedFDeriv_const_smul_apply'
    ((hsmooth x hx ω).of_le (by simp)))

/-- For a probability measure, a kernel derivative bound `B` and boundary
supremum bound `M` give the interior derivative bound `B * M`. -/
theorem norm_iteratedFDeriv_integral_smul_le
    [IsProbabilityMeasure μ]
    {K : E → Ω → ℝ} {g : Ω → ℝ}
    (hU : IsOpen U)
    (hsmooth : ∀ x ∈ U, ∀ ω, ContDiffAt ℝ SmoothOrder (fun y => K y ω) x)
    (hjoint : ∀ n : ℕ, ContinuousOn
      (fun p : E × Ω => iteratedFDeriv ℝ n (fun y => K y p.2) p.1)
      (U ×ˢ univ))
    (hg : Continuous g) (n : ℕ) {x : E} (hx : x ∈ U)
    {B M : ℝ} (hB0 : 0 ≤ B)
    (hB : ∀ ω, ‖iteratedFDeriv ℝ n (fun y => K y ω) x‖ ≤ B)
    (hM : ∀ ω, ‖g ω‖ ≤ M) :
    ‖iteratedFDeriv ℝ n (fun y => ∫ ω, g ω • K y ω ∂μ) x‖ ≤ B * M := by
  rw [iteratedFDeriv_integral_smul hU hsmooth hjoint hg n hx]
  have hbound : ∀ᵐ ω ∂μ, ‖g ω • iteratedFDeriv ℝ n (fun y => K y ω) x‖ ≤ B * M :=
    ae_of_all _ (fun ω => by
      rw [norm_smul, mul_comm]
      exact mul_le_mul (hB ω) (hM ω) (norm_nonneg _) hB0)
  simpa using norm_integral_le_of_norm_le_const hbound

omit [FiniteDimensional ℝ E] [MeasurableSpace Ω] [BorelSpace Ω] in
/-- Compactness gives a nonnegative bound for each derivative order, uniform
in both spatial points in `A` and parameters. The constant is independent of
any boundary data subsequently multiplied into the kernel. -/
theorem exists_uniform_bound_iteratedFDeriv
    {K : E → Ω → ℝ} {A : Set E}
    (hA : IsCompact A) (hAU : A ⊆ U)
    (hjoint : ∀ n : ℕ, ContinuousOn
      (fun p : E × Ω => iteratedFDeriv ℝ n (fun y => K y p.2) p.1)
      (U ×ˢ univ)) (n : ℕ) :
    ∃ B : ℝ, 0 ≤ B ∧ ∀ x ∈ A, ∀ ω,
      ‖iteratedFDeriv ℝ n (fun y => K y ω) x‖ ≤ B := by
  obtain ⟨C, hC⟩ := (hA.prod (isCompact_univ : IsCompact (univ : Set Ω))).bddAbove_image
    ((hjoint n).norm.mono (prod_mono hAU Subset.rfl))
  refine ⟨max C 0, le_max_right _ _, ?_⟩
  intro x hx ω
  exact (hC ⟨(x, ω), ⟨hx, mem_univ ω⟩, rfl⟩).trans (le_max_left _ _)

omit [FiniteDimensional ℝ E] [TopologicalSpace Ω] [CompactSpace Ω]
  [MeasurableSpace Ω] [BorelSpace Ω] in
/-- Joint ambient smoothness implies joint smoothness of every derivative
in the first variable. The parameter variable is not differentiated. -/
theorem contDiffAt_spatial_iteratedFDeriv
    {P : Type*} [NormedAddCommGroup P] [NormedSpace ℝ P]
    {G : E × P → ℝ} {x : E} {w : P}
    (hG : ContDiffAt ℝ SmoothOrder G (x, w)) (n : ℕ) :
    ContDiffAt ℝ SmoothOrder
      (fun p : E × P => iteratedFDeriv ℝ n (fun z => G (z, p.2)) p.1) (x, w) := by
  induction n with
  | zero =>
    exact hG.continuousLinearMap_comp
      (continuousMultilinearCurryFin0 ℝ E ℝ).symm.toContinuousLinearEquiv.toContinuousLinearMap
  | succ n ih =>
    let D : E × P → E [×n]→L[ℝ] ℝ :=
      fun p => iteratedFDeriv ℝ n (fun z => G (z, p.2)) p.1
    have hD : ContDiffAt ℝ SmoothOrder
        (Function.uncurry (fun (p : E × P) (z : E) => D (z, p.2))) ((x, w), x) :=
      ih.comp ((x, w), x) (contDiffAt_snd.prodMk contDiffAt_fst.snd)
    have hd := hD.fderiv (m := SmoothOrder)
          (g := fun p : E × P => p.1) contDiffAt_fst (by simp)
    exact hd.continuousLinearMap_comp
      (continuousMultilinearCurryLeftEquiv ℝ
        (fun _ : Fin (n + 1) => E) ℝ).symm.toContinuousLinearEquiv.toContinuousLinearMap

end General

section Poisson

variable {E Ω : Type*}
  [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]
  [TopologicalSpace Ω] [CompactSpace Ω] [MeasurableSpace Ω] [BorelSpace Ω]
  {μ : Measure Ω} [IsFiniteMeasure μ]

omit [InnerProductSpace ℝ E] [FiniteDimensional ℝ E] [TopologicalSpace Ω]
  [CompactSpace Ω] [MeasurableSpace Ω] [BorelSpace Ω] in
/-- The algebraic three-dimensional Poisson kernel with probability-normalized
sphere measure. Its analytic regularity below is valid in any real Euclidean
space; harmonic representation by this kernel is a separate PDE hypothesis. -/
noncomputable def poissonKernel (x w : E) : ℝ :=
  (1 - ‖x‖ ^ 2) / ‖x - w‖ ^ 3

omit [FiniteDimensional ℝ E] [TopologicalSpace Ω] [CompactSpace Ω]
  [MeasurableSpace Ω] [BorelSpace Ω] in
/-- The Poisson kernel is jointly smooth away from its diagonal singularity. -/
theorem contDiffAt_poissonKernel
    {x w : E} (h : x ≠ w) :
    ContDiffAt ℝ SmoothOrder (Function.uncurry (poissonKernel (E := E))) (x, w) := by
  exact (contDiffAt_const.sub (contDiffAt_fst.norm_sq ℝ)).div
    (((contDiffAt_fst.sub contDiffAt_snd).norm ℝ (sub_ne_zero.mpr h)).pow 3)
    (pow_ne_zero 3 (norm_ne_zero_iff.mpr (sub_ne_zero.mpr h)))

omit [FiniteDimensional ℝ E] [CompactSpace Ω] [MeasurableSpace Ω] [BorelSpace Ω] in
/-- All spatial Poisson-kernel derivatives are jointly continuous for a
continuous parameterization of the unit sphere and spatial points in the unit ball. -/
theorem continuousOn_iteratedFDeriv_poissonKernel
    {b : Ω → E} (hb : Continuous b) (hnorm : ∀ ω, ‖b ω‖ = 1) (n : ℕ) :
    ContinuousOn (fun p : E × Ω =>
      iteratedFDeriv ℝ n (fun y => poissonKernel y (b p.2)) p.1)
      (ball (0 : E) 1 ×ˢ univ) := by
  intro p hp
  have hne : p.1 ≠ b p.2 := by
    intro heq
    have hlt : ‖p.1‖ < 1 := by simpa using hp.1
    rw [heq, hnorm] at hlt
    exact (lt_irrefl _ hlt)
  have hc := (contDiffAt_spatial_iteratedFDeriv
    (contDiffAt_poissonKernel hne) n).continuousAt
  exact (hc.comp (f := fun q : E × Ω => (q.1, b q.2)) (x := p)
      (continuous_fst.prodMk (hb.comp continuous_snd)).continuousAt).continuousWithinAt

omit [FiniteDimensional ℝ E] [TopologicalSpace Ω] [CompactSpace Ω]
  [MeasurableSpace Ω] [BorelSpace Ω] in
/-- Spatial smoothness of a Poisson-kernel section inside the unit ball. -/
theorem contDiffAt_poissonKernel_left {x w : E}
    (hx : x ∈ ball (0 : E) 1) (hw : ‖w‖ = 1) :
    ContDiffAt ℝ SmoothOrder (fun y => poissonKernel y w) x := by
  have hne : x ≠ w := by
    intro heq
    have hlt : ‖x‖ < 1 := by simpa using hx
    rw [heq, hw] at hlt
    exact lt_irrefl _ hlt
  exact (contDiffAt_poissonKernel hne).comp x (contDiffAt_id.prodMk contDiffAt_const)

/-- All-order differentiation of the Poisson integral on the unit ball. No
harmonicity, differentiated representation, or interior estimate is assumed. -/
theorem iteratedFDeriv_poissonIntegral
    {b : Ω → E} {g : Ω → ℝ}
    (hb : Continuous b) (hnorm : ∀ ω, ‖b ω‖ = 1) (hg : Continuous g)
    (n : ℕ) {x : E} (hx : x ∈ ball (0 : E) 1) :
    iteratedFDeriv ℝ n (fun y => ∫ ω, g ω • poissonKernel y (b ω) ∂μ) x =
      ∫ ω, g ω • iteratedFDeriv ℝ n (fun y => poissonKernel y (b ω)) x ∂μ := by
  exact iteratedFDeriv_integral_smul isOpen_ball
    (fun y hy ω => contDiffAt_poissonKernel_left hy (hnorm ω))
    (fun n => continuousOn_iteratedFDeriv_poissonKernel hb hnorm n) hg n hx

/-- Transfer all derivatives to a function satisfying only the undifferentiated
Poisson representation on the unit ball. No regularity hypothesis on `u` is needed. -/
theorem iteratedFDeriv_of_poisson_eq
    {b : Ω → E} {g : Ω → ℝ} {u : E → ℝ}
    (hb : Continuous b) (hnorm : ∀ ω, ‖b ω‖ = 1) (hg : Continuous g)
    (hrepr : EqOn u (fun y => ∫ ω, g ω • poissonKernel y (b ω) ∂μ)
      (ball (0 : E) 1))
    (n : ℕ) {x : E} (hx : x ∈ ball (0 : E) 1) :
    iteratedFDeriv ℝ n u x =
      ∫ ω, g ω • iteratedFDeriv ℝ n (fun y => poissonKernel y (b ω)) x ∂μ := by
  have heq := hrepr.eventuallyEq_of_mem (isOpen_ball.mem_nhds hx)
  exact ((heq.iteratedFDeriv ℝ n).eq_of_nhds).trans
    (iteratedFDeriv_poissonIntegral hb hnorm hg n hx)

/-- On every smaller closed ball and at every derivative order there is an
interior estimate constant independent of the continuous boundary data.
Probability normalization is the only measure normalization required. -/
theorem exists_bound_poissonIntegral [IsProbabilityMeasure μ]
    {b : Ω → E} (hb : Continuous b) (hnorm : ∀ ω, ‖b ω‖ = 1)
    (ρ : ℝ) (hρ : ρ < 1) (n : ℕ) :
    ∃ B : ℝ, 0 ≤ B ∧ ∀ (g : Ω → ℝ), Continuous g →
      ∀ M : ℝ, (∀ ω, ‖g ω‖ ≤ M) → ∀ x ∈ closedBall (0 : E) ρ,
        ‖iteratedFDeriv ℝ n (fun y => ∫ ω, g ω • poissonKernel y (b ω) ∂μ) x‖ ≤ B * M := by
  have hsub : closedBall (0 : E) ρ ⊆ ball (0 : E) 1 := closedBall_subset_ball hρ
  have hjoint := fun n => continuousOn_iteratedFDeriv_poissonKernel hb hnorm n
  obtain ⟨B, hB0, hB⟩ := exists_uniform_bound_iteratedFDeriv
    (K := fun y ω => poissonKernel y (b ω)) (isCompact_closedBall (0 : E) ρ) hsub hjoint n
  refine ⟨B, hB0, ?_⟩
  intro g hg M hM x hx
  exact norm_iteratedFDeriv_integral_smul_le
    (K := fun y ω => poissonKernel y (b ω)) isOpen_ball
    (fun y hy ω => contDiffAt_poissonKernel_left hy (hnorm ω))
    hjoint hg n (hsub hx) hB0 (hB x hx) hM

end Poisson

end PDETranslations.ParametricIntegral
