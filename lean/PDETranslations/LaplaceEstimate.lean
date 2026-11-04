module

public import PDETranslations.LaplaceLegendreConnection
public import PDETranslations.LaplaceTailBound
public import PDETranslations.Truncation
import PDETranslations.LaplaceTaylorCoefficients
import PDETranslations.LaplaceTaylorDerivatives

@[expose] public section

/-!
# The homogeneous Taylor-term estimate for the three-dimensional Laplace kernel

This proves the proposition in `tail-bound.tex`, conditional on two classical
scalar inputs: the normalized Poisson formula and `LegendreTaylorFormula`.
The latter identifies the scalar Taylor coefficients of the reciprocal square
root with the integral-defined Legendre coefficients. Their sharp bound,
kernel harmonicity, mixed-derivative identities, multinomial coefficient
identification, and shrinking-ball estimates are all proved in Lean.

The coordinate kernel below uses the Euclidean norm, even though its domain
is the coordinate model `Fin 3 → ℝ` used by `Multiindex.deriv`.
-/

open scoped BigOperators Topology
open Set

namespace PDETranslations.LaplaceEstimate

open LaplaceHarmonic LaplaceTailBound LaplaceLegendreConnection LaplaceTaylorDerivatives

abbrev Space := EuclideanSpace ℝ (Fin 3)

/-- The Laplace kernel in the coordinate model used by the paper's derivatives.
Lean's totalized division assigns zero at the pole; all estimates exclude it. -/
noncomputable def laplaceKernel (a : ℝ) (x : Fin 3 → ℝ) : ℝ :=
  a / ‖(EuclideanSpace.equiv (Fin 3) ℝ).symm x‖

/-- The unnormalized estimate, with no restriction on the displacement length. -/
theorem exists_G0_estimate
    (μ : MeasureTheory.Measure UnitSphere) [MeasureTheory.IsProbabilityMeasure μ]
    (hPoisson : PoissonFormula μ) (hLegendre : LegendreTaylorFormula) (p : ℕ) :
    ∃ C : ℝ, 0 < C ∧
      ∀ r : Multiindex.Index 3, Multiindex.magnitude r ≤ p →
        ∀ (k : ℕ) (z h : Space), z ≠ 0 →
          |∑ q ∈ Multiindex.multiindicesOfMagnitude k,
            (coordinateDeriv (q + r) G0 z / (Multiindex.factorial q : ℝ)) *
              Multiindex.vectorPower (EuclideanSpace.equiv (Fin 3) ℝ h) q| ≤
            C * (1 + (k : ℝ)) ^ p / ‖z‖ ^ (Multiindex.magnitude r + 1) *
              (‖h‖ / ‖z‖) ^ k := by
  obtain ⟨C, hC, hb⟩ := exists_uniform_homogeneous_derivative_estimate μ hPoisson p
  refine ⟨C, hC, ?_⟩
  intro r hr k z h hz
  have he : ∀ i : Fin (Multiindex.directions r).length,
      ‖coordinate ((Multiindex.directions r).get i)‖ ≤ 1 := by simp
  have hbound := hb (Multiindex.directions r).length (by simpa using hr)
    (homogeneousTaylorTerm k G0 h) k ‖h‖ (norm_nonneg h)
    (harmonicOnNhd_G0_homogeneousTaylorTerm k h)
    (fun w hw => abs_homogeneousTaylorTerm_G0_le hLegendre k h w hw)
    z hz (fun i => coordinate ((Multiindex.directions r).get i)) he
  change |coordinateDeriv r (homogeneousTaylorTerm k G0 h) z| ≤ _ at hbound
  rw [coordinateDeriv_homogeneousTaylorTerm contDiffOn_G0
    isClosed_singleton.isOpen_compl (by simpa using hz)] at hbound
  simp only [Multiindex.length_directions] at hbound
  convert hbound using 1
  rw [div_pow, show k + Multiindex.magnitude r + 1 =
    (Multiindex.magnitude r + 1) + k by omega, pow_add]
  ring

/-- **The proposition in `tail-bound.tex`.** The constant is uniform in the
multiindex, Taylor degree, center, and displacement. Only the Poisson formula
and the classical scalar Legendre coefficient identity are assumed. -/
theorem exists_laplace_estimate
    (μ : MeasureTheory.Measure UnitSphere) [MeasureTheory.IsProbabilityMeasure μ]
    (hPoisson : PoissonFormula μ) (hLegendre : LegendreTaylorFormula)
    (a : ℝ) (ha : a ≠ 0) (p : ℕ) :
    ∃ C : ℝ, 0 < C ∧
      ∀ r : Multiindex.Index 3, Multiindex.magnitude r ≤ p →
        ∀ (k : ℕ) (z h : Space), z ≠ 0 →
          |∑ q ∈ Multiindex.multiindicesOfMagnitude k,
            (Multiindex.deriv (q + r) (laplaceKernel a) z / (Multiindex.factorial q : ℝ)) *
              Multiindex.vectorPower h q| ≤
            C * (1 + (k : ℝ)) ^ p / ‖z‖ ^ (Multiindex.magnitude r + 1) *
              (‖h‖ / ‖z‖) ^ k := by
  obtain ⟨C, hC, hb⟩ := exists_G0_estimate μ hPoisson hLegendre p
  refine ⟨|a| * C, mul_pos (abs_pos.mpr ha) hC, ?_⟩
  intro r hr k z h hz
  have hderiv (q : Multiindex.Index 3) :
      Multiindex.deriv q (laplaceKernel a) z = a * coordinateDeriv q G0 z := by
    have hbridge := coordinateDeriv_const_mul_G0_eq_multiindexDeriv q a z
    simp only [EuclideanSpace.equiv, PiLp.coe_continuousLinearEquiv] at hbridge
    change coordinateDeriv q (fun y => a * G0 y) z =
      Multiindex.deriv q (laplaceKernel a) z at hbridge
    rw [← hbridge]
    exact coordinateDeriv_const_mul contDiffOn_G0 isClosed_singleton.isOpen_compl
      (by simpa using hz) q a
  simp_rw [hderiv]
  have hsum : (∑ q ∈ Multiindex.multiindicesOfMagnitude k,
      (a * coordinateDeriv (q + r) G0 z / (Multiindex.factorial q : ℝ)) *
        Multiindex.vectorPower h q) =
      a * ∑ q ∈ Multiindex.multiindicesOfMagnitude k,
        (coordinateDeriv (q + r) G0 z / (Multiindex.factorial q : ℝ)) *
          Multiindex.vectorPower h q := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro q hq
    ring
  rw [hsum, abs_mul]
  have hbound := mul_le_mul_of_nonneg_left (hb r hr k z h hz) (abs_nonneg a)
  simp only [EuclideanSpace.equiv, PiLp.coe_continuousLinearEquiv] at hbound
  convert hbound using 1
  ring

/-- The paper's derivative assumption holds with `p' = 1` and `s_p = p`. -/
theorem laplaceKernel_satisfiesDerivativeEstimate
    (μ : MeasureTheory.Measure UnitSphere) [MeasureTheory.IsProbabilityMeasure μ]
    (hPoisson : PoissonFormula μ) (hLegendre : LegendreTaylorFormula)
    (a : ℝ) (ha : a ≠ 0) :
    SatisfiesDerivativeEstimate (fun x => (laplaceKernel a x : ℂ)) 1 (fun p => p) := by
  intro p
  obtain ⟨C, hC, hb⟩ := exists_laplace_estimate μ hPoisson hLegendre a ha p
  refine ⟨C, hC, ?_⟩
  intro r hr k z h hhz
  have hz : z ≠ 0 := hhz.1
  have hpow : ‖z‖ ^ ((Multiindex.magnitude r : ℝ) + 1) =
      ‖z‖ ^ (Multiindex.magnitude r + 1) := by
    rw [show (Multiindex.magnitude r : ℝ) + 1 =
      ((Multiindex.magnitude r + 1 : ℕ) : ℝ) by push_cast; rfl,
      Real.rpow_natCast]
  have hsmooth : ContDiffAt ℝ ⊤ (laplaceKernel a) (z : Fin 3 → ℝ) := by
    simpa only [laplaceKernel, G0, Function.comp_def, mul_one_div] using!
          (contDiffAt_const (c := a)).mul
      ((contDiffOn_G0.contDiffAt (x := z) (isClosed_singleton.isOpen_compl.mem_nhds
        (by simpa using hz))).comp _
        (EuclideanSpace.equiv (Fin 3) ℝ).symm.contDiff.contDiffAt)
  have hderiv (q : Multiindex.Index 3) :
      Multiindex.deriv q (fun x => (laplaceKernel a x : ℂ)) z =
        ((Multiindex.deriv q (laplaceKernel a) z : ℝ) : ℂ) := by
    unfold Multiindex.deriv
    have heq := Complex.ofRealCLM.iteratedFDeriv_comp_left hsmooth
      (i := (Multiindex.directions q).length) (by simp)
    exact congrArg (fun A => A
      (fun i => Pi.single ((Multiindex.directions q).get i) (1 : ℝ))) heq
  simp_rw [hderiv, ← Complex.ofReal_div, ← Complex.ofReal_mul,
    ← Complex.ofReal_sum]
  simpa only [hpow, Complex.norm_real, Real.norm_eq_abs] using
    hb r (Multiindex.mem_upTo.mp hr) k z h hz

end PDETranslations.LaplaceEstimate
