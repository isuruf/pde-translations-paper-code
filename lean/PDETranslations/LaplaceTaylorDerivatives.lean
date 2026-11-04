module

public import PDETranslations.LaplaceHarmonic
import PDETranslations.LaplaceTaylorCoefficients

@[expose] public section

/-!
# Center derivatives of homogeneous Taylor terms

Mixed coordinate derivatives are transported exactly to `Multiindex.deriv`.
Only smoothness on an open set is required; no derivative estimate or
harmonicity hypothesis is used. The displacement in a Taylor term stays fixed.
-/

open Set
open scoped BigOperators ContDiff

namespace PDETranslations.LaplaceTaylorDerivatives

open LaplaceHarmonic

/-- Mixed Euclidean coordinate derivatives compose by multiindex addition. -/
theorem coordinateDeriv_add {s : Set Space} {f : Space → ℝ}
    (hf : ContDiffOn ℝ ⊤ f s) (hs : IsOpen s) {z : Space} (hz : z ∈ s)
    (q r : Multiindex.Index 3) :
    coordinateDeriv (q + r) f z = coordinateDeriv q (coordinateDeriv r f) z := by
  let e := EuclideanSpace.equiv (Fin 3) ℝ
  have hf' : ContDiffOn ℝ ⊤ (fun y => f (e.symm y)) (e.symm ⁻¹' s) :=
    hf.comp e.symm.contDiff.contDiffOn (fun _ hy => hy)
  have hs' : IsOpen (e.symm ⁻¹' s) := hs.preimage e.symm.continuous
  have hz' : e z ∈ e.symm ⁻¹' s := by simpa using hz
  have he : (fun y => coordinateDeriv r f (e.symm y)) =
      Multiindex.deriv r (fun y => f (e.symm y)) := by
    funext y
    have hh := coordinateDeriv_eq_multiindexDeriv r f (e.symm y)
    change coordinateDeriv r f (e.symm y) =
      Multiindex.deriv r (fun y => f (e.symm y)) (e (e.symm y)) at hh
    simpa only [e.apply_symm_apply] using hh
  rw [coordinateDeriv_eq_multiindexDeriv, coordinateDeriv_eq_multiindexDeriv]
  change Multiindex.deriv (q + r) (fun y => f (e.symm y)) (e z) =
    Multiindex.deriv q (fun y => coordinateDeriv r f (e.symm y)) (e z)
  rw [he]
  exact Multiindex.deriv_add hf' hs' hz' q r

/-- A finite sum may be differentiated in any fixed coordinate multiindex. -/
theorem coordinateDeriv_sum {ι : Type*} (t : Finset ι) (g : ι → Space → ℝ)
    {s : Set Space} (hg : ∀ i ∈ t, ContDiffOn ℝ ⊤ (g i) s)
    (hs : IsOpen s) {z : Space} (hz : z ∈ s) (r : Multiindex.Index 3) :
    coordinateDeriv r (fun x => ∑ i ∈ t, g i x) z =
      ∑ i ∈ t, coordinateDeriv r (g i) z := by
  classical
  unfold coordinateDeriv
  rw [iteratedFDeriv_fun_sum_apply (fun i hi =>
    ((hg i hi).contDiffAt (hs.mem_nhds hz)).of_le (by simp))]
  simp

/-- Constant real factors commute with coordinate differentiation. -/
theorem coordinateDeriv_const_mul {s : Set Space} {f : Space → ℝ}
    (hf : ContDiffOn ℝ ⊤ f s) (hs : IsOpen s) {z : Space} (hz : z ∈ s)
    (r : Multiindex.Index 3) (a : ℝ) :
    coordinateDeriv r (fun x => a * f x) z = a * coordinateDeriv r f z := by
  unfold coordinateDeriv
  have hf' : ContDiffAt ℝ (Multiindex.directions r).length f z :=
    (hf.contDiffAt (hs.mem_nhds hz)).of_le (by simp)
  simpa only [smul_eq_mul, smul_apply] using
    congrArg (fun A => A (fun i => coordinate ((Multiindex.directions r).get i)))
      (iteratedFDeriv_const_smul_apply' (a := a) hf')

/-- Differentiate the center of a Taylor term, keeping its displacement fixed. -/
theorem coordinateDeriv_homogeneousTaylorTerm {s : Set Space} {f : Space → ℝ}
    (hf : ContDiffOn ℝ ⊤ f s) (hs : IsOpen s) {z : Space} (hz : z ∈ s)
    (r : Multiindex.Index 3) (k : ℕ) (h : Space) :
    coordinateDeriv r (homogeneousTaylorTerm k f h) z =
      ∑ q ∈ Multiindex.multiindicesOfMagnitude k,
        (coordinateDeriv (q + r) f z / (Multiindex.factorial q : ℝ)) *
          Multiindex.vectorPower (EuclideanSpace.equiv (Fin 3) ℝ h) q := by
  classical
  unfold homogeneousTaylorTerm
  simp only [smul_eq_mul]
  rw [coordinateDeriv_sum _ _ (fun q _ => by
    simpa only [smul_eq_mul] using
      (contDiffOn_coordinateDeriv hf hs q).const_smul
        (Multiindex.vectorPower (EuclideanSpace.equiv (Fin 3) ℝ h) q /
          (Multiindex.factorial q : ℝ))) hs hz r]
  apply Finset.sum_congr rfl
  intro q hq
  rw [coordinateDeriv_const_mul (contDiffOn_coordinateDeriv hf hs q) hs hz,
    ← coordinateDeriv_add hf hs hz r q, add_comm r q]
  ring

/-- Constant factors commute with Taylor terms of any smooth function. -/
theorem homogeneousTaylorTerm_const_mul {s : Set Space} {f : Space → ℝ}
    (hf : ContDiffOn ℝ ⊤ f s) (hs : IsOpen s) {x : Space} (hx : x ∈ s)
    (a : ℝ) (k : ℕ) (h : Space) :
    homogeneousTaylorTerm k (fun y => a * f y) h x =
      a * homogeneousTaylorTerm k f h x := by
  classical
  unfold homogeneousTaylorTerm
  simp only [smul_eq_mul, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro q hq
  rw [coordinateDeriv_const_mul hf hs hx]
  ring

/-- Taylor terms of a scaled Laplace kernel, away from its pole. -/
theorem homogeneousTaylorTerm_const_mul_G0 (a : ℝ) (k : ℕ) (h : Space)
    {x : Space} (hx : x ≠ 0) :
    homogeneousTaylorTerm k (fun y => a * G0 y) h x =
      a * homogeneousTaylorTerm k G0 h x :=
  homogeneousTaylorTerm_const_mul contDiffOn_G0 isClosed_singleton.isOpen_compl
    (by simpa using hx) a k h

/-- Exact sup-norm-domain bridge for the scaled Laplace kernel.
This transport identity also holds at the pole and needs no smoothness. -/
theorem coordinateDeriv_const_mul_G0_eq_multiindexDeriv
    (r : Multiindex.Index 3) (a : ℝ) (x : Space) :
    coordinateDeriv r (fun y => a * G0 y) x =
      Multiindex.deriv r
        (fun y => a / ‖(EuclideanSpace.equiv (Fin 3) ℝ).symm y‖)
        (EuclideanSpace.equiv (Fin 3) ℝ x) := by
  simpa only [G0, mul_one_div] using
    coordinateDeriv_eq_multiindexDeriv r (fun y => a * G0 y) x

end PDETranslations.LaplaceTaylorDerivatives
