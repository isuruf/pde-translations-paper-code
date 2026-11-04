module

public import PDETranslations.Biharmonic2DCoefficients
public import PDETranslations.Multiindex
public import Mathlib.Analysis.SpecialFunctions.Complex.Analytic
public import Mathlib.Analysis.SpecialFunctions.Complex.LogDeriv
public import Mathlib.Analysis.SpecialFunctions.Log.Deriv
public import Mathlib.Analysis.Complex.RealDeriv
import Mathlib.Analysis.Calculus.Deriv.ZPow
import Mathlib.Data.Nat.Choose.Cast
import Mathlib.Tactic

@[expose] public section

/-!
# Analytic ingredients for the two-dimensional biharmonic derivative bridge

The centered holomorphic potential avoids imposing a principal-log branch condition
on the base point. Its real part recovers the actual norm/log kernel, and all its
complex derivatives of order at least two are computed below.

The real multivariate Leibniz rule is proved by induction on the coordinate
word defining `Multiindex.deriv`. Normalized binomial sums then give equation (6)
in `sum_deriv_biharmonicKernel_eq_coefficient`, with no size restriction on the
displacement and no branch-cut hypothesis on the nonzero base point.
-/

namespace PDETranslations.Biharmonic2DEstimate

open scoped Topology

/-- The coordinate identification specified in equation (6). -/
def complexCoordinates (x : Fin 2 → ℝ) : ℂ :=
  (x 0 : ℂ) + Complex.I * (x 1 : ℂ)

/-- The actual two-dimensional kernel, with Lean's total logarithm convention at zero. -/
noncomputable def biharmonicKernelComplex (w : ℂ) : ℝ :=
  ‖w‖ ^ 2 * Real.log ‖w‖

/-- The actual kernel on real coordinates. -/
noncomputable def biharmonicKernel (x : Fin 2 → ℝ) : ℝ :=
  biharmonicKernelComplex (complexCoordinates x)

/-- A holomorphic potential near any nonzero base point `z`, including points on
 the branch cut of the principal logarithm. -/
noncomputable def biharmonicPotential (z w : ℂ) : ℂ :=
  w * (Complex.log (w / z) + (Real.log ‖z‖ : ℂ))

private theorem potential_log_re (z w : ℂ) (hz : z ≠ 0) (hw : w ≠ 0) :
    (Complex.log (w / z) + (Real.log ‖z‖ : ℂ)).re = Real.log ‖w‖ := by
  rw [Complex.add_re, Complex.ofReal_re, Complex.log_re, norm_div,
    Real.log_div (norm_ne_zero_iff.mpr hw) (norm_ne_zero_iff.mpr hz)]
  ring

/-- Exact real-kernel representation. There is no branch-cut hypothesis. -/
theorem biharmonicKernelComplex_eq_re_potential (z w : ℂ)
    (hz : z ≠ 0) (hw : w ≠ 0) :
    biharmonicKernelComplex w =
      (starRingEnd ℂ w * biharmonicPotential z w).re := by
  unfold biharmonicPotential biharmonicKernelComplex
  rw [← mul_assoc, Complex.conj_mul']
  simp only [← Complex.ofReal_pow, Complex.mul_re, Complex.ofReal_re,
    Complex.ofReal_im, zero_mul, sub_zero]
  rw [potential_log_re z w hz hw]

private theorem normalized_log_hasDerivAt (z w : ℂ) (hz : z ≠ 0)
    (hw : w / z ∈ Complex.slitPlane) :
    HasDerivAt (fun v : ℂ => Complex.log (v / z)) w⁻¹ w := by
  have hw0 : w ≠ 0 := by
    intro h
    simpa [h] using Complex.slitPlane_ne_zero hw
  convert (Complex.hasDerivAt_log hw).comp w ((hasDerivAt_id w).div_const z) using 1
  · rfl
  · simp [div_eq_mul_inv, mul_comm, hz]

private theorem potential_hasDerivAt (z w : ℂ) (hz : z ≠ 0)
    (hw : w / z ∈ Complex.slitPlane) :
    HasDerivAt (biharmonicPotential z)
      (Complex.log (w / z) + (Real.log ‖z‖ : ℂ) + 1) w := by
  have hw0 : w ≠ 0 := by
    intro h
    simpa [h] using Complex.slitPlane_ne_zero hw
  convert (hasDerivAt_id w).mul
    ((normalized_log_hasDerivAt z w hz hw).add_const (Real.log ‖z‖ : ℂ)) using 1
  · rfl
  · simp only [one_mul, id_eq, mul_inv_cancel₀ hw0]

private theorem normalized_slit_open (z : ℂ) :
    IsOpen {w : ℂ | w / z ∈ Complex.slitPlane} :=
  Complex.isOpen_slitPlane.preimage (continuous_id.div_const z)

private theorem potential_deriv_eventually (z w : ℂ) (hz : z ≠ 0)
    (hw : w / z ∈ Complex.slitPlane) :
    deriv (biharmonicPotential z) =ᶠ[nhds w]
      (fun v => Complex.log (v / z) + (Real.log ‖z‖ : ℂ) + 1) := by
  filter_upwards [(normalized_slit_open z).mem_nhds hw] with v hv
  exact (potential_hasDerivAt z v hz hv).deriv

private theorem potential_second_deriv (z w : ℂ) (hz : z ≠ 0)
    (hw : w / z ∈ Complex.slitPlane) :
    deriv (deriv (biharmonicPotential z)) w = w⁻¹ := by
  rw [(potential_deriv_eventually z w hz hw).deriv_eq]
  exact (((normalized_log_hasDerivAt z w hz hw).add_const
    (Real.log ‖z‖ : ℂ)).add_const 1).deriv

/-- All higher complex derivatives of the centered potential. The integer-power
form avoids truncated natural subtraction. -/
theorem iteratedDeriv_biharmonicPotential (m : ℕ) (z w : ℂ) (hz : z ≠ 0)
    (hw : w / z ∈ Complex.slitPlane) :
    iteratedDeriv (m + 2) (biharmonicPotential z) w =
      (-1 : ℂ) ^ m * (m.factorial : ℂ) * w ^ (-(m : ℤ) - 1) := by
  have heq : deriv (deriv (biharmonicPotential z)) =ᶠ[nhds w] Inv.inv := by
    filter_upwards [(normalized_slit_open z).mem_nhds hw] with v hv
    exact potential_second_deriv z v hz hv
  rw [show m + 2 = (m + 1) + 1 by omega, iteratedDeriv_succ',
    iteratedDeriv_succ', heq.iteratedDeriv_eq, iteratedDeriv_eq_iterate, iter_deriv_inv]
  congr 2
  omega

/-- The higher-derivative formula is available at every nonzero base point,
without assuming that the base point itself belongs to the principal slit plane. -/
theorem iteratedDeriv_biharmonicPotential_self (m : ℕ) (z : ℂ) (hz : z ≠ 0) :
    iteratedDeriv (m + 2) (biharmonicPotential z) z =
      (-1 : ℂ) ^ m * (m.factorial : ℂ) * z ^ (-(m : ℤ) - 1) := by
  apply iteratedDeriv_biharmonicPotential m z z hz
  simp [div_self hz]

/-- The same higher-derivative formula using only natural powers and division,
matching the shape of `biharmonicCoefficientComplex`. -/
theorem iteratedDeriv_biharmonicPotential_self_div (m : ℕ) (z : ℂ) (hz : z ≠ 0) :
    iteratedDeriv (m + 2) (biharmonicPotential z) z =
      (-1 : ℂ) ^ m * (m.factorial : ℂ) / z ^ (m + 1) := by
  rw [iteratedDeriv_biharmonicPotential_self m z hz]
  rw [show -(m : ℤ) - 1 = -((m + 1 : ℕ) : ℤ) by push_cast; ring,
    zpow_neg, zpow_natCast, div_eq_mul_inv]

/-- The centered potential really is holomorphic on a neighborhood of its base
point; this is not merely a formal formula for derivatives. -/
theorem analyticAt_biharmonicPotential (z : ℂ) (hz : z ≠ 0) :
    AnalyticAt ℂ (biharmonicPotential z) z := by
  have hs : z / z ∈ Complex.slitPlane := by simp [div_self hz]
  apply DifferentiableOn.analyticAt _ ((normalized_slit_open z).mem_nhds hs)
  intro w hw
  exact (potential_hasDerivAt z w hz hw).differentiableAt.differentiableWithinAt

/-- A polynomial/log representation convenient for proving real smoothness. -/
theorem biharmonicKernelComplex_eq_normSq_log (w : ℂ) :
    biharmonicKernelComplex w = Complex.normSq w * Real.log (Complex.normSq w) / 2 := by
  rw [Complex.normSq_eq_norm_sq, Real.log_pow]
  unfold biharmonicKernelComplex
  ring

/-- Squared Euclidean length in the specified coordinate identification. -/
theorem normSq_complexCoordinates (x : Fin 2 → ℝ) :
    Complex.normSq (complexCoordinates x) = (x 0) ^ 2 + (x 1) ^ 2 := by
  simp [complexCoordinates, Complex.normSq_apply, pow_two]

/-- The coordinate identification is injective at zero. -/
theorem complexCoordinates_eq_zero_iff (x : Fin 2 → ℝ) :
    complexCoordinates x = 0 ↔ x = 0 := by
  constructor
  · intro h
    have hr := congrArg Complex.re h
    have hi := congrArg Complex.im h
    simp [complexCoordinates] at hr hi
    funext i
    fin_cases i <;> assumption
  · rintro rfl
    simp [complexCoordinates]

/-- Infinite real smoothness of the actual kernel away from the origin. -/
theorem contDiffAt_biharmonicKernel {x : Fin 2 → ℝ} (hx : x ≠ 0) :
    ContDiffAt ℝ ⊤ biharmonicKernel x := by
  have hq : ContDiff ℝ ⊤ (fun y : Fin 2 → ℝ => (y 0) ^ 2 + (y 1) ^ 2) := by
    fun_prop
  have hq0 : (x 0) ^ 2 + (x 1) ^ 2 ≠ 0 := by
    rw [← normSq_complexCoordinates, Complex.normSq_eq_norm_sq]
    exact pow_ne_zero _ (norm_ne_zero_iff.mpr
      (mt (complexCoordinates_eq_zero_iff x).mp hx))
  have heq : biharmonicKernel = fun y : Fin 2 → ℝ =>
      ((y 0) ^ 2 + (y 1) ^ 2) * Real.log ((y 0) ^ 2 + (y 1) ^ 2) / 2 := by
    funext y
    exact (biharmonicKernelComplex_eq_normSq_log _).trans (by
      rw [normSq_complexCoordinates])
  rw [heq]
  exact (hq.contDiffAt.mul (hq.contDiffAt.log hq0)).div_const 2

/-- Smoothness in the form used by `Multiindex.deriv_add`. -/
theorem contDiffOn_biharmonicKernel :
    ContDiffOn ℝ ⊤ biharmonicKernel {x : Fin 2 → ℝ | x ≠ 0} := by
  intro x hx
  exact (contDiffAt_biharmonicKernel hx).contDiffWithinAt

/-- Every mixed derivative of the actual kernel is smooth off the origin. -/
theorem contDiffOn_deriv_biharmonicKernel (r : Multiindex.Index 2) :
    ContDiffOn ℝ ⊤ (Multiindex.deriv r biharmonicKernel)
      {x : Fin 2 → ℝ | x ≠ 0} := by
  exact Multiindex.contDiffOn_deriv contDiffOn_biharmonicKernel
    (isOpen_ne_fun continuous_id continuous_const) r

private noncomputable def coordinatesCLM : (Fin 2 → ℝ) →L[ℝ] ℂ :=
  Complex.ofRealCLM.comp (ContinuousLinearMap.proj 0) +
    Complex.I • Complex.ofRealCLM.comp (ContinuousLinearMap.proj 1)

private theorem coordinatesCLM_apply (x : Fin 2 → ℝ) :
    coordinatesCLM x = complexCoordinates x := rfl

private theorem jet_fderiv (A B : ℂ) (f g : ℂ → ℂ) (x v : Fin 2 → ℝ)
    (f' g' : ℂ) (hf : HasDerivAt f f' (complexCoordinates x))
    (hg : HasDerivAt g g' (complexCoordinates x)) :
    fderiv ℝ (fun y => (A * starRingEnd ℂ (complexCoordinates y) *
      f (complexCoordinates y) + B * g (complexCoordinates y)).re) x v =
      (A * starRingEnd ℂ (complexCoordinates v) * f (complexCoordinates x) +
        A * starRingEnd ℂ (complexCoordinates x) * (f' * complexCoordinates v) +
        B * (g' * complexCoordinates v)).re := by
  have hc : HasFDerivAt complexCoordinates coordinatesCLM x :=
    coordinatesCLM.hasFDerivAt
  have hconj := Complex.conjCLE.toContinuousLinearMap.hasFDerivAt.comp x hc
  have hff := hf.complexToReal_fderiv.comp x hc
  have hgg := hg.complexToReal_fderiv.comp x hc
  have hh := ((hconj.const_mul A).mul hff).add (hgg.const_mul B)
  have hr := Complex.reCLM.hasFDerivAt.comp x hh
  have he := congrArg (fun L : (Fin 2 → ℝ) →L[ℝ] ℝ => L v) hr.fderiv
  change fderiv ℝ (fun y => (A * starRingEnd ℂ (complexCoordinates y) *
    f (complexCoordinates y) + B * g (complexCoordinates y)).re) x v =
      (A * starRingEnd ℂ (complexCoordinates x) * (f' * complexCoordinates v) +
        f (complexCoordinates x) * (A * starRingEnd ℂ (complexCoordinates v)) +
        B * (g' * complexCoordinates v)).re at he
  rw [he]
  congr 1
  ring

private theorem differentiableAt_iterated_of_contDiff (F : ℂ → ℂ) (w : ℂ)
    (hF : ContDiffAt ℂ ⊤ F w) (n : ℕ) :
    DifferentiableAt ℂ (iteratedDeriv n F) w := by
  rw [iteratedDeriv_eq_equiv_comp]
  exact (ContinuousMultilinearMap.piFieldEquiv ℂ (Fin n) ℂ).symm.differentiableAt.comp w
    (hF.differentiableAt_iteratedFDeriv (by simp))

private noncomputable def listJet (l : List (Fin 2)) (F : ℂ → ℂ) (x : Fin 2 → ℝ) : ℝ :=
  (Complex.I ^ l.count 1 * (starRingEnd ℂ (complexCoordinates x) *
    iteratedDeriv l.length F (complexCoordinates x) +
    ((l.count 0 : ℂ) - (l.count 1 : ℂ)) *
      iteratedDeriv (l.length - 1) F (complexCoordinates x))).re

private theorem listJet_step (l : List (Fin 2)) (F : ℂ → ℂ) (x : Fin 2 → ℝ)
    (hF : ContDiffAt ℂ ⊤ F (complexCoordinates x)) (i : Fin 2) :
    fderiv ℝ (listJet l F) x (Pi.single i 1) = listJet (i :: l) F x := by
  have hd (n : ℕ) : HasDerivAt (iteratedDeriv n F)
      (iteratedDeriv (n + 1) F (complexCoordinates x)) (complexCoordinates x) := by
    simpa only [iteratedDeriv_succ] using
      (differentiableAt_iterated_of_contDiff F _ hF n).hasDerivAt
  have hj : listJet l F = fun y =>
      (Complex.I ^ l.count 1 * starRingEnd ℂ (complexCoordinates y) *
        iteratedDeriv l.length F (complexCoordinates y) +
        (Complex.I ^ l.count 1 * ((l.count 0 : ℂ) - (l.count 1 : ℂ))) *
          iteratedDeriv (l.length - 1) F (complexCoordinates y)).re := by
    funext y
    unfold listJet
    congr 1
    ring
  rw [hj, jet_fderiv _ _ _ _ _ _ _ _ (hd _) (hd _)]
  unfold listJet
  apply congrArg Complex.re
  fin_cases i <;> cases l with
  | nil => simp [complexCoordinates]; ring
  | cons j l =>
    simp only [List.length_cons, List.count_cons, Nat.add_sub_cancel,
      Nat.cast_add]
    simp [complexCoordinates, add_comm, add_left_comm]
    ring

private theorem iteratedFDeriv_eq_listJet (F : ℂ → ℂ) (g : (Fin 2 → ℝ) → ℝ)
    (s : Set (Fin 2 → ℝ)) (hs : IsOpen s) (hg : ContDiffOn ℝ ⊤ g s)
    (hF : ∀ y ∈ s, ContDiffAt ℂ ⊤ F (complexCoordinates y))
    (he : Set.EqOn g (listJet [] F) s) (l : List (Fin 2)) {x : Fin 2 → ℝ}
    (hx : x ∈ s) :
    iteratedFDeriv ℝ l.length g x (fun k => Pi.single (l.get k) (1 : ℝ)) =
      listJet l F x := by
  induction l generalizing x with
  | nil => simpa using he hx
  | cons i l ih =>
    have heq : (fun y => iteratedFDeriv ℝ l.length g y
        (fun k => Pi.single (l.get k) (1 : ℝ))) =ᶠ[nhds x] listJet l F := by
      filter_upwards [hs.mem_nhds hx] with y hy
      exact ih hy
    have hd : DifferentiableAt ℝ (iteratedFDeriv ℝ l.length g) x :=
      ((hg.contDiffAt (hs.mem_nhds hx)).iteratedFDeriv_right
        (m := ⊤) (by simp)).differentiableAt (by simp)
    change iteratedFDeriv ℝ (l.length + 1) g x
      (fun k => Pi.single ((i :: l).get k) (1 : ℝ)) = _
    rw [hd.iteratedFDeriv_succ_apply_left']
    change fderiv ℝ (fun y => iteratedFDeriv ℝ l.length g y
      (fun k => Pi.single (l.get k) (1 : ℝ))) x (Pi.single i 1) = _
    rw [heq.fderiv_eq]
    exact listJet_step l F x (hF x hx) i

private theorem directions_two_perm (r : Multiindex.Index 2) :
    (Multiindex.directions r).Perm (List.replicate (r 0) 0 ++ List.replicate (r 1) 1) := by
  have hp : (Finset.univ.toList : List (Fin 2)).Perm [0, 1] := by
    apply (List.perm_ext_iff_of_nodup (Finset.nodup_toList _) (by decide)).mpr
    intro i
    fin_cases i <;> simp
  simpa [Multiindex.directions] using hp.flatMap_right (fun i => List.replicate (r i) i)

private theorem directions_two_length (r : Multiindex.Index 2) :
    (Multiindex.directions r).length = r 0 + r 1 := by
  simpa using (directions_two_perm r).length_eq

private theorem directions_two_count (r : Multiindex.Index 2) (i : Fin 2) :
    (Multiindex.directions r).count i = r i := by
  have he := (directions_two_perm r).count_eq i
  fin_cases i <;> simpa [List.count_replicate] using he

/-- The exact mixed real derivative bridge to the centered holomorphic potential.
This includes the contribution of differentiating the conjugate coordinate factor. -/
theorem deriv_biharmonicKernel_eq_potential (r : Multiindex.Index 2)
    (x : Fin 2 → ℝ) (hx : x ≠ 0) :
    Multiindex.deriv r biharmonicKernel x =
      (Complex.I ^ r 1 * (starRingEnd ℂ (complexCoordinates x) *
        iteratedDeriv (r 0 + r 1) (biharmonicPotential (complexCoordinates x))
          (complexCoordinates x) +
        ((r 0 : ℂ) - (r 1 : ℂ)) *
          iteratedDeriv (r 0 + r 1 - 1) (biharmonicPotential (complexCoordinates x))
            (complexCoordinates x))).re := by
  let z := complexCoordinates x
  have hz : z ≠ 0 := mt (complexCoordinates_eq_zero_iff x).mp hx
  let s := {y : Fin 2 → ℝ | complexCoordinates y / z ∈ Complex.slitPlane}
  have hs : IsOpen s := (normalized_slit_open z).preimage coordinatesCLM.continuous
  have hxs : x ∈ s := by simp [s, z, div_self hz]
  have hnz (y : Fin 2 → ℝ) (hy : y ∈ s) : complexCoordinates y ≠ 0 := by
    intro he
    have := Complex.slitPlane_ne_zero hy
    simp [he] at this
  have hg : ContDiffOn ℝ ⊤ biharmonicKernel s := by
    intro y hy
    exact (contDiffAt_biharmonicKernel
      (mt (complexCoordinates_eq_zero_iff y).mpr (hnz y hy))).contDiffWithinAt
  have hF (y : Fin 2 → ℝ) (hy : y ∈ s) :
      ContDiffAt ℂ ⊤ (biharmonicPotential z) (complexCoordinates y) := by
    apply AnalyticAt.contDiffAt
    apply DifferentiableOn.analyticAt _ ((normalized_slit_open z).mem_nhds hy)
    intro w hw
    exact (potential_hasDerivAt z w hz hw).differentiableAt.differentiableWithinAt
  have he : Set.EqOn biharmonicKernel (listJet [] (biharmonicPotential z)) s := by
    intro y hy
    simpa [listJet, biharmonicKernel] using
      biharmonicKernelComplex_eq_re_potential z (complexCoordinates y) hz (hnz y hy)
  have hh := iteratedFDeriv_eq_listJet (biharmonicPotential z) biharmonicKernel s hs
    hg hF he (Multiindex.directions r) hxs
  simpa only [Multiindex.deriv, listJet, directions_two_length, directions_two_count] using hh

private theorem potential_deriv_explicit (n : ℕ) (hn : 2 ≤ n) (z : ℂ) (hz : z ≠ 0) :
    iteratedDeriv n (biharmonicPotential z) z =
      (-1 : ℂ) ^ n * ((n - 2).factorial : ℂ) / z ^ (n - 1) := by
  have he := iteratedDeriv_biharmonicPotential_self_div (n - 2) z hz
  have hn' : n - 2 + 2 = n := by omega
  have hs : (-1 : ℂ) ^ (n - 2) = (-1 : ℂ) ^ n := by
    conv_rhs => rw [← hn', pow_add]
    norm_num
  simpa only [hn', hs, show n - 2 + 1 = n - 1 by omega] using he

/-- Explicit high-order mixed derivatives of the actual norm/log kernel.
The derivative order hypothesis is exactly `|r| ≥ 3`; the formula is valid at
all nonzero points and contains no logarithm or branch-cut condition. -/
theorem deriv_biharmonicKernel_explicit (r : Multiindex.Index 2)
    (hr : 3 ≤ Multiindex.magnitude r) (x : Fin 2 → ℝ) (hx : x ≠ 0) :
    Multiindex.deriv r biharmonicKernel x =
      (Complex.I ^ r 1 *
        (starRingEnd ℂ (complexCoordinates x) *
          ((-1 : ℂ) ^ (r 0 + r 1) * ((r 0 + r 1 - 2).factorial : ℂ) /
            complexCoordinates x ^ (r 0 + r 1 - 1)) +
          ((r 0 : ℂ) - (r 1 : ℂ)) *
            ((-1 : ℂ) ^ (r 0 + r 1 - 1) * ((r 0 + r 1 - 3).factorial : ℂ) /
              complexCoordinates x ^ (r 0 + r 1 - 2)))).re := by
  have hn : 3 ≤ r 0 + r 1 := by simpa [Multiindex.magnitude, Fin.sum_univ_two] using hr
  have hz : complexCoordinates x ≠ 0 := mt (complexCoordinates_eq_zero_iff x).mp hx
  rw [deriv_biharmonicKernel_eq_potential r x hx,
    potential_deriv_explicit _ (by omega) _ hz,
    potential_deriv_explicit _ (by omega) _ hz]
  simp only [show r 0 + r 1 - 1 - 2 = r 0 + r 1 - 3 by omega,
    show r 0 + r 1 - 1 - 1 = r 0 + r 1 - 2 by omega]

private noncomputable def twoIndex (a b : ℕ) : Multiindex.Index 2 :=
  Finsupp.single 0 a + Finsupp.single 1 b

private theorem sum_layer_two {M : Type*} [AddCommMonoid M] (k : ℕ)
    (F : Multiindex.Index 2 → M) :
    ∑ q ∈ Multiindex.multiindicesOfMagnitude k, F q =
      ∑ j ∈ Finset.range (k + 1), F (twoIndex j (k - j)) := by
  classical
  apply Finset.sum_bij (fun q _ => q 0)
  · intro q hq
    have h := Multiindex.mem_multiindicesOfMagnitude.mp hq
    simp only [Multiindex.magnitude, Fin.sum_univ_two] at h
    simp only [Finset.mem_range]
    omega
  · intro q hq p hp he
    have hq' := Multiindex.mem_multiindicesOfMagnitude.mp hq
    have hp' := Multiindex.mem_multiindicesOfMagnitude.mp hp
    simp only [Multiindex.magnitude, Fin.sum_univ_two] at hq' hp'
    apply Finsupp.ext
    intro i
    fin_cases i
    · exact he
    · change q 1 = p 1
      omega
  · intro j hj
    have hj' : j ≤ k := by simpa only [Finset.mem_range, Nat.lt_succ_iff] using hj
    refine ⟨twoIndex j (k - j), ?_, ?_⟩
    · apply Multiindex.mem_multiindicesOfMagnitude.mpr
      simp [twoIndex, Multiindex.magnitude, Fin.sum_univ_two, Nat.add_sub_of_le hj']
    · simp [twoIndex]
  · intro q hq
    have h := Multiindex.mem_multiindicesOfMagnitude.mp hq
    simp only [Multiindex.magnitude, Fin.sum_univ_two] at h
    congr 1
    apply Finsupp.ext
    intro i
    fin_cases i <;> simp [twoIndex]
    omega

private theorem sum_layer_monomial (k : ℕ) (a b : ℂ) :
    ∑ q ∈ Multiindex.multiindicesOfMagnitude (d := 2) k,
      a ^ q 0 * b ^ q 1 / (Multiindex.factorial q : ℂ) =
        (a + b) ^ k / (k.factorial : ℂ) := by
  rw [sum_layer_two]
  simp only [twoIndex, Finsupp.add_apply, Finsupp.single_eq_same,
    Finsupp.single_eq_of_ne (by decide : (0 : Fin 2) ≠ 1),
    Finsupp.single_eq_of_ne (by decide : (1 : Fin 2) ≠ 0), zero_add, add_zero]
  rw [add_pow, Finset.sum_div]
  apply Finset.sum_congr rfl
  intro j hj
  have hj' : j ≤ k := by simpa only [Finset.mem_range, Nat.lt_succ_iff] using hj
  simp only [Multiindex.factorial, Fin.prod_univ_two, Finsupp.add_apply,
    Finsupp.single_eq_same, Finsupp.single_eq_of_ne (by decide : (0 : Fin 2) ≠ 1),
    Finsupp.single_eq_of_ne (by decide : (1 : Fin 2) ≠ 0), zero_add, add_zero,
    Nat.cast_mul]
  rw [Nat.cast_choose ℂ hj']
  have hkf : (k.factorial : ℂ) ≠ 0 := by exact_mod_cast k.factorial_ne_zero
  field_simp

private theorem sum_layer_weighted_monomial (k : ℕ) (hk : 1 ≤ k) (a b : ℂ) :
    ∑ q ∈ Multiindex.multiindicesOfMagnitude (d := 2) k,
      ((q 0 : ℂ) - (q 1 : ℂ)) * (a ^ q 0 * b ^ q 1 / (Multiindex.factorial q : ℂ)) =
        (a - b) * (a + b) ^ (k - 1) / ((k - 1).factorial : ℂ) := by
  classical
  let S := Multiindex.multiindicesOfMagnitude (d := 2) k
  have hd (q : Multiindex.Index 2) : HasDerivAt
      (fun t : ℂ => (a ^ q 0 * b ^ q 1 / (Multiindex.factorial q : ℂ)) *
        t ^ q 0 * (2 - t) ^ q 1)
      (((q 0 : ℂ) - (q 1 : ℂ)) *
        (a ^ q 0 * b ^ q 1 / (Multiindex.factorial q : ℂ))) 1 := by
    convert (((hasDerivAt_id (1 : ℂ)).pow (q 0)).const_mul
      (a ^ q 0 * b ^ q 1 / (Multiindex.factorial q : ℂ))).mul
        (((hasDerivAt_id (1 : ℂ)).const_sub 2).pow (q 1)) using 1
    · rfl
    · norm_num
      ring
  have hsum := HasDerivAt.fun_sum (u := S) (fun q _ => hd q)
  have he : (fun t : ℂ => ∑ q ∈ S,
      (a ^ q 0 * b ^ q 1 / (Multiindex.factorial q : ℂ)) * t ^ q 0 * (2 - t) ^ q 1) =
      fun t => (a * t + b * (2 - t)) ^ k / (k.factorial : ℂ) := by
    funext t
    rw [← sum_layer_monomial]
    apply Finset.sum_congr rfl
    intro q hq
    simp only [mul_pow]
    ring
  rw [he] at hsum
  have hright : HasDerivAt
      (fun t : ℂ => (a * t + b * (2 - t)) ^ k / (k.factorial : ℂ))
      ((a - b) * (k : ℂ) * (a + b) ^ (k - 1) / (k.factorial : ℂ)) 1 := by
    convert (((hasDerivAt_id (1 : ℂ)).const_mul a).add
      (((hasDerivAt_id (1 : ℂ)).const_sub 2).const_mul b)).pow k |>.div_const
        (k.factorial : ℂ) using 1
    · rfl
    · norm_num
      ring
  change _ = _
  rw [hsum.unique hright]
  have hf : (k.factorial : ℂ) = (k : ℂ) * ((k - 1).factorial : ℂ) := by
    conv_lhs => rw [show k = (k - 1) + 1 by omega, Nat.factorial_succ]
    rw [show k - 1 + 1 = k by omega]
    push_cast
    rfl
  rw [hf]
  have hk0 : (k : ℂ) ≠ 0 := by exact_mod_cast (show k ≠ 0 by omega)
  have hf0 : ((k - 1).factorial : ℂ) ≠ 0 := by exact_mod_cast (k - 1).factorial_ne_zero
  field_simp

private noncomputable def monomialWeight (h : Fin 2 → ℝ) (q : Multiindex.Index 2) : ℂ :=
  (Multiindex.vectorPower h q : ℂ) / (Multiindex.factorial q : ℂ)

private theorem monomialWeight_eq (h : Fin 2 → ℝ) (q : Multiindex.Index 2) :
    Complex.I ^ q 1 * monomialWeight h q =
      (h 0 : ℂ) ^ q 0 * (Complex.I * (h 1 : ℂ)) ^ q 1 / (Multiindex.factorial q : ℂ) := by
  simp only [monomialWeight, Multiindex.vectorPower, Fin.prod_univ_two,
    Complex.ofReal_mul, Complex.ofReal_pow, mul_pow]
  ring

private theorem sum_monomialWeight (k : ℕ) (h : Fin 2 → ℝ) :
    ∑ q ∈ Multiindex.multiindicesOfMagnitude k, Complex.I ^ q 1 * monomialWeight h q =
      complexCoordinates h ^ k / (k.factorial : ℂ) := by
  simp_rw [monomialWeight_eq]
  exact sum_layer_monomial k _ _

private theorem sum_weighted_monomialWeight (k : ℕ) (hk : 1 ≤ k) (h : Fin 2 → ℝ) :
    ∑ q ∈ Multiindex.multiindicesOfMagnitude k,
      ((q 0 : ℂ) - (q 1 : ℂ)) * (Complex.I ^ q 1 * monomialWeight h q) =
        starRingEnd ℂ (complexCoordinates h) * complexCoordinates h ^ (k - 1) /
          ((k - 1).factorial : ℂ) := by
  simp_rw [monomialWeight_eq]
  rw [sum_layer_weighted_monomial k hk]
  simp [complexCoordinates, sub_eq_add_neg]

private theorem real_scaled_re (c : ℂ) (a b : ℝ) :
    c.re / a * b = (c * ((b / a : ℝ) : ℂ)).re := by
  simp only [Complex.mul_re, Complex.ofReal_re, Complex.ofReal_im, mul_zero, sub_zero]
  ring

/-- The homogeneous mixed-derivative sum, before substituting the explicit
factorial formulas for the two holomorphic derivatives. -/
theorem sum_deriv_biharmonicKernel_eq_potential (k : ℕ) (hk : 1 ≤ k)
    (r : Multiindex.Index 2) (x h : Fin 2 → ℝ) (hx : x ≠ 0) :
    (∑ q ∈ Multiindex.multiindicesOfMagnitude k,
      Multiindex.deriv (q + r) biharmonicKernel x / (Multiindex.factorial q : ℝ) *
        Multiindex.vectorPower h q) =
      (Complex.I ^ r 1 *
        (starRingEnd ℂ (complexCoordinates x) *
          iteratedDeriv (k + (r 0 + r 1)) (biharmonicPotential (complexCoordinates x))
            (complexCoordinates x) *
            (complexCoordinates h ^ k / (k.factorial : ℂ)) +
          iteratedDeriv (k + (r 0 + r 1) - 1)
            (biharmonicPotential (complexCoordinates x)) (complexCoordinates x) *
            (starRingEnd ℂ (complexCoordinates h) * complexCoordinates h ^ (k - 1) /
              ((k - 1).factorial : ℂ) +
              ((r 0 : ℂ) - (r 1 : ℂ)) *
                (complexCoordinates h ^ k / (k.factorial : ℂ))))).re := by
  classical
  let S := Multiindex.multiindicesOfMagnitude (d := 2) k
  let z := complexCoordinates x
  let F := biharmonicPotential z
  let n := k + (r 0 + r 1)
  let U := starRingEnd ℂ z * iteratedDeriv n F z
  let V := iteratedDeriv (n - 1) F z
  let d : ℂ := (r 0 : ℂ) - (r 1 : ℂ)
  let W := fun q => Complex.I ^ q 1 * monomialWeight h q
  have ht (q : Multiindex.Index 2) (hq : q ∈ S) :
      Multiindex.deriv (q + r) biharmonicKernel x / (Multiindex.factorial q : ℝ) *
        Multiindex.vectorPower h q =
      (Complex.I ^ r 1 * (U * W q + V * (((q 0 : ℂ) - (q 1 : ℂ)) + d) * W q)).re := by
    have hq' := Multiindex.mem_multiindicesOfMagnitude.mp hq
    simp only [Multiindex.magnitude, Fin.sum_univ_two] at hq'
    rw [deriv_biharmonicKernel_eq_potential (q + r) x hx]
    simp only [Finsupp.add_apply]
    rw [show q 0 + r 0 + (q 1 + r 1) = n by dsimp [n]; omega, pow_add]
    rw [real_scaled_re]
    apply congrArg Complex.re
    dsimp [U, V, d, W, monomialWeight, F, z]
    push_cast
    ring
  calc
    _ = (∑ q ∈ S, Complex.I ^ r 1 *
        (U * W q + V * (((q 0 : ℂ) - (q 1 : ℂ)) + d) * W q)).re := by
      change _ = Complex.reCLM (∑ q ∈ S, _)
      rw [map_sum]
      exact Finset.sum_congr rfl ht
    _ = (Complex.I ^ r 1 * (U * (∑ q ∈ S, W q) +
        V * ((∑ q ∈ S, ((q 0 : ℂ) - (q 1 : ℂ)) * W q) + d * ∑ q ∈ S, W q))).re := by
      congr 1
      calc
        _ = ∑ q ∈ S, (Complex.I ^ r 1 * U * W q +
            Complex.I ^ r 1 * V * (((q 0 : ℂ) - (q 1 : ℂ)) * W q) +
            Complex.I ^ r 1 * V * d * W q) := by
          apply Finset.sum_congr rfl
          intro q hq
          ring
        _ = _ := by simp only [Finset.sum_add_distrib, ← Finset.mul_sum]; ring
    _ = _ := by rw [show (∑ q ∈ S, W q) = _ from sum_monomialWeight k h,
      show (∑ q ∈ S, ((q 0 : ℂ) - (q 1 : ℂ)) * W q) = _ from
        sum_weighted_monomialWeight k hk h]

/-- Equation (6): the actual homogeneous mixed-derivative sum equals the
explicit biharmonic coefficient. No restriction is imposed on the displacement. -/
theorem sum_deriv_biharmonicKernel_eq_coefficient (k : ℕ) (hk : 1 ≤ k)
    (r : Multiindex.Index 2) (hn : 3 ≤ k + Multiindex.magnitude r)
    (x h : Fin 2 → ℝ) (hx : x ≠ 0) :
    (∑ q ∈ Multiindex.multiindicesOfMagnitude k,
      Multiindex.deriv (q + r) biharmonicKernel x / (Multiindex.factorial q : ℝ) *
        Multiindex.vectorPower h q) =
      biharmonicCoefficient k (r 0) (r 1) (complexCoordinates x) (complexCoordinates h) := by
  have hn' : 3 ≤ k + (r 0 + r 1) := by
    simpa [Multiindex.magnitude, Fin.sum_univ_two] using hn
  have hz : complexCoordinates x ≠ 0 := mt (complexCoordinates_eq_zero_iff x).mp hx
  rw [sum_deriv_biharmonicKernel_eq_potential k hk r x h hx,
    potential_deriv_explicit _ (by omega) _ hz,
    potential_deriv_explicit _ (by omega) _ hz]
  simp only [show k + (r 0 + r 1) - 1 - 2 = k + (r 0 + r 1) - 3 by omega,
    show k + (r 0 + r 1) - 1 - 1 = k + (r 0 + r 1) - 2 by omega]
  unfold biharmonicCoefficient biharmonicCoefficientComplex
  dsimp only
  apply congrArg Complex.re
  push_cast
  ring

/-- Inverse to `complexCoordinates`. -/
def realCoordinates (z : ℂ) : Fin 2 → ℝ := ![z.re, z.im]

@[simp] theorem complexCoordinates_realCoordinates (z : ℂ) :
    complexCoordinates (realCoordinates z) = z := by
  apply Complex.ext <;> simp [complexCoordinates, realCoordinates]

/-- Equation (6) with arbitrary complex base point and displacement, using the
specified real/complex coordinate identification. The sole pointwise hypothesis
is `z ≠ 0`; in particular, `h = 0` is permitted. -/
theorem sum_deriv_biharmonicKernel_complex (k : ℕ) (hk : 1 ≤ k)
    (r : Multiindex.Index 2) (hn : 3 ≤ k + Multiindex.magnitude r)
    (z h : ℂ) (hz : z ≠ 0) :
    (∑ q ∈ Multiindex.multiindicesOfMagnitude k,
      Multiindex.deriv (q + r) biharmonicKernel (realCoordinates z) /
        (Multiindex.factorial q : ℝ) * Multiindex.vectorPower (realCoordinates h) q) =
      biharmonicCoefficient k (r 0) (r 1) z h := by
  have hx : realCoordinates z ≠ 0 := by
    intro he
    apply hz
    have hh := congrArg complexCoordinates he
    rw [complexCoordinates_realCoordinates] at hh
    simpa [complexCoordinates] using hh
  simpa only [complexCoordinates_realCoordinates] using
    sum_deriv_biharmonicKernel_eq_coefficient k hk r hn (realCoordinates z) (realCoordinates h) hx

end PDETranslations.Biharmonic2DEstimate
