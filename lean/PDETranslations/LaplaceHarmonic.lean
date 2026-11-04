module

public import Mathlib.Analysis.InnerProductSpace.Harmonic.Basic
public import PDETranslations.Multiindex
import Mathlib.Analysis.Calculus.FDeriv.Symmetric
import Mathlib.Analysis.InnerProductSpace.Calculus
import Mathlib.Analysis.SpecialFunctions.Pow.Deriv

@[expose] public section

/-!
# Harmonicity of evaluated derivatives and homogeneous Taylor terms

The derivative directions live in the inner product space itself, not in the
sup-norm coordinate model. Smoothness is explicit: mathlib's `HarmonicOnNhd`
only supplies two derivatives, not the elliptic regularity theorem.
-/

open Set InnerProductSpace Laplacian
open scoped Topology BigOperators

namespace PDETranslations.LaplaceHarmonic

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Successive derivatives in a fixed list of directions. -/
noncomputable def directionalList : List E → (E → ℝ) → E → ℝ
  | [], f => f
  | v :: l, f => fun x => fderiv ℝ (directionalList l f) x v

theorem contDiffOn_directionalList {s : Set E} {f : E → ℝ}
    (hf : ContDiffOn ℝ ⊤ f s) (hs : IsOpen s) (l : List E) :
    ContDiffOn ℝ ⊤ (directionalList l f) s := by
  induction l with
  | nil => exact hf
  | cons v l ih =>
    exact (ih.fderiv_of_isOpen hs (by simp)).clm_apply contDiffOn_const

private theorem directionalList_congr {s : Set E} (hs : IsOpen s)
    {f g : E → ℝ} (hfg : EqOn f g s) (l : List E) :
    EqOn (directionalList l f) (directionalList l g) s := by
  induction l with
  | nil => exact hfg
  | cons v l ih =>
    intro x hx
    exact congrArg (fun A => A v)
      (Filter.EventuallyEq.fderiv_eq (𝕜 := ℝ)
        (ih.eventuallyEq_of_mem (hs.mem_nhds hx)))

theorem directionalList_eq_iteratedFDeriv {s : Set E} {f : E → ℝ}
    (hf : ContDiffOn ℝ ⊤ f s) (hs : IsOpen s) (l : List E)
    {x : E} (hx : x ∈ s) :
    directionalList l f x = iteratedFDeriv ℝ l.length f x (fun i => l.get i) := by
  induction l generalizing x with
  | nil => simp [directionalList, iteratedFDeriv_zero_apply]
  | cons v l ih =>
    have heq : directionalList l f =ᶠ[nhds x]
        (fun y => iteratedFDeriv ℝ l.length f y (fun i => l.get i)) :=
      Filter.eventually_of_mem (hs.mem_nhds hx) (fun y hy => ih hy)
    have hd : DifferentiableAt ℝ (iteratedFDeriv ℝ l.length f) x :=
      ((hf.contDiffAt (hs.mem_nhds hx)).iteratedFDeriv_right
        (m := ⊤) (by simp)).differentiableAt (by simp)
    change directionalList (v :: l) f x = iteratedFDeriv ℝ (l.length + 1) f x _
    rw [hd.iteratedFDeriv_succ_apply_left']
    change fderiv ℝ (directionalList l f) x v =
      fderiv ℝ (fun y => iteratedFDeriv ℝ l.length f y (fun i => l.get i)) x v
    exact congrArg (fun A => A v) (heq.fderiv_eq (𝕜 := ℝ))

private theorem directionalList_append (l m : List E) (f : E → ℝ) :
    directionalList (l ++ m) f = directionalList l (directionalList m f) := by
  induction l with
  | nil => rfl
  | cons v l ih => simp only [List.cons_append, directionalList, ih]

private theorem directionalList_perm {s : Set E} {f : E → ℝ}
    (hf : ContDiffOn ℝ ⊤ f s) (hs : IsOpen s) {l m : List E} (h : l.Perm m) :
    EqOn (directionalList l f) (directionalList m f) s := by
  induction h with
  | nil => intro x hx; rfl
  | cons v h ih => exact directionalList_congr hs ih [v]
  | swap v w l =>
    intro x hx
    have hg := (contDiffOn_directionalList hf hs l).contDiffAt (hs.mem_nhds hx)
    have hd := (hg.fderiv_right (m := ⊤) (by simp)).differentiableAt (by simp)
    change fderiv ℝ (fun y => fderiv ℝ (directionalList l f) y v) x w =
      fderiv ℝ (fun y => fderiv ℝ (directionalList l f) y w) x v
    simpa [fderiv_clm_apply hd (differentiableAt_const _)] using
      (hg.isSymmSndFDerivAt (by simp)).eq w v
  | trans h₁ h₂ ih₁ ih₂ => exact ih₁.trans ih₂

private theorem directionalList_sum {ι : Type*} (t : Finset ι)
    {g : ι → E → ℝ} {s : Set E} (hs : IsOpen s)
    (hg : ∀ i ∈ t, ContDiffOn ℝ ⊤ (g i) s) (l : List E) :
    EqOn (directionalList l (fun x => ∑ i ∈ t, g i x))
      (fun x => ∑ i ∈ t, directionalList l (g i) x) s := by
  induction l with
  | nil => intro x hx; rfl
  | cons v l ih =>
    intro x hx
    have heq := ih.eventuallyEq_of_mem (hs.mem_nhds hx)
    change fderiv ℝ (directionalList l (fun x => ∑ i ∈ t, g i x)) x v = _
    rw [heq.fderiv_eq]
    rw [fderiv_fun_sum (fun i hi =>
      ((contDiffOn_directionalList (hg i hi) hs l).contDiffAt
        (hs.mem_nhds hx)).differentiableAt (by simp))]
    simp [directionalList]

variable [FiniteDimensional ℝ E]

private theorem laplacian_eq_directionalList (f : E → ℝ) {s : Set E}
    (hf : ContDiffOn ℝ ⊤ f s) (hs : IsOpen s) {x : E} (hx : x ∈ s) :
    laplacian f x = ∑ i, directionalList
      [stdOrthonormalBasis ℝ E i, stdOrthonormalBasis ℝ E i] f x := by
  rw [laplacian_eq_iteratedFDeriv_stdOrthonormalBasis]
  apply Finset.sum_congr rfl
  intro i hi
  rw [directionalList_eq_iteratedFDeriv hf hs _ hx]
  congr 1
  ext j
  fin_cases j <;> rfl

/-- Fixed evaluated iterated derivatives commute with the Laplacian on an open
set. The list formulation avoids any choice of coordinates. -/
theorem laplacian_directionalList {s : Set E} {f : E → ℝ}
    (hf : ContDiffOn ℝ ⊤ f s) (hs : IsOpen s) (l : List E) :
    EqOn (laplacian (directionalList l f)) (directionalList l (laplacian f)) s := by
  let b := stdOrthonormalBasis ℝ E
  have hsum : EqOn (laplacian f) (fun x => ∑ i, directionalList [b i, b i] f x) s :=
    fun x hx => laplacian_eq_directionalList f hf hs hx
  intro x hx
  rw [laplacian_eq_directionalList _ (contDiffOn_directionalList hf hs l) hs hx]
  rw [directionalList_congr hs hsum l hx]
  rw [directionalList_sum Finset.univ hs
    (fun i _ => contDiffOn_directionalList hf hs [b i, b i]) l hx]
  apply Finset.sum_congr rfl
  intro i hi
  rw [← directionalList_append, ← directionalList_append]
  exact directionalList_perm hf hs (List.perm_append_comm) hx

/-- Harmonicity is preserved by any fixed list of directional derivatives,
provided the original function is smooth on the open domain. -/
theorem harmonicOnNhd_directionalList {s : Set E} {f : E → ℝ}
    (hf : ContDiffOn ℝ ⊤ f s) (hs : IsOpen s) (hh : HarmonicOnNhd f s)
    (l : List E) : HarmonicOnNhd (directionalList l f) s := by
  have hzero : EqOn (laplacian f) (fun _ => 0) s :=
    fun x hx => (hh x hx).2.self_of_nhds
  have hlzero : EqOn (directionalList l (fun _ : E => (0 : ℝ))) (fun _ => 0) s := by
    induction l with
    | nil => intro x hx; rfl
    | cons v l ih =>
      intro x hx
      change fderiv ℝ (directionalList l (fun _ : E => (0 : ℝ))) x v = 0
      rw [(ih.eventuallyEq_of_mem (hs.mem_nhds hx)).fderiv_eq]
      simp
  intro x hx
  refine ⟨((contDiffOn_directionalList hf hs l).contDiffAt
    (hs.mem_nhds hx)).of_le (by simp), ?_⟩
  apply Filter.eventually_of_mem (hs.mem_nhds hx)
  intro y hy
  exact (laplacian_directionalList hf hs l hy).trans
    ((directionalList_congr hs hzero l hy).trans (hlzero hy))

omit [FiniteDimensional ℝ E] in
private theorem directionalList_ofFn {s : Set E} {f : E → ℝ}
    (hf : ContDiffOn ℝ ⊤ f s) (hs : IsOpen s) (m : ℕ) (v : Fin m → E)
    {x : E} (hx : x ∈ s) :
    directionalList (List.ofFn v) f x = iteratedFDeriv ℝ m f x v := by
  have aux (l : List E) (n : ℕ) (hn : l.length = n) (w : Fin n → E)
      (hw : ∀ i, l.get i = w (Fin.cast hn i)) :
      directionalList l f x = iteratedFDeriv ℝ n f x w := by
    subst n
    rw [directionalList_eq_iteratedFDeriv hf hs l hx]
    congr 1
    funext i
    exact hw i
  exact aux (List.ofFn v) m (List.length_ofFn) v (fun i => List.get_ofFn v i)

omit [FiniteDimensional ℝ E] in
/-- Any fixed evaluation of an iterated Fréchet derivative remains smooth. -/
theorem contDiffOn_evaluated_iteratedFDeriv {s : Set E} {f : E → ℝ}
    (hf : ContDiffOn ℝ ⊤ f s) (hs : IsOpen s) (m : ℕ) (v : Fin m → E) :
    ContDiffOn ℝ ⊤ (fun x => iteratedFDeriv ℝ m f x v) s := by
  have heq : EqOn (directionalList (List.ofFn v) f)
      (fun x => iteratedFDeriv ℝ m f x v) s := by
    intro x hx
    exact directionalList_ofFn hf hs m v hx
  exact (contDiffOn_directionalList hf hs (List.ofFn v)).congr
    (fun x hx => (heq hx).symm)

/-- Harmonicity of arbitrary fixed evaluated iterated derivatives. -/
theorem harmonicOnNhd_evaluated_iteratedFDeriv {s : Set E} {f : E → ℝ}
    (hf : ContDiffOn ℝ ⊤ f s) (hs : IsOpen s) (hh : HarmonicOnNhd f s)
    (m : ℕ) (v : Fin m → E) :
    HarmonicOnNhd (fun x => iteratedFDeriv ℝ m f x v) s := by
  have heq : EqOn (directionalList (List.ofFn v) f)
      (fun x => iteratedFDeriv ℝ m f x v) s := by
    intro x hx
    exact directionalList_ofFn hf hs m v hx
  intro x hx
  exact (harmonicAt_congr_nhds
    (heq.eventuallyEq_of_mem (hs.mem_nhds hx))).mp
    (harmonicOnNhd_directionalList hf hs hh _ x hx)

/-- Finite linear combinations of harmonic functions are harmonic. -/
theorem harmonicOnNhd_linearCombination {ι : Type*} (t : Finset ι)
    (c : ι → ℝ) (g : ι → E → ℝ) {s : Set E}
    (hg : ∀ i ∈ t, HarmonicOnNhd (g i) s) :
    HarmonicOnNhd (fun x => ∑ i ∈ t, c i • g i x) s := by
  classical
  induction t using Finset.induction_on with
  | empty => simp
  | @insert i t hi ih =>
    simp only [Finset.sum_insert hi]
    exact (hg i (Finset.mem_insert_self i t)).const_smul.add
      (ih (fun j hj => hg j (Finset.mem_insert_of_mem hj)))

abbrev Space := EuclideanSpace ℝ (Fin 3)

/-- The Euclidean coordinate unit vector. -/
noncomputable def coordinate (i : Fin 3) : Space :=
  (EuclideanSpace.equiv (Fin 3) ℝ).symm (Pi.single i 1)

@[simp] theorem norm_coordinate (i : Fin 3) : ‖coordinate i‖ = 1 := by
  change ‖(PiLp.single 2 i (1 : ℝ) : Space)‖ = 1
  simp

/-- Multi-index differentiation in Euclidean unit directions. Unlike
`Multiindex.deriv`, this definition uses the Euclidean norm on the domain. -/
noncomputable def coordinateDeriv (α : Multiindex.Index 3) (f : Space → ℝ)
    (x : Space) : ℝ :=
  iteratedFDeriv ℝ (Multiindex.directions α).length f x
    (fun i => coordinate ((Multiindex.directions α).get i))

/-- Exact bridge to the existing sup-norm coordinate derivative. The
continuous linear equivalence is enough; no isometry between the norms is
asserted. This identity even holds without a smoothness hypothesis. -/
theorem coordinateDeriv_eq_multiindexDeriv (α : Multiindex.Index 3)
    (f : Space → ℝ) (x : Space) :
    coordinateDeriv α f x =
      Multiindex.deriv α (fun y => f ((EuclideanSpace.equiv (Fin 3) ℝ).symm y))
        (EuclideanSpace.equiv (Fin 3) ℝ x) := by
  let g := (EuclideanSpace.equiv (Fin 3) ℝ).symm
  have h := g.iteratedFDerivWithin_comp_right f uniqueDiffOn_univ
    (x := EuclideanSpace.equiv (Fin 3) ℝ x) (mem_univ _)
    (Multiindex.directions α).length
  simp only [preimage_univ, iteratedFDerivWithin_univ] at h
  have he := congrArg (fun A => A
    (fun i => Pi.single ((Multiindex.directions α).get i) (1 : ℝ))) h
  simpa [coordinateDeriv, Multiindex.deriv, coordinate, g, Function.comp_def,
    ContinuousMultilinearMap.compContinuousLinearMap_apply] using he.symm

theorem contDiffOn_coordinateDeriv {s : Set Space} {f : Space → ℝ}
    (hf : ContDiffOn ℝ ⊤ f s) (hs : IsOpen s) (α : Multiindex.Index 3) :
    ContDiffOn ℝ ⊤ (coordinateDeriv α f) s :=
  contDiffOn_evaluated_iteratedFDeriv hf hs _ _

theorem harmonicOnNhd_coordinateDeriv {s : Set Space} {f : Space → ℝ}
    (hf : ContDiffOn ℝ ⊤ f s) (hs : IsOpen s) (hh : HarmonicOnNhd f s)
    (α : Multiindex.Index 3) : HarmonicOnNhd (coordinateDeriv α f) s :=
  harmonicOnNhd_evaluated_iteratedFDeriv hf hs hh _ _

/-- The degree-`k` Taylor term, as a function of the expansion center `x`,
with the displacement `h` held fixed. -/
noncomputable def homogeneousTaylorTerm (k : ℕ) (f : Space → ℝ) (h : Space)
    (x : Space) : ℝ :=
  ∑ α ∈ Multiindex.multiindicesOfMagnitude k,
    (Multiindex.vectorPower (EuclideanSpace.equiv (Fin 3) ℝ h) α /
      (Multiindex.factorial α : ℝ)) • coordinateDeriv α f x

theorem harmonicOnNhd_homogeneousTaylorTerm {s : Set Space} {f : Space → ℝ}
    (hf : ContDiffOn ℝ ⊤ f s) (hs : IsOpen s) (hh : HarmonicOnNhd f s)
    (k : ℕ) (h : Space) : HarmonicOnNhd (homogeneousTaylorTerm k f h) s :=
  harmonicOnNhd_linearCombination _ _ _
    (fun α _ => harmonicOnNhd_coordinateDeriv hf hs hh α)

theorem contDiffOn_homogeneousTaylorTerm {s : Set Space} {f : Space → ℝ}
    (hf : ContDiffOn ℝ ⊤ f s) (hs : IsOpen s) (k : ℕ) (h : Space) :
    ContDiffOn ℝ ⊤ (homogeneousTaylorTerm k f h) s := by
  apply ContDiffOn.sum
  intro α hα
  exact (contDiffOn_coordinateDeriv hf hs α).const_smul _

/-- The unnormalized three-dimensional Laplace kernel. -/
noncomputable def G0 (x : Space) : ℝ := 1 / ‖x‖

theorem contDiffOn_G0 : ContDiffOn ℝ ⊤ G0 ({0}ᶜ : Set Space) := by
  apply contDiffOn_const.div (contDiffOn_id.norm ℝ ?_) ?_
  · intro x hx
    simpa using hx
  · intro x hx
    exact norm_ne_zero_iff.mpr (by simpa using hx)

open scoped RealInnerProductSpace

omit [FiniteDimensional ℝ E] in
private theorem radial_fderiv (p : ℝ) {x : E} (hx : x ≠ 0) (v : E) :
    fderiv ℝ (fun y : E => (‖y‖ ^ 2) ^ p) x v =
      2 * p * (‖x‖ ^ 2) ^ (p - 1) * ⟪x, v⟫ := by
  have hd := (hasStrictFDerivAt_norm_sq x).hasFDerivAt.rpow_const
    (p := p) (Or.inl (pow_ne_zero 2 (norm_ne_zero_iff.mpr hx)))
  rw [hd.fderiv]
  simp only [smul_apply, innerSL_apply_apply, smul_eq_mul]
  ring

omit [FiniteDimensional ℝ E] in
private theorem radial_second (p : ℝ) {x : E} (hx : x ≠ 0) (v : E) :
    directionalList [v, v] (fun y : E => (‖y‖ ^ 2) ^ p) x =
      4 * p * (p - 1) * (‖x‖ ^ 2) ^ (p - 2) * ⟪x, v⟫ ^ 2 +
        2 * p * (‖x‖ ^ 2) ^ (p - 1) * ⟪v, v⟫ := by
  have heq : (fun y : E => fderiv ℝ (fun z : E => (‖z‖ ^ 2) ^ p) y v) =ᶠ[𝓝 x]
      (fun y => 2 * p * (‖y‖ ^ 2) ^ (p - 1) * ⟪y, v⟫) := by
    filter_upwards [isOpen_ne.mem_nhds hx] with y hy
    exact radial_fderiv p hy v
  have hr := (hasStrictFDerivAt_norm_sq x).hasFDerivAt.rpow_const
    (p := p - 1) (Or.inl (pow_ne_zero 2 (norm_ne_zero_iff.mpr hx)))
  have hi : HasFDerivAt (fun y : E => ⟪y, v⟫) (innerSL ℝ v) x := by
    have he : (fun y : E => ⟪y, v⟫) = (innerSL ℝ v) :=
      funext (fun y => (real_inner_comm y v).symm)
    rw [he]
    exact (innerSL ℝ v).hasFDerivAt
  have hd := (hr.const_mul (2 * p)).fun_mul hi
  change fderiv ℝ (fun y : E => fderiv ℝ (fun z : E => (‖z‖ ^ 2) ^ p) y v) x v = _
  rw [heq.fderiv_eq, hd.fderiv]
  simp only [add_apply, smul_apply,
    innerSL_apply_apply, smul_eq_mul]
  rw [real_inner_comm v x]
  have hp : p - 1 - 1 = p - 2 := by ring
  rw [hp]
  ring

private theorem radial_laplacian (p : ℝ) {x : E} (hx : x ≠ 0) :
    laplacian (fun y : E => (‖y‖ ^ 2) ^ p) x =
      4 * p * (p - 1) * (‖x‖ ^ 2) ^ (p - 2) * ‖x‖ ^ 2 +
        2 * p * (‖x‖ ^ 2) ^ (p - 1) * (Module.finrank ℝ E : ℝ) := by
  have hs : IsOpen ({0}ᶜ : Set E) := isClosed_singleton.isOpen_compl
  have hf : ContDiffOn ℝ ⊤ (fun y : E => (‖y‖ ^ 2) ^ p) ({0}ᶜ : Set E) := by
    exact (contDiffOn_id.norm_sq ℝ).rpow_const_of_ne
      (fun y hy => pow_ne_zero 2 (norm_ne_zero_iff.mpr (by simpa using hy)))
  rw [laplacian_eq_directionalList _ hf hs (by simpa using hx)]
  simp_rw [radial_second p hx, Finset.sum_add_distrib]
  let b := stdOrthonormalBasis ℝ E
  have hb : ∀ i, ⟪b i, b i⟫ = 1 := fun i => b.inner_eq_one i
  change (∑ i, 4 * p * (p - 1) * (‖x‖ ^ 2) ^ (p - 2) * ⟪x, b i⟫ ^ 2) +
    (∑ i, 2 * p * (‖x‖ ^ 2) ^ (p - 1) * ⟪b i, b i⟫) = _
  simp_rw [hb]
  rw [← Finset.mul_sum, b.sum_sq_inner_left]
  simp [mul_comm]

private theorem G0_eq_radial : G0 = fun x : Space => (‖x‖ ^ 2) ^ (- (1 / 2 : ℝ)) := by
  funext x
  have h := Real.rpow_mul (norm_nonneg x) 2 (- (1 / 2 : ℝ))
  norm_num at h
  simpa [G0, Real.rpow_neg_one, one_div] using h

/-- The actual three-dimensional Laplace kernel is harmonic away from its pole. -/
theorem harmonicOnNhd_G0 : HarmonicOnNhd G0 ({0}ᶜ : Set Space) := by
  have hs : IsOpen ({0}ᶜ : Set Space) := isClosed_singleton.isOpen_compl
  intro x hx
  refine ⟨(contDiffOn_G0.contDiffAt (hs.mem_nhds hx)).of_le (by simp), ?_⟩
  apply Filter.eventually_of_mem (hs.mem_nhds hx)
  intro y hy
  have hny : y ≠ 0 := by simpa using hy
  rw [G0_eq_radial, radial_laplacian _ hny]
  have hn : 0 < ‖y‖ ^ 2 := pow_pos (norm_pos_iff.mpr hny) 2
  have hp := Real.rpow_sub_one hn.ne' (- (1 / 2 : ℝ) - 1)
  norm_num at hp ⊢
  rw [hp]
  field_simp
  ring

theorem harmonicOnNhd_G0_evaluated_iteratedFDeriv (m : ℕ) (v : Fin m → Space) :
    HarmonicOnNhd (fun x => iteratedFDeriv ℝ m G0 x v) ({0}ᶜ : Set Space) :=
  harmonicOnNhd_evaluated_iteratedFDeriv contDiffOn_G0
    isClosed_singleton.isOpen_compl harmonicOnNhd_G0 m v

theorem harmonicOnNhd_G0_coordinateDeriv (α : Multiindex.Index 3) :
    HarmonicOnNhd (coordinateDeriv α G0) ({0}ᶜ : Set Space) :=
  harmonicOnNhd_coordinateDeriv contDiffOn_G0 isClosed_singleton.isOpen_compl
    harmonicOnNhd_G0 α

theorem contDiffOn_G0_homogeneousTaylorTerm (k : ℕ) (h : Space) :
    ContDiffOn ℝ ⊤ (homogeneousTaylorTerm k G0 h) ({0}ᶜ : Set Space) :=
  contDiffOn_homogeneousTaylorTerm contDiffOn_G0 isClosed_singleton.isOpen_compl k h

/-- Homogeneous Taylor terms of the actual Laplace kernel are harmonic in the
center variable, for every fixed displacement and every degree. -/
theorem harmonicOnNhd_G0_homogeneousTaylorTerm (k : ℕ) (h : Space) :
    HarmonicOnNhd (homogeneousTaylorTerm k G0 h) ({0}ᶜ : Set Space) :=
  harmonicOnNhd_homogeneousTaylorTerm contDiffOn_G0 isClosed_singleton.isOpen_compl
    harmonicOnNhd_G0 k h

end PDETranslations.LaplaceHarmonic
