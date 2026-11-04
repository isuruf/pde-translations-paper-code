module

public import Mathlib.Analysis.Calculus.ContDiff.Defs
import Mathlib.Analysis.Calculus.ContDiff.Comp
import Mathlib.Analysis.Calculus.FDeriv.Symmetric
import Mathlib.Analysis.Calculus.IteratedDeriv.Defs
import Mathlib.Data.List.Perm.Basic

@[expose] public section

open scoped BigOperators

namespace Multiindex

variable {d : ℕ} {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- A multi-index in `d` variables is a finitely supported family of natural numbers. -/
abbrev Index (d : ℕ) := Fin d →₀ ℕ

/-- A list containing each coordinate `i` exactly `α i` times.

The fixed order is immaterial for sufficiently smooth functions, whose mixed partial derivatives
commute.  It makes the definition usable without adding smoothness hypotheses. -/
noncomputable def directions (α : Index d) : List (Fin d) :=
  (Finset.univ.toList.flatMap fun i => List.replicate (α i) i)

/-- The magnitude `|α| = ∑ i, α i` of a multi-index `α`. -/
def magnitude (α : Index d) : ℕ :=
  ∑ i, α i

/-- The factorial `α! = ∏ i, (α i)!` of a multi-index `α`. -/
def factorial (α : Index d) : ℕ :=
  ∏ i, (α i).factorial

/-- The multi-index power `v^α = ∏ i, v_i ^ α_i`. -/
def vectorPower (v : Fin d → ℝ) (α : Index d) : ℝ :=
  ∏ i, v i ^ α i

/-- The `i`th coordinate partial derivative, over the real spatial variables.
The codomain may be any real normed vector space, including `ℝ` and `ℂ`. -/
noncomputable def partialDeriv (i : Fin d) (f : (Fin d → ℝ) → E) : (Fin d → ℝ) → E :=
  fun x => fderiv ℝ f x (Pi.single i 1)

/-- The derivative of `f` with respect to the multi-index `α`.

It is the iterated real Fréchet derivative in the coordinate directions prescribed by `α`,
also for complex-valued functions. -/
noncomputable def deriv (α : Index d) (f : (Fin d → ℝ) → E) (x : Fin d → ℝ) : E :=
  iteratedFDeriv ℝ (directions α).length f x
    (fun k => Pi.single ((directions α).get k) (1 : ℝ))

/-- Adding multi-indices concatenates their directions up to permutation. -/
theorem directions_add_perm (α β : Index d) :
    (directions (α + β)).Perm (directions α ++ directions β) := by
  unfold directions
  simp only [Finsupp.add_apply, List.replicate_add]
  exact (List.flatMap_append_perm _ _ _).symm

private noncomputable def listDeriv : List (Fin d) → ((Fin d → ℝ) → E) → (Fin d → ℝ) → E
  | [], f => f
  | i :: l, f => partialDeriv i (listDeriv l f)

private theorem contDiffOn_listDeriv {s : Set (Fin d → ℝ)}
    {f : (Fin d → ℝ) → E} (hG : ContDiffOn ℝ ⊤ f s) (hs : IsOpen s)
    (l : List (Fin d)) : ContDiffOn ℝ ⊤ (listDeriv l f) s := by
  induction l with
  | nil => exact hG
  | cons i l ih =>
    exact (ih.fderiv_of_isOpen hs (by simp)).clm_apply contDiffOn_const

private theorem listDeriv_congr {s : Set (Fin d → ℝ)} (hs : IsOpen s)
    {f g : (Fin d → ℝ) → E} (hfg : Set.EqOn f g s) (l : List (Fin d)) :
    Set.EqOn (listDeriv l f) (listDeriv l g) s := by
  induction l with
  | nil => exact hfg
  | cons i l ih =>
    intro x hx
    exact congrArg (fun A => A (Pi.single i 1))
      (Filter.EventuallyEq.fderiv_eq (𝕜 := ℝ)
        (ih.eventuallyEq_of_mem (hs.mem_nhds hx)))

private theorem listDeriv_eq_iteratedFDeriv {s : Set (Fin d → ℝ)}
    {f : (Fin d → ℝ) → E} (hG : ContDiffOn ℝ ⊤ f s) (hs : IsOpen s)
    (l : List (Fin d)) {x : Fin d → ℝ} (hx : x ∈ s) :
    listDeriv l f x = iteratedFDeriv ℝ l.length f x
      (fun k => Pi.single (l.get k) (1 : ℝ)) := by
  induction l generalizing x with
  | nil => simp [listDeriv, iteratedFDeriv_zero_apply]
  | cons i l ih =>
    have heq : (listDeriv l f) =ᶠ[nhds x]
        (fun y => iteratedFDeriv ℝ l.length f y
          (fun k => Pi.single (l.get k) (1 : ℝ))) :=
      Filter.eventually_of_mem (hs.mem_nhds hx) (fun y hy => ih hy)
    have hd : DifferentiableAt ℝ (iteratedFDeriv ℝ l.length f) x :=
      ((hG.contDiffAt (hs.mem_nhds hx)).iteratedFDeriv_right
        (m := ⊤) (by simp)).differentiableAt (by simp)
    change listDeriv (i :: l) f x = iteratedFDeriv ℝ (l.length + 1) f x
      (fun k => Pi.single ((i :: l).get k) (1 : ℝ))
    rw [hd.iteratedFDeriv_succ_apply_left']
    change fderiv ℝ (listDeriv l f) x (Pi.single i 1) =
      fderiv ℝ (fun y => iteratedFDeriv ℝ l.length f y
        (fun k => Pi.single (l.get k) (1 : ℝ))) x (Pi.single i 1)
    exact congrArg (fun A => A (Pi.single i 1)) (heq.fderiv_eq (𝕜 := ℝ))

/-- Coordinate derivatives of a smooth function on an open set remain smooth. -/
theorem contDiffOn_deriv {s : Set (Fin d → ℝ)} {f : (Fin d → ℝ) → E}
    (hG : ContDiffOn ℝ ⊤ f s) (hs : IsOpen s) (α : Index d) :
    ContDiffOn ℝ ⊤ (deriv α f) s := by
  apply (contDiffOn_listDeriv hG hs (directions α)).congr
  intro x hx
  exact (listDeriv_eq_iteratedFDeriv hG hs (directions α) hx).symm

private theorem listDeriv_append (l m : List (Fin d)) (f : (Fin d → ℝ) → E) :
    listDeriv (l ++ m) f = listDeriv l (listDeriv m f) := by
  induction l with
  | nil => rfl
  | cons i l ih => simp only [List.cons_append, listDeriv, ih]

private theorem listDeriv_perm {s : Set (Fin d → ℝ)} {f : (Fin d → ℝ) → E}
    (hG : ContDiffOn ℝ ⊤ f s) (hs : IsOpen s)
    {l m : List (Fin d)} (h : l.Perm m) :
    Set.EqOn (listDeriv l f) (listDeriv m f) s := by
  induction h with
  | nil => intro x hx; rfl
  | cons i h ih =>
    exact listDeriv_congr hs ih [i]
  | swap i j l =>
    intro x hx
    have hg := (contDiffOn_listDeriv hG hs l).contDiffAt (hs.mem_nhds hx)
    have hd := (hg.fderiv_right (m := ⊤) (by simp)).differentiableAt (by simp)
    change fderiv ℝ (fun y => fderiv ℝ (listDeriv l f) y (Pi.single i 1)) x
        (Pi.single j 1) =
      fderiv ℝ (fun y => fderiv ℝ (listDeriv l f) y (Pi.single j 1)) x
        (Pi.single i 1)
    simpa [fderiv_clm_apply hd (differentiableAt_const _)] using
      (hg.isSymmSndFDerivAt (by simp)).eq (Pi.single j 1) (Pi.single i 1)
  | trans h₁ h₂ ih₁ ih₂ => exact ih₁.trans ih₂

/-- Mixed coordinate derivatives compose by addition of their multi-indices. -/
theorem deriv_add {s : Set (Fin d → ℝ)} {f : (Fin d → ℝ) → E}
    (hG : ContDiffOn ℝ ⊤ f s) (hs : IsOpen s) {x} (hx : x ∈ s)
    (α β : Index d) : deriv (α + β) f x = deriv α (deriv β f) x := by
  have hβ : Set.EqOn (listDeriv (directions β) f) (deriv β f) s :=
    fun y hy => listDeriv_eq_iteratedFDeriv hG hs (directions β) hy
  calc
    deriv (α + β) f x = listDeriv (directions (α + β)) f x :=
      (listDeriv_eq_iteratedFDeriv hG hs _ hx).symm
    _ = listDeriv (directions α ++ directions β) f x :=
      listDeriv_perm hG hs (directions_add_perm α β) hx
    _ = listDeriv (directions α) (listDeriv (directions β) f) x := by
      rw [listDeriv_append]
    _ = listDeriv (directions α) (deriv β f) x :=
      listDeriv_congr hs hβ _ hx
    _ = deriv α (deriv β f) x :=
      listDeriv_eq_iteratedFDeriv (contDiffOn_deriv hG hs β) hs _ hx

/-- Multi-indices of magnitude at most `n`. -/
noncomputable def upTo (n : ℕ) : Finset (Index d) :=
  ((Finset.univ.pi fun _ : Fin d => Finset.range (n + 1)).image
    fun g => Finsupp.equivFunOnFinite.symm (fun i => g i (by simp))).filter
    fun α => magnitude α ≤ n

/-- Multi-indices of magnitude exactly `k`. -/
noncomputable def multiindicesOfMagnitude (k : ℕ) : Finset (Index d) :=
  (upTo k).filter fun q => magnitude q = k

/-- Membership in `upTo` is exactly the bound on the magnitude. -/
theorem mem_upTo {p : ℕ} {α : Index d} :
    α ∈ upTo p ↔ magnitude α ≤ p := by
  constructor
  · intro h
    exact (Finset.mem_filter.mp h).2
  · intro h
    rw [upTo]
    refine Finset.mem_filter.mpr ⟨?_, h⟩
    rw [Finset.mem_image]
    refine ⟨fun i _ => α i, ?_, ?_⟩
    · apply Finset.mem_pi.mpr
      intro i hi
      simp only [Finset.mem_range]
      apply Nat.lt_succ_iff.mpr
      apply le_trans _ h
      simp only [magnitude]
      exact Finset.single_le_sum (fun _ _ => Nat.zero_le _) (Finset.mem_univ i)
    · apply Finsupp.ext
      intro i
      rfl

/-- Membership in a magnitude layer is exactly equality of magnitudes. -/
theorem mem_multiindicesOfMagnitude {k : ℕ} {α : Index d} :
    α ∈ multiindicesOfMagnitude k ↔ magnitude α = k := by
  simp only [multiindicesOfMagnitude, Finset.mem_filter, mem_upTo]
  constructor
  · exact fun h => h.2
  · intro h
    exact ⟨h.le, h⟩

/-- Splitting the multi-indices of magnitude at most `p` into their exact-magnitude layers. -/
theorem sum_upTo_eq_sum_range_multiindicesOfMagnitude
    {M : Type*} [AddCommMonoid M] (p : ℕ) (F : Index d → M) :
    ∑ α ∈ upTo p, F α =
      ∑ k ∈ Finset.range (p + 1), ∑ α ∈ multiindicesOfMagnitude k, F α := by
  let layers : ℕ → Finset (Index d) := multiindicesOfMagnitude
  have hdisjoint : (↑(Finset.range (p + 1)) : Set ℕ).PairwiseDisjoint layers := by
    rw [Finset.pairwiseDisjoint_iff]
    intro i hi j hj hij
    obtain ⟨α, hα⟩ := hij
    exact (mem_multiindicesOfMagnitude.mp (Finset.mem_of_mem_inter_left hα)).symm.trans
      (mem_multiindicesOfMagnitude.mp (Finset.mem_of_mem_inter_right hα))
  have hunion : (Finset.range (p + 1)).biUnion layers = upTo p := by
    ext α
    simp only [Finset.mem_biUnion, layers, mem_multiindicesOfMagnitude, Finset.mem_range,
      mem_upTo]
    constructor
    · rintro ⟨k, hk, hmag⟩
      exact Nat.le_of_lt_succ (hmag.symm ▸ hk)
    · intro hmag
      exact ⟨magnitude α, Nat.lt_succ_iff.mpr hmag, rfl⟩
  rw [← Finset.sum_biUnion hdisjoint, hunion]

/-- The Taylor polynomial of order `n` at `a`, evaluated at `x`, written using multi-indices. -/
noncomputable def taylorPolynomial (n : ℕ) (f : (Fin d → ℝ) → E)
    (a x : Fin d → ℝ) : E :=
  ∑ α ∈ upTo n,
    (vectorPower (x - a) α / (factorial α : ℝ)) • deriv α f a

/-- The exact remainder after the multi-index Taylor polynomial. -/
noncomputable def taylorRemainder (n : ℕ) (f : (Fin d → ℝ) → E)
    (a x : Fin d → ℝ) : E :=
  f x - taylorPolynomial n f a x

/-- **Multivariate Taylor's theorem, indexed by multi-indices.**

For every order `n`, a function is its multi-index Taylor polynomial about `a` plus the exact
remainder.  Analytic hypotheses can subsequently be used to estimate or eliminate
`taylorRemainder`; the identity itself needs none. -/
theorem taylor (n : ℕ) (f : (Fin d → ℝ) → E) (a x : Fin d → ℝ) :
    f x =
      ∑ α ∈ upTo n,
        (vectorPower (x - a) α / (factorial α : ℝ)) • deriv α f a
      + taylorRemainder n f a x := by
  rw [taylorRemainder, taylorPolynomial, ← add_sub_assoc, add_sub_cancel_left]

end Multiindex
