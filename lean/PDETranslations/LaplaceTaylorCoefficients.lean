module

public import Mathlib.Data.Nat.Choose.Multinomial
public import PDETranslations.LaplaceHarmonic
import Mathlib.Algebra.MvPolynomial.Coeff
import Mathlib.Analysis.Calculus.FDeriv.Symmetric

@[expose] public section

/-!
# Identification of homogeneous Laplace Taylor coefficients

Multilinearity expands the actual iterated derivative into ordered coordinate
words. Extracting a coefficient from the formal polynomial multinomial theorem
counts each multiplicity fiber. Finite-order differentiability and symmetry of
the second derivative prove permutation invariance of its evaluations.

Consequently the project's factorial-weighted `homogeneousTaylorTerm` equals
`(1 / k!) * iteratedFDeriv ℝ k f x (fun _ => h)` whenever `f` is `C^k` at `x`.
No Taylor-coefficient identity, mixed-derivative symmetry, or word-counting
formula is assumed. The Laplace-kernel specialization provides the coefficient
bridge for a separate Legendre generating-function argument.
-/

open scoped BigOperators ContDiff

namespace Multiindex

@[simp] theorem length_directions {d : ℕ} (α : Index d) :
    (directions α).length = magnitude α := by
  classical
  simp [directions, magnitude, List.length_flatMap]

/-- The multinomial multiplicity has exactly the expected real factorial
normalization; no natural-number division is silently coerced to real division. -/
theorem multinomial_div_factorial {d k : ℕ} (α : Index d)
    (hα : magnitude α = k) :
    (Nat.multinomial Finset.univ α : ℝ) / (k.factorial : ℝ) =
      1 / (factorial α : ℝ) := by
  have hs := Nat.multinomial_spec Finset.univ α
  change factorial α * Nat.multinomial Finset.univ α = (magnitude α).factorial at hs
  rw [hα] at hs
  have hsR : (factorial α : ℝ) * (Nat.multinomial Finset.univ α : ℝ) =
      (k.factorial : ℝ) := by exact_mod_cast hs
  have ha : (factorial α : ℝ) ≠ 0 := by
    exact_mod_cast (Finset.prod_ne_zero_iff.mpr (fun i _ => Nat.factorial_ne_zero (α i)))
  have hk : (k.factorial : ℝ) ≠ 0 := by exact_mod_cast Nat.factorial_ne_zero k
  apply (div_eq_div_iff hk ha).mpr
  simpa [mul_comm] using hsR

end Multiindex

namespace PDETranslations.LaplaceTaylorCoefficients

open LaplaceHarmonic

/-- An ordered choice of a coordinate in each derivative slot. -/
abbrev Word (k : ℕ) := Fin k → Fin 3

/-- Multiplicities of the coordinates in an ordered word. -/
noncomputable def wordIndex {k : ℕ} (w : Word k) : Multiindex.Index 3 :=
  Finsupp.equivFunOnFinite.symm (fun i => (Finset.univ.filter (fun j => w j = i)).card)

@[simp] theorem wordIndex_apply {k : ℕ} (w : Word k) (i : Fin 3) :
    wordIndex w i = (Finset.univ.filter (fun j => w j = i)).card := rfl

@[simp] theorem magnitude_wordIndex {k : ℕ} (w : Word k) :
    Multiindex.magnitude (wordIndex w) = k := by
  classical
  have hh := Finset.sum_fiberwise_of_maps_to
    (s := Finset.univ) (t := Finset.univ) (g := w)
    (fun _ _ => Finset.mem_univ _) (fun _ => (1 : ℕ))
  simpa [Multiindex.magnitude, wordIndex_apply] using hh

/-- The product of the ordered coordinate weights is the multiindex monomial. -/
theorem prod_word_eq_vectorPower {k : ℕ} (w : Word k) (v : Fin 3 → ℝ) :
    (∏ j, v (w j)) = Multiindex.vectorPower v (wordIndex w) := by
  classical
  have hh := Finset.prod_fiberwise_of_maps_to
    (s := Finset.univ) (t := Finset.univ) (g := w)
    (fun _ _ => Finset.mem_univ _) (fun j => v (w j))
  rw [← hh]
  apply Finset.prod_congr rfl
  intro i hi
  have he : (∏ j ∈ Finset.univ.filter (fun j => w j = i), v (w j)) =
      ∏ j ∈ Finset.univ.filter (fun j => w j = i), v i := by
    apply Finset.prod_congr rfl
    intro j hj
    rw [(Finset.mem_filter.mp hj).2]
  simpa [Multiindex.vectorPower, wordIndex_apply] using he

private theorem prod_word_eq_prod_pow {R : Type*} [CommMonoid R]
    {k : ℕ} (w : Word k) (v : Fin 3 → R) :
    (∏ j, v (w j)) = ∏ i, v i ^ wordIndex w i := by
  classical
  rw [← Finset.prod_fiberwise_of_maps_to
    (s := Finset.univ) (t := Finset.univ) (g := w)
    (fun _ _ => Finset.mem_univ _) (fun j => v (w j))]
  apply Finset.prod_congr rfl
  intro i hi
  calc
    _ = ∏ j ∈ Finset.univ.filter (fun j => w j = i), v i := by
      apply Finset.prod_congr rfl
      intro j hj
      rw [(Finset.mem_filter.mp hj).2]
    _ = _ := by simp

/-- The number of ordered words with specified multiplicities is the
multinomial coefficient. We extract coefficients of a polynomial identity,
rather than assume an enumeration formula. -/
theorem card_word_fiber {k : ℕ} (α : Multiindex.Index 3)
    (hα : Multiindex.magnitude α = k) :
    (Finset.univ.filter (fun w : Word k => wordIndex w = α)).card =
      Nat.multinomial Finset.univ α := by
  classical
  have hm : α.sum (fun _ n => n) = k := by
    rw [α.sum_of_support_subset (Finset.subset_univ α.support)
      (fun _ n => n) (fun _ _ => rfl)]
    exact hα
  have he := MvPolynomial.coeff_sum_X_pow_of_fintype (R := ℕ) α k
  rw [ite_eq_left hm, Finsupp.multinomial_eq_of_support_subset (Finset.subset_univ _)] at he
  rw [Fintype.sum_pow] at he
  simp only [MvPolynomial.coeff_sum] at he
  have hc (w : Word k) :
      (∏ j, (MvPolynomial.X (w j) : MvPolynomial (Fin 3) ℕ)).coeff α =
        if wordIndex w = α then 1 else 0 := by
    rw [prod_word_eq_prod_pow, MvPolynomial.coeff_prod_X_pow]
    have hi : Finsupp.indicator Finset.univ (fun i _ => wordIndex w i) = wordIndex w := by
      ext i
      simp
    rw [hi]
    simp [eq_comm]
  simp_rw [hc] at he
  simpa using he

/-- Decomposition of a Euclidean vector into its three coordinate directions. -/
theorem sum_coordinates (h : Space) :
    (∑ i : Fin 3, (EuclideanSpace.equiv (Fin 3) ℝ h) i • coordinate i) = h := by
  apply (EuclideanSpace.equiv (Fin 3) ℝ).injective
  simp only [map_sum, map_smul, coordinate, ContinuousLinearEquiv.apply_symm_apply]
  ext j
  simp [Pi.single_apply]

/-- Multilinearity expands diagonal evaluation into all ordered coordinate words.
This is the combinatorial expansion before grouping by multiplicity. -/
theorem diagonal_eq_sum_words (k : ℕ)
    (A : ContinuousMultilinearMap ℝ (fun _ : Fin k => Space) ℝ) (h : Space) :
    A (fun _ => h) =
      ∑ w : Word k, (∏ j, (EuclideanSpace.equiv (Fin 3) ℝ h) (w j)) *
        A (fun j => coordinate (w j)) := by
  classical
  have hh := A.map_sum (fun (_ : Fin k) (i : Fin 3) =>
    (EuclideanSpace.equiv (Fin 3) ℝ h) i • coordinate i)
  simp only [sum_coordinates] at hh
  rw [hh]
  apply Finset.sum_congr rfl
  intro w hw
  exact A.map_smul_univ _ _

/-- Ordered expansion of the actual derivative; no smoothness is needed for
this algebraic identity (even when the totalized derivative is zero). -/
theorem iteratedFDeriv_eq_sum_words (k : ℕ) (f : Space → ℝ) (h x : Space) :
    iteratedFDeriv ℝ k f x (fun _ => h) =
      ∑ w : Word k, Multiindex.vectorPower
        (EuclideanSpace.equiv (Fin 3) ℝ h) (wordIndex w) *
        iteratedFDeriv ℝ k f x (fun j => coordinate (w j)) := by
  rw [diagonal_eq_sum_words]
  simp_rw [prod_word_eq_vectorPower]

/-- Exact regrouping by the same magnitude layers used by
`homogeneousTaylorTerm`. The inner sum still contains every ordered word. -/
theorem iteratedFDeriv_eq_sum_multiindex_fibers
    (k : ℕ) (f : Space → ℝ) (h x : Space) :
    iteratedFDeriv ℝ k f x (fun _ => h) =
      ∑ α ∈ Multiindex.multiindicesOfMagnitude k,
        Multiindex.vectorPower (EuclideanSpace.equiv (Fin 3) ℝ h) α *
          ∑ w ∈ Finset.univ.filter (fun w : Word k => wordIndex w = α),
            iteratedFDeriv ℝ k f x (fun j => coordinate (w j)) := by
  classical
  rw [iteratedFDeriv_eq_sum_words]
  have hh := Finset.sum_fiberwise_of_maps_to
    (s := Finset.univ) (t := Multiindex.multiindicesOfMagnitude k)
    (g := fun w : Word k => wordIndex w)
    (fun w _ => Multiindex.mem_multiindicesOfMagnitude.mpr (magnitude_wordIndex w))
    (fun w => Multiindex.vectorPower (EuclideanSpace.equiv (Fin 3) ℝ h) (wordIndex w) *
      iteratedFDeriv ℝ k f x (fun j => coordinate (w j)))
  rw [← hh]
  apply Finset.sum_congr rfl
  intro α hα
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro w hw
  rw [(Finset.mem_filter.mp hw).2]

/-- Normalized actual derivative coefficient, expressed in exact multiindex
fibers. This API can be used to identify the coefficient in a generating
function without postulating any identity for `homogeneousTaylorTerm`. -/
theorem normalized_iteratedFDeriv_eq_sum_multiindex_fibers
    (k : ℕ) (f : Space → ℝ) (h x : Space) :
    (1 / (k.factorial : ℝ)) * iteratedFDeriv ℝ k f x (fun _ => h) =
      ∑ α ∈ Multiindex.multiindicesOfMagnitude k,
        (Multiindex.vectorPower (EuclideanSpace.equiv (Fin 3) ℝ h) α /
          (k.factorial : ℝ)) *
          ∑ w ∈ Finset.univ.filter (fun w : Word k => wordIndex w = α),
            iteratedFDeriv ℝ k f x (fun j => coordinate (w j)) := by
  rw [iteratedFDeriv_eq_sum_multiindex_fibers, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro α hα
  ring

private theorem count_list_eq_sum (l : List (Fin 3)) (i : Fin 3) :
    l.count i = (l.map (fun j => if j = i then 1 else 0)).sum := by
  induction l with
  | nil => simp
  | cons j l ih =>
    by_cases h : j = i
    · subst j; simp [ih, Nat.add_comm]
    · simp [ih, h]

private theorem count_ofFn_word {k : ℕ} (w : Word k) (i : Fin 3) :
    (List.ofFn w).count i = wordIndex w i := by
  classical
  rw [count_list_eq_sum, List.map_ofFn, List.sum_ofFn]
  simp

private theorem count_directions (α : Multiindex.Index 3) (i : Fin 3) :
    (Multiindex.directions α).count i = α i := by
  classical
  simp [Multiindex.directions, List.count_flatMap, List.count_replicate]

private theorem contDiffAt_directionalList (l : List Space) {m : ℕ}
    {f : Space → ℝ} {x : Space}
    (hf : ContDiffAt ℝ (↑(m + l.length) : ℕ∞ω) f x) :
    ContDiffAt ℝ m (directionalList l f) x := by
  induction l generalizing m with
  | nil => simpa [directionalList] using hf
  | cons v l ih =>
    have hg : ContDiffAt ℝ (m + 1) (directionalList l f) x :=
      ih (by simpa [List.length_cons, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using hf)
    exact (hg.fderiv_right (m := m) (by simp)).clm_apply contDiffAt_const

private theorem directionalList_eq_iteratedFDeriv_at (l : List Space)
    {f : Space → ℝ} {x : Space} (hf : ContDiffAt ℝ l.length f x) :
    directionalList l f x = iteratedFDeriv ℝ l.length f x (fun i => l.get i) := by
  induction l generalizing x with
  | nil => simp [directionalList, iteratedFDeriv_zero_apply]
  | cons v l ih =>
    have heq : directionalList l f =ᶠ[nhds x]
        (fun y => iteratedFDeriv ℝ l.length f y (fun i => l.get i)) := by
      have hl : ContDiffAt ℝ (l.length : ℕ) f x := hf.of_le (by simp)
      filter_upwards [hl.eventually (by simp)] with y hy
      exact ih hy
    have hd : DifferentiableAt ℝ (iteratedFDeriv ℝ l.length f) x :=
      hf.differentiableAt_iteratedFDeriv (by
        change (l.length : ℕ∞ω) < ((l.length + 1 : ℕ) : ℕ∞ω)
        exact_mod_cast Nat.lt_succ_self l.length)
    change directionalList (v :: l) f x = iteratedFDeriv ℝ (l.length + 1) f x _
    rw [hd.iteratedFDeriv_succ_apply_left']
    change fderiv ℝ (directionalList l f) x v =
      fderiv ℝ (fun y => iteratedFDeriv ℝ l.length f y (fun i => l.get i)) x v
    exact congrArg (fun A => A v) (heq.fderiv_eq (𝕜 := ℝ))

private theorem directionalList_perm_at {l m : List Space} (hp : l.Perm m)
    {f : Space → ℝ} {x : Space} (hf : ContDiffAt ℝ l.length f x) :
    directionalList l f x = directionalList m f x := by
  induction hp generalizing x with
  | nil => rfl
  | @cons v l₁ l₂ hp ih =>
    have heq : directionalList l₁ f =ᶠ[nhds x] directionalList l₂ f := by
      have hl : ContDiffAt ℝ (l₁.length : ℕ) f x := hf.of_le (by simp)
      filter_upwards [hl.eventually (by simp)] with y hy
      exact ih hy
    exact congrArg (fun A => A v) (heq.fderiv_eq (𝕜 := ℝ))
  | swap v w l =>
    have hg : ContDiffAt ℝ 2 (directionalList l f) x :=
      contDiffAt_directionalList l (m := 2) (by
        have hl : (2 + l.length : ℕ) = (w :: v :: l).length := by simp; omega
        rw [hl]
        exact hf)
    have hd := (hg.fderiv_right (m := 1) (by norm_num)).differentiableAt (by simp)
    change fderiv ℝ (fun y => fderiv ℝ (directionalList l f) y v) x w =
      fderiv ℝ (fun y => fderiv ℝ (directionalList l f) y w) x v
    simpa [fderiv_clm_apply hd (differentiableAt_const _)] using
      (hg.isSymmSndFDerivAt (by norm_num)).eq w v
  | trans hp hq ih₁ ih₂ =>
    exact (ih₁ hf).trans (ih₂ (by simpa [← hp.length_eq] using hf))

private theorem directionalList_eq_iteratedFDeriv_of_length_at
    (l : List Space) (n : ℕ) (hn : l.length = n) (v : Fin n → Space)
    (hv : ∀ i, l.get i = v (Fin.cast hn i))
    {f : Space → ℝ} {x : Space} (hf : ContDiffAt ℝ n f x) :
    directionalList l f x = iteratedFDeriv ℝ n f x v := by
  subst n
  rw [directionalList_eq_iteratedFDeriv_at l hf]
  congr 1
  funext i
  exact hv i

/-- All ordered evaluations in a multiplicity fiber agree with the canonical
coordinate derivative. Only `k` derivatives at the center are required. -/
theorem iteratedFDeriv_word_eq_coordinateDeriv {k : ℕ} {f : Space → ℝ}
    {x : Space} (hf : ContDiffAt ℝ k f x) (α : Multiindex.Index 3)
    (w : Word k) (hw : wordIndex w = α) :
    iteratedFDeriv ℝ k f x (fun j => coordinate (w j)) = coordinateDeriv α f x := by
  classical
  have hα : Multiindex.magnitude α = k := by rw [← hw]; exact magnitude_wordIndex w
  have hp : (List.ofFn w).Perm (Multiindex.directions α) :=
    List.perm_iff_count.mpr (fun i => by rw [count_ofFn_word, hw, count_directions])
  have hpm := hp.map coordinate
  rw [List.map_ofFn] at hpm
  have he : directionalList (List.ofFn (fun j => coordinate (w j))) f x =
      iteratedFDeriv ℝ k f x (fun j => coordinate (w j)) := by
    exact directionalList_eq_iteratedFDeriv_of_length_at
      (List.ofFn (fun j => coordinate (w j))) k List.length_ofFn _
      (fun i => List.get_ofFn _ i) hf
  rw [← he]
  calc
    _ = directionalList ((Multiindex.directions α).map coordinate) f x :=
      directionalList_perm_at hpm (by simpa using hf)
    _ = coordinateDeriv α f x := by
      exact directionalList_eq_iteratedFDeriv_of_length_at
        ((Multiindex.directions α).map coordinate) (Multiindex.directions α).length
        (List.length_map coordinate) _ (fun i => by simp)
        (by simpa [hα] using hf)

/-- Exact word-fiber identity: symmetry supplies equal summands and the
polynomial counting theorem supplies their multinomial multiplicity. -/
theorem sum_word_fiber_eq_multinomial_coordinateDeriv {k : ℕ} {f : Space → ℝ}
    {x : Space} (hf : ContDiffAt ℝ k f x) (α : Multiindex.Index 3)
    (hα : Multiindex.magnitude α = k) :
    (∑ w ∈ Finset.univ.filter (fun w : Word k => wordIndex w = α),
      iteratedFDeriv ℝ k f x (fun j => coordinate (w j))) =
        (Nat.multinomial Finset.univ α : ℝ) * coordinateDeriv α f x := by
  classical
  calc
    _ = ∑ _w ∈ Finset.univ.filter (fun w : Word k => wordIndex w = α),
        coordinateDeriv α f x := by
      apply Finset.sum_congr rfl
      intro w hw
      exact iteratedFDeriv_word_eq_coordinateDeriv hf α w (Finset.mem_filter.mp hw).2
    _ = _ := by simp [card_word_fiber α hα, nsmul_eq_mul]

/-- The project’s multiindex Taylor term is the actual normalized diagonal
Fréchet derivative. Finite-order smoothness at the center suffices. -/
theorem homogeneousTaylorTerm_eq_normalized_iteratedFDeriv_of_contDiffAt
    (k : ℕ) {f : Space → ℝ} (h x : Space) (hf : ContDiffAt ℝ k f x) :
    homogeneousTaylorTerm k f h x =
      (1 / (k.factorial : ℝ)) * iteratedFDeriv ℝ k f x (fun _ => h) := by
  classical
  rw [homogeneousTaylorTerm, normalized_iteratedFDeriv_eq_sum_multiindex_fibers]
  apply Finset.sum_congr rfl
  intro α hα
  have hm := Multiindex.mem_multiindicesOfMagnitude.mp hα
  rw [sum_word_fiber_eq_multinomial_coordinateDeriv hf α hm]
  simp only [smul_eq_mul]
  have hn := Multiindex.multinomial_div_factorial α hm
  calc
    _ = Multiindex.vectorPower (EuclideanSpace.equiv (Fin 3) ℝ h) α *
        ((Nat.multinomial Finset.univ α : ℝ) / (k.factorial : ℝ)) *
          coordinateDeriv α f x := by rw [hn]; ring
    _ = _ := by ring

/-- Smooth-at-the-center API for coefficient identification in a Legendre
generating formula. No assumptions about Taylor coefficients are needed. -/
theorem homogeneousTaylorTerm_eq_normalized_iteratedFDeriv
    (k : ℕ) {f : Space → ℝ} (h x : Space) (hf : ContDiffAt ℝ ⊤ f x) :
    homogeneousTaylorTerm k f h x =
      (1 / (k.factorial : ℝ)) * iteratedFDeriv ℝ k f x (fun _ => h) :=
  homogeneousTaylorTerm_eq_normalized_iteratedFDeriv_of_contDiffAt k h x
    (hf.of_le le_top)

/-- Infinitely differentiable (not necessarily analytic) coefficient API. -/
theorem homogeneousTaylorTerm_eq_normalized_iteratedFDeriv_of_smooth
    (k : ℕ) {f : Space → ℝ} (h x : Space) (hf : ContDiffAt ℝ ∞ f x) :
    homogeneousTaylorTerm k f h x =
      (1 / (k.factorial : ℝ)) * iteratedFDeriv ℝ k f x (fun _ => h) :=
  homogeneousTaylorTerm_eq_normalized_iteratedFDeriv_of_contDiffAt k h x
    (hf.of_le (by simp))

/-- Coefficient identification for the actual Laplace kernel away from its pole.
This theorem has no classical Legendre or coefficient assumptions. -/
theorem homogeneousTaylorTerm_G0_eq_normalized_iteratedFDeriv
    (k : ℕ) (h x : Space) (hx : x ≠ 0) :
    homogeneousTaylorTerm k G0 h x =
      (1 / (k.factorial : ℝ)) * iteratedFDeriv ℝ k G0 x (fun _ => h) := by
  have hx' : x ∈ ({0}ᶜ : Set Space) := by simpa using hx
  exact homogeneousTaylorTerm_eq_normalized_iteratedFDeriv k h x
    (contDiffOn_G0.contDiffAt (isClosed_singleton.isOpen_compl.mem_nhds hx'))

end PDETranslations.LaplaceTaylorCoefficients
