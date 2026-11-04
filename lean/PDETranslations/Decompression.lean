module

public import Mathlib.LinearAlgebra.Matrix.Block
public import PDETranslations.PDE
import Mathlib.Algebra.BigOperators.Group.Finset.Piecewise
import Mathlib.Analysis.Complex.Basic
import Mathlib.Logic.Equiv.Set

@[expose] public section

open scoped BigOperators Classical

namespace PDETranslations

variable {d : ℕ}

variable (ν : GradedMonomialOrdering d)

/-- The vector of derivative operators `𝒟 q` from equation `(define_v)` of the paper,
 applied to `f` at `x`.  Entry `i` is the derivative with multi-index `ν(i)`;
 Lean's zero-based index `i` corresponds to the paper's index `i + 1`.
 Differentiation is over the real spatial variables, with values in any real
 normed vector space, including the complex-valued functions of the paper. -/
noncomputable def derivativeVector {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (q : ℕ) (f : (Fin d → ℝ) → E) (x : Fin d → ℝ) :
    Fin (numberOfMultiindices d q) → E :=
  fun i => Multiindex.deriv (ν.equiv i.val) f x

/-- The shifted PDE coefficient `b_{n,m}` from equation `(eq:coeff-shift)`
 of the paper.  The comparison `n ≤ m` is componentwise, and subtraction is
 componentwise natural-number subtraction.  This defines the coefficient for
 all multi-indices; the paper uses it for `m ∈ 𝓜(p)`. -/
noncomputable def shiftedPDECoefficient (c : ℕ)
    (a : Multiindex.Index d → ℂ) (n m : Multiindex.Index d) : ℂ := by
  classical
  exact if n ≤ m ∧ m - n ∈ multiindicesUpTo d c then a (m - n) else 0

/-- The index `h(i)` of the last nonzero column in row `i`, from `(eq:define_h)`.
Indices are zero-based, so the paper's `h(1)` is `lastNonzeroColumn ν p c a 0`.
The value is defined to be zero for an empty row.  Under `c ≤ p` and a nonzero
coefficient of degree at most `c`, the rows indexed below `N(p-c)` are nonempty;
`lastNonzeroColumn_spec` then identifies the value as their maximum. -/
noncomputable def lastNonzeroColumn
    (p c : ℕ) (a : Multiindex.Index d → ℂ) (i : ℕ) : ℕ := by
  classical
  exact ((Finset.range (numberOfMultiindices d p)).filter fun l =>
    shiftedPDECoefficient c a (ν.equiv i) (ν.equiv l) ≠ 0).sup id

private theorem enumeration_mem_iff
    {p l : ℕ} :
    ν.equiv l ∈ multiindicesUpTo d p ↔ l < numberOfMultiindices d p := by
  classical
  rw [← initial_segment_eq_multiindicesUpTo ν p]
  simp

private theorem shiftedPDECoefficient_ne_zero_iff (c : ℕ)
    (a : Multiindex.Index d → ℂ) (n m : Multiindex.Index d) :
    shiftedPDECoefficient c a n m ≠ 0 ↔
      n ≤ m ∧ m - n ∈ multiindicesUpTo d c ∧ a (m - n) ≠ 0 := by
  classical
  unfold shiftedPDECoefficient
  split_ifs with h
  · simp [h.1, h.2]
  · simp only [ne_eq, not_true_eq_false, false_iff]
    exact fun h' => h ⟨h'.1, h'.2.1⟩

private theorem translated_column_mem
    {p c i : ℕ} (hcp : c ≤ p) (hi : i < numberOfMultiindices d (p - c))
    {α : Multiindex.Index d} (hα : α ∈ multiindicesUpTo d c) :
    ν.equiv.symm (α + ν.equiv i) < numberOfMultiindices d p := by
  apply (enumeration_mem_iff ν).mp
  simp only [Equiv.apply_symm_apply, multiindicesUpTo, Multiindex.mem_upTo]
  have hα' := Multiindex.mem_upTo.mp hα
  have hi' := Multiindex.mem_upTo.mp ((enumeration_mem_iff ν).mpr hi)
  calc
    Multiindex.magnitude (α + ν.equiv i) =
        Multiindex.magnitude α + Multiindex.magnitude (ν.equiv i) := by
      simp [Multiindex.magnitude, Finset.sum_add_distrib]
    _ ≤ c + (p - c) := Nat.add_le_add hα' hi'
    _ = p := Nat.add_sub_of_le hcp

variable {p c : ℕ} (L : LinearPDEOperator d c) (hcp : c ≤ p)

/-- For the rows used in the paper, `h(i)` is a valid column with a nonzero
coefficient, and every nonzero column is at most `h(i)`.  Nontriviality of the
PDE coefficients is necessary for the maximum in `(eq:define_h)` to exist
and is supplied by the bundled operator's `nonzero` field. -/
theorem lastNonzeroColumn_spec (hcp : c ≤ p)
    {i : ℕ} (hi : i < numberOfMultiindices d (p - c)) :
    lastNonzeroColumn ν p c L.coefficients i < numberOfMultiindices d p ∧
      shiftedPDECoefficient c L.coefficients (ν.equiv i)
        (ν.equiv (lastNonzeroColumn ν p c L.coefficients i)) ≠ 0 ∧
      ∀ l, l < numberOfMultiindices d p →
        shiftedPDECoefficient c L.coefficients (ν.equiv i) (ν.equiv l) ≠ 0 →
        l ≤ lastNonzeroColumn ν p c L.coefficients i := by
  classical
  let s := (Finset.range (numberOfMultiindices d p)).filter fun l =>
    shiftedPDECoefficient c L.coefficients (ν.equiv i) (ν.equiv l) ≠ 0
  obtain ⟨α, hdegree, haα⟩ := L.nonzero
  have hα : α ∈ multiindicesUpTo d c := Multiindex.mem_upTo.mpr hdegree.le
  have hs : s.Nonempty := by
    refine ⟨ν.equiv.symm (α + ν.equiv i), Finset.mem_filter.mpr ⟨?_, ?_⟩⟩
    · exact Finset.mem_range.mpr (translated_column_mem ν hcp hi hα)
    · rw [Equiv.apply_symm_apply, shiftedPDECoefficient_ne_zero_iff]
      simpa only [add_tsub_cancel_right] using
        (show ν.equiv i ≤ α + ν.equiv i ∧ α ∈ multiindicesUpTo d c ∧
            L.coefficients α ≠ 0 from
          ⟨le_add_of_nonneg_left (show 0 ≤ α from zero_le), hα, haα⟩)
  have hmem : s.sup id ∈ s := by
    simpa only [id_eq, Set.mem_image, Finset.mem_coe, exists_eq_right] using
      s.sup_mem_of_nonempty (f := id) hs
  have hmax := Finset.mem_filter.mp hmem
  refine ⟨Finset.mem_range.mp hmax.1, hmax.2, ?_⟩
  intro l hl hn
  exact Finset.le_sup (f := id) (Finset.mem_filter.mpr ⟨Finset.mem_range.mpr hl, hn⟩)

/-- Lemma `(lemma:h_eq)`: the last nonzero multi-index in each row is the
last nonzero multi-index in the first row plus the row's multi-index.
The paper uses one-based indices; its first row is row zero here. -/
theorem lastNonzeroColumn_shift (hcp : c ≤ p)
    {i : ℕ} (hi : i < numberOfMultiindices d (p - c)) :
    ν.equiv (lastNonzeroColumn ν p c L.coefficients i) =
      ν.equiv (lastNonzeroColumn ν p c L.coefficients 0) + ν.equiv i := by
  classical
  have hzero : 0 < numberOfMultiindices d (p - c) := by
    apply (enumeration_mem_iff ν).mp
    simp [ν.map_zero, multiindicesUpTo, Multiindex.mem_upTo, Multiindex.magnitude]
  obtain ⟨h0bound, h0nz, h0max⟩ := lastNonzeroColumn_spec ν L hcp hzero
  obtain ⟨hibound, hinz, himax⟩ := lastNonzeroColumn_spec ν L hcp hi
  rw [shiftedPDECoefficient_ne_zero_iff, ν.map_zero] at h0nz
  simp only [zero_le, tsub_zero, true_and] at h0nz
  let z := ν.equiv.symm (ν.equiv (lastNonzeroColumn ν p c L.coefficients 0) + ν.equiv i)
  have hzbound : z < numberOfMultiindices d p :=
    translated_column_mem ν hcp hi h0nz.1
  have hznz : shiftedPDECoefficient c L.coefficients (ν.equiv i) (ν.equiv z) ≠ 0 := by
    rw [shiftedPDECoefficient_ne_zero_iff]
    simp only [z, Equiv.apply_symm_apply, add_tsub_cancel_right]
    exact ⟨le_add_of_nonneg_left
      (show 0 ≤ ν.equiv (lastNonzeroColumn ν p c L.coefficients 0) from zero_le), h0nz⟩
  have hzle := himax z hzbound hznz
  rw [shiftedPDECoefficient_ne_zero_iff] at hinz
  let m := ν.equiv.symm (ν.equiv (lastNonzeroColumn ν p c L.coefficients i) - ν.equiv i)
  have hmbound : m < numberOfMultiindices d p := by
    apply (enumeration_mem_iff ν).mp
    simp only [m, Equiv.apply_symm_apply, multiindicesUpTo, Multiindex.mem_upTo]
    exact le_trans (Multiindex.mem_upTo.mp hinz.2.1) hcp
  have hmnz : shiftedPDECoefficient c L.coefficients (ν.equiv 0) (ν.equiv m) ≠ 0 := by
    rw [shiftedPDECoefficient_ne_zero_iff, ν.map_zero]
    simpa only [m, Equiv.apply_symm_apply, zero_le, tsub_zero, true_and] using hinz.2
  have hmle := h0max m hmbound hmnz
  have hsum : ν.equiv m + ν.equiv i =
      ν.equiv (lastNonzeroColumn ν p c L.coefficients i) := by
    simp only [m, Equiv.apply_symm_apply]
    exact tsub_add_cancel_of_le hinz.1
  have hile : lastNonzeroColumn ν p c L.coefficients i ≤ z := by
    rcases lt_or_eq_of_le hmle with hlt | heq
    · have hlt' := ν.add_lt_add hlt (k := i)
      rw [hsum, Equiv.symm_apply_apply] at hlt'
      exact hlt'.le
    · have heq' := congrArg (fun l => ν.equiv.symm (ν.equiv l + ν.equiv i)) heq
      rw [hsum, Equiv.symm_apply_apply] at heq'
      exact heq'.le
  have heq := Nat.le_antisymm hile hzle
  rw [heq]
  exact ν.equiv.apply_symm_apply _

/-- Lemma `(lemma:h_increasing)`: the last nonzero column index is strictly
increasing with the row index. As in the paper, the indices are restricted to
rows below `N(p-c)`. The explicit `c ≤ p` and nonzero-operator assumptions
ensure that these rows have the maxima used to define `h`; the proof also uses
the addition-compatibility axiom of `GradedMonomialOrdering`. -/
theorem lastNonzeroColumn_strictMono (hcp : c ≤ p)
    {i j : ℕ} (hi : i < numberOfMultiindices d (p - c))
    (hj : j < numberOfMultiindices d (p - c)) (hij : i < j) :
    lastNonzeroColumn ν p c L.coefficients i <
      lastNonzeroColumn ν p c L.coefficients j := by
  let h₀ := lastNonzeroColumn ν p c L.coefficients 0
  have hi_shift := lastNonzeroColumn_shift ν L hcp hi
  have hj_shift := lastNonzeroColumn_shift ν L hcp hj
  have hleft : lastNonzeroColumn ν p c L.coefficients i =
      ν.equiv.symm (ν.equiv i + ν.equiv h₀) := by
    apply ν.equiv.injective
    calc
      ν.equiv (lastNonzeroColumn ν p c L.coefficients i) =
          ν.equiv h₀ + ν.equiv i := hi_shift
      _ = ν.equiv i + ν.equiv h₀ := add_comm _ _
      _ = ν.equiv (ν.equiv.symm (ν.equiv i + ν.equiv h₀)) := by simp
  have hright : lastNonzeroColumn ν p c L.coefficients j =
      ν.equiv.symm (ν.equiv j + ν.equiv h₀) := by
    apply ν.equiv.injective
    calc
      ν.equiv (lastNonzeroColumn ν p c L.coefficients j) =
          ν.equiv h₀ + ν.equiv j := hj_shift
      _ = ν.equiv j + ν.equiv h₀ := add_comm _ _
      _ = ν.equiv (ν.equiv.symm (ν.equiv j + ν.equiv h₀)) := by simp
  rw [hleft, hright]
  exact ν.add_lt_add hij (k := h₀)

/-- The PDE coefficients matrix `P` in `(eq:define_P)`, with zero-based indices.
This is a coefficient-level definition; the analytic identity `(eq:Pv)` is not
asserted here. -/
noncomputable def pdeCoefficientMatrix (p c : ℕ) (L : LinearPDEOperator d c) :
    Matrix (Fin (numberOfMultiindices d (p - c)))
      (Fin (numberOfMultiindices d p)) ℂ :=
  fun i j => shiftedPDECoefficient c L.coefficients (ν.equiv i.val) (ν.equiv j.val)

include hcp in
/-- Reindex a row of `P` by the PDE multi-index `α`: its contributing
 columns are exactly `α + ν(i)` for `|α| ≤ c`. This is purely algebraic. -/
theorem pdeCoefficientMatrix_mulVec_eq_sum
    (jet : Multiindex.Index d → ℂ) (β : Multiindex.Index d)
    (i : Fin (numberOfMultiindices d (p - c))) :
    (pdeCoefficientMatrix ν p c L).mulVec
        (fun j => jet (ν.equiv j.val + β)) i =
      ∑ α ∈ multiindicesUpTo d c,
        L.coefficients α * jet ((α + ν.equiv i.val) + β) := by
  classical
  change (∑ j : Fin (numberOfMultiindices d p),
    shiftedPDECoefficient c L.coefficients (ν.equiv i.val) (ν.equiv j.val) *
      jet (ν.equiv j.val + β)) = _
  simp only [shiftedPDECoefficient, ite_mul, zero_mul]
  rw [← Finset.sum_filter]
  symm
  let idx (α : Multiindex.Index d) (hα : α ∈ multiindicesUpTo d c) :
      Fin (numberOfMultiindices d p) :=
    ⟨ν.equiv.symm (α + ν.equiv i.val), translated_column_mem ν hcp i.isLt hα⟩
  apply Finset.sum_bij (fun α hα => idx α hα)
  · intro α hα
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, idx,
      Equiv.apply_symm_apply, add_tsub_cancel_right]
    exact ⟨le_add_of_nonneg_left (show 0 ≤ α from zero_le), hα⟩
  · intro α hα α' hα' heq
    have heq' := congrArg (fun j : Fin (numberOfMultiindices d p) => ν.equiv j.val) heq
    simp only [idx, Equiv.apply_symm_apply] at heq'
    exact add_right_cancel heq'
  · intro j hj
    obtain ⟨hle, hmem⟩ := (Finset.mem_filter.mp hj).2
    refine ⟨ν.equiv j.val - ν.equiv i.val, hmem, ?_⟩
    apply Fin.ext
    simp only [idx, tsub_add_cancel_of_le hle, Equiv.symm_apply_apply]
  · intro α hα
    simp only [idx, Equiv.apply_symm_apply, add_tsub_cancel_right]

include hcp in
/-- The differentiated PDE relations `(eq:Pv)` for every shifted derivative
 jet of a smooth solution on an arbitrary open domain. No degree bound on
 the shift `β` is needed. The bridge lives here because `PDE` cannot import
 the coefficient matrix without an import cycle. -/
theorem pdeCoefficientMatrix_mulVec_shifted_deriv_eq_zero
    (G : (Fin d → ℝ) → ℂ) {s : Set (Fin d → ℝ)} (hs : IsOpen s)
    (hG : ContDiffOn ℝ ⊤ G s)
    (hPDE : ∀ x ∈ s, linearPDEOperator L G x = 0)
    (β : Multiindex.Index d) {x : Fin d → ℝ} (hx : x ∈ s) :
    (pdeCoefficientMatrix ν p c L).mulVec
      (fun i => Multiindex.deriv (ν.equiv i.val + β) G x) = 0 := by
  funext i
  rw [pdeCoefficientMatrix_mulVec_eq_sum ν L hcp
    (fun α => Multiindex.deriv α G x) β i]
  simp_rw [add_assoc, Multiindex.deriv_add hG hs hx]
  exact linearPDEOperator_deriv_eq_zero L G hs hG hPDE (ν.equiv i.val + β) hx

include hcp in
/-- The unshifted derivative vector of a smooth PDE solution lies in the
 kernel of `P` at every point of its open domain. -/
theorem pdeCoefficientMatrix_mulVec_derivativeVector_eq_zero
    (G : (Fin d → ℝ) → ℂ) {s : Set (Fin d → ℝ)} (hs : IsOpen s)
    (hG : ContDiffOn ℝ ⊤ G s)
    (hPDE : ∀ x ∈ s, linearPDEOperator L G x = 0)
    {x : Fin d → ℝ} (hx : x ∈ s) :
    (pdeCoefficientMatrix ν p c L).mulVec (derivativeVector ν p G x) = 0 := by
  change (pdeCoefficientMatrix ν p c L).mulVec
    (fun i => Multiindex.deriv (ν.equiv i.val) G x) = 0
  simpa only [add_zero] using
    pdeCoefficientMatrix_mulVec_shifted_deriv_eq_zero ν L hcp G hs hG hPDE 0 hx

/-- The ordered pivot columns `j̄` in `(eq:define_bar_j)`. The bound proof
requires `c ≤ p`; exact order supplies a nonzero coefficient in every row. -/
noncomputable def pivotColumn (hcp : c ≤ p)
    (i : Fin (numberOfMultiindices d (p - c))) : Fin (numberOfMultiindices d p) :=
  ⟨lastNonzeroColumn ν p c L.coefficients i.val,
    (lastNonzeroColumn_spec ν L hcp i.isLt).1⟩

/-- The pivot columns occur in increasing order, so are distinct. -/
theorem pivotColumn_strictMono :
    StrictMono (pivotColumn ν L hcp) := by
  intro i j hij
  exact lastNonzeroColumn_strictMono ν L hcp i.isLt j.isLt hij

/-- The square pivot block `P_{[:,j̄]}` from `(lemma:lower_triangular)`. -/
noncomputable def pivotMatrix (hcp : c ≤ p) :
    Matrix (Fin (numberOfMultiindices d (p - c)))
      (Fin (numberOfMultiindices d (p - c))) ℂ :=
  (pdeCoefficientMatrix ν p c L).submatrix id (pivotColumn ν L hcp)

/-- The diagonal entries of the pivot block are the last nonzero entries of
its rows. This is the nonzero-diagonal step in `(lemma:lower_triangular)`. -/
theorem pivotMatrix_diagonal_ne_zero
    (i : Fin (numberOfMultiindices d (p - c))) :
    pivotMatrix ν L hcp i i ≠ 0 :=
  (lastNonzeroColumn_spec ν L hcp i.isLt).2.1

/-- Lemma `(lemma:lower_triangular)`: the selected block is lower triangular
and invertible. `IsUnit` expresses two-sided matrix invertibility, not merely
nonzero entries. The assumptions are the paper's `p ≥ c` and exact PDE order. -/
theorem pivotMatrix_lower_triangular :
    (pivotMatrix ν L hcp).IsLowerTriangular ∧ IsUnit (pivotMatrix ν L hcp) := by
  classical
  have htri : (pivotMatrix ν L hcp).IsLowerTriangular := by
    intro i j hij
    by_contra hn
    have hmax := (lastNonzeroColumn_spec ν L hcp i.isLt).2.2
      (pivotColumn ν L hcp j).val (pivotColumn ν L hcp j).isLt hn
    have hlt := lastNonzeroColumn_strictMono ν L hcp i.isLt j.isLt hij
    exact (not_le_of_gt hlt) hmax
  refine ⟨htri, ?_⟩
  rw [Matrix.isUnit_iff_isUnit_det, Matrix.det_of_isLowerTriangular _ htri]
  exact isUnit_iff_ne_zero.mpr
    (Finset.prod_ne_zero_iff.mpr fun i _ => pivotMatrix_diagonal_ne_zero ν L hcp i)

namespace AbstractDecompression

variable {R C : Type*} [Fintype R] [Fintype C] [DecidableEq R] [DecidableEq C]

/-- Columns not selected by the pivot injection. -/
private abbrev FreeColumns (h : R → C) := {j : C // j ∉ Set.range h}

omit [DecidableEq R] in
/-- Split a column sum into its pivot and free parts. -/
private theorem sum_columns (h : R → C) (hh : Function.Injective h) (f : C → ℂ) :
    (∑ c, f c) = (∑ r, f (h r)) + ∑ j : FreeColumns h, f j.val := by
  let : Fintype (Set.range h) := Subtype.fintype (fun c => c ∈ Set.range h)
  have hp : (∑ r, f (h r)) = ∑ c : Set.range h, f c.val :=
    Fintype.sum_equiv (Equiv.ofInjective h hh) _ _ (fun _ => rfl)
  rw [hp]
  exact (Fintype.sum_subtype_add_sum_subtype (fun c => c ∈ Set.range h) f).symm

/-- The pivot block is `A = P.submatrix id h`; the free block is `B`.
Pivot rows of the decompression matrix are `-A⁻¹ B`, and free rows are identity rows. -/
private noncomputable def decompressionMatrix (P : Matrix R C ℂ) (h : R → C)
    (hh : Function.Injective h) : Matrix C (FreeColumns h) ℂ :=
  fun c j => if hc : c ∈ Set.range h then
    -((P.submatrix id h)⁻¹ * P.submatrix id Subtype.val)
      ((Equiv.ofInjective h hh).symm ⟨c, hc⟩) j
    else if c = j.val then 1 else 0

omit [Fintype C] in
@[simp]
private theorem decompressionMatrix_pivot (P : Matrix R C ℂ) (h : R → C)
    (hh : Function.Injective h) (r : R) (j : FreeColumns h) :
    decompressionMatrix P h hh (h r) j =
      -((P.submatrix id h)⁻¹ * P.submatrix id Subtype.val) r j := by
  simp [decompressionMatrix, show h r ∈ Set.range h from ⟨r, rfl⟩,
    Equiv.ofInjective_symm_apply]

omit [Fintype C] in
/-- Entrywise pivot-row formula, with the inverse multiplication expanded as a sum. -/
private theorem decompressionMatrix_pivot_eq_sum (P : Matrix R C ℂ) (h : R → C)
    (hh : Function.Injective h) (r : R) (j : FreeColumns h) :
    decompressionMatrix P h hh (h r) j =
      -(∑ r' : R, (P.submatrix id h)⁻¹ r r' * P r' j.val) := by
  rw [decompressionMatrix_pivot]
  rfl

omit [Fintype C] in
@[simp]
private theorem decompressionMatrix_free (P : Matrix R C ℂ) (h : R → C)
    (hh : Function.Injective h) (j k : FreeColumns h) :
    decompressionMatrix P h hh j.val k = if j = k then 1 else 0 := by
  simp [decompressionMatrix, j.property, Subtype.ext_iff]

/-- Decompression preserves every prescribed free coordinate. -/
@[simp]
private theorem decompressionMatrix_mulVec_free (P : Matrix R C ℂ) (h : R → C)
    (hh : Function.Injective h) (t : FreeColumns h → ℂ) (j : FreeColumns h) :
    (decompressionMatrix P h hh).mulVec t j.val = t j := by
  simp [Matrix.mulVec, dotProduct, ite_mul]

/-- The pivot coordinates are explicitly `-A⁻¹ (B t)`. -/
private theorem decompressionMatrix_mulVec_pivot (P : Matrix R C ℂ) (h : R → C)
    (hh : Function.Injective h) (t : FreeColumns h → ℂ) :
    (fun r => (decompressionMatrix P h hh).mulVec t (h r)) =
      -((P.submatrix id h)⁻¹.mulVec ((P.submatrix id Subtype.val).mulVec t)) := by
  calc
    _ = (-((P.submatrix id h)⁻¹ * P.submatrix id Subtype.val)).mulVec t := by
      funext r
      change (∑ j, decompressionMatrix P h hh (h r) j * t j) =
        ∑ j, (-((P.submatrix id h)⁻¹ * P.submatrix id Subtype.val) r j) * t j
      apply Finset.sum_congr rfl
      intro j _
      rw [decompressionMatrix_pivot]
    _ = _ := by rw [Matrix.neg_mulVec, ← Matrix.mulVec_mulVec]

omit [DecidableEq R] in
/-- The row equations split as `P s = A (s ∘ h) + B (s|J)`. -/
private theorem mulVec_split (P : Matrix R C ℂ) (h : R → C)
    (hh : Function.Injective h) (s : C → ℂ) :
    P.mulVec s = (P.submatrix id h).mulVec (fun r => s (h r)) +
      (P.submatrix id Subtype.val).mulVec (fun j : FreeColumns h => s j.val) := by
  funext r
  exact sum_columns h hh (fun c => P r c * s c)

/-- Every vector reconstructed from free coordinates lies in the kernel. -/
private theorem decompressionMatrix_mulVec_kernel (P : Matrix R C ℂ) (h : R → C)
    (hh : Function.Injective h) (hA : IsUnit (P.submatrix id h))
    (t : FreeColumns h → ℂ) :
    P.mulVec ((decompressionMatrix P h hh).mulVec t) = 0 := by
  have hdet := (Matrix.isUnit_iff_isUnit_det (P.submatrix id h)).mp hA
  rw [mulVec_split P h hh, decompressionMatrix_mulVec_pivot]
  have hfree : (fun j : FreeColumns h =>
      (decompressionMatrix P h hh).mulVec t j.val) = t := by
    funext j
    exact decompressionMatrix_mulVec_free P h hh t j
  rw [hfree, Matrix.mulVec_neg, Matrix.mulVec_mulVec,
    Matrix.mul_nonsing_inv _ hdet, Matrix.one_mulVec, neg_add_cancel]

/-- The kernel is exactly the graph of the explicit decompression matrix. -/
private theorem decompressionMatrix_spec (P : Matrix R C ℂ) (h : R → C)
    (hh : Function.Injective h) (hA : IsUnit (P.submatrix id h)) (s : C → ℂ) :
    P.mulVec s = 0 ↔
      s = (decompressionMatrix P h hh).mulVec (fun j : FreeColumns h => s j.val) := by
  constructor
  · intro hs
    have hdet := (Matrix.isUnit_iff_isUnit_det (P.submatrix id h)).mp hA
    have heq : (P.submatrix id h).mulVec (fun r => s (h r)) =
        -((P.submatrix id Subtype.val).mulVec
          (fun j : FreeColumns h => s j.val)) := by
      have hsplit := mulVec_split P h hh s
      rw [hs] at hsplit
      exact eq_neg_of_add_eq_zero_left hsplit.symm
    have hpivot : (fun r => s (h r)) =
        -((P.submatrix id h)⁻¹.mulVec
          ((P.submatrix id Subtype.val).mulVec
            (fun j : FreeColumns h => s j.val))) := by
      have hinv := congrArg (fun v => (P.submatrix id h)⁻¹.mulVec v) heq
      simpa only [Matrix.mulVec_mulVec, Matrix.nonsing_inv_mul _ hdet,
        Matrix.one_mulVec, Matrix.mulVec_neg] using hinv
    funext c
    by_cases hc : c ∈ Set.range h
    · obtain ⟨r, rfl⟩ := hc
      have hrow := congrFun hpivot r
      rw [← decompressionMatrix_mulVec_pivot P h hh] at hrow
      exact hrow
    · exact (decompressionMatrix_mulVec_free P h hh
        (fun j : FreeColumns h => s j.val) ⟨c, hc⟩).symm
  · intro hs
    rw [hs]
    exact decompressionMatrix_mulVec_kernel P h hh hA _

/-- Unique matrix decompression for an arbitrary finite pivot injection.
No ordering, triangularity, or nonemptiness assumptions are needed. -/
private theorem existsUnique_decompressionMatrix (P : Matrix R C ℂ) (h : R → C)
    (hh : Function.Injective h) (hA : IsUnit (P.submatrix id h)) :
    ∃! M : Matrix C (FreeColumns h) ℂ,
      ∀ s : C → ℂ, P.mulVec s = 0 ↔
        s = M.mulVec (fun j : FreeColumns h => s j.val) := by
  refine ⟨decompressionMatrix P h hh, decompressionMatrix_spec P h hh hA, ?_⟩
  intro N hN
  have heq : ∀ t : FreeColumns h → ℂ,
      N.mulVec t = (decompressionMatrix P h hh).mulVec t := by
    intro t
    have hs := (hN ((decompressionMatrix P h hh).mulVec t)).mp
      (decompressionMatrix_mulVec_kernel P h hh hA t)
    have hfree : (fun j : FreeColumns h =>
        (decompressionMatrix P h hh).mulVec t j.val) = t := by
      funext j
      exact decompressionMatrix_mulVec_free P h hh t j
    rw [hfree] at hs
    exact hs.symm
  ext c j
  have hcol := congrFun (heq (Pi.single j 1)) c
  simpa only [Matrix.mulVec_single_one, Matrix.col_apply] using hcol

end AbstractDecompression

/-- The complement `j` of the pivot columns. Rather than choose a numbering,
Lean indexes it by the actual nonpivot columns. Choosing an enumeration yields
the paper's `N(p) × |j|` matrix with the corresponding column permutation. -/
abbrev FreeColumn :=
  {j : Fin (numberOfMultiindices d p) // j ∉ Set.range (pivotColumn ν L hcp)}

/-- The number of free coordinates is exactly `N(p) - N(p-c)`, as in the
paper's matrix dimensions; no enumeration of the complement is required. -/
theorem card_freeColumn :
    Fintype.card (FreeColumn ν L hcp) =
      numberOfMultiindices d p - numberOfMultiindices d (p - c) := by
  classical
  rw [Fintype.card_subtype_compl]
  have hcard := Fintype.card_congr
    (Equiv.ofInjective (pivotColumn ν L hcp) (pivotColumn_strictMono ν L hcp).injective)
  simpa only [Fintype.card_fin] using congrArg
    (fun n => Fintype.card (Fin (numberOfMultiindices d p)) - n) hcard.symm

/-- The decompression matrix `M` in `(eq:m_1)` and `(eq:m_2)`, in the original
column order of `P`. The free rows are identity rows and the pivot rows are
`-A⁻¹ B`. Invertibility is proved, not imposed as another assumption. -/
@[no_expose] noncomputable def decompressionMatrix :
    Matrix (Fin (numberOfMultiindices d p)) (FreeColumn ν L hcp) ℂ :=
  AbstractDecompression.decompressionMatrix (pdeCoefficientMatrix ν p c L)
    (pivotColumn ν L hcp) (pivotColumn_strictMono ν L hcp).injective

/-- Equation `(eq:m_1)`: the free rows form the identity matrix. -/
@[simp]
theorem decompressionMatrix_free
    (j k : FreeColumn ν L hcp) :
    decompressionMatrix ν L hcp j.val k = if j = k then 1 else 0 := by
  simpa only [decompressionMatrix] using
    AbstractDecompression.decompressionMatrix_free (pdeCoefficientMatrix ν p c L)
      (pivotColumn ν L hcp) (pivotColumn_strictMono ν L hcp).injective j k

/-- Equation `(eq:m_2)`: the pivot rows are `-A⁻¹ B`. -/
@[simp]
theorem decompressionMatrix_pivot
    (i : Fin (numberOfMultiindices d (p - c))) (j : FreeColumn ν L hcp) :
    decompressionMatrix ν L hcp (pivotColumn ν L hcp i) j =
      -((pivotMatrix ν L hcp)⁻¹ *
        (pdeCoefficientMatrix ν p c L).submatrix id Subtype.val) i j :=
  AbstractDecompression.decompressionMatrix_pivot _ _ _ i j

/-- Every choice of free coordinates extends to a kernel vector. This fact
also supplies the surjectivity needed by the paper's uniqueness claim. -/
theorem decompressionMatrix_mulVec_kernel
    (t : FreeColumn ν L hcp → ℂ) :
    (pdeCoefficientMatrix ν p c L).mulVec ((decompressionMatrix ν L hcp).mulVec t) = 0 :=
  AbstractDecompression.decompressionMatrix_mulVec_kernel _ _ _
    (pivotMatrix_lower_triangular ν L hcp).2 t

/-- The equivalence in `(thm:define_M)` for the explicitly constructed matrix.
This is purely algebraic: no claim that a derivative vector lies in the kernel
is used or proved. -/
theorem decompressionMatrix_spec
    (s : Fin (numberOfMultiindices d p) → ℂ) :
    (pdeCoefficientMatrix ν p c L).mulVec s = 0 ↔
      s = (decompressionMatrix ν L hcp).mulVec (fun j => s j.val) :=
  AbstractDecompression.decompressionMatrix_spec _ _ _
    (pivotMatrix_lower_triangular ν L hcp).2 s

/-- Theorem `(thm:define_M)`, including uniqueness among all complex matrices
with these row and free-column indices. Uniqueness follows by testing any
candidate on reconstructed kernel vectors with arbitrary free coordinates,
then on standard basis vectors. -/
theorem existsUnique_decompressionMatrix :
    ∃! M : Matrix (Fin (numberOfMultiindices d p)) (FreeColumn ν L hcp) ℂ,
      ∀ s : Fin (numberOfMultiindices d p) → ℂ,
        (pdeCoefficientMatrix ν p c L).mulVec s = 0 ↔
          s = M.mulVec (fun j => s j.val) :=
  AbstractDecompression.existsUnique_decompressionMatrix _ _
    (pivotColumn_strictMono ν L hcp).injective (pivotMatrix_lower_triangular ν L hcp).2

/-- The inverse of the pivot block is lower triangular, as used in the proof
of `(prop:m_property)`. Its invertibility follows from the existing PDE
assumptions, rather than being an additional hypothesis. -/
theorem pivotMatrix_inv_lower_triangular :
    ((pivotMatrix ν L hcp)⁻¹).IsLowerTriangular := by
  let : Invertible (pivotMatrix ν L hcp) :=
    Matrix.invertibleOfIsUnitDet _
      ((Matrix.isUnit_iff_isUnit_det _).mp (pivotMatrix_lower_triangular ν L hcp).2)
  exact Matrix.blockTriangular_inv_of_blockTriangular
    (pivotMatrix_lower_triangular ν L hcp).1

/-- Part 2 of `(prop:m_property)`: a free column strictly after a row has zero
entry in that row of `M`. The comparison is between actual columns of `P`,
not between positions in an enumeration of its free columns. -/
theorem decompressionMatrix_eq_zero_of_lt
    (i : Fin (numberOfMultiindices d p)) (k : FreeColumn ν L hcp)
    (hik : i < k.val) :
    decompressionMatrix ν L hcp i k = 0 := by
  classical
  by_cases hi : i ∈ Set.range (pivotColumn ν L hcp)
  · obtain ⟨w, rfl⟩ := hi
    rw [decompressionMatrix_pivot, Matrix.mul_apply]
    suffices hs : (∑ r, (pivotMatrix ν L hcp)⁻¹ w r *
        (pdeCoefficientMatrix ν p c L).submatrix id Subtype.val r k) = 0 by
      rw [hs, neg_zero]
    apply Finset.sum_eq_zero
    intro r _
    by_cases hwr : w < r
    · rw [pivotMatrix_inv_lower_triangular ν L hcp hwr, zero_mul]
    · have hrw : r ≤ w := le_of_not_gt hwr
      have hrk : pivotColumn ν L hcp r < k.val :=
        lt_of_le_of_lt ((pivotColumn_strictMono ν L hcp).monotone hrw) hik
      have hzero : pdeCoefficientMatrix ν p c L r k.val = 0 := by
        by_contra hn
        have hmax := (lastNonzeroColumn_spec ν L hcp r.isLt).2.2
          k.val.val k.val.isLt hn
        exact (not_le_of_gt hrk) hmax
      change (pivotMatrix ν L hcp)⁻¹ w r * pdeCoefficientMatrix ν p c L r k.val = 0
      rw [hzero, mul_zero]
  · have hik' : (⟨i, hi⟩ : FreeColumn ν L hcp) ≠ k := by
      intro heq
      have hval := congrArg Subtype.val heq
      exact (ne_of_lt hik) hval
    change decompressionMatrix ν L hcp (⟨i, hi⟩ : FreeColumn ν L hcp).val k = 0
    rw [decompressionMatrix_free, ite_eq_right hik']

/-- Proposition `(prop:m_property)`: free rows are Kronecker delta rows, and
entries whose actual free column is greater than the row vanish. Indices are
zero-based, and `k.val` is the paper's `j_k`; the free-column enumeration need
not be increasing. This is purely algebraic and needs no analytic assumptions. -/
theorem decompressionMatrix_property
    (i : Fin (numberOfMultiindices d p)) (k : FreeColumn ν L hcp) :
    (i ∉ Set.range (pivotColumn ν L hcp) →
      decompressionMatrix ν L hcp i k = if i = k.val then 1 else 0) ∧
    (i < k.val → decompressionMatrix ν L hcp i k = 0) := by
  classical
  constructor
  · intro hi
    simpa only [Subtype.ext_iff] using
      (decompressionMatrix_free ν L hcp ⟨i, hi⟩ k)
  · exact decompressionMatrix_eq_zero_of_lt ν L hcp i k

/-- The paper's condition "PDEs with only terms of equal order": every
nonzero coefficient used by the order-`c` operator has degree exactly `c`.
Stored coefficients above `c` are ignored by the operator and are unrestricted. -/
def LinearPDEOperator.HasOnlyEqualOrder : Prop :=
  ∀ α ∈ multiindicesUpTo d c, L.coefficients α ≠ 0 → Multiindex.magnitude α = c

/-- The degree constraint behind `(eq:p-equal-order)`, before restricting to
matrix indices. Componentwise subtraction is only used when `n ≤ m`, so no
integer-subtraction interpretation of degrees is needed. -/
theorem shiftedPDECoefficient_magnitude_of_ne_zero
    (hequal : L.HasOnlyEqualOrder) (n m : Multiindex.Index d)
    (hnz : shiftedPDECoefficient c L.coefficients n m ≠ 0) :
    Multiindex.magnitude n + c = Multiindex.magnitude m := by
  obtain ⟨hnm, hmem, ha⟩ :=
    (shiftedPDECoefficient_ne_zero_iff c L.coefficients n m).mp hnz
  have hdegree := hequal (m - n) hmem ha
  calc
    Multiindex.magnitude n + c =
        Multiindex.magnitude n + Multiindex.magnitude (m - n) := by rw [hdegree]
    _ = Multiindex.magnitude (n + (m - n)) := by
      simp [Multiindex.magnitude, Finset.sum_add_distrib]
    _ = Multiindex.magnitude m := by rw [add_tsub_cancel_of_le hnm]

/-- Equation `(eq:p-equal-order)`: the PDE matrix only couples degrees
separated by exactly the PDE order. -/
theorem pdeCoefficientMatrix_eq_zero_of_magnitude_ne
    (hequal : L.HasOnlyEqualOrder)
    (i : Fin (numberOfMultiindices d (p - c)))
    (j : Fin (numberOfMultiindices d p))
    (hdegree : Multiindex.magnitude (ν.equiv i.val) + c ≠
      Multiindex.magnitude (ν.equiv j.val)) :
    pdeCoefficientMatrix ν p c L i j = 0 := by
  by_contra hnz
  exact hdegree (shiftedPDECoefficient_magnitude_of_ne_zero L hequal _ _ hnz)

/-- Equation `(eq:h-increment-eq-order)`: a pivot column has degree equal to
its row degree plus `c`. This follows from its nonzero coefficient. -/
theorem pivotColumn_magnitude (hequal : L.HasOnlyEqualOrder)
    (i : Fin (numberOfMultiindices d (p - c))) :
    Multiindex.magnitude (ν.equiv (pivotColumn ν L hcp i).val) =
      Multiindex.magnitude (ν.equiv i.val) + c :=
  (shiftedPDECoefficient_magnitude_of_ne_zero L hequal _ _
    (lastNonzeroColumn_spec ν L hcp i.isLt).2.1).symm

/-- The pivot block has zero entries between different row-degree blocks.
This strengthens the block-zero observation in the proof of
`(thm:m_equal_order_property)`. -/
theorem pivotMatrix_eq_zero_of_magnitude_ne (hequal : L.HasOnlyEqualOrder)
    (i j : Fin (numberOfMultiindices d (p - c)))
    (hdegree : Multiindex.magnitude (ν.equiv i.val) ≠
      Multiindex.magnitude (ν.equiv j.val)) :
    pivotMatrix ν L hcp i j = 0 := by
  apply pdeCoefficientMatrix_eq_zero_of_magnitude_ne ν L hequal
  rw [pivotColumn_magnitude ν L hcp hequal j]
  exact fun h => hdegree (Nat.add_right_cancel h)

/-- The inverse pivot block also has zero entries between different degrees.
We apply preservation of block triangularity in both degree orders instead
of choosing the paper's boundary index `underline w`. Invertibility is
already a consequence of exact PDE order and `c ≤ p`. -/
theorem pivotMatrix_inv_eq_zero_of_magnitude_ne (hequal : L.HasOnlyEqualOrder)
    (i j : Fin (numberOfMultiindices d (p - c)))
    (hdegree : Multiindex.magnitude (ν.equiv i.val) ≠
      Multiindex.magnitude (ν.equiv j.val)) :
    (pivotMatrix ν L hcp)⁻¹ i j = 0 := by
  let : Invertible (pivotMatrix ν L hcp) :=
    Matrix.invertibleOfIsUnitDet _
      ((Matrix.isUnit_iff_isUnit_det _).mp (pivotMatrix_lower_triangular ν L hcp).2)
  have hupper : (pivotMatrix ν L hcp).BlockTriangular
      (fun r => Multiindex.magnitude (ν.equiv r.val)) := by
    intro r s hlt
    exact pivotMatrix_eq_zero_of_magnitude_ne ν L hcp hequal r s (ne_of_gt hlt)
  have hlower : (pivotMatrix ν L hcp).BlockTriangular
      (fun r => OrderDual.toDual (Multiindex.magnitude (ν.equiv r.val))) := by
    intro r s hlt
    exact pivotMatrix_eq_zero_of_magnitude_ne ν L hcp hequal r s (ne_of_lt hlt)
  rcases lt_or_gt_of_ne hdegree with hlt | hgt
  · exact Matrix.blockTriangular_inv_of_blockTriangular hlower hlt
  · exact Matrix.blockTriangular_inv_of_blockTriangular hupper hgt

/-- Theorem `(thm:m_equal_order_property)`: decompression preserves total
degree for PDEs with only equal-order terms. `k.val` is the paper's actual
free column `j_k`; indices are zero-based. The proof uses the identity free
rows and the degree-block structure of the pivot block and its inverse,
rather than splitting at a particular degree boundary. -/
theorem decompressionMatrix_equal_order_property (hequal : L.HasOnlyEqualOrder)
    (i : Fin (numberOfMultiindices d p)) (k : FreeColumn ν L hcp)
    (hdegree : Multiindex.magnitude (ν.equiv i.val) ≠
      Multiindex.magnitude (ν.equiv k.val.val)) :
    decompressionMatrix ν L hcp i k = 0 := by
  classical
  by_cases hi : i ∈ Set.range (pivotColumn ν L hcp)
  · obtain ⟨w, rfl⟩ := hi
    rw [decompressionMatrix_pivot, Matrix.mul_apply]
    suffices hs : (∑ r, (pivotMatrix ν L hcp)⁻¹ w r *
        (pdeCoefficientMatrix ν p c L).submatrix id Subtype.val r k) = 0 by
      rw [hs, neg_zero]
    apply Finset.sum_eq_zero
    intro r _
    by_cases hwr : Multiindex.magnitude (ν.equiv w.val) =
        Multiindex.magnitude (ν.equiv r.val)
    · have hrk : Multiindex.magnitude (ν.equiv r.val) + c ≠
          Multiindex.magnitude (ν.equiv k.val.val) := by
        rw [← hwr, ← pivotColumn_magnitude ν L hcp hequal w]
        exact hdegree
      change (pivotMatrix ν L hcp)⁻¹ w r * pdeCoefficientMatrix ν p c L r k.val = 0
      rw [pdeCoefficientMatrix_eq_zero_of_magnitude_ne ν L hequal r k.val hrk,
        mul_zero]
    · rw [pivotMatrix_inv_eq_zero_of_magnitude_ne ν L hcp hequal w r hwr, zero_mul]
  · have hik : (⟨i, hi⟩ : FreeColumn ν L hcp) ≠ k := by
      intro heq
      exact hdegree (congrArg (fun j : FreeColumn ν L hcp =>
        Multiindex.magnitude (ν.equiv j.val.val)) heq)
    change decompressionMatrix ν L hcp (⟨i, hi⟩ : FreeColumn ν L hcp).val k = 0
    rw [decompressionMatrix_free, ite_eq_right hik]

/-- Algebraic form of `(eq:deriv-representation)`: a kernel coordinate only
 uses free coordinates of at most its degree. -/
theorem kernel_coordinate_eq_sum_degree_le
    (s : Fin (numberOfMultiindices d p) → ℂ)
    (hs : (pdeCoefficientMatrix ν p c L).mulVec s = 0)
    (i : Fin (numberOfMultiindices d p)) :
    s i = ∑ k : FreeColumn ν L hcp with
        Multiindex.magnitude (ν.equiv k.val.val) ≤ Multiindex.magnitude (ν.equiv i.val),
      decompressionMatrix ν L hcp i k * s k.val := by
  classical
  have hi := congrFun ((decompressionMatrix_spec ν L hcp s).mp hs) i
  change s i = ∑ k, decompressionMatrix ν L hcp i k * s k.val at hi
  rw [hi]
  symm
  apply Finset.sum_filter_of_ne
  intro k _ hnz
  by_contra hdegree
  have hik : i < k.val := ν.graded (Nat.lt_of_not_ge hdegree)
  exact hnz (by rw [decompressionMatrix_eq_zero_of_lt ν L hcp i k hik, zero_mul])

/-- Algebraic form of `(eq:deriv-representation-equal)`: for equal-order
 operators, a kernel coordinate uses only free coordinates of the same degree. -/
theorem kernel_coordinate_eq_sum_degree_eq (hequal : L.HasOnlyEqualOrder)
    (s : Fin (numberOfMultiindices d p) → ℂ)
    (hs : (pdeCoefficientMatrix ν p c L).mulVec s = 0)
    (i : Fin (numberOfMultiindices d p)) :
    s i = ∑ k : FreeColumn ν L hcp with
        Multiindex.magnitude (ν.equiv k.val.val) = Multiindex.magnitude (ν.equiv i.val),
      decompressionMatrix ν L hcp i k * s k.val := by
  classical
  have hi := congrFun ((decompressionMatrix_spec ν L hcp s).mp hs) i
  change s i = ∑ k, decompressionMatrix ν L hcp i k * s k.val at hi
  rw [hi]
  symm
  apply Finset.sum_filter_of_ne
  intro k _ hnz
  by_contra hdegree
  exact hnz (by
    rw [decompressionMatrix_equal_order_property ν L hcp hequal i k
      (Ne.symm hdegree), zero_mul])

/-- Corollary `(corollary:deriv_less_order)`, first formula, for a smooth
 solution of the constant-coefficient PDE away from the origin. The kernel
 identity `(eq:Pv)` follows by differentiating the PDE. Free columns are
 indexed by their actual columns. -/
theorem derivative_eq_sum_degree_le
    (G : (Fin d → ℝ) → ℂ)
    (hG : ContDiffOn ℝ ⊤ G {x | x ≠ 0})
    (hPDE : ∀ x, x ≠ 0 → linearPDEOperator L G x = 0)
    (x : Fin d → ℝ) (hx : x ≠ 0) (i : Fin (numberOfMultiindices d p)) :
    derivativeVector ν p G x i = ∑ k : FreeColumn ν L hcp with
        Multiindex.magnitude (ν.equiv k.val.val) ≤ Multiindex.magnitude (ν.equiv i.val),
      decompressionMatrix ν L hcp i k * derivativeVector ν p G x k.val :=
  kernel_coordinate_eq_sum_degree_le ν L hcp _
    (pdeCoefficientMatrix_mulVec_derivativeVector_eq_zero ν L hcp G
      (isOpen_ne_fun continuous_id continuous_const) hG hPDE hx) i

/-- Corollary `(corollary:deriv_less_order)`, equal-order formula, with the
 same smoothness and PDE hypotheses as `derivative_eq_sum_degree_le`. -/
theorem derivative_eq_sum_degree_eq (hequal : L.HasOnlyEqualOrder)
    (G : (Fin d → ℝ) → ℂ)
    (hG : ContDiffOn ℝ ⊤ G {x | x ≠ 0})
    (hPDE : ∀ x, x ≠ 0 → linearPDEOperator L G x = 0)
    (x : Fin d → ℝ) (hx : x ≠ 0) (i : Fin (numberOfMultiindices d p)) :
    derivativeVector ν p G x i = ∑ k : FreeColumn ν L hcp with
        Multiindex.magnitude (ν.equiv k.val.val) = Multiindex.magnitude (ν.equiv i.val),
      decompressionMatrix ν L hcp i k * derivativeVector ν p G x k.val :=
  kernel_coordinate_eq_sum_degree_eq ν L hcp hequal _
    (pdeCoefficientMatrix_mulVec_derivativeVector_eq_zero ν L hcp G
      (isOpen_ne_fun continuous_id continuous_const) hG hPDE hx) i

include ν in
private theorem zero_lt_numberOfMultiindices (q : ℕ) :
    0 < numberOfMultiindices d q := by
  apply (enumeration_mem_iff ν).mp
  simp [ν.map_zero, multiindicesUpTo, Multiindex.mem_upTo, Multiindex.magnitude]

include hcp in
/-- The first pivot multi-index has degree exactly `c`, also for mixed-order
 PDEs. This uses exact order and grading, not `HasOnlyEqualOrder`, and justifies
 the degree bound in the proof of `(prop:j-character)`. -/
theorem firstPivot_magnitude :
    Multiindex.magnitude (ν.equiv (lastNonzeroColumn ν p c L.coefficients 0)) = c := by
  have hzero := zero_lt_numberOfMultiindices ν (p - c)
  obtain ⟨_, hnz, hmax⟩ := lastNonzeroColumn_spec ν L hcp hzero
  rw [shiftedPDECoefficient_ne_zero_iff, ν.map_zero] at hnz
  simp only [zero_le, tsub_zero, true_and] at hnz
  have hle := Multiindex.mem_upTo.mp hnz.1
  obtain ⟨α, hα, ha⟩ := L.nonzero
  have hmem : α ∈ multiindicesUpTo d c := Multiindex.mem_upTo.mpr hα.le
  have hbound : ν.equiv.symm α < numberOfMultiindices d p := by
    apply (enumeration_mem_iff ν).mp
    simpa only [Equiv.apply_symm_apply] using
      (show α ∈ multiindicesUpTo d p from Multiindex.mem_upTo.mpr (hα ▸ hcp))
  have hnonzero : shiftedPDECoefficient c L.coefficients (ν.equiv 0)
      (ν.equiv (ν.equiv.symm α)) ≠ 0 := by
    rw [shiftedPDECoefficient_ne_zero_iff, ν.map_zero, Equiv.apply_symm_apply]
    simpa only [zero_le, tsub_zero, true_and] using (show
      α ∈ multiindicesUpTo d c ∧ L.coefficients α ≠ 0 from ⟨hmem, ha⟩)
  have hge := magnitude_monotone ν (hmax _ hbound hnonzero)
  rw [Equiv.apply_symm_apply, hα] at hge
  exact Nat.le_antisymm hle hge

/-- The pivot part of `(prop:j-character)`: among columns of degree at most
 `p`, being a pivot is equivalent to componentwise dominance of the first
 pivot multi-index. -/
theorem mem_range_pivotColumn_iff
    (j : Fin (numberOfMultiindices d p)) :
    j ∈ Set.range (pivotColumn ν L hcp) ↔
      ν.equiv (lastNonzeroColumn ν p c L.coefficients 0) ≤ ν.equiv j.val := by
  constructor
  · rintro ⟨i, rfl⟩
    change ν.equiv (lastNonzeroColumn ν p c L.coefficients 0) ≤
      ν.equiv (lastNonzeroColumn ν p c L.coefficients i.val)
    rw [lastNonzeroColumn_shift ν L hcp i.isLt]
    exact le_add_of_nonneg_right zero_le
  · intro hle
    let t := ν.equiv (lastNonzeroColumn ν p c L.coefficients 0)
    let n := ν.equiv j.val - t
    have hadd : t + n = ν.equiv j.val := add_tsub_cancel_of_le hle
    have hn : n ∈ multiindicesUpTo d (p - c) := by
      apply Multiindex.mem_upTo.mpr
      have hj := Multiindex.mem_upTo.mp ((enumeration_mem_iff ν).mpr j.isLt)
      have hsum : c + Multiindex.magnitude n = Multiindex.magnitude (ν.equiv j.val) := by
        calc
          c + Multiindex.magnitude n = Multiindex.magnitude t + Multiindex.magnitude n := by
            rw [show Multiindex.magnitude t = c from firstPivot_magnitude ν L hcp]
          _ = Multiindex.magnitude (t + n) := by
            simp [Multiindex.magnitude, Finset.sum_add_distrib]
          _ = Multiindex.magnitude (ν.equiv j.val) := congrArg Multiindex.magnitude hadd
      omega
    let i : Fin (numberOfMultiindices d (p - c)) :=
      ⟨ν.equiv.symm n, (enumeration_mem_iff ν).mp (by simpa using hn)⟩
    refine ⟨i, ?_⟩
    apply Fin.ext
    apply ν.equiv.injective
    change ν.equiv (lastNonzeroColumn ν p c L.coefficients i.val) = ν.equiv j.val
    rw [lastNonzeroColumn_shift ν L hcp i.isLt]
    simpa only [i, Equiv.apply_symm_apply] using hadd

/-- Pointwise form of `(prop:j-character)`: a free column has some coordinate
 strictly smaller than the corresponding coordinate of the first pivot. -/
theorem freeColumn_iff_exists_lt
    (j : Fin (numberOfMultiindices d p)) :
    j ∉ Set.range (pivotColumn ν L hcp) ↔
      ∃ r : Fin d, ν.equiv j.val r <
        ν.equiv (lastNonzeroColumn ν p c L.coefficients 0) r := by
  rw [mem_range_pivotColumn_iff ν L hcp]
  change (¬ ∀ r, ν.equiv (lastNonzeroColumn ν p c L.coefficients 0) r ≤
    ν.equiv j.val r) ↔ _
  simp only [not_forall, not_le]

/-- Proposition `(prop:j-character)` as an equality of sets of multi-indices.
 The explicit hypotheses are `c ≤ p`, an exact-order operator, and a graded
 monomial ordering. No equal-order or analytic hypothesis is needed. -/
theorem freeColumn_multiindices_eq :
    Set.range (fun j : FreeColumn ν L hcp => ν.equiv j.val.val) =
      {m : Multiindex.Index d | m ∈ multiindicesUpTo d p ∧
        ∃ r : Fin d, m r < ν.equiv (lastNonzeroColumn ν p c L.coefficients 0) r} := by
  ext m
  constructor
  · rintro ⟨j, rfl⟩
    exact ⟨(enumeration_mem_iff ν).mpr j.val.isLt,
      (freeColumn_iff_exists_lt ν L hcp j.val).mp j.property⟩
  · rintro ⟨hm, hlt⟩
    let j : Fin (numberOfMultiindices d p) :=
      ⟨ν.equiv.symm m, (enumeration_mem_iff ν).mp (by simpa using hm)⟩
    have hj : j ∉ Set.range (pivotColumn ν L hcp) := by
      apply (freeColumn_iff_exists_lt ν L hcp j).mpr
      simpa only [j, Equiv.apply_symm_apply] using hlt
    exact ⟨⟨j, hj⟩, ν.equiv.apply_symm_apply m⟩

/-- Compressibility of local expansion coefficients, as in
`(eq:compressed-local-form)`: multiply the free coefficients by their
multi-index factorials, apply `M`, then divide each output by its factorial.
The factorials are cast to `ℂ` before multiplication and division.

This is an algebraic predicate on any coefficient vector, not a claim that
it arises from a particular kernel or a convergent Taylor expansion. No
analytic or equal-order hypothesis is imposed. The existing construction of
`M` uses an exact-order operator `L`, a graded monomial ordering `ν`, and
`hcp : c ≤ p`; these are the ambient decompression assumptions, made explicit
here. Indices are zero-based, and the paper's numbered subset `j` is indexed
by the actual free columns, so no choice of its enumeration is needed. -/
def IsCompressibleLocalExpansion
    (s : Fin (numberOfMultiindices d p) → ℂ) : Prop :=
  ∀ i : Fin (numberOfMultiindices d p),
    s i = (decompressionMatrix ν L hcp).mulVec
      (fun j => s j.val * (Multiindex.factorial (ν.equiv j.val.val) : ℂ)) i /
        (Multiindex.factorial (ν.equiv i.val) : ℂ)

end PDETranslations
