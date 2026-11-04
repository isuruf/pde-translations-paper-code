module

public import PDETranslations.Decompression
public import PDETranslations.SourceMoments
public import PDETranslations.Truncation
import Mathlib.Analysis.Asymptotics.Defs
import Mathlib.Data.Nat.Choose.Sum
import Mathlib.LinearAlgebra.Matrix.ToLin
import PDETranslations.DegreeLex
import PDETranslations.M2MBounds

@[expose] public section

open scoped BigOperators Classical

namespace PDETranslations

variable {d c p : ℕ}

private theorem enumeration_mem (ν : GradedMonomialOrdering d) (n p : ℕ) :
    Multiindex.magnitude (ν.equiv n) ≤ p ↔ n < numberOfMultiindices d p := by
  have h := initial_segment_eq_multiindicesUpTo ν p
  have : ν.equiv n ∈ multiindicesUpTo d p ↔ n < numberOfMultiindices d p := by
    rw [← h]
    simp
  simpa only [multiindicesUpTo, Multiindex.mem_upTo] using this

/-- The normalized monomial `v^α / α!` used for both source moments and
translation weights. Spatial coordinates are real. -/
noncomputable def normalizedMonomial (v : Fin d → ℝ) (α : Multiindex.Index d) : ℝ :=
  Multiindex.vectorPower v α / (Multiindex.factorial α : ℝ)

private theorem normalizedMonomial_eq_prod (v : Fin d → ℝ) (α : Multiindex.Index d) :
    normalizedMonomial v α = ∏ k, v k ^ α k / (Nat.factorial (α k) : ℝ) := by
  simp [normalizedMonomial, Multiindex.vectorPower, Multiindex.factorial,
    Nat.cast_prod, Finset.prod_div_distrib]

private theorem normalized_pow_add (x y : ℝ) (n : ℕ) :
    (x + y) ^ n / (n.factorial : ℝ) =
      ∑ m ∈ Finset.range (n + 1),
        (x ^ (n - m) / ((n - m).factorial : ℝ)) *
          (y ^ m / (m.factorial : ℝ)) := by
  rw [add_comm x y, add_pow, div_eq_mul_inv, Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro m hm
  have hm' : m ≤ n := Nat.lt_succ_iff.mp (Finset.mem_range.mp hm)
  have hfact : (n.choose m : ℝ) * (m.factorial : ℝ) *
      ((n - m).factorial : ℝ) = (n.factorial : ℝ) := by
    exact_mod_cast Nat.choose_mul_factorial_mul_factorial hm'
  have hn : (n.factorial : ℝ) ≠ 0 := by exact_mod_cast n.factorial_ne_zero
  have hm0 : (m.factorial : ℝ) ≠ 0 := by exact_mod_cast m.factorial_ne_zero
  have hnm : ((n - m).factorial : ℝ) ≠ 0 := by
    exact_mod_cast (n - m).factorial_ne_zero
  field_simp [hn, hm0, hnm]
  rw [← hfact]
  ring

/-- The normalized multi-index binomial theorem, indexed exactly as in the
paper's M2M derivation. The finite index `q` supplies the implicit assumption
`|ν(q)| ≤ p`, so every multi-index below it is included in the sum. -/
theorem normalizedMonomial_add (ν : GradedMonomialOrdering d) (p : ℕ)
    (q : Fin (numberOfMultiindices d p)) (h v : Fin d → ℝ) :
    normalizedMonomial (h + v) (ν.equiv q.val) =
      ∑ i : Fin (numberOfMultiindices d p),
        if ν.equiv i.val ≤ ν.equiv q.val then
          normalizedMonomial h (ν.equiv q.val - ν.equiv i.val) *
            normalizedMonomial v (ν.equiv i.val)
        else 0 := by
  let α := ν.equiv q.val
  rw [normalizedMonomial_eq_prod]
  simp_rw [Pi.add_apply, normalized_pow_add]
  rw [Finset.prod_univ_sum, ← Finset.sum_filter]
  let idx (b : Fin d → ℕ)
      (hb : b ∈ Fintype.piFinset (fun k => Finset.range (α k + 1))) :
      Fin (numberOfMultiindices d p) :=
    ⟨ν.equiv.symm (Finsupp.equivFunOnFinite.symm b), by
      apply (enumeration_mem ν _ p).mp
      rw [Equiv.apply_symm_apply]
      apply le_trans _ ((enumeration_mem ν q.val p).mpr q.isLt)
      apply Finset.sum_le_sum
      intro k _
      exact Nat.lt_succ_iff.mp (Finset.mem_range.mp (Fintype.mem_piFinset.mp hb k))⟩
  refine Finset.sum_bij idx ?_ ?_ ?_ ?_
  · intro b hb
    simp only [Finset.mem_filter, Finset.mem_univ, true_and]
    change ν.equiv (ν.equiv.symm (Finsupp.equivFunOnFinite.symm b)) ≤ α
    rw [Equiv.apply_symm_apply]
    intro k
    exact Nat.lt_succ_iff.mp (Finset.mem_range.mp (Fintype.mem_piFinset.mp hb k))
  · intro b hb b' hb' heq
    have heq' := congrArg (fun i : Fin (numberOfMultiindices d p) => ν.equiv i.val) heq
    simp only [idx, Equiv.apply_symm_apply] at heq'
    exact Finsupp.equivFunOnFinite.symm.injective heq'
  · intro i hi
    have hi' := (Finset.mem_filter.mp hi).2
    let b : Fin d → ℕ := fun k => ν.equiv i.val k
    have hb : b ∈ Fintype.piFinset (fun k => Finset.range (α k + 1)) := by
      apply Fintype.mem_piFinset.mpr
      intro k
      exact Finset.mem_range.mpr (Nat.lt_succ_iff.mpr (hi' k))
    refine ⟨b, hb, ?_⟩
    apply Fin.ext
    change ν.equiv.symm (Finsupp.equivFunOnFinite.symm b) = i.val
    have : Finsupp.equivFunOnFinite.symm b = ν.equiv i.val := by
      ext k
      rfl
    rw [this, Equiv.symm_apply_apply]
  · intro b hb
    simp only [idx, Equiv.apply_symm_apply, normalizedMonomial_eq_prod,
      Finsupp.tsub_apply, ← Finset.prod_mul_distrib]
    rfl

/-- Order-`p` source multipole coefficients about `center`, embedded in `ℂ`
to match the coefficient space of the paper's translation operator. -/
noncomputable def multipoleCoefficients (ν : GradedMonomialOrdering d) (p : ℕ)
    (center source : Fin d → ℝ) : Fin (numberOfMultiindices d p) → ℂ :=
  fun q => (normalizedMonomial (center - source) (ν.equiv q.val) : ℂ)

/-- The uncompressed M2M matrix: row `q`, column `i` is the normalized
monomial of `h` at `ν(q) - ν(i)` when `ν(i) ≤ ν(q)`, and zero otherwise. -/
noncomputable def m2mMatrix (ν : GradedMonomialOrdering d) (p : ℕ) (h : Fin d → ℝ) :
    Matrix (Fin (numberOfMultiindices d p)) (Fin (numberOfMultiindices d p)) ℂ :=
  fun q i => if ν.equiv i.val ≤ ν.equiv q.val then
    (normalizedMonomial h (ν.equiv q.val - ν.equiv i.val) : ℂ) else 0

/-- The complex-linear M2M operator from `c₁` to `c₂`. -/
noncomputable def m2mOperator (ν : GradedMonomialOrdering d) (p : ℕ)
    (c₁ c₂ : Fin d → ℝ) :
    (Fin (numberOfMultiindices d p) → ℂ) →ₗ[ℂ]
      (Fin (numberOfMultiindices d p) → ℂ) :=
  (m2mMatrix ν p (c₂ - c₁)).mulVecLin

/-- The component form of the M2M operator, for arbitrary coefficient vectors. -/
theorem m2mOperator_apply (ν : GradedMonomialOrdering d) (p : ℕ)
    (c₁ c₂ : Fin d → ℝ) (a : Fin (numberOfMultiindices d p) → ℂ)
    (q : Fin (numberOfMultiindices d p)) :
    m2mOperator ν p c₁ c₂ a q =
      ∑ i : Fin (numberOfMultiindices d p),
        if ν.equiv i.val ≤ ν.equiv q.val then
          (normalizedMonomial (c₂ - c₁) (ν.equiv q.val - ν.equiv i.val) : ℂ) * a i
        else 0 := by
  simp [m2mOperator, Matrix.mulVec, dotProduct, m2mMatrix,
    ite_mul]

/-- The M2M matrix is lower triangular in any graded monomial ordering. -/
theorem m2mMatrix_lower_triangular (ν : GradedMonomialOrdering d) (p : ℕ)
    (h : Fin d → ℝ) (q i : Fin (numberOfMultiindices d p)) (hqi : q < i) :
    m2mMatrix ν p h q i = 0 := by
  apply ite_eq_right
  intro hle
  let β := ν.equiv q.val - ν.equiv i.val
  have hadd : ν.equiv i.val + β = ν.equiv q.val := add_tsub_cancel_of_le hle
  let n := ν.equiv.symm β
  have hn : ν.equiv n = β := ν.equiv.apply_symm_apply β
  by_cases hzero : n = 0
  · have hβ : β = 0 := by simpa [hzero, ν.map_zero] using hn.symm
    have heq : i.val = q.val := ν.equiv.injective (by simpa [hβ] using hadd)
    exact (Nat.ne_of_lt hqi) heq.symm
  · have hlt := ν.add_lt_add (i := 0) (j := n) (k := i.val) (Nat.pos_of_ne_zero hzero)
    rw [ν.map_zero, zero_add, Equiv.symm_apply_apply, hn, add_comm β, hadd,
      Equiv.symm_apply_apply] at hlt
    exact (Nat.not_lt_of_ge (Nat.le_of_lt hqi)) hlt

/-- Exact source-moment translation: the binomial theorem introduces no
algebraic error. This does not assert analytic convergence of an expansion. -/
theorem m2mOperator_multipoleCoefficients (ν : GradedMonomialOrdering d) (p : ℕ)
    (c₁ c₂ source : Fin d → ℝ) :
    m2mOperator ν p c₁ c₂ (multipoleCoefficients ν p c₁ source) =
      multipoleCoefficients ν p c₂ source := by
  funext q
  rw [m2mOperator_apply]
  have h := normalizedMonomial_add ν p q (c₂ - c₁) (c₁ - source)
  have hcenters : (c₂ - c₁) + (c₁ - source) = c₂ - source := by
    ext k
    simp only [Pi.add_apply, Pi.sub_apply]
    ring
  rw [hcenters] at h
  simp only [multipoleCoefficients]
  rw [h]
  simp only [Complex.ofReal_sum, Complex.ofReal_mul, apply_ite, Complex.ofReal_zero]

private theorem magnitude_add (α β : Multiindex.Index d) :
    Multiindex.magnitude (α + β) = Multiindex.magnitude α + Multiindex.magnitude β := by
  simp [Multiindex.magnitude, Finset.sum_add_distrib]

/-- Reindex a truncated translation sum by `q = α + β`, as in the proof of
`(prop:m2m_single_coeff)`. The degree bound ensures that every shifted index
still lies in `𝓜(p)`. Natural subtraction agrees with ordinary subtraction
here because `α ≤ ν(q)` is included in the filter. -/
theorem m2m_sum_reindex (ν : GradedMonomialOrdering d) (p r : ℕ)
    (α : Multiindex.Index d) (hbound : Multiindex.magnitude α + r ≤ p)
    (h : Fin d → ℝ) (D : Multiindex.Index d → ℂ) :
    (∑ q : Fin (numberOfMultiindices d p) with
        α ≤ ν.equiv q.val ∧ Multiindex.magnitude (ν.equiv q.val) -
          Multiindex.magnitude α ≤ r,
      (normalizedMonomial h (ν.equiv q.val - α) : ℂ) * D (ν.equiv q.val)) =
    ∑ b : Fin (numberOfMultiindices d r),
      (normalizedMonomial h (ν.equiv b.val) : ℂ) * D (α + ν.equiv b.val) := by
  classical
  let idx (b : Fin (numberOfMultiindices d r)) : Fin (numberOfMultiindices d p) :=
    ⟨ν.equiv.symm (α + ν.equiv b.val), (enumeration_mem ν _ p).mp (by
      rw [Equiv.apply_symm_apply, magnitude_add]
      exact le_trans (Nat.add_le_add_left ((enumeration_mem ν _ r).mpr b.isLt) _) hbound)⟩
  symm
  apply Finset.sum_bij (fun b _ => idx b)
  · intro b _
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, idx,
      Equiv.apply_symm_apply, magnitude_add, Nat.add_sub_cancel_left]
    exact ⟨le_add_right le_rfl, (enumeration_mem ν _ r).mpr b.isLt⟩
  · intro b₁ _ b₂ _ heq
    apply Fin.ext
    apply ν.equiv.injective
    have heq' := congrArg (fun q : Fin (numberOfMultiindices d p) => ν.equiv q.val) heq
    simp only [idx, Equiv.apply_symm_apply] at heq'
    exact add_left_cancel heq'
  · intro q hq
    obtain ⟨hle, hdegree⟩ := (Finset.mem_filter.mp hq).2
    have hadd : α + (ν.equiv q.val - α) = ν.equiv q.val := add_tsub_cancel_of_le hle
    have hmag : Multiindex.magnitude (ν.equiv q.val - α) ≤ r := by
      have hm := congrArg Multiindex.magnitude hadd
      rw [magnitude_add] at hm
      omega
    let b : Fin (numberOfMultiindices d r) :=
      ⟨ν.equiv.symm (ν.equiv q.val - α), (enumeration_mem ν _ r).mp (by simpa using hmag)⟩
    refine ⟨b, Finset.mem_univ _, ?_⟩
    apply Fin.ext
    simp only [idx, b, Equiv.apply_symm_apply]
    simpa only [Equiv.symm_apply_apply] using congrArg ν.equiv.symm hadd
  · intro b _
    simp only [idx, Equiv.apply_symm_apply, add_tsub_cancel_left]

/-- Algebraic derivative-jet version of `(prop:m2m_single_coeff)`, with the
actual decompression matrix `M` and translation matrix `T`.

hkernel says that the shifted jets satisfy the differentiated PDE
relations. It is not an assumption of the conclusion: decompression of each
shifted jet follows from `decompressionMatrix_spec`. We then use the zero
entries of `M` above the row degree and reindex the finite sums exactly as in
the paper. No source-moment assumption is needed for the scalar `a`.

This algebraic theorem explicitly assumes shifted PDE relations. Each shifted
jet has all `N(p)` entries, so the hypothesis uses `jet` through degree
`p + (p - |ν(k)|)`, whereas the final identity uses only degrees at most `p`.
The analytic wrapper below derives these relations from smoothness and the
PDE on the punctured domain. All `Fin` indices are zero-based
(the paper's index is one greater). Free columns are indexed by their actual
columns, not by a chosen one-based enumeration `s`. -/
theorem m2m_single_coeff_jet (ν : GradedMonomialOrdering d)
    (L : LinearPDEOperator d c) (hcp : c ≤ p)
    (h : Fin d → ℝ) (jet : Multiindex.Index d → ℂ)
    (k : Fin (numberOfMultiindices d p)) (a : ℂ)
    (hkernel : ∀ β : Multiindex.Index d,
      Multiindex.magnitude β ≤ p - Multiindex.magnitude (ν.equiv k.val) →
      (pdeCoefficientMatrix ν p c L).mulVec
        (fun i => jet (ν.equiv i.val + β)) = 0) :
    (∑ q : Fin (numberOfMultiindices d p),
      jet (ν.equiv q.val) * (m2mMatrix ν p h).mulVec (Pi.single k a) q) =
    ∑ q : Fin (numberOfMultiindices d p),
      (∑ s : FreeColumn ν L hcp with
          ν.equiv s.val.val ≤ ν.equiv q.val ∧
          Multiindex.magnitude (ν.equiv q.val) -
            Multiindex.magnitude (ν.equiv s.val.val) ≤
              p - Multiindex.magnitude (ν.equiv k.val),
        (normalizedMonomial h (ν.equiv q.val - ν.equiv s.val.val) : ℂ) *
          (decompressionMatrix ν L hcp).transpose.mulVec (Pi.single k a) s) *
        jet (ν.equiv q.val) := by
  classical
  let r := p - Multiindex.magnitude (ν.equiv k.val)
  let M := decompressionMatrix ν L hcp
  have hk := (enumeration_mem ν k.val p).mpr k.isLt
  have hkr : Multiindex.magnitude (ν.equiv k.val) + r ≤ p := by dsimp [r]; omega
  have hsupport (s : FreeColumn ν L hcp)
      (hs : ¬ Multiindex.magnitude (ν.equiv s.val.val) ≤
        Multiindex.magnitude (ν.equiv k.val)) : M k s = 0 := by
    apply decompressionMatrix_eq_zero_of_lt
    exact ν.graded (Nat.lt_of_not_ge hs)
  have hshift (b : Fin (numberOfMultiindices d r)) :
      jet (ν.equiv k.val + ν.equiv b.val) =
        ∑ s : FreeColumn ν L hcp, M k s * jet (ν.equiv s.val.val + ν.equiv b.val) := by
    have heq := (decompressionMatrix_spec ν L hcp _).mp
      (hkernel (ν.equiv b.val) ((enumeration_mem ν _ r).mpr b.isLt))
    exact congrFun heq k
  have hleft : (∑ q : Fin (numberOfMultiindices d p),
      jet (ν.equiv q.val) * (m2mMatrix ν p h).mulVec (Pi.single k a) q) =
      a * ∑ b : Fin (numberOfMultiindices d r),
        (normalizedMonomial h (ν.equiv b.val) : ℂ) *
          jet (ν.equiv k.val + ν.equiv b.val) := by
    simp only [Matrix.mulVec_single, Pi.smul_apply, op_smul_eq_mul, Matrix.col_apply,
          m2mMatrix]
    have hfilter : (∑ q : Fin (numberOfMultiindices d p),
        jet (ν.equiv q.val) *
          ((if ν.equiv k.val ≤ ν.equiv q.val then
            (normalizedMonomial h (ν.equiv q.val - ν.equiv k.val) : ℂ) else 0) * a)) =
        a * ∑ q : Fin (numberOfMultiindices d p) with
            ν.equiv k.val ≤ ν.equiv q.val ∧
            Multiindex.magnitude (ν.equiv q.val) - Multiindex.magnitude (ν.equiv k.val) ≤ r,
          (normalizedMonomial h (ν.equiv q.val - ν.equiv k.val) : ℂ) *
            jet (ν.equiv q.val) := by
      rw [Finset.mul_sum, Finset.sum_filter]
      apply Finset.sum_congr rfl
      intro q _
      have hq := (enumeration_mem ν q.val p).mpr q.isLt
      have hdegree : Multiindex.magnitude (ν.equiv q.val) -
          Multiindex.magnitude (ν.equiv k.val) ≤ r := by dsimp [r]; omega
      by_cases hle : ν.equiv k.val ≤ ν.equiv q.val
      · simp only [hle, hdegree, and_self, ite_true]
        ring
      · simp [hle]
    rw [hfilter, m2m_sum_reindex ν p r _ hkr]
  rw [hleft]
  simp only [Matrix.mulVec_single, Pi.smul_apply, op_smul_eq_mul, Matrix.col_apply,
      Matrix.transpose_apply]
  calc
    a * (∑ b : Fin (numberOfMultiindices d r),
        (normalizedMonomial h (ν.equiv b.val) : ℂ) *
          jet (ν.equiv k.val + ν.equiv b.val)) =
        ∑ s : FreeColumn ν L hcp, (M k s * a) *
          ∑ b : Fin (numberOfMultiindices d r),
            (normalizedMonomial h (ν.equiv b.val) : ℂ) *
              jet (ν.equiv s.val.val + ν.equiv b.val) := by
      simp_rw [hshift, Finset.mul_sum]
      rw [Finset.sum_comm]
      apply Finset.sum_congr rfl
      intro s _
      apply Finset.sum_congr rfl
      intro b _
      ring
    _ = ∑ s : FreeColumn ν L hcp, (M k s * a) *
        ∑ q : Fin (numberOfMultiindices d p) with
            ν.equiv s.val.val ≤ ν.equiv q.val ∧
            Multiindex.magnitude (ν.equiv q.val) -
              Multiindex.magnitude (ν.equiv s.val.val) ≤ r,
          (normalizedMonomial h (ν.equiv q.val - ν.equiv s.val.val) : ℂ) *
            jet (ν.equiv q.val) := by
      apply Finset.sum_congr rfl
      intro s _
      by_cases hs : Multiindex.magnitude (ν.equiv s.val.val) ≤
          Multiindex.magnitude (ν.equiv k.val)
      · rw [m2m_sum_reindex ν p r _ (le_trans (Nat.add_le_add_right hs r) hkr)]
      · rw [hsupport s hs]
        simp
    _ = _ := by
      simp_rw [Finset.mul_sum, Finset.sum_filter]
      rw [Finset.sum_comm]
      apply Finset.sum_congr rfl
      intro q _
      rw [Finset.sum_mul]
      apply Finset.sum_congr rfl
      intro s _
      split_ifs <;> ring

/-- Proposition `(prop:m2m_single_coeff)` evaluated at `z = x - c₂`.
`Pi.single k (a k)` is the paper's `α_k e_k`; `transpose`, not conjugate
transpose, is used on both sides.

Smoothness and the constant-coefficient PDE away from the origin imply all
shifted PDE relations required by `m2m_single_coeff_jet`, by the open-domain
bridge in `Decompression`. We explicitly exclude the singular point `x = c₂`, and require
`c ≤ p` and the bundled exact-order PDE assumptions for construction of `M`.
The theorem asserts equality of finite expansions, not convergence or an
error estimate, and holds for arbitrary complex coefficient vectors. -/
theorem m2m_single_coeff (ν : GradedMonomialOrdering d)
    (L : LinearPDEOperator d c) (hcp : c ≤ p)
    (c₁ c₂ x : Fin d → ℝ) (G : (Fin d → ℝ) → ℂ)
    (a : Fin (numberOfMultiindices d p) → ℂ) (k : Fin (numberOfMultiindices d p))
    (hx : x - c₂ ≠ 0)
    (hG : ContDiffOn ℝ ⊤ G {z | z ≠ 0})
    (hPDE : ∀ z, z ≠ 0 → linearPDEOperator L G z = 0) :
    (∑ q : Fin (numberOfMultiindices d p),
      derivativeVector ν p G (x - c₂) q *
        m2mOperator ν p c₁ c₂ (Pi.single k (a k)) q) =
    ∑ q : Fin (numberOfMultiindices d p),
      (∑ s : FreeColumn ν L hcp with
          ν.equiv s.val.val ≤ ν.equiv q.val ∧
          Multiindex.magnitude (ν.equiv q.val) -
            Multiindex.magnitude (ν.equiv s.val.val) ≤
              p - Multiindex.magnitude (ν.equiv k.val),
        (normalizedMonomial (c₂ - c₁) (ν.equiv q.val - ν.equiv s.val.val) : ℂ) *
          (decompressionMatrix ν L hcp).transpose.mulVec (Pi.single k (a k)) s) *
        derivativeVector ν p G (x - c₂) q := by
  exact m2m_single_coeff_jet ν L hcp (c₂ - c₁)
    (fun α => Multiindex.deriv α G (x - c₂)) k (a k)
    (fun β _ => pdeCoefficientMatrix_mulVec_shifted_deriv_eq_zero ν L hcp G
      (isOpen_ne_fun continuous_id continuous_const) hG hPDE β hx)

/-- Proposition `(prop:m2m_single_coeff_equal_order)` at `z = x - c₂`.
The proof follows the paper: equal-order decompression makes all terms with
`|ν(j_s)| ≠ |ν(k)|` zero; for the remaining terms the degree cutoff is
automatic since `|ν(q)| ≤ p`.

As in `m2m_single_coeff`, the ambient analytic assumptions are explicit:
`G` is smooth and satisfies the constant-coefficient PDE off the origin,
`x ≠ c₂`, and `c ≤ p`. `L` bundles exact order `c`, while
`HasOnlyEqualOrder` excludes nonzero lower-order terms among the coefficients
used by the operator (stored coefficients above `c` are ignored).
These hypotheses are implicit in the paper's Green-function setting, not
consequences of the displayed proposition alone. No source-moment condition
on `a` is needed: the identity holds for arbitrary complex coefficients.
Indices are zero-based and free columns are indexed directly rather than
by the paper's enumeration `s`. Both transposes are ordinary, not conjugate,
and the conclusion is an exact finite-expansion identity, not convergence. -/
theorem m2m_single_coeff_equal_order (ν : GradedMonomialOrdering d)
    (L : LinearPDEOperator d c) (hcp : c ≤ p) (hequal : L.HasOnlyEqualOrder)
    (c₁ c₂ x : Fin d → ℝ) (G : (Fin d → ℝ) → ℂ)
    (a : Fin (numberOfMultiindices d p) → ℂ) (k : Fin (numberOfMultiindices d p))
    (hx : x - c₂ ≠ 0)
    (hG : ContDiffOn ℝ ⊤ G {z | z ≠ 0})
    (hPDE : ∀ z, z ≠ 0 → linearPDEOperator L G z = 0) :
    (∑ q : Fin (numberOfMultiindices d p),
      derivativeVector ν p G (x - c₂) q *
        m2mOperator ν p c₁ c₂ (Pi.single k (a k)) q) =
    ∑ q : Fin (numberOfMultiindices d p),
      (∑ s : FreeColumn ν L hcp with ν.equiv s.val.val ≤ ν.equiv q.val,
        (normalizedMonomial (c₂ - c₁) (ν.equiv q.val - ν.equiv s.val.val) : ℂ) *
          (decompressionMatrix ν L hcp).transpose.mulVec (Pi.single k (a k)) s) *
        derivativeVector ν p G (x - c₂) q := by
  rw [m2m_single_coeff ν L hcp c₁ c₂ x G a k hx hG hPDE]
  apply Finset.sum_congr rfl
  intro q _
  congr 1
  simp only [Finset.sum_filter]
  apply Finset.sum_congr rfl
  intro s _
  by_cases hdegree : Multiindex.magnitude (ν.equiv k.val) =
      Multiindex.magnitude (ν.equiv s.val.val)
  · have hq := (enumeration_mem ν q.val p).mpr q.isLt
    have hcutoff : Multiindex.magnitude (ν.equiv q.val) -
        Multiindex.magnitude (ν.equiv s.val.val) ≤
          p - Multiindex.magnitude (ν.equiv k.val) := by
      rw [← hdegree]
      exact Nat.sub_le_sub_right hq _
    simp only [hcutoff, and_true]
  · have hzero : (decompressionMatrix ν L hcp).transpose.mulVec
        (Pi.single k (a k)) s = 0 := by
      simp only [Matrix.mulVec_single, Pi.smul_apply, op_smul_eq_mul,
        Matrix.col_apply, Matrix.transpose_apply]
      rw [decompressionMatrix_equal_order_property ν L hcp hequal k s hdegree,
        zero_mul]
    simp [hzero]

/-- The paper's zero-filling matrix `E`: embed the free coordinates in the
full coefficient vector without decompressing them. Free coordinates are
indexed by actual columns rather than the paper's enumeration `s`. -/
noncomputable def multipoleEmbeddingMatrix (ν : GradedMonomialOrdering d)
    (L : LinearPDEOperator d c) (hcp : c ≤ p) :
    Matrix (Fin (numberOfMultiindices d p)) (FreeColumn ν L hcp) ℂ :=
  fun q s => if q = s.val then 1 else 0

private theorem m2m_embed_apply (ν : GradedMonomialOrdering d)
    (L : LinearPDEOperator d c) (hcp : c ≤ p) (h : Fin d → ℝ)
    (b : FreeColumn ν L hcp → ℂ) (q : Fin (numberOfMultiindices d p)) :
    (m2mMatrix ν p h).mulVec ((multipoleEmbeddingMatrix ν L hcp).mulVec b) q =
      ∑ s : FreeColumn ν L hcp with ν.equiv s.val.val ≤ ν.equiv q.val,
        (normalizedMonomial h (ν.equiv q.val - ν.equiv s.val.val) : ℂ) * b s := by
  rw [Matrix.mulVec_mulVec]
  simp [Matrix.mulVec, dotProduct, Matrix.mul_apply, multipoleEmbeddingMatrix,
    mul_ite, m2mMatrix, Finset.sum_filter, ite_mul]

-- Reindex only the tail, so that the bounds are the paper's exact integer
-- interval. `N(r) ≤ b.val` is its one-based lower bound `N(r) + 1`.
private theorem m2m_tail_reindex (ν : GradedMonomialOrdering d) (p r : ℕ)
    (α : Multiindex.Index d) (hα : Multiindex.magnitude α ≤ p)
    (h : Fin d → ℝ) (jet : Multiindex.Index d → ℂ) :
    (∑ q : Fin (numberOfMultiindices d p) with
        α ≤ ν.equiv q.val ∧ r < Multiindex.magnitude (ν.equiv q.val) -
          Multiindex.magnitude α,
      (normalizedMonomial h (ν.equiv q.val - α) : ℂ) * jet (ν.equiv q.val)) =
    ∑ b : Fin (numberOfMultiindices d p) with
        numberOfMultiindices d r ≤ b.val ∧
          b.val < numberOfMultiindices d (p - Multiindex.magnitude α),
      (normalizedMonomial h (ν.equiv b.val) : ℂ) * jet (ν.equiv b.val + α) := by
  classical
  symm
  let idx (b : Fin (numberOfMultiindices d p))
      (hb : b ∈ Finset.univ.filter (fun b : Fin (numberOfMultiindices d p) =>
        numberOfMultiindices d r ≤ b.val ∧
          b.val < numberOfMultiindices d (p - Multiindex.magnitude α))) :
      Fin (numberOfMultiindices d p) :=
    ⟨ν.equiv.symm (α + ν.equiv b.val), (enumeration_mem ν _ p).mp (by
      rw [Equiv.apply_symm_apply, magnitude_add]
      have := (enumeration_mem ν b.val (p - Multiindex.magnitude α)).mpr
        (Finset.mem_filter.mp hb).2.2
      omega)⟩
  apply Finset.sum_bij idx
  · intro b hb
    obtain ⟨hlow, hupp⟩ := (Finset.mem_filter.mp hb).2
    have hlow' : r < Multiindex.magnitude (ν.equiv b.val) := by
      have := (enumeration_mem ν b.val r)
      omega
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, idx,
      Equiv.apply_symm_apply, magnitude_add, Nat.add_sub_cancel_left]
    exact ⟨le_add_right le_rfl, hlow'⟩
  · intro b hb b' hb' heq
    apply Fin.ext
    apply ν.equiv.injective
    have := congrArg (fun q : Fin (numberOfMultiindices d p) => ν.equiv q.val) heq
    simp only [idx, Equiv.apply_symm_apply] at this
    exact add_left_cancel this
  · intro q hq
    obtain ⟨hle, htail⟩ := (Finset.mem_filter.mp hq).2
    have hadd : α + (ν.equiv q.val - α) = ν.equiv q.val := add_tsub_cancel_of_le hle
    have hm := congrArg Multiindex.magnitude hadd
    rw [magnitude_add] at hm
    have hqp := (enumeration_mem ν q.val p).mpr q.isLt
    let b : Fin (numberOfMultiindices d p) :=
      ⟨ν.equiv.symm (ν.equiv q.val - α), (enumeration_mem ν _ p).mp (by
        rw [Equiv.apply_symm_apply]; omega)⟩
    have hbν : ν.equiv b.val = ν.equiv q.val - α := Equiv.apply_symm_apply _ _
    have hb : b ∈ Finset.univ.filter (fun b : Fin (numberOfMultiindices d p) =>
        numberOfMultiindices d r ≤ b.val ∧
          b.val < numberOfMultiindices d (p - Multiindex.magnitude α)) := by
      simp only [Finset.mem_filter, Finset.mem_univ, true_and]
      constructor
      · have := enumeration_mem ν b.val r
        rw [hbν] at this
        omega
      · apply (enumeration_mem ν _ _).mp
        rw [hbν]
        omega
    refine ⟨b, hb, ?_⟩
    apply Fin.ext
    simp only [idx, b, Equiv.apply_symm_apply]
    simpa only [Equiv.symm_apply_apply] using congrArg ν.equiv.symm hadd
  · intro b hb
    simp only [idx, Equiv.apply_symm_apply, add_tsub_cancel_left]
    rw [add_comm α]

/-- The finite-expansion error `Dᵀ T E Mᵀ α - Dᵀ T α`, as a complex-linear
functional of the uncompressed coefficients. This is not error against the
true potential, and transposes are ordinary (not Hermitian) transposes. -/
noncomputable def m2mTranslationError (ν : GradedMonomialOrdering d)
    (L : LinearPDEOperator d c) (hcp : c ≤ p)
    (c₁ c₂ x : Fin d → ℝ) (G : (Fin d → ℝ) → ℂ) :
    (Fin (numberOfMultiindices d p) → ℂ) →ₗ[ℂ] ℂ :=
  let eval : (Fin (numberOfMultiindices d p) → ℂ) →ₗ[ℂ] ℂ :=
    ∑ q, derivativeVector ν p G (x - c₂) q • LinearMap.proj q
  eval.comp ((m2mOperator ν p c₁ c₂).comp
    ((multipoleEmbeddingMatrix ν L hcp).mulVecLin.comp
      (decompressionMatrix ν L hcp).transpose.mulVecLin)) -
    eval.comp (m2mOperator ν p c₁ c₂)

/-- Lemma `(lemma:m2m_error_single_coeff_varying_order)`: the compressed
translation minus the uncompressed translation is exactly the surplus tail
from free columns of strictly smaller degree. The proof subtracts
`m2m_single_coeff` and reindexes that tail; no estimate or convergence claim
is involved. `multipoleEmbeddingMatrix` is `E`, and both matrix transposes
are ordinary transposes, not conjugate transposes.

The paper leaves the Green-function/PDE setting implicit. Here `c ≤ p`,
the bundled exact-order constant-coefficient operator, smoothness and the PDE
off the origin, and `x - c₂ ≠ 0` are explicit assumptions. Smoothness is
stronger than the finite differentiability needed for this identity, as in
the preceding proposition. No source-moment assumption is needed: `a` can
be any complex coefficient vector. Nor must the PDE actually contain terms
of different orders (the equal-order case gives a zero tail).

All indices are zero-based. The inner filtered sum is precisely the paper's
one-based interval `N(p - |ν(k)|) + 1, ..., N(p - |ν(j_s)|)`; its ambient
`Fin N(p)` introduces no extra terms. Free columns are indexed directly,
without choosing the paper's enumeration `s`. -/
theorem m2m_error_single_coeff_varying_order (ν : GradedMonomialOrdering d)
    (L : LinearPDEOperator d c) (hcp : c ≤ p)
    (c₁ c₂ x : Fin d → ℝ) (G : (Fin d → ℝ) → ℂ)
    (a : Fin (numberOfMultiindices d p) → ℂ) (k : Fin (numberOfMultiindices d p))
    (hx : x - c₂ ≠ 0)
    (hG : ContDiffOn ℝ ⊤ G {z | z ≠ 0})
    (hPDE : ∀ z, z ≠ 0 → linearPDEOperator L G z = 0) :
    m2mTranslationError ν L hcp c₁ c₂ x G (Pi.single k (a k)) =
    ∑ s : FreeColumn ν L hcp with
        Multiindex.magnitude (ν.equiv s.val.val) < Multiindex.magnitude (ν.equiv k.val),
      (decompressionMatrix ν L hcp) k s * a k *
        ∑ b : Fin (numberOfMultiindices d p) with
            numberOfMultiindices d (p - Multiindex.magnitude (ν.equiv k.val)) ≤ b.val ∧
              b.val < numberOfMultiindices d (p - Multiindex.magnitude (ν.equiv s.val.val)),
          (normalizedMonomial (c₂ - c₁) (ν.equiv b.val) : ℂ) *
            Multiindex.deriv (ν.equiv b.val + ν.equiv s.val.val) G (x - c₂) := by
  classical
  simp only [m2mTranslationError, LinearMap.sub_apply, LinearMap.comp_apply,
    LinearMap.sum_apply, LinearMap.smul_apply, LinearMap.proj_apply,
    smul_eq_mul, Matrix.mulVecLin_apply]
  rw [m2m_single_coeff ν L hcp c₁ c₂ x G a k hx hG hPDE]
  simp only [m2mOperator, Matrix.mulVecLin_apply, m2m_embed_apply,
    Matrix.mulVec_single, Pi.smul_apply, op_smul_eq_mul, Matrix.col_apply,
    Matrix.transpose_apply]
  let r := p - Multiindex.magnitude (ν.equiv k.val)
  trans ∑ s : FreeColumn ν L hcp,
      (decompressionMatrix ν L hcp) k s * a k *
        ∑ q : Fin (numberOfMultiindices d p) with
            ν.equiv s.val.val ≤ ν.equiv q.val ∧
              r < Multiindex.magnitude (ν.equiv q.val) -
                Multiindex.magnitude (ν.equiv s.val.val),
          (normalizedMonomial (c₂ - c₁) (ν.equiv q.val - ν.equiv s.val.val) : ℂ) *
            derivativeVector ν p G (x - c₂) q
  · simp_rw [Finset.sum_filter, Finset.sum_mul, Finset.mul_sum]
    conv_rhs => rw [Finset.sum_comm]
    rw [← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro q _
    rw [← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro s _
    dsimp [r]
    by_cases hle : ν.equiv s.val.val ≤ ν.equiv q.val <;>
      by_cases hdegree : Multiindex.magnitude (ν.equiv q.val) -
          Multiindex.magnitude (ν.equiv s.val.val) ≤
            p - Multiindex.magnitude (ν.equiv k.val) <;>
      simp only [← not_le, hle, hdegree, not_true_eq_false, not_false_eq_true,
        and_true, and_false, ite_true, ite_false] <;> ring
  · rw [Finset.sum_filter]
    apply Finset.sum_congr rfl
    intro s _
    have hs := (enumeration_mem ν s.val.val p).mpr s.val.isLt
    by_cases hsk : Multiindex.magnitude (ν.equiv s.val.val) <
        Multiindex.magnitude (ν.equiv k.val)
    · simp only [hsk, ite_true]
      rw [show (∑ q : Fin (numberOfMultiindices d p) with
          ν.equiv s.val.val ≤ ν.equiv q.val ∧
            r < Multiindex.magnitude (ν.equiv q.val) - Multiindex.magnitude (ν.equiv s.val.val),
          (normalizedMonomial (c₂ - c₁) (ν.equiv q.val - ν.equiv s.val.val) : ℂ) *
            derivativeVector ν p G (x - c₂) q) = _ from
        m2m_tail_reindex ν p r _ hs (c₂ - c₁)
          (fun α => Multiindex.deriv α G (x - c₂))]
    · simp only [hsk, ite_false]
      by_cases heq : Multiindex.magnitude (ν.equiv s.val.val) =
          Multiindex.magnitude (ν.equiv k.val)
      · have hempty : ∀ q : Fin (numberOfMultiindices d p),
            ¬ (ν.equiv s.val.val ≤ ν.equiv q.val ∧
              r < Multiindex.magnitude (ν.equiv q.val) - Multiindex.magnitude (ν.equiv s.val.val)) := by
          intro q
          have hq := (enumeration_mem ν q.val p).mpr q.isLt
          dsimp [r]
          rw [heq]
          omega
        simp [hempty]
      · have hzero : decompressionMatrix ν L hcp k s = 0 := by
          apply decompressionMatrix_eq_zero_of_lt
          apply ν.graded
          omega
        simp [hzero]


/-- Theorem `(thm:varying)` for a fixed PDE and finite complex point sources.

The decay exponent `p'` is the same explicit exponent supplied to
`SatisfiesDerivativeEstimate`; only the prefactor is existentially quantified.
The existential constant is chosen before the centers, source locations, radius,
and target, so it is uniform in all of them. It may depend on the fixed upper
radius `a`, as well as the PDE, order, kernel and source strengths. The paper's
`K'` condition is vacuous for a fixed finite matrix: its entries and
`max 1 (a^p)` are absorbed directly into the constant, with no parameter-uniform
claim. Dependence on `a` is implicit in the paper's choice of `K'`.

We explicitly require the earlier derivative estimate `(assumption_g)`, smoothness
and the constant-coefficient PDE off the origin, `c ≤ p`, and a nonnegative
source radius. Smoothness is stronger than finite differentiability needed here.
Euclidean vectors ensure the distances are 2-norms. Closed source balls and
complex strengths are allowed. We drop reciprocal factorials and use finite
sums instead of `e^d` and `p |j|`; this changes only the unspecified prefactor,
not the exponent or the uniformity in `x,R₂`. There is no Taylor reconstruction
hypothesis: this compares two finite expansions, not either one to the potential. -/
theorem m2m_error_varying_order {ι : Type*} [Fintype ι]
    (ν : GradedMonomialOrdering d) (L : LinearPDEOperator d c) (hcp : c ≤ p)
    (G : (Fin d → ℝ) → ℂ) {p' : ℝ} {s : ℕ → ℕ}
    (hDerivative : SatisfiesDerivativeEstimate G p' s)
    (hG : ContDiffOn ℝ ⊤ G {z | z ≠ 0})
    (hPDE : ∀ z, z ≠ 0 → linearPDEOperator L G z = 0)
    (w : ι → ℂ) (a : ℝ) (ha : 0 < a) :
    ∃ B : ℝ, 0 ≤ B ∧
      ∀ (c₁ c₂ : EuclideanSpace ℝ (Fin d))
        (source : ι → EuclideanSpace ℝ (Fin d)) (R₁ : ℝ),
        0 ≤ R₁ → (∀ i, ‖source i - c₁‖ ≤ R₁) →
        ∀ x : EuclideanSpace ℝ (Fin d),
          R₁ + ‖c₂ - c₁‖ < ‖x - c₂‖ → ‖x - c₂‖ ≤ a →
          ‖m2mTranslationError ν L hcp c₁ c₂ x G
            (finiteSourceMoments ν p c₁ source w)‖ ≤
            B / ‖x - c₂‖ ^ p' *
              ((R₁ + ‖c₂ - c₁‖) / ‖x - c₂‖) ^ (p + 1) := by
  classical
  obtain ⟨C, hC, hbound⟩ := hDerivative p
  let A : ℝ := ∑ i, ‖w i‖
  let D := fun (k : Fin (numberOfMultiindices d p)) (j : FreeColumn ν L hcp) =>
    ‖decompressionMatrix ν L hcp k j‖ * (p+1 : ℝ) * C * (p+1 : ℝ)^(s p) * max 1 (a^p)
  refine ⟨A * ∑ k, ∑ j, D k j, by dsimp [A, D]; positivity, ?_⟩
  intro c₁ c₂ source R₁ hR₁ hsource x hRx hxa
  let R := R₁ + ‖c₂ - c₁‖
  let z := x - c₂
  let h := c₂ - c₁
  let α := finiteSourceMoments ν p c₁ source w
  have hR : 0 ≤ R := by dsimp [R]; positivity
  have hz : 0 < ‖z‖ := lt_of_le_of_lt hR hRx
  have hx : (x : Fin d → ℝ) - c₂ ≠ 0 := by
    intro he
    have : z = 0 := by ext i; exact congrFun he i
    simp [this] at hz
  have hsingle (k : Fin (numberOfMultiindices d p)) :
      ‖m2mTranslationError ν L hcp c₁ c₂ x G (Pi.single k (α k))‖ ≤
        A * (∑ j, D k j) * (R / ‖z‖)^(p+1) / ‖z‖^p' := by
    rw [m2m_error_single_coeff_varying_order ν L hcp c₁ c₂ x G α k hx hG hPDE]
    calc
      _ ≤ ∑ j ∈ Finset.univ.filter (fun j : FreeColumn ν L hcp =>
          Multiindex.magnitude (ν.equiv j.val.val) < Multiindex.magnitude (ν.equiv k.val)),
          ‖decompressionMatrix ν L hcp k j * α k *
            ∑ b : Fin (numberOfMultiindices d p) with
              numberOfMultiindices d (p - Multiindex.magnitude (ν.equiv k.val)) ≤ b.val ∧
                b.val < numberOfMultiindices d (p - Multiindex.magnitude (ν.equiv j.val.val)),
              (normalizedMonomial (c₂ - c₁) (ν.equiv b.val) : ℂ) *
                Multiindex.deriv (ν.equiv b.val + ν.equiv j.val.val) G (x - c₂)‖ :=
          norm_sum_le _ _
      _ ≤ ∑ j ∈ Finset.univ.filter (fun j : FreeColumn ν L hcp =>
          Multiindex.magnitude (ν.equiv j.val.val) < Multiindex.magnitude (ν.equiv k.val)),
          A * D k j * (R / ‖z‖)^(p+1) / ‖z‖^p' := by
        apply Finset.sum_le_sum
        intro j hj
        have hjk := (Finset.mem_filter.mp hj).2
        have hkp := (enumeration_mem ν k.val p).mpr k.isLt
        have hjp := (enumeration_mem ν j.val.val p).mpr j.val.isLt
        let F := fun q : Multiindex.Index d =>
          (normalizedMonomial h q : ℂ) * Multiindex.deriv (q + ν.equiv j.val.val) G z
        have hlayer (n : ℕ) (hn : 0 < n) :
            ‖∑ q ∈ Multiindex.multiindicesOfMagnitude n, F q‖ ≤
              C * (1+n : ℝ)^(s p) / ‖z‖^((Multiindex.magnitude (ν.equiv j.val.val) : ℝ)+p') *
                (‖h‖/‖z‖)^n := by
          have hb := hbound (ν.equiv j.val.val) (Multiindex.mem_upTo.mpr hjp)
            n z h ⟨norm_pos_iff.mp hz, hn⟩
          simpa [F, normalizedMonomial, Complex.ofReal_div, Complex.ofReal_mul,
            mul_div_assoc, div_mul_eq_mul_div, mul_comm, mul_left_comm, mul_assoc] using hb
        have hb := m2m_weighted_tail_bound ν p _ _ hjk hkp h z R R₁ a C
          (decompressionMatrix ν L hcp k j) (s p) p' hR hR₁
          (by dsimp [R]; linarith [norm_nonneg (c₂-c₁)]) ha
          (by dsimp [h, R]; linarith) hRx hxa hC F hlayer
        have hα := norm_finiteSourceMoments_le ν p c₁ source w hR₁ hsource k
        simp only [norm_mul]
        change ‖decompressionMatrix ν L hcp k j‖ * ‖α k‖ *
          ‖∑ b : Fin (numberOfMultiindices d p) with
            numberOfMultiindices d (p - Multiindex.magnitude (ν.equiv k.val)) ≤ b.val ∧
              b.val < numberOfMultiindices d (p - Multiindex.magnitude (ν.equiv j.val.val)),
            F (ν.equiv b.val)‖ ≤ _
        calc
          _ ≤ A * (‖decompressionMatrix ν L hcp k j‖ * R₁^Multiindex.magnitude
              (ν.equiv k.val) * ‖∑ b : Fin (numberOfMultiindices d p) with
                numberOfMultiindices d (p - Multiindex.magnitude (ν.equiv k.val)) ≤ b.val ∧
                  b.val < numberOfMultiindices d (p - Multiindex.magnitude (ν.equiv j.val.val)),
                F (ν.equiv b.val)‖) := by
            dsimp [α, A] at *
            nlinarith [mul_le_mul_of_nonneg_left hα (show 0 ≤
              ‖decompressionMatrix ν L hcp k j‖ * ‖∑ b : Fin (numberOfMultiindices d p) with
                numberOfMultiindices d (p - Multiindex.magnitude (ν.equiv k.val)) ≤ b.val ∧
                  b.val < numberOfMultiindices d (p - Multiindex.magnitude (ν.equiv j.val.val)),
                F (ν.equiv b.val)‖ by positivity)]
          _ ≤ A * (D k j * (R / ‖z‖)^(p+1) / ‖z‖^p') :=
            mul_le_mul_of_nonneg_left hb (by dsimp [A]; positivity)
          _ = _ := by ring
      _ ≤ ∑ j : FreeColumn ν L hcp, A * D k j * (R / ‖z‖)^(p+1) / ‖z‖^p' := by
        apply Finset.sum_le_sum_of_subset_of_nonneg (Finset.filter_subset _ _)
        intro j _ _
        dsimp [A, D]
        positivity
      _ = _ := by rw [← Finset.sum_div, ← Finset.sum_mul, ← Finset.mul_sum]
  have hdecomp : α = ∑ k, Pi.single k (α k) := by
    ext k
    simp
  change ‖m2mTranslationError ν L hcp c₁ c₂ x G α‖ ≤ _
  rw [hdecomp, map_sum]
  calc
    _ ≤ ∑ k, ‖m2mTranslationError ν L hcp c₁ c₂ x G (Pi.single k (α k))‖ :=
      norm_sum_le _ _
    _ ≤ ∑ k, A * (∑ j, D k j) * (R / ‖z‖)^(p+1) / ‖z‖^p' :=
      Finset.sum_le_sum (fun k _ => hsingle k)
    _ = _ := by rw [← Finset.sum_div, ← Finset.sum_mul, ← Finset.mul_sum]; ring

end PDETranslations
