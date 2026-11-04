module

public import PDETranslations.Decompression
public import PDETranslations.DegreeLex
import Mathlib.Data.Nat.Choose.Sum
import Mathlib.LinearAlgebra.Matrix.ToLin

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

private theorem magnitude_single (k : Fin d) (n : ℕ) :
    Multiindex.magnitude (Finsupp.single k n) = n := by
  simp [Multiindex.magnitude]
  
/-- For the preferred degree-lex ordering, the first pivot is the nonzero
pure power in coordinate `k`. This is the paper's assertion about `h(1)`. -/
theorem firstPivot_purePower (L : LinearPDEOperator d c) (hcp : c ≤ p)
    (k : Fin d) (hk : L.coefficients (Finsupp.single k c) ≠ 0) :
    (degreeLexOrdering k).equiv
      (lastNonzeroColumn (degreeLexOrdering k) p c L.coefficients 0) =
      Finsupp.single k c := by
  let ν := degreeLexOrdering k
  have hzero : 0 < numberOfMultiindices d (p - c) := by
    apply (enumeration_mem ν 0 (p - c)).mp
    simp [ν.map_zero, Multiindex.magnitude]
  obtain ⟨hb, hn, hm⟩ := lastNonzeroColumn_spec ν L hcp hzero
  have hn' : Multiindex.magnitude
      (ν.equiv (lastNonzeroColumn ν p c L.coefficients 0)) ≤ c := by
    unfold shiftedPDECoefficient at hn
    rw [ν.map_zero] at hn
    split_ifs at hn with h
    · simpa only [tsub_zero, multiindicesUpTo, Multiindex.mem_upTo] using h.2
    · exact (hn rfl).elim
  have hle := degreeLexOrdering_purePower_maximal k c _ hn'
  change ν.equiv.symm (ν.equiv (lastNonzeroColumn ν p c L.coefficients 0)) ≤
    ν.equiv.symm (Finsupp.single k c) at hle
  rw [Equiv.symm_apply_apply] at hle
  have hbound : ν.equiv.symm (Finsupp.single k c) < numberOfMultiindices d p := by
    apply (enumeration_mem ν _ p).mp
    simpa only [Equiv.apply_symm_apply, magnitude_single] using hcp
  have hnz : shiftedPDECoefficient c L.coefficients (ν.equiv 0)
      (ν.equiv (ν.equiv.symm (Finsupp.single k c))) ≠ 0 := by
    simpa [shiftedPDECoefficient, ν.map_zero, multiindicesUpTo,
      Multiindex.mem_upTo, magnitude_single] using hk
  have heq := Nat.le_antisymm hle (hm _ hbound hnz)
  change ν.equiv _ = _
  rw [heq, Equiv.apply_symm_apply]

/-- The preferred pure power is the last entry of `𝓜(c)`. Lean's index is
`N(c) - 1`, corresponding to the paper's one-based `N(c)`. -/
theorem purePower_lastIndex (k : Fin d) (c : ℕ) :
    (degreeLexOrdering k).equiv.symm (Finsupp.single k c) + 1 =
      numberOfMultiindices d c := by
  let ν := degreeLexOrdering k
  have hbound : ν.equiv.symm (Finsupp.single k c) < numberOfMultiindices d c := by
    apply (enumeration_mem ν _ c).mp
    simp only [Equiv.apply_symm_apply, magnitude_single, le_refl]
  have hlast : numberOfMultiindices d c - 1 < numberOfMultiindices d c := by omega
  have hmag := (enumeration_mem ν _ c).mpr hlast
  have hmax := degreeLexOrdering_purePower_maximal k c _ hmag
  change ν.equiv.symm (ν.equiv (numberOfMultiindices d c - 1)) ≤
    ν.equiv.symm (Finsupp.single k c) at hmax
  rw [Equiv.symm_apply_apply] at hmax
  dsimp only [ν] at hbound hmax
  omega

/-- The pivot multi-indices are the translates by the preferred pure power. -/
theorem pivotColumn_purePower_shift (L : LinearPDEOperator d c) (hcp : c ≤ p)
    (k : Fin d) (hk : L.coefficients (Finsupp.single k c) ≠ 0)
    (i : Fin (numberOfMultiindices d (p - c))) :
    (degreeLexOrdering k).equiv (pivotColumn (degreeLexOrdering k) L hcp i).val =
      Finsupp.single k c + (degreeLexOrdering k).equiv i.val := by
  exact (lastNonzeroColumn_shift (degreeLexOrdering k) L hcp i.isLt).trans
    (congrArg (fun α => α + (degreeLexOrdering k).equiv i.val)
      (firstPivot_purePower L hcp k hk))

/-- Exactly the columns whose preferred coordinate is at least `c` are pivots.
This strengthens the inclusion asserted in the paper's proof. -/
theorem pivotColumn_range_iff (L : LinearPDEOperator d c) (hcp : c ≤ p)
    (k : Fin d) (hk : L.coefficients (Finsupp.single k c) ≠ 0)
    (j : Fin (numberOfMultiindices d p)) :
    j ∈ Set.range (pivotColumn (degreeLexOrdering k) L hcp) ↔
      c ≤ (degreeLexOrdering k).equiv j.val k := by
  let ν := degreeLexOrdering k
  have hshift (i : Fin (numberOfMultiindices d (p - c))) :
      ν.equiv (pivotColumn ν L hcp i).val = Finsupp.single k c + ν.equiv i.val := by
    exact pivotColumn_purePower_shift L hcp k hk i
  constructor
  · rintro ⟨i, rfl⟩
    rw [hshift]
    simp
  · intro hj
    let α := ν.equiv j.val - Finsupp.single k c
    have hle : Finsupp.single k c ≤ ν.equiv j.val := by
      intro i
      by_cases hi : i = k
      · subst i; simpa using hj
      · simp [hi]
    have hadd : Finsupp.single k c + α = ν.equiv j.val := by
      exact add_tsub_cancel_of_le hle
    have hmag : Multiindex.magnitude α ≤ p - c := by
      have hsum := congrArg Multiindex.magnitude hadd
      have hbound := (enumeration_mem ν j.val p).mpr j.isLt
      simp only [Multiindex.magnitude, Finsupp.add_apply, Finset.sum_add_distrib] at hsum
      have hc := magnitude_single k c
      unfold Multiindex.magnitude at hc hbound ⊢
      omega
    let i : Fin (numberOfMultiindices d (p - c)) :=
      ⟨ν.equiv.symm α, (enumeration_mem ν _ _).mp (by simpa using hmag)⟩
    refine ⟨i, ?_⟩
    apply Fin.ext
    apply ν.equiv.injective
    rw [hshift]
    exact (congrArg (fun β => Finsupp.single k c + β)
      (ν.equiv.apply_symm_apply α)).trans hadd

/-- Stored columns are precisely the multi-indices with preferred coordinate
less than the PDE order. In particular, none remain when `c = 0`. -/
theorem freeColumn_iff (L : LinearPDEOperator d c) (hcp : c ≤ p)
    (k : Fin d) (hk : L.coefficients (Finsupp.single k c) ≠ 0)
    (j : Fin (numberOfMultiindices d p)) :
    j ∉ Set.range (pivotColumn (degreeLexOrdering k) L hcp) ↔
      (degreeLexOrdering k).equiv j.val k < c := by
  rw [pivotColumn_range_iff L hcp k hk j, not_le]

/-- The maximum stored exponent in a coordinate, with the empty maximum
assigned zero. The columns are the actual `FreeColumn`s from decompression. -/
noncomputable def storedCoordinateMax (ν : GradedMonomialOrdering d)
    (L : LinearPDEOperator d c) (hcp : c ≤ p) (i : Fin d) : ℕ :=
  Finset.univ.sup fun j : FreeColumn ν L hcp => ν.equiv j.val.val i

/-- The product in `(eq:max_prod_nice)`, extended to all expansion orders using
`max p c`. For `p ≥ c` this is exactly the product over stored columns. -/
noncomputable def storedMaxProduct (ν : GradedMonomialOrdering d)
    (L : LinearPDEOperator d c) (p : ℕ) : ℕ :=
  ∏ i : Fin d, storedCoordinateMax ν L (Nat.le_max_right p c) i

/-- All stored coordinate exponents are bounded by the expansion order. -/
theorem storedCoordinateMax_le_order (ν : GradedMonomialOrdering d)
    (L : LinearPDEOperator d c) (hcp : c ≤ p) (i : Fin d) :
    storedCoordinateMax ν L hcp i ≤ p := by
  apply Finset.sup_le
  intro j _
  exact le_trans
    (Finset.single_le_sum (fun _ _ => Nat.zero_le _) (Finset.mem_univ i))
    ((enumeration_mem ν j.val.val p).mpr j.val.isLt)

/-- The preferred coordinate is bounded independently of expansion order.
The inequality includes order zero under the empty-maximum convention. -/
theorem storedCoordinateMax_preferred_le (L : LinearPDEOperator d c)
    (hcp : c ≤ p) (k : Fin d) (hk : L.coefficients (Finsupp.single k c) ≠ 0) :
    storedCoordinateMax (degreeLexOrdering k) L hcp k ≤ c - 1 := by
  apply Finset.sup_le
  intro j _
  have h := (freeColumn_iff L hcp k hk j.val).mp j.property
  omega

/-- The preferred maximum is exactly `c - 1` for positive order. With natural
subtraction and our empty-maximum convention the identity also holds at zero. -/
theorem storedCoordinateMax_preferred_eq (L : LinearPDEOperator d c)
    (hcp : c ≤ p) (k : Fin d) (hk : L.coefficients (Finsupp.single k c) ≠ 0) :
    storedCoordinateMax (degreeLexOrdering k) L hcp k = c - 1 := by
  apply Nat.le_antisymm (storedCoordinateMax_preferred_le L hcp k hk)
  by_cases hc : c = 0
  · simp [hc]
  · let ν := degreeLexOrdering k
    have hbound : ν.equiv.symm (Finsupp.single k (c - 1)) < numberOfMultiindices d p := by
      apply (enumeration_mem ν _ p).mp
      simp only [Equiv.apply_symm_apply, magnitude_single]
      omega
    let j : Fin (numberOfMultiindices d p) :=
      ⟨ν.equiv.symm (Finsupp.single k (c - 1)), hbound⟩
    have hj : j ∉ Set.range (pivotColumn ν L hcp) := by
      apply (freeColumn_iff L hcp k hk j).mpr
      change ν.equiv (ν.equiv.symm (Finsupp.single k (c - 1))) k < c
      rw [Equiv.apply_symm_apply, Finsupp.single_eq_same]
      omega
    have h := Finset.le_sup (f := fun s : FreeColumn ν L hcp => ν.equiv s.val.val k)
      (Finset.mem_univ (⟨j, hj⟩ : FreeColumn ν L hcp))
    simpa only [storedCoordinateMax, ν, j, Equiv.apply_symm_apply,
      Finsupp.single_eq_same] using h

/-- An explicit bound underlying `(lemma:max_prod_nice)`. -/
theorem storedMaxProduct_le (L : LinearPDEOperator d c)
    (k : Fin d) (hk : L.coefficients (Finsupp.single k c) ≠ 0) (p : ℕ) :
    storedMaxProduct (degreeLexOrdering k) L p ≤ (c - 1) * (max p c) ^ (d - 1) := by
  classical
  unfold storedMaxProduct
  rw [← Finset.mul_prod_erase _ _ (Finset.mem_univ k)]
  apply Nat.mul_le_mul (storedCoordinateMax_preferred_le L _ k hk)
  calc
    _ ≤ ∏ _i ∈ Finset.univ.erase k, max p c :=
      Finset.prod_le_prod fun i _ => storedCoordinateMax_le_order _ L _ i
    _ = (max p c) ^ (d - 1) := by simp

/-- The paper's asymptotic FFT-size bound. The ordering is chosen once for the
PDE, independently of `p`; the implicit constant can depend on its order and
dimension. Natural maxima are cast to reals for Mathlib's `IsBigO`. -/
theorem exists_ordering_storedMaxProduct_isBigO (L : LinearPDEOperator d c)
    (hnice : L.HasNonzeroPurePower) :
    ∃ ν : GradedMonomialOrdering d,
      Asymptotics.IsBigO Filter.atTop (fun p : ℕ => (storedMaxProduct ν L p : ℝ))
        (fun p : ℕ => (p : ℝ) ^ (d - 1)) := by
  obtain ⟨k, hk⟩ := hnice
  refine ⟨degreeLexOrdering k, Asymptotics.isBigO_iff.mpr ⟨(c - 1 : ℕ), ?_⟩⟩
  filter_upwards [Filter.eventually_ge_atTop c] with p hp
  have h := storedMaxProduct_le L k hk p
  rw [max_eq_left hp] at h

  rw [Real.norm_eq_abs, Real.norm_eq_abs,
    abs_of_nonneg (Nat.cast_nonneg (storedMaxProduct (degreeLexOrdering k) L p)),
    abs_of_nonneg (pow_nonneg (Nat.cast_nonneg p) (d - 1))]
  exact_mod_cast h

/-- The M2L local coefficients in `(eq:m2l_toeplitz)`, for compressed
multipole coefficients `β` and displacement `z = c₂ - c₁`. The sum is bilinear,
not a Hermitian inner product: there is no complex conjugation.

We use the combined derivative `∂^(ν(i) + ν(j)) G z`. For a smooth kernel this
is the iterated derivative in the right-hand expression of the paper's
formula, by `Multiindex.deriv_add` and commutation of mixed derivatives.
This definition specifies the coefficient formula directly; it does not
assert the preceding identity obtained by differentiating the translated
multipole expansion, or convergence of that expansion. As with `FreeColumn`,
free indices are actual columns rather than a chosen numbering. -/
noncomputable def m2lLocalCoefficients
    (ν : GradedMonomialOrdering d)
    (L : LinearPDEOperator d c) (hcp : c ≤ p)
    (G : (Fin d → ℝ) → ℂ) (β : FreeColumn ν L hcp → ℂ)
    (z : Fin d → ℝ) : Fin (numberOfMultiindices d p) → ℂ :=
  fun i => (∑ j, β j * Multiindex.deriv (ν.equiv i.val + ν.equiv j.val.val) G z) /
    (Multiindex.factorial (ν.equiv i.val) : ℂ)

/-- Compressibility of the M2L result, `(prop:m2l_error)`. This holds for
arbitrary compressed coefficients `β`, hence in particular for the paper's
`β = Mᵀ α`; no assumption that `α` comes from actual sources is needed.

The kernel must solve the constant-coefficient PDE on an open domain
containing the displacement `z`. Smoothness there justifies differentiated
PDE identities and mixed-derivative commutation; it is stronger than the
finite differentiability needed for this finite computation. For a kernel
singular at zero, take the domain `{z | z ≠ 0}` and require `c₂ - c₁ ≠ 0`.
The other hypotheses are the ambient exact-order and `c ≤ p` assumptions.
No equal-order or geometric separation condition is needed for this identity.

The conclusion is exact recovery of every coefficient from the free ones,
so compression adds no coefficient error in exact arithmetic. It does not
bound floating-point error or establish either Taylor truncation estimate.
Unlike the paper's chain of differentiated vector identities, the proof
reconstructs each shifted jet using its PDE kernel relation and then sums. -/
theorem m2lLocalCoefficients_isCompressible
    (ν : GradedMonomialOrdering d)
    (L : LinearPDEOperator d c) (hcp : c ≤ p)
    (G : (Fin d → ℝ) → ℂ) {s : Set (Fin d → ℝ)} (hs : IsOpen s)
    (hG : ContDiffOn ℝ ⊤ G s)
    (hPDE : ∀ x ∈ s, linearPDEOperator L G x = 0)
    (β : FreeColumn ν L hcp → ℂ) {z : Fin d → ℝ} (hz : z ∈ s) :
    IsCompressibleLocalExpansion ν L hcp (m2lLocalCoefficients ν L hcp G β z) := by
  classical
  let jet : Fin (numberOfMultiindices d p) → ℂ :=
    fun i => ∑ j, β j * Multiindex.deriv (ν.equiv i.val + ν.equiv j.val.val) G z
  have hshift (j : FreeColumn ν L hcp) :=
    (decompressionMatrix_spec ν L hcp _).mp
      (pdeCoefficientMatrix_mulVec_shifted_deriv_eq_zero ν L hcp G hs hG hPDE
        (ν.equiv j.val.val) hz)
  have hjet : jet = (decompressionMatrix ν L hcp).mulVec (fun j => jet j.val) := by
    funext i
    calc
      jet i = ∑ j : FreeColumn ν L hcp, β j *
          (decompressionMatrix ν L hcp).mulVec
            (fun k => Multiindex.deriv (ν.equiv k.val.val + ν.equiv j.val.val) G z) i := by
        apply Finset.sum_congr rfl
        intro j _
        exact congrArg (fun v : Fin (numberOfMultiindices d p) → ℂ => β j * v i)
          (hshift j)
      _ = (decompressionMatrix ν L hcp).mulVec (fun j => jet j.val) i := by
        simp only [Matrix.mulVec, dotProduct, jet, Finset.mul_sum]
        rw [Finset.sum_comm]
        apply Finset.sum_congr rfl
        intro k _
        apply Finset.sum_congr rfl
        intro j _
        ring
  have hfactorial (α : Multiindex.Index d) : (Multiindex.factorial α : ℂ) ≠ 0 := by
    exact_mod_cast (Finset.prod_ne_zero_iff.mpr
      (fun k _ => Nat.factorial_ne_zero (α k)))
  intro i
  change jet i / (Multiindex.factorial (ν.equiv i.val) : ℂ) =
    (decompressionMatrix ν L hcp).mulVec
      (fun j => (jet j.val / (Multiindex.factorial (ν.equiv j.val.val) : ℂ)) *
        (Multiindex.factorial (ν.equiv j.val.val) : ℂ)) i /
      (Multiindex.factorial (ν.equiv i.val) : ℂ)
  simp_rw [div_mul_cancel₀ _ (hfactorial _)]
  exact congrArg (fun v : Fin (numberOfMultiindices d p) → ℂ =>
    v i / (Multiindex.factorial (ν.equiv i.val) : ℂ)) hjet

end PDETranslations
