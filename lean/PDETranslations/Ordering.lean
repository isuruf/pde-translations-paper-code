module

public import PDETranslations.Multiindex

@[expose] public section

open scoped BigOperators

namespace PDETranslations

/-- The multi-indices in `d` variables with total degree at most `p`. -/
noncomputable def multiindicesUpTo (d p : ℕ) : Finset (Multiindex.Index d) :=
  Multiindex.upTo p

/-- The number of multi-indices in `d` variables with total degree at most `p`. -/
noncomputable def numberOfMultiindices (d p : ℕ) : ℕ :=
  (multiindicesUpTo d p).card

/-- A graded monomial ordering enumerates multi-indices in nondecreasing total
 degree and is compatible with addition.  Lean indices the enumeration from
 zero; this corresponds to the paper's one-based indexing after adding one. -/
structure GradedMonomialOrdering (d : ℕ) where
  equiv : ℕ ≃ Multiindex.Index d
  map_zero : equiv 0 = 0
  graded : ∀ ⦃i j : ℕ⦄,
    Multiindex.magnitude (equiv i) < Multiindex.magnitude (equiv j) → i < j
  add_lt_add : ∀ ⦃i j k : ℕ⦄, i < j →
    equiv.symm (equiv i + equiv k) < equiv.symm (equiv j + equiv k)

variable {d : ℕ} (ν : GradedMonomialOrdering d)

private theorem mem_multiindicesUpTo_iff {α : Multiindex.Index d} {p : ℕ} :
    α ∈ multiindicesUpTo d p ↔ Multiindex.magnitude α ≤ p := by
  constructor
  · intro h
    have h' := h
    simp [multiindicesUpTo, Multiindex.upTo] at h'
    exact h'.2
  · intro h
    simp only [multiindicesUpTo, Multiindex.upTo, Finset.mem_filter,
      Finset.mem_image, Finset.mem_pi, Finset.mem_univ]
    refine ⟨?_, h⟩
    refine ⟨fun i _ => α i, ?_, Finsupp.equivFunOnFinite.symm_apply_apply α⟩
    intro i _
    simp only [Finset.mem_range]
    exact Nat.lt_succ_iff.mpr (calc
      α i ≤ ∑ j, α j := Finset.single_le_sum (fun _ _ => Nat.zero_le _) (Finset.mem_univ i)
      _ = Multiindex.magnitude α := rfl
      _ ≤ p := h)

/-- The notation for the enumeration associated with a graded monomial ordering. -/
abbrev toFun (i : ℕ) : Multiindex.Index d := ν.equiv i

/-- The initial `N(p)` entries enumerate precisely the multi-indices of degree at most `p`. -/
theorem initial_segment_eq_multiindicesUpTo (p : ℕ) :
    (Finset.range (numberOfMultiindices d p)).image ν.equiv = multiindicesUpTo d p := by
  classical
  apply Finset.eq_of_subset_of_card_le
  · intro x hx
    rcases Finset.mem_image.mp hx with ⟨i, hi, hix⟩
    rw [Finset.mem_range] at hi
    subst x
    apply mem_multiindicesUpTo_iff.mpr
    by_contra hdegree
    have hp : p < Multiindex.magnitude (ν.equiv i) := Nat.lt_of_not_ge hdegree
    have hsub : (multiindicesUpTo d p).image ν.equiv.symm ⊆ Finset.range i := by
      intro n hn
      rcases Finset.mem_image.mp hn with ⟨α, hα, hαn⟩
      rw [← hαn, Finset.mem_range]
      apply ν.graded
      simpa only [Equiv.apply_symm_apply] using
        lt_of_le_of_lt (mem_multiindicesUpTo_iff.mp hα) hp
    have hcard := Finset.card_le_card hsub
    rw [Finset.card_image_of_injective _ ν.equiv.symm.injective,
      Finset.card_range] at hcard
    exact (not_le_of_gt hi) (by simpa [numberOfMultiindices] using hcard)
  · rw [Finset.card_image_of_injective _ ν.equiv.injective, Finset.card_range]
    exact Nat.le_refl _

/-- Addition compatibility reflects strict inequalities as well as preserving them. -/
theorem add_lt_add_iff {i j k : ℕ} :
    ν.equiv.symm (ν.equiv i + ν.equiv k) <
        ν.equiv.symm (ν.equiv j + ν.equiv k) ↔ i < j := by
  constructor
  · intro h
    rcases lt_trichotomy i j with hij | hij | hji
    · exact hij
    · subst j
      exact (lt_irrefl _ h).elim
    · have h' := ν.add_lt_add hji (k := k)
      exact False.elim ((not_lt_of_ge (Nat.le_of_lt h')) h)
  · intro hij
    exact ν.add_lt_add hij

/-- The enumeration is monotone with respect to total degree. -/
theorem magnitude_monotone {i j : ℕ} (hij : i ≤ j) :
    Multiindex.magnitude (ν.equiv i) ≤ Multiindex.magnitude (ν.equiv j) := by
  by_contra h
  have hlt : Multiindex.magnitude (ν.equiv j) < Multiindex.magnitude (ν.equiv i) :=
    Nat.lt_of_not_ge h
  have hji : j < i := ν.graded hlt
  exact (not_lt_of_ge hij) hji

end PDETranslations
