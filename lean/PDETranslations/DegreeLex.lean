module

public import PDETranslations.Ordering
import Mathlib.Data.Finsupp.MonomialOrder.DegLex
import Mathlib.Order.SuccPred.LinearLocallyFinite

@[expose] public section

open scoped BigOperators

namespace PDETranslations

variable {d : ℕ}

@[instance_reducible]
private noncomputable def degreeLexLocallyFinite :
    LocallyFiniteOrder (DegLex (Multiindex.Index d)) :=
  LocallyFiniteOrder.ofFiniteIcc fun a b => by
    have h := (Finsupp.finite_of_degree_le (σ := Fin d) (ofDegLex b).degree).image toDegLex
    apply h.subset
    intro x hx
    exact ⟨ofDegLex x, Finsupp.DegLex.monotone_degree hx.2, rfl⟩

private noncomputable def degreeLexNatIso (k : Fin d) :
    DegLex (Multiindex.Index d) ≃o ℕ := by
  letI := degreeLexLocallyFinite (d := d)
  letI := LinearLocallyFiniteOrder.succOrder (DegLex (Multiindex.Index d))
  letI := LinearLocallyFiniteOrder.predOrder (DegLex (Multiindex.Index d))
  letI : NoMaxOrder (DegLex (Multiindex.Index d)) := ⟨fun a => by
    refine ⟨a + toDegLex (Finsupp.single k 1), ?_⟩
    apply Finsupp.DegLex.lt_iff.mpr
    left
    simp [ofDegLex_add]⟩
  exact orderIsoNatOfLinearSuccPredArch

private def preferredCoordinate (k : Fin d) : Fin d := ⟨0, Nat.zero_lt_of_lt k.isLt⟩

private noncomputable def coordinateSwap (k : Fin d) :
    Multiindex.Index d ≃+ Multiindex.Index d :=
  Finsupp.domCongr (Equiv.swap k (preferredCoordinate k))

private theorem coordinateSwap_degree (k : Fin d) (α : Multiindex.Index d) :
    (coordinateSwap k α).degree = Multiindex.magnitude α := by
  change (Finsupp.equivMapDomain (Equiv.swap k (preferredCoordinate k)) α).degree = _
  rw [Finsupp.equivMapDomain_eq_mapDomain, Finsupp.degree_mapDomain,
    Finsupp.degree_eq_sum]
  rfl

private noncomputable def orderingOfDegreeLexIso
    (e : DegLex (Multiindex.Index d) ≃o ℕ)
    (s : Multiindex.Index d ≃+ Multiindex.Index d)
    (hs : ∀ α, (s α).degree = Multiindex.magnitude α) : GradedMonomialOrdering d where
  equiv := e.symm.toEquiv.trans (ofDegLex.trans s.symm.toEquiv)
  map_zero := by
    have h : e.symm 0 = ⊥ := e.symm.map_bot
    change s.symm (ofDegLex (e.symm 0)) = 0
    rw [h]
    exact s.symm.map_zero
  graded := by
    intro i j h
    apply e.symm.lt_iff_lt.mp
    apply Finsupp.DegLex.lt_iff.mpr
    left
    have hi := hs (s.symm (ofDegLex (e.symm i)))
    have hj := hs (s.symm (ofDegLex (e.symm j)))
    simpa only [AddEquiv.apply_symm_apply] using hi.trans_lt (h.trans_eq hj.symm)
  add_lt_add := by
    intro i j l hij
    change e (toDegLex (s
      (s.symm (ofDegLex (e.symm i)) + s.symm (ofDegLex (e.symm l))))) <
      e (toDegLex (s
      (s.symm (ofDegLex (e.symm j)) + s.symm (ofDegLex (e.symm l)))))
    simp only [map_add, AddEquiv.apply_symm_apply, toDegLex_add, toDegLex_ofDegLex]
    have hij' : e.symm i < e.symm j := e.symm.strictMono hij
    have hadd : e.symm i + e.symm l < e.symm j + e.symm l :=
      by simpa only [add_comm] using add_lt_add_right hij' (e.symm l)
    exact e.strictMono hadd

/-- Degree-lexicographic enumeration, with coordinate `k` given highest priority.
Coordinates are read in increasing order after swapping `k` with zero; at the
first differing coordinate the smaller exponent comes first. -/
@[no_expose] noncomputable def degreeLexOrdering (k : Fin d) : GradedMonomialOrdering d :=
  orderingOfDegreeLexIso (degreeLexNatIso k) (coordinateSwap k) (coordinateSwap_degree k)

private theorem degreeLex_purePower_maximal (z : Fin d) (hz : z.val = 0)
    (c : ℕ) (α : Multiindex.Index d) (h : α.degree ≤ c) :
    toDegLex α ≤ toDegLex (Finsupp.single z c) := by
  have hcoord : α z ≤ α.degree := by
    rw [Finsupp.degree_eq_sum]
    exact Finset.single_le_sum (fun _ _ => Nat.zero_le _) (Finset.mem_univ z)
  by_cases heq : α z = c
  · have hle : Finsupp.single z c ≤ α := by
      intro i
      by_cases hi : i = z
      · subst i; simp [heq]
      · simp [hi]
    have hsum := congrArg Finsupp.degree (tsub_add_cancel_of_le hle)
    have hzero : (α - Finsupp.single z c).degree = 0 := by
      simp only [map_add, Finsupp.degree_single] at hsum
      omega
    have he := (Finsupp.degree_eq_zero_iff _).mp hzero
    have ha := tsub_add_cancel_of_le hle
    rw [he, zero_add] at ha
    rw [ha]
  · apply Finsupp.DegLex.le_iff.mpr
    simp only [ofDegLex_toDegLex, Finsupp.degree_single]
    rcases lt_or_eq_of_le h with hlt | he
    · exact Or.inl hlt
    · refine Or.inr ⟨he, le_of_lt ?_⟩
      apply Finsupp.Lex.lt_iff.mpr
      refine ⟨z, ?_, ?_⟩
      · intro j hj
        have : j.val < z.val := hj
        omega
      · change α z < (Finsupp.single z c) z
        rw [Finsupp.single_eq_same]
        omega

/-- Every multi-index of degree at most `c` occurs no later than the pure
power of degree `c` in the preferred coordinate. -/
theorem degreeLexOrdering_purePower_maximal (k : Fin d)
    (c : ℕ) (α : Multiindex.Index d) (h : Multiindex.magnitude α ≤ c) :
    (degreeLexOrdering k).equiv.symm α ≤
      (degreeLexOrdering k).equiv.symm (Finsupp.single k c) := by
  change degreeLexNatIso k (toDegLex (coordinateSwap k α)) ≤
    degreeLexNatIso k (toDegLex (coordinateSwap k (Finsupp.single k c)))
  apply (degreeLexNatIso k).monotone
  have hsingle : coordinateSwap k (Finsupp.single k c) =
      Finsupp.single (preferredCoordinate k) c := by
    simp [coordinateSwap, Finsupp.domCongr_apply, Finsupp.equivMapDomain_single]
  rw [hsingle]
  exact degreeLex_purePower_maximal (preferredCoordinate k) rfl c
    (coordinateSwap k α) (by simpa only [coordinateSwap_degree] using h)

/-- A graded monomial ordering with a prescribed pure power last in every
bounded-degree initial segment. -/
theorem exists_gradedMonomialOrdering_purePower_maximal (k : Fin d) :
    ∃ ν : GradedMonomialOrdering d, ∀ c α, Multiindex.magnitude α ≤ c →
      ν.equiv.symm α ≤ ν.equiv.symm (Finsupp.single k c) :=
  ⟨degreeLexOrdering k, degreeLexOrdering_purePower_maximal k⟩

end PDETranslations
