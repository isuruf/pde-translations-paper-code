module

public import Mathlib.Analysis.Complex.Basic
public import PDETranslations.Ordering
import Mathlib.Analysis.Calculus.ContDiff.Operations

@[expose] public section

open scoped BigOperators Classical

namespace PDETranslations

variable {d c : ℕ}

/-- Coefficients of a constant-coefficient, scalar linear PDE operator of exact
order `c`, as in the paper.  Only coefficients of degree at most `c` contribute
to `linearPDEOperator`, and a nonzero coefficient of degree exactly `c` ensures
that the order is attained.  Coefficients outside this range are ignored. -/
structure LinearPDEOperator (d c : ℕ) where
  coefficients : Multiindex.Index d → ℂ
  nonzero : ∃ α, Multiindex.magnitude α = c ∧ coefficients α ≠ 0

/-- The constant-coefficient, scalar linear PDE operator `ℒ` from equation
`(define_pde_op)` of the paper, applied to `f` at `x`.  The bundled operator
includes a nonzero coefficient of degree exactly `c`.  The coefficients and
function values are complex, while differentiation is over the real spatial
variables.  Smoothness and the equation `ℒ f = 0` are separate assumptions,
not part of this coefficient-level definition. -/
noncomputable def linearPDEOperator {c : ℕ} (L : LinearPDEOperator d c)
    (f : (Fin d → ℝ) → ℂ) (x : Fin d → ℝ) : ℂ :=
  ∑ α ∈ multiindicesUpTo d c,
    L.coefficients α * Multiindex.deriv α f x


/-- Constant-coefficient PDE operators commute with every multi-index derivative
of a smooth function on an open domain. -/
theorem linearPDEOperator_deriv (L : LinearPDEOperator d c)
    (G : (Fin d → ℝ) → ℂ) {s : Set (Fin d → ℝ)} (hs : IsOpen s)
    (hG : ContDiffOn ℝ ⊤ G s) (β : Multiindex.Index d) {x} (hx : x ∈ s) :
    linearPDEOperator L (Multiindex.deriv β G) x =
      Multiindex.deriv β (linearPDEOperator L G) x := by
  have hα (α : Multiindex.Index d) : ContDiffAt ℝ ⊤ (Multiindex.deriv α G) x :=
    (Multiindex.contDiffOn_deriv hG hs α).contDiffAt (hs.mem_nhds hx)
  change (∑ α ∈ multiindicesUpTo d c,
      L.coefficients α * Multiindex.deriv α (Multiindex.deriv β G) x) =
    iteratedFDeriv ℝ (Multiindex.directions β).length
      (fun y => ∑ α ∈ multiindicesUpTo d c,
        L.coefficients α • Multiindex.deriv α G y) x
      (fun k => Pi.single ((Multiindex.directions β).get k) (1 : ℝ))
  rw [iteratedFDeriv_fun_sum_apply (fun α _ =>
    ((hα α).const_smul (L.coefficients α)).of_le (by simp))]
  simp only [sum_apply]
  apply Finset.sum_congr rfl
  intro α hmem
  rw [iteratedFDeriv_const_smul_apply' ((hα α).of_le (by simp))]
  change L.coefficients α * Multiindex.deriv α (Multiindex.deriv β G) x =
    L.coefficients α * Multiindex.deriv β (Multiindex.deriv α G) x
  rw [← Multiindex.deriv_add hG hs hx α β,
    ← Multiindex.deriv_add hG hs hx β α, add_comm α β]

/-- Every multi-index derivative of a smooth solution of a constant-coefficient
PDE is again a solution on the same open domain. Smoothness and the PDE are
required only on the domain, not at singularities outside it. -/
theorem linearPDEOperator_deriv_eq_zero (L : LinearPDEOperator d c)
    (G : (Fin d → ℝ) → ℂ) {s : Set (Fin d → ℝ)} (hs : IsOpen s)
    (hG : ContDiffOn ℝ ⊤ G s) (hPDE : ∀ x ∈ s, linearPDEOperator L G x = 0)
    (β : Multiindex.Index d) {x} (hx : x ∈ s) :
    linearPDEOperator L (Multiindex.deriv β G) x = 0 := by
  rw [linearPDEOperator_deriv L G hs hG β hx]
  have heq : linearPDEOperator L G =ᶠ[nhds x] (fun _ => (0 : ℂ)) :=
    Filter.eventually_of_mem (hs.mem_nhds hx) (fun y hy => hPDE y hy)
  unfold Multiindex.deriv
  rw [(heq.iteratedFDeriv ℝ _).eq_of_nhds]
  simp

/-- Property `(prop:nice)`: a nonzero pure-power coefficient of principal order. -/
def LinearPDEOperator.HasNonzeroPurePower (L : LinearPDEOperator d c) : Prop :=
  ∃ k : Fin d, L.coefficients (Finsupp.single k c) ≠ 0

/-- The principal symbol, evaluated on a real vector, with complex coefficients.
No Fourier-transform factors of `I` are included, as in the paper. -/
noncomputable def LinearPDEOperator.principalSymbol (L : LinearPDEOperator d c)
    (ξ : Fin d → ℝ) : ℂ :=
  ∑ α ∈ Multiindex.multiindicesOfMagnitude c,
    L.coefficients α * (Multiindex.vectorPower ξ α : ℂ)

/-- Ellipticity in exactly the nonvanishing-symbol sense used in the paper,
not a positivity or uniform strong-ellipticity condition. -/
def LinearPDEOperator.IsElliptic (L : LinearPDEOperator d c) : Prop :=
  ∀ ξ : Fin d → ℝ, ξ ≠ 0 → L.principalSymbol ξ ≠ 0

private theorem magnitude_single (k : Fin d) (n : ℕ) :
    Multiindex.magnitude (Finsupp.single k n) = n := by
  simp [Multiindex.magnitude]

/-- Evaluating the principal symbol on a coordinate unit vector isolates the
corresponding pure-power coefficient, also when `c = 0`. -/
theorem LinearPDEOperator.principalSymbol_unit (L : LinearPDEOperator d c) (k : Fin d) :
    L.principalSymbol (Pi.single k 1) = L.coefficients (Finsupp.single k c) := by
  classical
  unfold principalSymbol
  rw [Finset.sum_eq_single (Finsupp.single k c)]
  · simp [Multiindex.vectorPower, Pi.single_apply, Finsupp.single_apply]
  · intro α hα hne
    have hmag := Multiindex.mem_multiindicesOfMagnitude.mp hα
    have hex : ∃ i, i ≠ k ∧ α i ≠ 0 := by
      by_contra h
      push Not at h
      have heq : α = Finsupp.single k (α k) := by
        ext i
        by_cases hi : i = k
        · subst i; simp
        · simp [hi, h i hi]
      have hk : α k = c := by
        exact (magnitude_single k (α k)).symm.trans
                  ((congrArg Multiindex.magnitude heq).symm.trans hmag)
      exact hne (heq.trans (congrArg (Finsupp.single k) hk))
    obtain ⟨i, hik, hi⟩ := hex
    have hz : Multiindex.vectorPower (Pi.single k (1 : ℝ)) α = 0 := by
      apply Finset.prod_eq_zero (Finset.mem_univ i)
      simp [hik, hi]
    simp [hz]
  · intro h
    exact (h (Multiindex.mem_multiindicesOfMagnitude.mpr (magnitude_single k c))).elim

/-- Ellipticity implies Property `(prop:nice)` in positive dimension. In fact
ellipticity makes every pure-power coefficient nonzero. Positive dimension is
necessary: in dimension zero ellipticity is vacuous but the property is false. -/
theorem LinearPDEOperator.IsElliptic.hasNonzeroPurePower
    (L : LinearPDEOperator d c) (hell : L.IsElliptic) (hd : 0 < d) :
    L.HasNonzeroPurePower := by
  let k : Fin d := ⟨0, hd⟩
  refine ⟨k, ?_⟩
  rw [← L.principalSymbol_unit k]
  apply hell
  intro h
  have := congrFun h k
  simp at this

end PDETranslations
