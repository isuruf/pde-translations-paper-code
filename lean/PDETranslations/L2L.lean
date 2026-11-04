module

public import PDETranslations.M2M
public import PDETranslations.L2LBounds

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

private theorem factorial_ne_zero (α : Multiindex.Index d) :
    (Multiindex.factorial α : ℂ) ≠ 0 := by
  exact_mod_cast (Finset.prod_ne_zero_iff.mpr
    (fun k _ => Nat.factorial_ne_zero (α k)))

/-- Order-`p` local coefficients of a single source about `center`.
The derivative is evaluated at `center - source`; the factorial is the
multi-index factorial, not the factorial of the total degree. The definition
specifies finite coefficients even without regularity; it makes no claim
about convergence of the associated local expansion. -/
noncomputable def localCoefficients (ν : GradedMonomialOrdering d) (p : ℕ)
    (G : (Fin d → ℝ) → ℂ) (center source : Fin d → ℝ) :
    Fin (numberOfMultiindices d p) → ℂ :=
  fun i => Multiindex.deriv (ν.equiv i.val) G (center - source) /
    (Multiindex.factorial (ν.equiv i.val) : ℂ)

/-- Finite local-to-local translation from `c₁` to `c₂`, implemented as the
factorial-scaled transpose of the M2M matrix (without complex conjugation).
`l2lOperator_apply` gives the paper's binomial formula exactly: the factorial
quotient is division in `ℂ`, not truncated natural-number division.
All indices are zero-based and range over the actual initial segment `𝓜(p)`.
This finite polynomial translation does not assert convergence of an infinite
local expansion or equality with the kernel's local coefficients at `c₂`. -/
noncomputable def l2lOperator (ν : GradedMonomialOrdering d) (p : ℕ)
    (c₁ c₂ : Fin d → ℝ) (γ : Fin (numberOfMultiindices d p) → ℂ) :
    Fin (numberOfMultiindices d p) → ℂ :=
  fun i => (∑ ell : Fin (numberOfMultiindices d p),
    m2mMatrix ν p (c₂ - c₁) ell i *
      (γ ell * (Multiindex.factorial (ν.equiv ell.val) : ℂ))) /
    (Multiindex.factorial (ν.equiv i.val) : ℂ)

/-- Component form of L2L. On the componentwise filter `ν(i) ≤ ν(ell)`,
the quotient `ν(ell)! / (ν(i)! * (ν(ell) - ν(i))!)` is exactly the
paper's multi-index binomial coefficient. Spatial monomials are real and
then embedded in `ℂ`; there is no conjugation. -/
theorem l2lOperator_apply (ν : GradedMonomialOrdering d) (p : ℕ)
    (c₁ c₂ : Fin d → ℝ) (γ : Fin (numberOfMultiindices d p) → ℂ)
    (i : Fin (numberOfMultiindices d p)) :
    l2lOperator ν p c₁ c₂ γ i =
      ∑ ell : Fin (numberOfMultiindices d p) with ν.equiv i.val ≤ ν.equiv ell.val,
        γ ell * ((Multiindex.factorial (ν.equiv ell.val) : ℂ) /
          ((Multiindex.factorial (ν.equiv i.val) : ℂ) *
            (Multiindex.factorial (ν.equiv ell.val - ν.equiv i.val) : ℂ))) *
          (Multiindex.vectorPower (c₂ - c₁) (ν.equiv ell.val - ν.equiv i.val) : ℂ) := by
  unfold l2lOperator
  rw [Finset.sum_div, Finset.sum_filter]
  apply Finset.sum_congr rfl
  intro ell _
  by_cases hle : ν.equiv i.val ≤ ν.equiv ell.val
  · simp only [m2mMatrix, ite_eq_left hle, normalizedMonomial, Complex.ofReal_div,
      Complex.ofReal_natCast]
    ring
  · simp [m2mMatrix, hle]

/-- Recover local coefficients from the stored free columns using the actual
PDE decompression matrix `M`. Stored values are factorial-scaled into a jet
before applying `M`, and the output is scaled back. A free index `s` is an
actual column, not a separate enumeration of stored entries. -/
noncomputable def compressedLocalCoefficients (ν : GradedMonomialOrdering d)
    (L : LinearPDEOperator d c) (hcp : c ≤ p)
    (μ : Fin (numberOfMultiindices d p) → ℂ) :
    Fin (numberOfMultiindices d p) → ℂ :=
  fun i => (decompressionMatrix ν L hcp).mulVec
    (fun s => μ s.val * (Multiindex.factorial (ν.equiv s.val.val) : ℂ)) i /
    (Multiindex.factorial (ν.equiv i.val) : ℂ)

/-- Algebraic L2L reindexing for an arbitrary derivative jet. The finite input
index supplies `|ν(i)| ≤ p`, so the residual order is `p - |ν(i)|`.
The proof uses `m2m_sum_reindex` with the degree condition redundant on
`𝓜(p)`. No PDE or regularity assumption is needed for this finite identity.
 -/
theorem l2lOperator_jet (ν : GradedMonomialOrdering d) (p : ℕ)
    (c₁ c₂ : Fin d → ℝ) (jet : Multiindex.Index d → ℂ)
    (i : Fin (numberOfMultiindices d p)) :
    l2lOperator ν p c₁ c₂
        (fun ell => jet (ν.equiv ell.val) /
          (Multiindex.factorial (ν.equiv ell.val) : ℂ)) i =
      (∑ q : Fin (numberOfMultiindices d (p - Multiindex.magnitude (ν.equiv i.val))),
        jet (ν.equiv q.val + ν.equiv i.val) *
          (normalizedMonomial (c₂ - c₁) (ν.equiv q.val) : ℂ)) /
        (Multiindex.factorial (ν.equiv i.val) : ℂ) := by
  have hi : Multiindex.magnitude (ν.equiv i.val) ≤ p :=
    (enumeration_mem ν i.val p).mpr i.isLt
  have hreindex := m2m_sum_reindex ν p
    (p - Multiindex.magnitude (ν.equiv i.val)) (ν.equiv i.val)
    (by omega) (c₂ - c₁) jet
  unfold l2lOperator
  simp_rw [div_mul_cancel₀ _ (factorial_ne_zero _)]
  apply congrArg (fun z : ℂ => z / (Multiindex.factorial (ν.equiv i.val) : ℂ))
  calc
    _ = ∑ ell : Fin (numberOfMultiindices d p) with
        ν.equiv i.val ≤ ν.equiv ell.val ∧
          Multiindex.magnitude (ν.equiv ell.val) -
            Multiindex.magnitude (ν.equiv i.val) ≤
              p - Multiindex.magnitude (ν.equiv i.val),
        (normalizedMonomial (c₂ - c₁) (ν.equiv ell.val - ν.equiv i.val) : ℂ) *
          jet (ν.equiv ell.val) := by
      rw [Finset.sum_filter]
      apply Finset.sum_congr rfl
      intro ell _
      have hdegree : Multiindex.magnitude (ν.equiv ell.val) -
          Multiindex.magnitude (ν.equiv i.val) ≤
            p - Multiindex.magnitude (ν.equiv i.val) :=
        Nat.sub_le_sub_right ((enumeration_mem ν ell.val p).mpr ell.isLt) _
      simp [m2mMatrix, hdegree, ite_mul]
    _ = _ := by
      rw [hreindex]
      apply Finset.sum_congr rfl
      intro q _
      rw [add_comm (ν.equiv i.val), mul_comm]

/-- Translation of single-source local coefficients, reindexed as a truncated
Taylor sum of derivatives at `c₁ - source`. This is an exact finite algebraic
identity, not a Taylor remainder estimate or a claim that the translated
coefficients equal those of the kernel at `c₂`. In particular it requires
neither smoothness nor a PDE hypothesis; those enter subsequent analytic or
compression results. The shifted derivative uses `ν(q) + ν(i)` and its total
degree is at most `p`. -/
theorem l2lOperator_localCoefficients (ν : GradedMonomialOrdering d) (p : ℕ)
    (G : (Fin d → ℝ) → ℂ) (c₁ c₂ source : Fin d → ℝ)
    (i : Fin (numberOfMultiindices d p)) :
    l2lOperator ν p c₁ c₂ (localCoefficients ν p G c₁ source) i =
      (∑ q : Fin (numberOfMultiindices d (p - Multiindex.magnitude (ν.equiv i.val))),
        Multiindex.deriv (ν.equiv q.val + ν.equiv i.val) G (c₁ - source) *
          (normalizedMonomial (c₂ - c₁) (ν.equiv q.val) : ℂ)) /
        (Multiindex.factorial (ν.equiv i.val) : ℂ) := by
  exact l2lOperator_jet ν p c₁ c₂
    (fun α => Multiindex.deriv α G (c₁ - source)) i

/-- Lemma `(lemma:l2l_difference)`, for a unit source at `source`.

The paper's Green-function setting is made explicit: `G` is smooth and
satisfies the constant-coefficient PDE off the origin, and `c₁ - source ≠ 0`.
The construction of `M` also requires `c ≤ p` and the exact-order nonzero
operator bundled in `L`. These assumptions are needed for the differentiated
PDE/decompression step in the paper's proof; no equal-order assumption is used.
Smoothness of all orders is convenient but stronger than the finite regularity
needed for this identity. The bridge used below constructs full shifted jets,
including derivatives above order `p`; the resulting formula only uses
derivatives of order at most `p`. Any smooth solution of the homogeneous PDE
off the origin suffices: fundamental-solution normalization is not required,
so this generalizes the paper's Green-function setting.

Indices are zero-based: the paper's inclusive tail from `N(r)+1` to `N(t)`
is `Finset.Ico (N(r)) (N(t))` here. Free columns directly index the paper's
`j_s`. The equality concerns finite translated coefficients, not convergence
of a local expansion, so no radius restriction or condition on `c₂` is needed. -/
theorem l2l_difference (ν : GradedMonomialOrdering d)
    (L : LinearPDEOperator d c) (hcp : c ≤ p)
    (G : (Fin d → ℝ) → ℂ) (c₁ c₂ source : Fin d → ℝ)
    (hz : c₁ - source ≠ 0)
    (hG : ContDiffOn ℝ ⊤ G {z | z ≠ 0})
    (hPDE : ∀ z, z ≠ 0 → linearPDEOperator L G z = 0)
    (i : Fin (numberOfMultiindices d p)) :
    compressedLocalCoefficients ν L hcp
        (l2lOperator ν p c₁ c₂ (localCoefficients ν p G c₁ source)) i -
      l2lOperator ν p c₁ c₂ (localCoefficients ν p G c₁ source) i =
    (∑ s : FreeColumn ν L hcp with
        Multiindex.magnitude (ν.equiv s.val.val) < Multiindex.magnitude (ν.equiv i.val),
      decompressionMatrix ν L hcp i s *
        ∑ q ∈ Finset.Ico
            (numberOfMultiindices d (p - Multiindex.magnitude (ν.equiv i.val)))
            (numberOfMultiindices d (p - Multiindex.magnitude (ν.equiv s.val.val))),
          Multiindex.deriv (ν.equiv q + ν.equiv s.val.val) G (c₁ - source) *
            (normalizedMonomial (c₂ - c₁) (ν.equiv q) : ℂ)) /
      (Multiindex.factorial (ν.equiv i.val) : ℂ) := by
  let M := decompressionMatrix ν L hcp
  let degree (k : Fin (numberOfMultiindices d p)) := Multiindex.magnitude (ν.equiv k.val)
  let r := p - degree i
  let S (t : ℕ) (α : Multiindex.Index d) : ℂ :=
    ∑ q ∈ Finset.range (numberOfMultiindices d t),
      Multiindex.deriv (ν.equiv q + α) G (c₁ - source) *
        (normalizedMonomial (c₂ - c₁) (ν.equiv q) : ℂ)
  let μ := l2lOperator ν p c₁ c₂ (localCoefficients ν p G c₁ source)
  have hμ (k : Fin (numberOfMultiindices d p)) :
      μ k = S (p - degree k) (ν.equiv k.val) /
        (Multiindex.factorial (ν.equiv k.val) : ℂ) := by
    dsimp only [μ]
    rw [l2lOperator_localCoefficients]
    exact congrArg (fun z : ℂ => z / (Multiindex.factorial (ν.equiv k.val) : ℂ))
      (Fin.sum_univ_eq_sum_range
        (fun q => Multiindex.deriv (ν.equiv q + ν.equiv k.val) G (c₁ - source) *
          (normalizedMonomial (c₂ - c₁) (ν.equiv q) : ℂ)) _)
  have hsupport (s : FreeColumn ν L hcp) (hs : degree i < degree s.val) :
      M i s = 0 := decompressionMatrix_eq_zero_of_lt ν L hcp i s (ν.graded hs)
  have hshift (q : ℕ) :
      Multiindex.deriv (ν.equiv q + ν.equiv i.val) G (c₁ - source) =
        ∑ s : FreeColumn ν L hcp, M i s *
          Multiindex.deriv (ν.equiv q + ν.equiv s.val.val) G (c₁ - source) := by
    have heq := (decompressionMatrix_spec ν L hcp _).mp
      (pdeCoefficientMatrix_mulVec_shifted_deriv_eq_zero ν L hcp G
        (isOpen_ne_fun continuous_id continuous_const) hG hPDE (ν.equiv q) hz)
    simpa only [Matrix.mulVec, dotProduct, add_comm] using congrFun heq i
  have hcommon : S r (ν.equiv i.val) =
      ∑ s : FreeColumn ν L hcp, M i s * S r (ν.equiv s.val.val) := by
    dsimp only [S]
    simp_rw [hshift, Finset.sum_mul, Finset.mul_sum]
    rw [Finset.sum_comm]
    apply Finset.sum_congr rfl
    intro s _
    apply Finset.sum_congr rfl
    intro q _
    ring
  have hcompressed : compressedLocalCoefficients ν L hcp μ i =
      (∑ s : FreeColumn ν L hcp, M i s *
        S (p - degree s.val) (ν.equiv s.val.val)) /
          (Multiindex.factorial (ν.equiv i.val) : ℂ) := by
    unfold compressedLocalCoefficients
    simp only [Matrix.mulVec, dotProduct, hμ, div_mul_cancel₀ _ (factorial_ne_zero _)]
    rfl
  change compressedLocalCoefficients ν L hcp μ i - μ i = _
  rw [hcompressed, hμ, ← sub_div, hcommon, ← Finset.sum_sub_distrib]
  congr 1
  rw [Finset.sum_filter]
  apply Finset.sum_congr rfl
  intro s _
  change M i s * S (p - degree s.val) (ν.equiv s.val.val) -
    M i s * S r (ν.equiv s.val.val) = _
  by_cases hs : degree s.val < degree i
  · simp only [show Multiindex.magnitude (ν.equiv s.val.val) <
        Multiindex.magnitude (ν.equiv i.val) from hs, ite_true]
    rw [← mul_sub]
    congr 1
    apply (Finset.sum_Ico_eq_sub _ ?_).symm
    have horder : r ≤ p - degree s.val := Nat.sub_le_sub_left hs.le p
    by_contra hn
    have hmem := (enumeration_mem ν (numberOfMultiindices d (p - degree s.val)) r).mpr
      (Nat.lt_of_not_ge hn)
    have hmem' := (enumeration_mem ν _ (p - degree s.val)).mp (hmem.trans horder)
    exact (Nat.lt_irrefl _) hmem'
  · simp only [show ¬ Multiindex.magnitude (ν.equiv s.val.val) <
        Multiindex.magnitude (ν.equiv i.val) from hs, ite_false]
    rcases eq_or_lt_of_le (Nat.le_of_not_gt hs) with heq | hlt
    · simp only [show p - degree s.val = r by omega, sub_self]
    · rw [hsupport s hlt]
      simp

private theorem norm_vectorPower_le (v : EuclideanSpace ℝ (Fin d))
    (α : Multiindex.Index d) {R : ℝ} (_hR : 0 ≤ R) (hv : ‖v‖ ≤ R) :
    ‖Multiindex.vectorPower v α‖ ≤ R ^ Multiindex.magnitude α := by
  unfold Multiindex.vectorPower Multiindex.magnitude
  rw [norm_prod, ← Finset.prod_pow_eq_pow_sum]
  apply Finset.prod_le_prod₀
  · intro k _; exact norm_nonneg _
  · intro k _
    rw [norm_pow]
    exact pow_le_pow_left₀ (norm_nonneg _) ((PiLp.norm_apply_le v k).trans hv) _

/-- The coefficientwise estimate in the proof of `thm:l2l_error`, for a unit
source. The layer estimate is Assumption (g) instantiated at `m = p`, only
at this source and displacement; unlike a bound on individual derivatives,
it retains cancellation within a homogeneous layer. The matrix estimate is
Assumption (v), only for the strictly lower-degree entries used here.

Spatial norms are Euclidean, not the sup norm on functions. Radii are
explicitly nonnegative. Smoothness and the homogeneous PDE off zero, the
exact-order operator `L`, and `c ≤ p` justify `l2l_difference`; smoothness of
all orders is stronger than necessary. Neither a convergent infinite Taylor
series nor fundamental-solution normalization is assumed. The target may be
on the closed radius-`R₂` ball, slightly generalizing the paper's open ball.
`s : ℕ` follows the existing derivative-estimate convention. -/
theorem l2l_error_single_coeff (ν : GradedMonomialOrdering d)
    (L : LinearPDEOperator d c) (hcp : c ≤ p)
    (G : (Fin d → ℝ) → ℂ) (c₁ c₂ source x : EuclideanSpace ℝ (Fin d))
    (R₁ R₂ r a C K : ℝ) (s : ℕ) (t : ℝ)
    (hR₁ : 0 ≤ R₁) (hR₂ : 0 ≤ R₂) (hgeom : ‖c₂ - c₁‖ + R₂ ≤ R₁)
    (hsep : R₁ < r) (hrz : r ≤ ‖c₁ - source‖) (hra : r ≤ a)
    (hx : ‖x - c₂‖ ≤ R₂) (hC : 0 < C) (hK : 0 ≤ K)
    (hG : ContDiffOn ℝ ⊤ G {z | z ≠ 0})
    (hPDE : ∀ z, z ≠ 0 → linearPDEOperator L G z = 0)
    (hM : ∀ (i : Fin (numberOfMultiindices d p)) (j : FreeColumn ν L hcp),
      Multiindex.magnitude (ν.equiv j.val.val) < Multiindex.magnitude (ν.equiv i.val) →
      ‖decompressionMatrix ν L hcp i j‖ *
        a ^ (Multiindex.magnitude (ν.equiv i.val) -
          Multiindex.magnitude (ν.equiv j.val.val)) ≤ K)
    (hlayer : ∀ α : Multiindex.Index d, Multiindex.magnitude α ≤ p →
      ∀ n : ℕ, 0 < n →
      ‖∑ q ∈ Multiindex.multiindicesOfMagnitude n,
        Multiindex.deriv (q + α) G (c₁ - source) *
          (normalizedMonomial (c₂ - c₁) q : ℂ)‖ ≤
        C * (1+n : ℝ)^s / ‖c₁ - source‖^((Multiindex.magnitude α : ℝ)+t) *
          (‖c₂ - c₁‖/‖c₁ - source‖)^n)
    (i : Fin (numberOfMultiindices d p)) :
    ‖(compressedLocalCoefficients ν L hcp
          (l2lOperator ν p c₁ c₂ (localCoefficients ν p G c₁ source)) i -
        l2lOperator ν p c₁ c₂ (localCoefficients ν p G c₁ source) i) *
      (Multiindex.vectorPower (x - c₂) (ν.equiv i.val) : ℂ)‖ ≤
      ((p : ℝ) * (p+1 : ℝ)^s * Fintype.card (FreeColumn ν L hcp) * K * C *
        (R₁/r)^(p+1) / ‖c₁ - source‖^t) /
        (Multiindex.factorial (ν.equiv i.val) : ℝ) := by
  have hr : 0 < r := lt_of_le_of_lt hR₁ hsep
  have hz : 0 < ‖c₁ - source‖ := hr.trans_le hrz
  have hz' : (c₁ : Fin d → ℝ) - source ≠ 0 := by
    intro he
    have : c₁ - source = 0 := by ext k; exact congrFun he k
    simp [this] at hz
  have hR₂R₁ : R₂ ≤ R₁ := by linarith [norm_nonneg (c₂ - c₁)]
  have hh : ‖c₂ - c₁‖ ≤ R₁ := by linarith
  let S := (Finset.univ : Finset (FreeColumn ν L hcp)).filter
    (fun j => Multiindex.magnitude (ν.equiv j.val.val) < Multiindex.magnitude (ν.equiv i.val))
  let T (j : FreeColumn ν L hcp) : ℂ :=
    ∑ q ∈ Finset.Ico
      (numberOfMultiindices d (p - Multiindex.magnitude (ν.equiv i.val)))
      (numberOfMultiindices d (p - Multiindex.magnitude (ν.equiv j.val.val))),
      Multiindex.deriv (ν.equiv q + ν.equiv j.val.val) G (c₁ - source) *
        (normalizedMonomial (c₂ - c₁) (ν.equiv q) : ℂ)
  let B : ℝ := (p : ℝ) * C * (p+1 : ℝ)^s * K * (R₁/r)^(p+1) / ‖c₁ - source‖^t
  have hB : 0 ≤ B := by dsimp [B]; positivity
  have hpoint (j : FreeColumn ν L hcp) (hj : j ∈ S) :
      ‖decompressionMatrix ν L hcp i j‖ * R₂^Multiindex.magnitude (ν.equiv i.val) *
        ‖T j‖ ≤ B :=
    l2l_weighted_tail_bound ν p _ _ (Finset.mem_filter.mp hj).2
      ((enumeration_mem ν i.val p).mpr i.isLt) (c₂ - c₁) (c₁ - source)
      R₁ R₂ r a C K _ s t hR₁ hR₂ hR₂R₁ hsep hrz hra hh hC hK
      (hM i j (Finset.mem_filter.mp hj).2) _
      (hlayer _ ((enumeration_mem ν j.val.val p).mpr j.val.isLt))
  have hfac : 0 < (Multiindex.factorial (ν.equiv i.val) : ℝ) := by
    unfold Multiindex.factorial
    positivity
  rw [l2l_difference ν L hcp G c₁ c₂ source hz' hG hPDE i,
    norm_mul, norm_div, Complex.norm_natCast, Complex.norm_real]
  change ‖∑ j ∈ S, decompressionMatrix ν L hcp i j * T j‖ /
    (Multiindex.factorial (ν.equiv i.val) : ℝ) *
    ‖Multiindex.vectorPower (x - c₂) (ν.equiv i.val)‖ ≤ _
  calc
    _ ≤ (∑ j ∈ S, ‖decompressionMatrix ν L hcp i j‖ * ‖T j‖) /
        (Multiindex.factorial (ν.equiv i.val) : ℝ) *
        R₂^Multiindex.magnitude (ν.equiv i.val) := by
      gcongr
      · exact (norm_sum_le _ _).trans (by simp only [norm_mul]; rfl)
      · exact norm_vectorPower_le (x - c₂) _ hR₂ hx
    _ = (∑ j ∈ S, ‖decompressionMatrix ν L hcp i j‖ *
        R₂^Multiindex.magnitude (ν.equiv i.val) * ‖T j‖) /
        (Multiindex.factorial (ν.equiv i.val) : ℝ) := by
      rw [Finset.sum_div, Finset.sum_mul, Finset.sum_div]
      apply Finset.sum_congr rfl
      intro j _
      ring
    _ ≤ ((S.card : ℝ) * B) / (Multiindex.factorial (ν.equiv i.val) : ℝ) := by
      gcongr
      simpa using Finset.sum_le_sum hpoint
    _ ≤ ((Fintype.card (FreeColumn ν L hcp) : ℝ) * B) /
        (Multiindex.factorial (ν.equiv i.val) : ℝ) := by
      gcongr
      exact_mod_cast Finset.card_le_card (Finset.filter_subset _ _)
    _ = _ := by dsimp [B]; ring

/-- Finite-source local coefficients, with complex source weights. This is a
finite sum of derivative coefficients, not an assertion of series convergence. -/
noncomputable def finiteSourceLocalCoefficients {κ : Type*} [Fintype κ]
    (ν : GradedMonomialOrdering d) (p : ℕ) (G : (Fin d → ℝ) → ℂ)
    (c₁ : Fin d → ℝ) (source : κ → Fin d → ℝ) (w : κ → ℂ) :
    Fin (numberOfMultiindices d p) → ℂ :=
  fun i => ∑ k, w k * localCoefficients ν p G c₁ (source k) i

private theorem l2l_finite_linearity {κ : Type*} [Fintype κ]
    (ν : GradedMonomialOrdering d)
    (c₁ c₂ : Fin d → ℝ) (γ : κ → Fin (numberOfMultiindices d p) → ℂ)
    (w : κ → ℂ) (i : Fin (numberOfMultiindices d p)) :
    l2lOperator ν p c₁ c₂ (fun j => ∑ k, w k * γ k j) i =
      ∑ k, w k * l2lOperator ν p c₁ c₂ (γ k) i := by
  simp only [l2lOperator, Finset.sum_mul, Finset.mul_sum, Finset.sum_div,
    mul_div_assoc]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro k _
  apply Finset.sum_congr rfl
  intro j _
  ring

private theorem compressed_finite_linearity {κ : Type*} [Fintype κ]
    (ν : GradedMonomialOrdering d) (L : LinearPDEOperator d c) (hcp : c ≤ p)
    (γ : κ → Fin (numberOfMultiindices d p) → ℂ) (w : κ → ℂ)
    (i : Fin (numberOfMultiindices d p)) :
    compressedLocalCoefficients ν L hcp (fun j => ∑ k, w k * γ k j) i =
      ∑ k, w k * compressedLocalCoefficients ν L hcp (γ k) i := by
  simp only [compressedLocalCoefficients, Matrix.mulVec, dotProduct,
    Finset.sum_mul, Finset.mul_sum, Finset.sum_div, mul_div_assoc]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro k _
  apply Finset.sum_congr rfl
  intro j _
  ring

private theorem l2l_weighted_factor (u B D F : ℝ) :
    u * ((B / D) / F) = (1 / F) * (B * (u / D)) := by ring

/-- Finite-source version of the estimate in `thm:l2l_error`, retaining the
weighted source distances. The error is the actual decompressed translated
polynomial minus the uncompressed translated polynomial.

Compared with the paper, smoothness and the homogeneous PDE are explicit off
the origin, as is `c ≤ p`. Radii are nonnegative, coordinates are Euclidean,
and `s : ℕ` uses the existing derivative-estimate convention. The positive
constant `C`, derivative-layer bounds (in normalized-monomial form), and
weighted matrix bound are assumptions, not inferred kernel estimates. These
are the paper's Assumptions (g) and (v) restricted to the source locations,
the actual displacement, and the matrix entries used by the proof; their
global versions imply these hypotheses. Here `r` is `R_loc`, `t` is `p'`,
`C` is `C_{G,p}`, `K` is `K'`, and `s` is `s_p`. Natural subtraction in
`hM` is used only under a strict degree inequality, so it equals the paper's
integer degree gap. Only finite sums occur: real analyticity and fundamental
solution normalization can be weakened to smooth homogeneous PDE solutions;
no series convergence is needed, and the closed target ball is allowed.
The full smooth shifted jets used by `l2l_difference` justify the proof's
common-tail substitution, even though the final tails use only orders at
most `p`. Source indices form any finite type (including the empty type),
rather than a specified finite subset. -/
theorem l2l_error_finite_sources {κ : Type*} [Fintype κ]
    (ν : GradedMonomialOrdering d) (L : LinearPDEOperator d c) (hcp : c ≤ p)
    (G : (Fin d → ℝ) → ℂ) (c₁ c₂ x : EuclideanSpace ℝ (Fin d))
    (source : κ → EuclideanSpace ℝ (Fin d)) (w : κ → ℂ)
    (R₁ R₂ r a C K : ℝ) (s : ℕ) (t : ℝ)
    (hR₁ : 0 ≤ R₁) (hR₂ : 0 ≤ R₂) (hgeom : ‖c₂ - c₁‖ + R₂ ≤ R₁)
    (hsep : R₁ < r) (hrz : ∀ k, r ≤ ‖c₁ - source k‖) (hra : r ≤ a)
    (hx : ‖x - c₂‖ ≤ R₂) (hC : 0 < C) (hK : 0 ≤ K)
    (hG : ContDiffOn ℝ ⊤ G {z | z ≠ 0})
    (hPDE : ∀ z, z ≠ 0 → linearPDEOperator L G z = 0)
    (hM : ∀ (i : Fin (numberOfMultiindices d p)) (j : FreeColumn ν L hcp),
      Multiindex.magnitude (ν.equiv j.val.val) < Multiindex.magnitude (ν.equiv i.val) →
      ‖decompressionMatrix ν L hcp i j‖ *
        a ^ (Multiindex.magnitude (ν.equiv i.val) -
          Multiindex.magnitude (ν.equiv j.val.val)) ≤ K)
    (hlayer : ∀ k, ∀ α : Multiindex.Index d, Multiindex.magnitude α ≤ p →
      ∀ n : ℕ, 0 < n →
      ‖∑ q ∈ Multiindex.multiindicesOfMagnitude n,
        Multiindex.deriv (q + α) G (c₁ - source k) *
          (normalizedMonomial (c₂ - c₁) q : ℂ)‖ ≤
        C * (1+n : ℝ)^s / ‖c₁ - source k‖^((Multiindex.magnitude α : ℝ)+t) *
          (‖c₂ - c₁‖/‖c₁ - source k‖)^n) :
    ‖(∑ i, compressedLocalCoefficients ν L hcp
          (l2lOperator ν p c₁ c₂
            (finiteSourceLocalCoefficients ν p G c₁ (fun k => source k) w)) i *
          (Multiindex.vectorPower (x - c₂) (ν.equiv i.val) : ℂ)) -
      (∑ i, l2lOperator ν p c₁ c₂
          (finiteSourceLocalCoefficients ν p G c₁ (fun k => source k) w) i *
          (Multiindex.vectorPower (x - c₂) (ν.equiv i.val) : ℂ))‖ ≤
      Real.exp (d : ℝ) * (p : ℝ) * (p+1 : ℝ)^s *
        Fintype.card (FreeColumn ν L hcp) * K * C * (R₁/r)^(p+1) *
          (∑ k, ‖w k‖ / ‖c₁ - source k‖^t) := by
  let γ := fun k => l2lOperator ν p c₁ c₂ (localCoefficients ν p G c₁ (source k))
  let B : ℝ := (p : ℝ) * (p+1 : ℝ)^s *
    Fintype.card (FreeColumn ν L hcp) * K * C * (R₁/r)^(p+1)
  have hr : 0 < r := lt_of_le_of_lt hR₁ hsep
  have hB : 0 ≤ B := by dsimp [B]; positivity
  have hlin : l2lOperator ν p c₁ c₂
      (finiteSourceLocalCoefficients ν p G c₁ (fun k => source k) w) =
      fun i => ∑ k, w k * γ k i := by
    funext i
    exact l2l_finite_linearity ν c₁ c₂ _ w i
  rw [hlin, ← Finset.sum_sub_distrib]
  have herr (i : Fin (numberOfMultiindices d p)) :
      compressedLocalCoefficients ν L hcp (fun j => ∑ k, w k * γ k j) i *
          (Multiindex.vectorPower (x - c₂) (ν.equiv i.val) : ℂ) -
        (∑ k, w k * γ k i) *
          (Multiindex.vectorPower (x - c₂) (ν.equiv i.val) : ℂ) =
      ∑ k, (w k * compressedLocalCoefficients ν L hcp (γ k) i - w k * γ k i) *
        (Multiindex.vectorPower (x - c₂) (ν.equiv i.val) : ℂ) := by
    rw [compressed_finite_linearity ν L hcp γ w i, ← sub_mul,
      ← Finset.sum_sub_distrib, Finset.sum_mul]
  simp only [herr]
  have hterm (i : Fin (numberOfMultiindices d p)) :
      ‖∑ k, (w k * compressedLocalCoefficients ν L hcp (γ k) i - w k * γ k i) *
        (Multiindex.vectorPower (x - c₂) (ν.equiv i.val) : ℂ)‖ ≤
      ∑ k, ‖w k‖ * ((B / ‖c₁ - source k‖^t) /
        (Multiindex.factorial (ν.equiv i.val) : ℝ)) := by
    apply (norm_sum_le _ _).trans
    apply Finset.sum_le_sum
    intro k _
    rw [← mul_sub, mul_assoc, norm_mul]
    apply mul_le_mul_of_nonneg_left _ (norm_nonneg _)
    exact l2l_error_single_coeff ν L hcp G c₁ c₂ (source k) x
      R₁ R₂ r a C K s t hR₁ hR₂ hgeom hsep (hrz k) hra hx hC hK hG hPDE hM
      (hlayer k) i
  calc
    _ ≤ ∑ i : Fin (numberOfMultiindices d p), ∑ k, ‖w k‖ * ((B / ‖c₁ - source k‖^t) /
        (Multiindex.factorial (ν.equiv i.val) : ℝ)) :=
      (norm_sum_le _ _).trans (Finset.sum_le_sum (fun i _ => hterm i))
    _ = (∑ i : Fin (numberOfMultiindices d p),
        1 / (Multiindex.factorial (ν.equiv i.val) : ℝ)) *
        (B * ∑ k, ‖w k‖ / ‖c₁ - source k‖^t) := by
      simp only [Finset.mul_sum, Finset.sum_mul]
      rw [Finset.sum_comm]
      apply Finset.sum_congr rfl
      intro i _
      apply Finset.sum_congr rfl
      intro k _
      exact l2l_weighted_factor _ _ _ _
    _ ≤ Real.exp (d : ℝ) * (B * ∑ k, ‖w k‖ / ‖c₁ - source k‖^t) := by
      apply mul_le_mul_of_nonneg_right (l2l_sum_inv_factorial_le_exp ν p)
      exact mul_nonneg hB (Finset.sum_nonneg (fun k _ => by positivity))
    _ = _ := by simp only [B, mul_assoc]


-- For fixed ν, L, p, and a, a finite sum bounds every scaled matrix entry.
-- In particular this constant is independent of the centers, sources, weights,
-- and target. This proves existence, not uniformity in PDE parameters or p.
private theorem exists_l2l_matrix_bound (ν : GradedMonomialOrdering d)
    (L : LinearPDEOperator d c) (hcp : c ≤ p) (a : ℝ) (ha : 0 ≤ a) :
    ∃ K : ℝ, 0 ≤ K ∧
      ∀ (i : Fin (numberOfMultiindices d p)) (j : FreeColumn ν L hcp),
        Multiindex.magnitude (ν.equiv j.val.val) < Multiindex.magnitude (ν.equiv i.val) →
        ‖decompressionMatrix ν L hcp i j‖ *
          a ^ (Multiindex.magnitude (ν.equiv i.val) -
            Multiindex.magnitude (ν.equiv j.val.val)) ≤ K := by
  let entry (i : Fin (numberOfMultiindices d p)) (j : FreeColumn ν L hcp) : ℝ :=
    ‖decompressionMatrix ν L hcp i j‖ *
      a ^ (Multiindex.magnitude (ν.equiv i.val) -
        Multiindex.magnitude (ν.equiv j.val.val))
  have hentry (i : Fin (numberOfMultiindices d p)) (j : FreeColumn ν L hcp) :
      0 ≤ entry i j := mul_nonneg (norm_nonneg _) (pow_nonneg ha _)
  refine ⟨∑ i, ∑ j, entry i j,
    Finset.sum_nonneg (fun i _ => Finset.sum_nonneg (fun j _ => hentry i j)), ?_⟩
  intro i j _
  exact (Finset.single_le_sum (fun k _ => hentry i k) (Finset.mem_univ j)).trans
    (Finset.single_le_sum
      (fun k _ => Finset.sum_nonneg (fun l _ => hentry k l)) (Finset.mem_univ i))

/-- The nonnegative-`t` case of `thm:l2l_error`, with the exact paper constant
`A = exp d * p * (p+1)^(s p) * card free * K * C * ∑ k, ‖w k‖`.
Assumption (g) is supplied as `SatisfiesDerivativeEstimate G t s`. Since it
existentially quantifies its constants, the conclusion supplies a positive
`C`, obtained from that assumption at `m = p`, rather than accepting an
arbitrary external constant. Only the local layer bounds used by the shared
proof are then derived.

In the fixed-PDE viewpoint, Assumption (v) is not a separate hypothesis:
`exists_l2l_matrix_bound` constructs a nonnegative `K` from the finite matrix
at scale `a`. Thus `K` is existential in the conclusion. The constructed
constant depends only on `ν`, `L`, `p`, and `a`, not on the centers, sources,
weights, target, or inner radii. No bound uniform in PDE parameters or
truncation order is asserted; obtaining a prescribed or uniform `K` still
requires a quantitative matrix estimate. The constant need not be sharp.
All other analytic and geometric conventions are those of
`l2l_error_finite_sources`. -/
theorem l2l_error_nonneg {κ : Type*} [Fintype κ]
    (ν : GradedMonomialOrdering d) (L : LinearPDEOperator d c) (hcp : c ≤ p)
    (G : (Fin d → ℝ) → ℂ) (c₁ c₂ x : EuclideanSpace ℝ (Fin d))
    (source : κ → EuclideanSpace ℝ (Fin d)) (w : κ → ℂ)
    (R₁ R₂ Rloc a : ℝ) (s : ℕ → ℕ) (p' : ℝ)
    (hR₁ : 0 ≤ R₁) (hR₂ : 0 ≤ R₂) (hgeom : ‖c₂ - c₁‖ + R₂ ≤ R₁)
    (hsep : R₁ < Rloc) (hrz : ∀ k, Rloc ≤ ‖c₁ - source k‖) (hra : Rloc ≤ a)
    (hx : ‖x - c₂‖ ≤ R₂)
    (hG : ContDiffOn ℝ ⊤ G {z | z ≠ 0})
    (hPDE : ∀ z, z ≠ 0 → linearPDEOperator L G z = 0)
    (hDerivative : SatisfiesDerivativeEstimate G p' s)
    (hpprime : 0 ≤ p') :
    ∃ K : ℝ, 0 ≤ K ∧ ∃ C : ℝ, 0 < C ∧
    ‖(∑ i, compressedLocalCoefficients ν L hcp
          (l2lOperator ν p c₁ c₂
            (finiteSourceLocalCoefficients ν p G c₁ (fun k => source k) w)) i *
          (Multiindex.vectorPower (x - c₂) (ν.equiv i.val) : ℂ)) -
      (∑ i, l2lOperator ν p c₁ c₂
          (finiteSourceLocalCoefficients ν p G c₁ (fun k => source k) w) i *
          (Multiindex.vectorPower (x - c₂) (ν.equiv i.val) : ℂ))‖ ≤
      (Real.exp (d : ℝ) * (p : ℝ) * (p+1 : ℝ)^(s p) *
        Fintype.card (FreeColumn ν L hcp) * K * C * (∑ k, ‖w k‖)) *
          (R₁/Rloc)^(p+1) / Rloc^p' := by
  obtain ⟨K, hK, hM⟩ := exists_l2l_matrix_bound ν L hcp a
    ((lt_of_le_of_lt hR₁ hsep).le.trans hra)
  obtain ⟨C, hC, hbound⟩ := hDerivative p
  refine ⟨K, hK, C, hC, ?_⟩
  have hlayer (k : κ) (α : Multiindex.Index d) (hα : Multiindex.magnitude α ≤ p)
      (n : ℕ) (hn : 0 < n) :=
    hbound α (Multiindex.mem_upTo.mpr hα) n (c₁ - source k) (c₂ - c₁)
      ⟨norm_pos_iff.mp ((lt_of_le_of_lt hR₁ hsep).trans_le (hrz k)), hn⟩
  have hr : 0 < Rloc := lt_of_le_of_lt hR₁ hsep

  have hsum : (∑ k, ‖w k‖ / ‖c₁ - source k‖^p') ≤
      (∑ k, ‖w k‖) / Rloc^p' := by
    rw [Finset.sum_div]
    apply Finset.sum_le_sum
    intro k _
    apply div_le_div_of_nonneg_left (norm_nonneg _)
      (Real.rpow_pos_of_pos hr p')
    exact Real.rpow_le_rpow hr.le (hrz k) hpprime
  have hbound := l2l_error_finite_sources ν L hcp G c₁ c₂ x source w
    R₁ R₂ Rloc a C K (s p) p' hR₁ hR₂ hgeom hsep hrz hra hx hC hK hG hPDE hM (by
      intro k α hα n hn
      simpa only [normalizedMonomial, Complex.ofReal_div, Complex.ofReal_natCast,
        WithLp.ofLp_sub, mul_div_assoc, div_mul_eq_mul_div] using hlayer k α hα n hn)
  apply hbound.trans
  calc
    _ ≤ (Real.exp (d : ℝ) * (p : ℝ) * (p+1 : ℝ)^(s p) *
        Fintype.card (FreeColumn ν L hcp) * K * C * (R₁/Rloc)^(p+1)) *
        ((∑ k, ‖w k‖) / Rloc^p') :=
      mul_le_mul_of_nonneg_left hsum (by positivity)
    _ = _ := by ring

/-- The negative-`t` case of `thm:l2l_error`, with the same exact constant
as `l2l_error_nonneg`. As there, Assumption (g) is supplied globally as
`SatisfiesDerivativeEstimate`, and the positive constant `C` in the conclusion
is extracted at `m = p`. The matrix constant `K` is constructed for the fixed
PDE, order, ordering, and scale `a`, not assumed via Assumption (v); the same
uniformity caveat as in `l2l_error_nonneg` applies. The additional outer
source-radius bound remains explicit and is not removed by finiteness of
the decompression matrix.
The paper's written proof ends with the nonnegative case; here the negative
case is also proved, using that positive-base real powers are decreasing in
the base for negative exponents. In particular the outer bound cannot be
dropped in this case. All other conventions are those of
`l2l_error_finite_sources`. -/
theorem l2l_error_neg {κ : Type*} [Fintype κ]
    (ν : GradedMonomialOrdering d) (L : LinearPDEOperator d c) (hcp : c ≤ p)
    (G : (Fin d → ℝ) → ℂ) (c₁ c₂ x : EuclideanSpace ℝ (Fin d))
    (source : κ → EuclideanSpace ℝ (Fin d)) (w : κ → ℂ)
    (R₁ R₂ Rloc a : ℝ) (s : ℕ → ℕ) (p' : ℝ)
    (hR₁ : 0 ≤ R₁) (hR₂ : 0 ≤ R₂) (hgeom : ‖c₂ - c₁‖ + R₂ ≤ R₁)
    (hsep : R₁ < Rloc) (hrz : ∀ k, Rloc ≤ ‖c₁ - source k‖) (hra : Rloc ≤ a)
    (hx : ‖x - c₂‖ ≤ R₂)
    (hG : ContDiffOn ℝ ⊤ G {z | z ≠ 0})
    (hPDE : ∀ z, z ≠ 0 → linearPDEOperator L G z = 0)
    (hDerivative : SatisfiesDerivativeEstimate G p' s)
    (hpprime : p' < 0) (hza : ∀ k, ‖c₁ - source k‖ ≤ a) :
    ∃ K : ℝ, 0 ≤ K ∧ ∃ C : ℝ, 0 < C ∧
    ‖(∑ i, compressedLocalCoefficients ν L hcp
          (l2lOperator ν p c₁ c₂
            (finiteSourceLocalCoefficients ν p G c₁ (fun k => source k) w)) i *
          (Multiindex.vectorPower (x - c₂) (ν.equiv i.val) : ℂ)) -
      (∑ i, l2lOperator ν p c₁ c₂
          (finiteSourceLocalCoefficients ν p G c₁ (fun k => source k) w) i *
          (Multiindex.vectorPower (x - c₂) (ν.equiv i.val) : ℂ))‖ ≤
      (Real.exp (d : ℝ) * (p : ℝ) * (p+1 : ℝ)^(s p) *
        Fintype.card (FreeColumn ν L hcp) * K * C * (∑ k, ‖w k‖)) *
          (R₁/Rloc)^(p+1) / a^p' := by
  obtain ⟨K, hK, hM⟩ := exists_l2l_matrix_bound ν L hcp a
    ((lt_of_le_of_lt hR₁ hsep).le.trans hra)
  obtain ⟨C, hC, hbound⟩ := hDerivative p
  refine ⟨K, hK, C, hC, ?_⟩
  have hlayer (k : κ) (α : Multiindex.Index d) (hα : Multiindex.magnitude α ≤ p)
      (n : ℕ) (hn : 0 < n) :=
    hbound α (Multiindex.mem_upTo.mpr hα) n (c₁ - source k) (c₂ - c₁)
      ⟨norm_pos_iff.mp ((lt_of_le_of_lt hR₁ hsep).trans_le (hrz k)), hn⟩
  have hr : 0 < Rloc := lt_of_le_of_lt hR₁ hsep
  have hbase : 0 < a := hr.trans_le hra
  have hsum : (∑ k, ‖w k‖ / ‖c₁ - source k‖^p') ≤
      (∑ k, ‖w k‖) / a^p' := by
    rw [Finset.sum_div]
    apply Finset.sum_le_sum
    intro k _
    apply div_le_div_of_nonneg_left (norm_nonneg _)
      (Real.rpow_pos_of_pos hbase p')
    exact Real.rpow_le_rpow_of_nonpos (hr.trans_le (hrz k)) (hza k) hpprime.le
  have hbound := l2l_error_finite_sources ν L hcp G c₁ c₂ x source w
    R₁ R₂ Rloc a C K (s p) p' hR₁ hR₂ hgeom hsep hrz hra hx hC hK hG hPDE hM (by
      intro k α hα n hn
      simpa only [normalizedMonomial, Complex.ofReal_div, Complex.ofReal_natCast,
        WithLp.ofLp_sub, mul_div_assoc, div_mul_eq_mul_div] using hlayer k α hα n hn)
  apply hbound.trans
  calc
    _ ≤ (Real.exp (d : ℝ) * (p : ℝ) * (p+1 : ℝ)^(s p) *
        Fintype.card (FreeColumn ν L hcp) * K * C * (R₁/Rloc)^(p+1)) *
        ((∑ k, ‖w k‖) / a^p') :=
      mul_le_mul_of_nonneg_left hsum (by positivity)
    _ = _ := by ring

end PDETranslations
