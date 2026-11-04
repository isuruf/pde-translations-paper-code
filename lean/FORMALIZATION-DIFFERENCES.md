# Differences between `paper.tex` and the Lean formalization

## Major differences (hypotheses, constants, smoothness)

- **Smoothness / PDE assumptions (all analytic results).** The paper leaves the Green's-function
  setting implicit (C^p, or just "a Green function of the PDE"). Lean assumes `ContDiffOn ℝ ⊤` on
  `{z ≠ 0}` (or an open set `s`) together with the PDE holding there, and adds `c ≤ p`,
  `x - c₂ ≠ 0`, and exact order `c` (bundled in `LinearPDEOperator`). Affects:
  `derivative_eq_sum_degree_le/eq`, `m2lLocalCoefficients_isCompressible`, `m2m_single_coeff`,
  `m2m_single_coeff_equal_order`, `m2m_error_single_coeff_varying_order`, `m2m_error_varying_order`,
  `l2l_difference`, `l2l_error_*`.
- **`local_truncation_bound`, `multipole_truncation_bound`**: both add the hypothesis
  `HasLocalMultiindexTaylorSeries G` (G equals its Taylor series on `‖h‖ < ‖z‖`), which the paper
  assumes tacitly via real-analyticity.
- **`m2m_error_varying_order`**: the constant is existential and explicit in Lean,
  `A·Σ‖M_kj‖(p+1)C(p+1)^{s_p}·max(1,a^p)`, versus the paper's `e^d p (p+1)^{s_p} |j| A K' C_{G,p}`.
  It depends on the radius `a`, takes `SatisfiesDerivativeEstimate` as a hypothesis in the statement,
  and uses finite complex-weighted sources in a closed ball.
- **`l2l_error_nonneg`, `l2l_error_neg`**: `K` (Assumption (v)) and `C` (Assumption (g)) are
  existentially quantified in the conclusion rather than given constants, so no quantitative or
  uniform constant is exposed.

## Minor notational differences

- **Zero-based indexing** throughout (`ν.equiv : ℕ ≃ Index d` with `equiv 0 = 0`; paper has
  `ν(1) = 0`). Tail sums `N(r)+1..N(t)` become `Finset.Ico N(r) N(t)`.
- **Free columns** are the actual non-pivot columns (subtype `FreeColumn`), not a numbered list
  `j_1, j_2, ...`; `|j|` is `Fintype.card (FreeColumn ...)`. `α_k e_k` is `Pi.single k (a k)`;
  `multipoleEmbeddingMatrix` is a matrix on `Fin N(p) × FreeColumn`.
- `storedMaxProduct` is extended to `p < c` via `max p c`.
- `s_p` is `s p` with `s : ℕ → ℕ`; `r ∈ M(m)` is `Multiindex.upTo m`; powers use `rpow`.
- **Recurrences**: `logRadius` is `log √(x²+y²)` with `y` fixed (total at the origin); all theorems
  add `x²+y²(+z²) ≠ 0`; natural-number subtraction in derivative orders (harmless: only truncates where
  the coefficient is 0); `invRadius_mixed_recurrence_2D` is the 3D kernel with `z` fixed; mixed
  derivatives are ordered iterated directional derivatives (x, then y, then z); the paper's
  "defined and nonzero" becomes "coefficients vanish outside the range".
