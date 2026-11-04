module

public import Mathlib.Analysis.Calculus.IteratedDeriv.Defs
public import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.Calculus.IteratedDeriv.Lemmas
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Tactic

@[expose] public section

/-!
# Pure and mixed derivative recurrences for Laplace and biharmonic kernels

We interpret the paper's partial x-derivatives as ordinary iterated derivatives
with the other coordinates fixed. The statement explicitly excludes the origin:
the classical real function log r is not defined there. Mathlib totalizes `Real.log` and `deriv`, so
omitting this restriction would not express the intended classical assertion.
The proofs use induction by differentiation, not Chebyshev polynomials.
The original one-y log-radius recurrence takes y first, then x. The arbitrary
mixed recurrences below take x first, then y, then z when present. These choices
fix the order left implicit by the paper's multi-index notation; we do not
separately formalize interchange of partial derivatives here.
All inverse-radius recurrences use the three-dimensional Laplace kernel
with r = sqrt (x² + y² + z²), including when only x or x and y are differentiated.
All classical kernels and derivatives are considered off the singular origin.
Lower-order terms with negative formal indices are represented using natural
subtraction and zero coefficients. The paper's description of such terms as
"nonzero" is read as "potentially nonzero", not pointwise nonvanishing.
The operation-count claims following the recurrences are not formalized.
-/

namespace PDETranslations.Recurrences

open Filter Set
open scoped Topology

/-- The paper's `log r`, with `y` fixed so that `logRadius y` is a function of `x`.
As with `Real.log`, this definition is totalized at the origin; the recurrence
below explicitly excludes that point to retain the classical meaning. -/
noncomputable def logRadius (y x : ℝ) : ℝ := Real.log (Real.sqrt (x ^ 2 + y ^ 2))

-- Work directly with the paper's function; no alternative kernel is postulated.
private theorem logRadius_hasDerivAt (x y : ℝ) (h : x ^ 2 + y ^ 2 ≠ 0) :
    HasDerivAt (fun t : ℝ => Real.log (Real.sqrt (t ^ 2 + y ^ 2)))
      (x / (x ^ 2 + y ^ 2)) x := by
  have heq : (fun t : ℝ => Real.log (Real.sqrt (t ^ 2 + y ^ 2))) =
      (fun t : ℝ => Real.log (t ^ 2 + y ^ 2) / 2) := by
    funext t
    exact Real.log_sqrt (by positivity)
  rw [heq]
  convert (((hasDerivAt_id x).pow 2).add_const (y ^ 2)).log h |>.div_const 2 using 1 <;> dsimp
  all_goals ring

private theorem smooth_iterated (k : ℕ) (x y : ℝ) (h : x ^ 2 + y ^ 2 ≠ 0) :
    ContDiffAt ℝ ⊤
      (iteratedDeriv k (fun t : ℝ => Real.log (Real.sqrt (t ^ 2 + y ^ 2)))) x := by
  induction k with
  | zero =>
    simp only [iteratedDeriv_zero]
    exact ((contDiffAt_id.pow 2).add contDiffAt_const).sqrt h |>.log
      (Real.sqrt_ne_zero'.mpr (lt_of_le_of_ne (by positivity) (Ne.symm h)))
  | succ k ih =>
    rw [iteratedDeriv_succ]
    exact ih.derivWithin (by simp)

-- Differentiating a locally valid quadratic-coefficient identity advances its
-- order. The constant is 1 for the second-order seed and 0 thereafter.
private theorem advance (f : ℝ → ℝ) (y x : ℝ) (k : ℕ) (c : ℝ)
    (hd : ∀ j, HasDerivAt (iteratedDeriv j f) (iteratedDeriv (j + 1) f x) x)
    (he : ∀ᶠ t in 𝓝 x,
      (t ^ 2 + y ^ 2) * iteratedDeriv (k + 2) f t +
        2 * (k + 1 : ℝ) * t * iteratedDeriv (k + 1) f t +
        (k : ℝ) * (k + 1) * iteratedDeriv k f t = c) :
    (x ^ 2 + y ^ 2) * iteratedDeriv (k + 3) f x +
      2 * (k + 2 : ℝ) * x * iteratedDeriv (k + 2) f x +
      (k + 1 : ℝ) * (k + 2) * iteratedDeriv (k + 1) f x = 0 := by
  have hdq := ((hasDerivAt_id x).pow 2).add_const (y ^ 2)
  have hdx := (hasDerivAt_id x).const_mul (2 * (k + 1 : ℝ))
  have hdall := ((hdq.mul (hd (k + 2))).add (hdx.mul (hd (k + 1)))).add
    ((hd k).const_mul ((k : ℝ) * (k + 1)))
  have hz := hdall.congr_of_eventuallyEq (he.mono fun _ ht => ht.symm)
  have hz' := hz.unique (hasDerivAt_const x c)
  dsimp at hz'
  convert hz' using 1
  all_goals ring

/-- The paper's pure x-derivative recurrence for `log r`, for every `n ≥ 3`.

Here `y` is held fixed and `r = sqrt (x² + y²)` exactly as in the paper.
The additional explicit hypothesis `x² + y² ≠ 0` excludes the singular origin;
it is the classical domain restriction implicit in the paper, not a global
smoothness assumption. Natural-number subtraction in derivative orders agrees
with ordinary subtraction because `3 ≤ n`. No claim about mixed derivatives or
operation counts is included in this theorem. -/
theorem logRadius_x_recurrence (n : ℕ) (hn : 3 ≤ n) (x y : ℝ)
    (h : x ^ 2 + y ^ 2 ≠ 0) :
    (Real.sqrt (x ^ 2 + y ^ 2)) ^ 2 *
        iteratedDeriv n (logRadius y) x =
      -2 * ((n : ℝ) - 1) * x *
        iteratedDeriv (n - 1) (logRadius y) x -
      ((n : ℝ) - 1) * ((n : ℝ) - 2) *
        iteratedDeriv (n - 2) (logRadius y) x := by
  let f := logRadius y
  have hd (j : ℕ) (t : ℝ) (ht : t ^ 2 + y ^ 2 ≠ 0) :
      HasDerivAt (iteratedDeriv j f) (iteratedDeriv (j + 1) f t) t := by
    rw [iteratedDeriv_succ]
    exact (smooth_iterated j t y ht).differentiableAt (by simp) |>.hasDerivAt
  have hnhood (t : ℝ) (ht : t ^ 2 + y ^ 2 ≠ 0) :
      ∀ᶠ s in 𝓝 t, s ^ 2 + y ^ 2 ≠ 0 :=
    ((continuousAt_id.pow 2).add continuousAt_const).eventually_ne ht
  have seed (t : ℝ) (ht : t ^ 2 + y ^ 2 ≠ 0) :
      (t ^ 2 + y ^ 2) * iteratedDeriv 2 f t +
        2 * t * iteratedDeriv 1 f t = 1 := by
    have he : ∀ᶠ s in 𝓝 t, (s ^ 2 + y ^ 2) * iteratedDeriv 1 f s = s := by
      filter_upwards [hnhood t ht] with s hs
      rw [iteratedDeriv_one]
      change (s ^ 2 + y ^ 2) * deriv (fun t : ℝ => Real.log (Real.sqrt (t ^ 2 + y ^ 2))) s = s
      rw [(logRadius_hasDerivAt s y hs).deriv]
      field_simp
    have hh := ((((hasDerivAt_id t).pow 2).add_const (y ^ 2)).mul
      (hd 1 t ht)).congr_of_eventuallyEq (he.mono fun _ hs => hs.symm)
    have hh' := hh.unique (hasDerivAt_id t)
    dsimp at hh'
    nlinarith
  have hrec (k : ℕ) : ∀ t : ℝ, t ^ 2 + y ^ 2 ≠ 0 →
      (t ^ 2 + y ^ 2) * iteratedDeriv (k + 3) f t +
        2 * (k + 2 : ℝ) * t * iteratedDeriv (k + 2) f t +
        (k + 1 : ℝ) * (k + 2) * iteratedDeriv (k + 1) f t = 0 := by
    induction k with
    | zero =>
      intro t ht
      apply advance f y t 0 1 (fun j => hd j t ht)
      filter_upwards [hnhood t ht] with s hs
      simpa using seed s hs
    | succ k ih =>
      intro t ht
      convert advance f y t (k + 1) 0 (fun j => hd j t ht) ?_ using 1
      filter_upwards [hnhood t ht] with s hs
      simpa [Nat.cast_add, Nat.cast_one, Nat.add_assoc, add_assoc, one_add_one_eq_two] using ih s hs
  have hr := hrec (n - 3) x h
  have h1 : n - 3 + 3 = n := by omega
  have h2 : n - 3 + 2 = n - 1 := by omega
  have h3 : n - 3 + 1 = n - 2 := by omega
  rw [h1, h2, h3] at hr
  rw [Real.sq_sqrt (by positivity)]
  have hc : ((n - 3 : ℕ) : ℝ) = (n : ℝ) - 3 := by
    rw [Nat.cast_sub (by omega : 3 ≤ n)]; norm_num
  dsimp [f, logRadius] at hr ⊢
  rw [hc] at hr
  nlinarith

/-- Equation `eq:deriv_laplace_2d` of the paper, with one y derivative.

The mixed derivative is represented explicitly as one y derivative followed by
`n` x derivatives. The paper's multi-index notation leaves this order implicit;
we do not prove a separate theorem interchanging partial derivatives. Off the
origin the classical kernel is smooth, so this is the intended mixed derivative.
As in the pure recurrence, we require `x² + y² ≠ 0`, rather than asserting an
identity about totalized derivatives at the singular origin. We retain the
preceding formula's range `n ≥ 3` (not repeated after the mixed formula in the
paper); thus all displayed derivative orders are nonnegative without truncated
subtraction. The proof actually establishes a seed at `n = 2` as well.

We use induction by differentiating local identities, as suggested in the paper,
starting from the actual y derivative of `log r`, not a postulated auxiliary
kernel. This theorem does not formalize the subsequent operation-count claim. -/
theorem logRadius_xy_recurrence (n : ℕ) (hn : 3 ≤ n) (x y : ℝ)
    (h : x ^ 2 + y ^ 2 ≠ 0) :
    (Real.sqrt (x ^ 2 + y ^ 2)) ^ 2 *
        iteratedDeriv n (fun t => deriv (fun s => logRadius s t) y) x =
      -2 * ((n : ℝ) - 1) * x *
        iteratedDeriv (n - 1) (fun t => deriv (fun s => logRadius s t) y) x -
      ((n : ℝ) - 1) * ((n : ℝ) - 2) *
        iteratedDeriv (n - 2) (fun t => deriv (fun s => logRadius s t) y) x -
      2 * y * iteratedDeriv n (logRadius y) x := by
  let f := logRadius y
  let g := fun t => deriv (fun s => logRadius s t) y
  have hnhood (t : ℝ) (ht : t ^ 2 + y ^ 2 ≠ 0) :
      ∀ᶠ s in 𝓝 t, s ^ 2 + y ^ 2 ≠ 0 :=
    ((continuousAt_id.pow 2).add continuousAt_const).eventually_ne ht
  have hg (t : ℝ) (ht : t ^ 2 + y ^ 2 ≠ 0) : g t = y / (t ^ 2 + y ^ 2) := by
    have hh := logRadius_hasDerivAt y t (by simpa [add_comm] using ht)
    simpa [g, logRadius, add_comm] using hh.deriv
  have hgsmooth (j : ℕ) (t : ℝ) (ht : t ^ 2 + y ^ 2 ≠ 0) :
      ContDiffAt ℝ ⊤ (iteratedDeriv j g) t := by
    induction j with
    | zero =>
      simp only [iteratedDeriv_zero]
      apply (contDiffAt_const.div ((contDiffAt_id.pow 2).add contDiffAt_const) ht
        : ContDiffAt ℝ ⊤ (fun s : ℝ => y / (s ^ 2 + y ^ 2)) t).congr_of_eventuallyEq
      filter_upwards [hnhood t ht] with s hs using hg s hs
    | succ j ih =>
      rw [iteratedDeriv_succ]
      exact ih.derivWithin (by simp)
  have hdf (j : ℕ) (t : ℝ) (ht : t ^ 2 + y ^ 2 ≠ 0) :
      HasDerivAt (iteratedDeriv j f) (iteratedDeriv (j + 1) f t) t := by
    rw [iteratedDeriv_succ]
    exact (smooth_iterated j t y ht).differentiableAt (by simp) |>.hasDerivAt
  have hdg (j : ℕ) (t : ℝ) (ht : t ^ 2 + y ^ 2 ≠ 0) :
      HasDerivAt (iteratedDeriv j g) (iteratedDeriv (j + 1) g t) t := by
    rw [iteratedDeriv_succ]
    exact (hgsmooth j t ht).differentiableAt (by simp) |>.hasDerivAt
  -- The first-order identity avoids differentiating a parameter-dependent
  -- iterated derivative or assuming any interchange of differentiation.
  have first (t : ℝ) (ht : t ^ 2 + y ^ 2 ≠ 0) :
      (t ^ 2 + y ^ 2) * iteratedDeriv 1 g t =
        -2 * y * iteratedDeriv 1 f t := by
    have he : ∀ᶠ s in 𝓝 t, (s ^ 2 + y ^ 2) * g s = y := by
      filter_upwards [hnhood t ht] with s hs
      rw [hg s hs]
      field_simp
    have hh := ((((hasDerivAt_id t).pow 2).add_const (y ^ 2)).mul
      (hdg 0 t ht)).congr_of_eventuallyEq (he.mono fun _ hs => hs.symm)
    have hh' := hh.unique (hasDerivAt_const t y)
    simp only [iteratedDeriv_zero] at hh'
    have hf : iteratedDeriv 1 f t = t / (t ^ 2 + y ^ 2) := by
      rw [iteratedDeriv_one]
      exact (logRadius_hasDerivAt t y ht).deriv
    rw [hg t ht] at hh'
    rw [hf]
    dsimp at hh'
    linear_combination hh'
  have seed (t : ℝ) (ht : t ^ 2 + y ^ 2 ≠ 0) :
      (t ^ 2 + y ^ 2) * iteratedDeriv 2 g t +
        2 * t * iteratedDeriv 1 g t = -2 * y * iteratedDeriv 2 f t := by
    have he : ∀ᶠ s in 𝓝 t, (s ^ 2 + y ^ 2) * iteratedDeriv 1 g s =
        -2 * y * iteratedDeriv 1 f s := by
      filter_upwards [hnhood t ht] with s hs using first s hs
    have hh := ((((hasDerivAt_id t).pow 2).add_const (y ^ 2)).mul
      (hdg 1 t ht)).congr_of_eventuallyEq (he.mono fun _ hs => hs.symm)
    have hh' := hh.unique ((hdf 1 t ht).const_mul (-2 * y))
    dsimp at hh'
    nlinarith
  have hrec (k : ℕ) : ∀ t : ℝ, t ^ 2 + y ^ 2 ≠ 0 →
      (t ^ 2 + y ^ 2) * iteratedDeriv (k + 2) g t +
        2 * (k + 1 : ℝ) * t * iteratedDeriv (k + 1) g t +
        (k : ℝ) * (k + 1) * iteratedDeriv k g t =
          -2 * y * iteratedDeriv (k + 2) f t := by
    induction k with
    | zero =>
      intro t ht
      simpa using seed t ht
    | succ k ih =>
      intro t ht
      have he : ∀ᶠ s in 𝓝 t,
          (s ^ 2 + y ^ 2) * iteratedDeriv (k + 2) g s +
            2 * (k + 1 : ℝ) * s * iteratedDeriv (k + 1) g s +
            (k : ℝ) * (k + 1) * iteratedDeriv k g s =
              -2 * y * iteratedDeriv (k + 2) f s := by
        filter_upwards [hnhood t ht] with s hs using ih s hs
      have hdq := ((hasDerivAt_id t).pow 2).add_const (y ^ 2)
      have hdx := (hasDerivAt_id t).const_mul (2 * (k + 1 : ℝ))
      have hdall := ((hdq.mul (hdg (k + 2) t ht)).add
        (hdx.mul (hdg (k + 1) t ht))).add
        ((hdg k t ht).const_mul ((k : ℝ) * (k + 1)))
      have hh := hdall.congr_of_eventuallyEq (he.mono fun _ hs => hs.symm)
      have hh' := hh.unique ((hdf (k + 2) t ht).const_mul (-2 * y))
      dsimp at hh'
      simp only [Nat.cast_succ, Nat.add_assoc] at ⊢
      convert hh' using 1; ring
  have hr := hrec (n - 2) x h
  have h1 : n - 2 + 2 = n := by omega
  have h2 : n - 2 + 1 = n - 1 := by omega
  rw [h1, h2] at hr
  have hc : ((n - 2 : ℕ) : ℝ) = (n : ℝ) - 2 := by
    rw [Nat.cast_sub (by omega : 2 ≤ n)]; norm_num
  rw [hc] at hr
  rw [Real.sq_sqrt (by positivity)]
  change (x ^ 2 + y ^ 2) * iteratedDeriv n g x = _
  nlinarith

/-- The three-dimensional Laplace kernel `1/r` in x, y, z argument order.
Lean totalizes division by zero; the recurrences below explicitly exclude
the singular origin. -/
noncomputable def invRadius (x y z : ℝ) : ℝ :=
  1 / Real.sqrt (x ^ 2 + y ^ 2 + z ^ 2)

/-- The pure x-derivative recurrence for the three-dimensional Laplace kernel.

Partial x-derivatives are ordinary iterated derivatives with `y` and `z` held
fixed. The hypothesis `x² + y² + z² ≠ 0` excludes the singular origin. The bound
`n ≥ 2` ensures subtraction in derivative orders is not truncated.

The proof is induction by differentiation of local identities away from the
origin, starting with `(x² + y² + z²) f' = -x f` for the actual kernel.
No mixed-derivative recurrence or operation-count claim is included here. -/
theorem invRadius_x_recurrence (n : ℕ) (hn : 2 ≤ n) (x y z : ℝ)
    (h : x ^ 2 + y ^ 2 + z ^ 2 ≠ 0) :
    (Real.sqrt (x ^ 2 + y ^ 2 + z ^ 2)) ^ 2 *
        iteratedDeriv n (fun t => invRadius t y z) x =
      -(2 * (n : ℝ) - 1) * x * iteratedDeriv (n - 1) (fun t => invRadius t y z) x -
        ((n : ℝ) - 1) ^ 2 * iteratedDeriv (n - 2) (fun t => invRadius t y z) x := by
  let f := fun t => invRadius t y z
  have hnhood (t : ℝ) (ht : t ^ 2 + y ^ 2 + z ^ 2 ≠ 0) :
      ∀ᶠ s in 𝓝 t, s ^ 2 + y ^ 2 + z ^ 2 ≠ 0 :=
    (((continuousAt_id.pow 2).add continuousAt_const).add continuousAt_const).eventually_ne ht
  have hsqrt (t : ℝ) (ht : t ^ 2 + y ^ 2 + z ^ 2 ≠ 0) :
      Real.sqrt (t ^ 2 + y ^ 2 + z ^ 2) ≠ 0 :=
    Real.sqrt_ne_zero'.mpr (lt_of_le_of_ne (by positivity) (Ne.symm ht))
  have smooth (j : ℕ) (t : ℝ) (ht : t ^ 2 + y ^ 2 + z ^ 2 ≠ 0) :
      ContDiffAt ℝ ⊤ (iteratedDeriv j f) t := by
    induction j with
    | zero =>
      simp only [iteratedDeriv_zero]
      exact contDiffAt_const.div
        ((((contDiffAt_id.pow 2).add contDiffAt_const).add contDiffAt_const).sqrt ht) (hsqrt t ht)
    | succ j ih =>
      rw [iteratedDeriv_succ]
      exact ih.derivWithin (by simp)
  have hd (j : ℕ) (t : ℝ) (ht : t ^ 2 + y ^ 2 + z ^ 2 ≠ 0) :
      HasDerivAt (iteratedDeriv j f) (iteratedDeriv (j + 1) f t) t := by
    rw [iteratedDeriv_succ]
    exact (smooth j t ht).differentiableAt (by simp) |>.hasDerivAt
  have first (t : ℝ) (ht : t ^ 2 + y ^ 2 + z ^ 2 ≠ 0) :
      (t ^ 2 + y ^ 2 + z ^ 2) * iteratedDeriv 1 f t = -t * f t := by
    have hh := (((((hasDerivAt_id t).pow 2).add_const (y ^ 2)).add_const (z ^ 2)).sqrt ht).inv
      (hsqrt t ht)
    rw [iteratedDeriv_one]
    change (t ^ 2 + y ^ 2 + z ^ 2) * deriv (fun s : ℝ =>
      1 / Real.sqrt (s ^ 2 + y ^ 2 + z ^ 2)) t = -t * (1 / Real.sqrt (t ^ 2 + y ^ 2 + z ^ 2))
    simp only [one_div]
    simp only [Pi.inv_def] at hh
    dsimp at hh
    rw [hh.deriv]
    have hsq := Real.sq_sqrt (show 0 ≤ t ^ 2 + y ^ 2 + z ^ 2 by positivity)
    field_simp
    rw [hsq]
  have seed (t : ℝ) (ht : t ^ 2 + y ^ 2 + z ^ 2 ≠ 0) :
      (t ^ 2 + y ^ 2 + z ^ 2) * iteratedDeriv 2 f t +
        3 * t * iteratedDeriv 1 f t + f t = 0 := by
    have he : ∀ᶠ s in 𝓝 t, (s ^ 2 + y ^ 2 + z ^ 2) * iteratedDeriv 1 f s = -s * f s := by
      filter_upwards [hnhood t ht] with s hs using first s hs
    have hh := (((((hasDerivAt_id t).pow 2).add_const (y ^ 2)).add_const (z ^ 2)).mul
      (hd 1 t ht)).congr_of_eventuallyEq (he.mono fun _ hs => hs.symm)
    have hh' := hh.unique ((hasDerivAt_id t).neg.mul (hd 0 t ht))
    dsimp at hh'
    nlinarith
  have hrec (k : ℕ) : ∀ t : ℝ, t ^ 2 + y ^ 2 + z ^ 2 ≠ 0 →
      (t ^ 2 + y ^ 2 + z ^ 2) * iteratedDeriv (k + 2) f t +
        (2 * (k : ℝ) + 3) * t * iteratedDeriv (k + 1) f t +
        ((k : ℝ) + 1) ^ 2 * iteratedDeriv k f t = 0 := by
    induction k with
    | zero =>
      intro t ht
      simpa using seed t ht
    | succ k ih =>
      intro t ht
      have he : ∀ᶠ s in 𝓝 t,
          (s ^ 2 + y ^ 2 + z ^ 2) * iteratedDeriv (k + 2) f s +
            (2 * (k : ℝ) + 3) * s * iteratedDeriv (k + 1) f s +
            ((k : ℝ) + 1) ^ 2 * iteratedDeriv k f s = 0 := by
        filter_upwards [hnhood t ht] with s hs using ih s hs
      have hdq := (((hasDerivAt_id t).pow 2).add_const (y ^ 2)).add_const (z ^ 2)
      have hdx := (hasDerivAt_id t).const_mul (2 * (k : ℝ) + 3)
      have hdall := ((hdq.mul (hd (k + 2) t ht)).add
        (hdx.mul (hd (k + 1) t ht))).add
        ((hd k t ht).const_mul (((k : ℝ) + 1) ^ 2))
      have hh := hdall.congr_of_eventuallyEq (he.mono fun _ hs => hs.symm)
      have hh' := hh.unique (hasDerivAt_const t 0)
      dsimp at hh'
      simp only [Nat.cast_succ, Nat.add_assoc] at ⊢
      convert hh' using 1; ring
  have hr := hrec (n - 2) x h
  have h1 : n - 2 + 2 = n := by omega
  have h2 : n - 2 + 1 = n - 1 := by omega
  rw [h1, h2] at hr
  have hc : ((n - 2 : ℕ) : ℝ) = (n : ℝ) - 2 := by
    rw [Nat.cast_sub hn]; norm_num
  rw [hc] at hr
  rw [Real.sq_sqrt (by positivity)]
  change (x ^ 2 + y ^ 2 + z ^ 2) * iteratedDeriv n f x = _
  nlinarith

/-!
# Arbitrary ordered mixed derivatives of the inverse radius

The order is x first, then y, then z. Directional Fréchet derivatives along
coordinate unit vectors are the partial derivatives used here; no interchange
of partial derivatives is asserted. All identities are on the explicit open
nonorigin domain. Natural subtractions totalize the lower derivative orders;
their coefficients vanish when those orders would otherwise be negative.
-/

/-- Iterate the partial derivative in direction `v`. For a coordinate unit
vector this is the usual coordinate partial, with the other coordinates fixed. -/
noncomputable def iteratedDirectionalPartial {E : Type*} [NormedAddCommGroup E]
    [NormedSpace ℝ E] (v : E) : ℕ → (E → ℝ) → E → ℝ
  | 0, f => f
  | k + 1, f => fun p => fderiv ℝ (iteratedDirectionalPartial v k f) p v

private noncomputable abbrev J {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] :=
  @iteratedDirectionalPartial E _ _

section Helpers
variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
variable {U : Set E}

private def Smooth (U : Set E) (f : E → ℝ) :=
  ∀ p ∈ U, ContDiffAt ℝ ⊤ f p

private lemma smooth_jet {f : E → ℝ} (hf : Smooth U f) (v : E) (k : ℕ) :
    Smooth U (J v k f) := by
  induction k with
  | zero => exact hf
  | succ k ih =>
    intro p hp
    exact ((ih p hp).fderiv_right (by simp)).clm_apply contDiffAt_const

private lemma diff {f : E → ℝ} (hf : Smooth U f) {p : E} (hp : p ∈ U) :
    DifferentiableAt ℝ f p := (hf p hp).differentiableAt (by simp)

private lemma d_congr (hU : IsOpen U) {f g : E → ℝ} (h : EqOn f g U) {p : E} (hp : p ∈ U)
    (v : E) : fderiv ℝ f p v = fderiv ℝ g p v := by
  have he : f =ᶠ[𝓝 p] g := Filter.mem_of_superset (hU.mem_nhds hp) h
  rw [he.fderiv_eq]

private lemma d_mul {p : E} {f g : E → ℝ} (hf : DifferentiableAt ℝ f p)
    (hg : DifferentiableAt ℝ g p) (v : E) :
    fderiv ℝ (fun t => f t * g t) p v =
      fderiv ℝ f p v * g p + f p * fderiv ℝ g p v := by
  rw [fderiv_fun_mul hf hg]
  simp only [add_apply, smul_apply, smul_eq_mul]; ring

private lemma jet_congr (hU : IsOpen U) {f g : E → ℝ} (h : EqOn f g U) (v : E) (k : ℕ) :
    EqOn (J v k f) (J v k g) U := by
  induction k with
  | zero => exact h
  | succ k ih => exact fun p hp => d_congr hU ih hp v

private lemma jet_add (hU : IsOpen U) {f g : E → ℝ} (hf : Smooth U f) (hg : Smooth U g)
    (v : E) (k : ℕ) :
    EqOn (J v k (fun p => f p + g p)) (fun p => J v k f p + J v k g p) U := by
  induction k with
  | zero => intro p hp; rfl
  | succ k ih =>
    intro p hp
    change fderiv ℝ (J v k (fun p => f p + g p)) p v = _
    rw [d_congr hU ih hp v, fderiv_fun_add (diff (smooth_jet hf v k) hp)
      (diff (smooth_jet hg v k) hp)]
    rfl

private lemma jet_scale (hU : IsOpen U) {f : E → ℝ} (hf : Smooth U f) (v : E) (k : ℕ) (c : ℝ) :
    EqOn (J v k (fun p => c * f p)) (fun p => c * J v k f p) U := by
  induction k with
  | zero => intro p hp; rfl
  | succ k ih =>
    intro p hp
    change fderiv ℝ (J v k (fun p => c * f p)) p v = _
    rw [d_congr hU ih hp v, fderiv_const_mul (diff (smooth_jet hf v k) hp)]
    simp only [smul_apply, smul_eq_mul]
    rfl

private lemma jet_affine (hU : IsOpen U) {a f : E → ℝ} (ha : Smooth U a) (hf : Smooth U f)
    (v : E) (c : ℝ) (hda : ∀ p ∈ U, fderiv ℝ a p v = c) (k : ℕ) :
    EqOn (J v k (fun p => a p * f p))
      (fun p => a p * J v k f p + (k : ℝ) * c * J v (k - 1) f p) U := by
  induction k with
  | zero => intro p hp; simp [J, iteratedDirectionalPartial]
  | succ k ih =>
    intro p hp
    change fderiv ℝ (J v k (fun p => a p * f p)) p v = _
    rw [d_congr hU ih hp v]
    have hs := smooth_jet hf v k
    have ht := smooth_jet hf v (k - 1)
    rw [fderiv_fun_add (f := fun p => a p * J v k f p) ((diff ha hp).mul (diff hs hp))
      ((diff ht hp).const_mul _)]
    simp only [add_apply]
    rw [d_mul (diff ha hp) (diff hs hp), hda p hp,
      fderiv_const_mul (diff ht hp)]
    simp only [smul_apply, smul_eq_mul]
    cases k with
    | zero => simp [J, iteratedDirectionalPartial]; ring
    | succ k =>
      simp only [Nat.succ_sub_one, Nat.cast_succ]
      change c * J v (k + 1) f p + a p * J v (k + 2) f p +
        ((k : ℝ) + 1) * c * J v (k + 1) f p = _
      ring

private lemma jet_quadratic (hU : IsOpen U) {q a f : E → ℝ} (hq : Smooth U q) (ha : Smooth U a)
    (hf : Smooth U f) (v : E)
    (hdq : ∀ p ∈ U, fderiv ℝ q p v = 2 * a p)
    (hda : ∀ p ∈ U, fderiv ℝ a p v = 1) (k : ℕ) :
    EqOn (J v k (fun p => q p * f p))
      (fun p => q p * J v k f p + 2 * (k : ℝ) * a p * J v (k - 1) f p +
        (k : ℝ) * ((k - 1 : ℕ) : ℝ) * J v (k - 2) f p) U := by
  induction k with
  | zero => intro p hp; simp [J, iteratedDirectionalPartial]
  | succ k ih =>
    intro p hp
    change fderiv ℝ (J v k (fun p => q p * f p)) p v = _
    rw [d_congr hU ih hp v]
    have hs := smooth_jet hf v k
    have ht := smooth_jet hf v (k - 1)
    have hu := smooth_jet hf v (k - 2)
    rw [fderiv_fun_add (f := fun p => q p * J v k f p + 2 * (k : ℝ) * a p * J v (k - 1) f p) (((diff hq hp).mul (diff hs hp)).add
      (((diff ha hp).const_mul (2 * (k : ℝ))).mul (diff ht hp))) ((diff hu hp).const_mul _)]
    simp only [add_apply]
    rw [fderiv_fun_add (f := fun p => q p * J v k f p) (g := fun p => 2 * (k : ℝ) * a p * J v (k - 1) f p) ((diff hq hp).mul (diff hs hp))
      (((diff ha hp).const_mul (2 * (k : ℝ))).mul (diff ht hp))]
    simp only [add_apply]
    rw [d_mul (diff hq hp) (diff hs hp), hdq p hp,
      d_mul ((diff ha hp).const_mul (2 * (k : ℝ))) (diff ht hp),
      fderiv_const_mul (diff ha hp),
      fderiv_const_mul (diff hu hp)]
    simp only [smul_apply, smul_eq_mul]
    rw [hda p hp]
    cases k with
    | zero => simp [J, iteratedDirectionalPartial]; ring
    | succ k =>
      cases k with
      | zero => simp [J, iteratedDirectionalPartial]; ring
      | succ k =>
        simp only [Nat.succ_sub_one, Nat.cast_succ]
        have he : k + 1 + 1 - 2 = k := by omega
        simp only [he, show k + 1 + 1 + 1 - 2 = k + 1 by omega]
        change 2 * a p * J v (k + 2) f p + q p * J v (k + 3) f p +
          (2 * ((k : ℝ) + 1 + 1) * 1 * J v (k + 1) f p +
            2 * ((k : ℝ) + 1 + 1) * a p * J v (k + 2) f p) +
          ((k : ℝ) + 1 + 1) * ((k : ℝ) + 1) * J v (k + 1) f p = _
        ring

private lemma jet_shift (v : E) (k r : ℕ) (f : E → ℝ) :
    J v k (J v r f) = J v (k + r) f := by
  induction k with
  | zero => simp [J, iteratedDirectionalPartial]
  | succ k ih =>
    simp only [J, iteratedDirectionalPartial, ih, Nat.add_right_comm k 1 r]

-- The first-order kernel identity implies the pure recurrence, including n=1.
private lemma pure_recurrence (hU : IsOpen U) {q a f : E → ℝ}
    (hq : Smooth U q) (ha : Smooth U a) (hf : Smooth U f) (v : E)
    (hdq : ∀ p ∈ U, fderiv ℝ q p v = 2 * a p)
    (hda : ∀ p ∈ U, fderiv ℝ a p v = 1)
    (first : EqOn (fun p => q p * J v 1 f p) (fun p => -1 * (a p * f p)) U)
    (n : ℕ) (hn : 1 ≤ n) {p : E} (hp : p ∈ U) :
    q p * J v n f p = -(2 * (n : ℝ) - 1) * a p * J v (n - 1) f p -
      ((n - 1 : ℕ) : ℝ) ^ 2 * J v (n - 2) f p := by
  obtain ⟨k, rfl⟩ : ∃ k, n = k + 1 := ⟨n - 1, by omega⟩
  have he := jet_congr hU first v k hp
  simp only [jet_quadratic hU hq ha (smooth_jet hf v 1) v hdq hda k hp,
    jet_scale hU (fun t ht => (ha t ht).mul (hf t ht)) v k (-1) hp,
    jet_affine hU ha hf v 1 hda k hp] at he
  simp only [jet_shift] at he
  cases k with
  | zero =>
    norm_num [J, iteratedDirectionalPartial] at he ⊢
    nlinarith
  | succ k =>
    cases k with
    | zero =>
      norm_num at he ⊢
      nlinarith
    | succ k =>

      have h2 : k + 1 + 1 - 2 + 1 = k + 1 := by omega
      simp only [h2, Nat.succ_sub_one, Nat.cast_succ] at he
      simp only [Nat.succ_sub_one, Nat.cast_succ,
        show k + 1 + 1 + 1 - 2 = k + 1 by omega]
      nlinarith

-- A transverse differentiation: only the radius factor depends quadratically
-- on the new coordinate; `a` is constant in that direction.
private lemma transverse (hU : IsOpen U) {q b a f g h : E → ℝ}
    (hq : Smooth U q) (hb : Smooth U b) (ha : Smooth U a)
    (hf : Smooth U f) (hg : Smooth U g) (hh : Smooth U h)
    (v : E) (hdq : ∀ p ∈ U, fderiv ℝ q p v = 2 * b p)
    (hdb : ∀ p ∈ U, fderiv ℝ b p v = 1)
    (hda : ∀ p ∈ U, fderiv ℝ a p v = 0)
    (A B : ℝ)
    (he : EqOn (fun p => q p * f p) (fun p => A * (a p * g p) + B * h p) U)
    (k : ℕ) {p : E} (hp : p ∈ U) :
    q p * J v k f p = A * a p * J v k g p + B * J v k h p -
      2 * (k : ℝ) * b p * J v (k - 1) f p -
      (k : ℝ) * ((k - 1 : ℕ) : ℝ) * J v (k - 2) f p := by
  have hj := jet_congr hU he v k hp
  simp only [jet_quadratic hU hq hb hf v hdq hdb k hp,
    jet_add hU (fun t ht => contDiffAt_const.mul ((ha t ht).mul (hg t ht)))
      (fun t ht => contDiffAt_const.mul (hh t ht)) v k hp,
    jet_scale hU (fun t ht => (ha t ht).mul (hg t ht)) v k A hp,
    jet_scale hU hh v k B hp, jet_affine hU ha hg v 0 hda k hp] at hj
  nlinarith

private lemma jet_linear (hU : IsOpen U) {a g h : E → ℝ}
    (ha : Smooth U a) (hg : Smooth U g) (hh : Smooth U h)
    (v : E) (hda : ∀ p ∈ U, fderiv ℝ a p v = 0) (A B : ℝ)
    (k : ℕ) {p : E} (hp : p ∈ U) :
    J v k (fun p => A * (a p * g p) + B * h p) p =
      A * (a p * J v k g p) + B * J v k h p := by
  simp only [jet_add hU (fun t ht => contDiffAt_const.mul ((ha t ht).mul (hg t ht)))
      (fun t ht => contDiffAt_const.mul (hh t ht)) v k hp,
    jet_scale hU (fun t ht => (ha t ht).mul (hg t ht)) v k A hp,
    jet_scale hU hh v k B hp, jet_affine hU ha hg v 0 hda k hp]
  ring

end Helpers

private def radiusSq {d : ℕ} (p : Fin d → ℝ) : ℝ := ∑ i, p i ^ 2
private noncomputable def radial {d : ℕ} (p : Fin d → ℝ) : ℝ :=
  1 / Real.sqrt (radiusSq p)
private def unit {d : ℕ} (i : Fin d) : Fin d → ℝ := Pi.single i 1
private def away (d : ℕ) : Set (Fin d → ℝ) := {p | radiusSq p ≠ 0}

private lemma radiusSq_smooth {d : ℕ} (p : Fin d → ℝ) :
    ContDiffAt ℝ ⊤ radiusSq p := by unfold radiusSq; fun_prop

private lemma away_open (d : ℕ) : IsOpen (away d) := by
  exact isOpen_ne_fun (continuous_iff_continuousAt.mpr fun p =>
    (radiusSq_smooth p).continuousAt) continuous_const

private lemma coordinate_smooth {d : ℕ} (i : Fin d) :
    Smooth (away d) (fun p => p i) := by intro p hp; fun_prop

private lemma coordinate_d {d : ℕ} (i j : Fin d) (p : Fin d → ℝ) :
    fderiv ℝ (fun p : Fin d → ℝ => p j) p (unit i) = if i = j then 1 else 0 := by
  rw [(hasFDerivAt_apply j p).fderiv]
  simp [unit, Pi.single_apply, eq_comm]

private lemma radiusSq_d {d : ℕ} (i : Fin d) (p : Fin d → ℝ) :
    fderiv ℝ radiusSq p (unit i) = 2 * p i := by
  unfold radiusSq
  rw [fderiv_fun_sum (fun j hj => (differentiableAt_apply j p).pow 2)]
  have hd (j : Fin d) :
      fderiv ℝ (fun t : Fin d → ℝ => t j ^ 2) p (unit i) =
        2 * p j * (if i = j then 1 else 0) := by
    rw [fderiv_fun_pow 2 (differentiableAt_apply (𝕜 := ℝ) j p)]
    simp [smul_apply, coordinate_d]
  simp only [sum_apply, hd]
  simp [mul_ite]

private lemma radiusSq_nonneg {d : ℕ} (p : Fin d → ℝ) : 0 ≤ radiusSq p := by
  unfold radiusSq; positivity

private lemma radial_smooth (d : ℕ) : Smooth (away d) radial := by
  intro p hp
  have hroot : Real.sqrt (radiusSq p) ≠ 0 := Real.sqrt_ne_zero'.mpr
    (lt_of_le_of_ne (radiusSq_nonneg p) (Ne.symm hp))
  exact contDiffAt_const.div ((radiusSq_smooth p).sqrt hp) hroot

private lemma radial_first {d : ℕ} (i : Fin d) :
    EqOn (fun p => radiusSq p * J (unit i) 1 radial p)
      (fun p => -1 * (p i * radial p)) (away d) := by
  intro p hp
  have hroot : Real.sqrt (radiusSq p) ≠ 0 := Real.sqrt_ne_zero'.mpr
    (lt_of_le_of_ne (radiusSq_nonneg p) (Ne.symm hp))
  have hh := (hasDerivAt_inv hroot).comp_hasFDerivAt p
    (((radiusSq_smooth p).differentiableAt (by simp)).hasFDerivAt.sqrt hp)
  change radiusSq p * fderiv ℝ (fun t => 1 / Real.sqrt (radiusSq t)) p (unit i) = _
  simp only [one_div] at hh ⊢
  dsimp only [Function.comp_def] at hh
  rw [hh.fderiv]
  simp only [smul_apply, smul_eq_mul, radiusSq_d]
  have hsq := Real.sq_sqrt (radiusSq_nonneg p)
  dsimp [radial]
  field_simp
  rw [hsq]

private lemma radial_pure {d : ℕ} (i : Fin d) (n : ℕ) (hn : 1 ≤ n)
    {p : Fin d → ℝ} (hp : p ∈ away d) :
    radiusSq p * J (unit i) n radial p =
      -(2 * (n : ℝ) - 1) * p i * J (unit i) (n - 1) radial p -
      ((n - 1 : ℕ) : ℝ) ^ 2 * J (unit i) (n - 2) radial p := by
  exact pure_recurrence (away_open d) (fun p hp => radiusSq_smooth p)
    (coordinate_smooth i) (radial_smooth d) (unit i)
    (fun p hp => radiusSq_d i p) (fun p hp => by simp [coordinate_d])
    (radial_first i) n hn hp

private noncomputable def mixed {d : ℕ} (i j : Fin d) (n m : ℕ) :
    (Fin d → ℝ) → ℝ := J (unit j) m (J (unit i) n radial)

private lemma radial_mixed {d : ℕ} (i j : Fin d) (hij : i ≠ j)
    (n m : ℕ) (hn : 1 ≤ n) {p : Fin d → ℝ} (hp : p ∈ away d) :
    radiusSq p * mixed i j n m p =
      -(2 * (n : ℝ) - 1) * p i * mixed i j (n - 1) m p -
      ((n - 1 : ℕ) : ℝ) ^ 2 * mixed i j (n - 2) m p -
      2 * (m : ℝ) * p j * mixed i j n (m - 1) p -
      (m : ℝ) * ((m - 1 : ℕ) : ℝ) * mixed i j n (m - 2) p := by
  have he : EqOn (fun p => radiusSq p * J (unit i) n radial p)
      (fun p => -(2 * (n : ℝ) - 1) * (p i * J (unit i) (n - 1) radial p) +
        -((n - 1 : ℕ) : ℝ) ^ 2 * J (unit i) (n - 2) radial p) (away d) := by
    intro t ht
    have h := radial_pure i n hn ht
    nlinarith
  have h := transverse (away_open d) (fun p hp => radiusSq_smooth p)
    (coordinate_smooth j) (coordinate_smooth i)
    (smooth_jet (radial_smooth d) (unit i) n)
    (smooth_jet (radial_smooth d) (unit i) (n - 1))
    (smooth_jet (radial_smooth d) (unit i) (n - 2))
    (unit j) (fun p hp => radiusSq_d j p)
    (fun p hp => by simp [coordinate_d])
    (fun p hp => by simp [coordinate_d, Ne.symm hij])
    (-(2 * (n : ℝ) - 1)) (-((n - 1 : ℕ) : ℝ) ^ 2) he m hp
  dsimp [mixed]
  nlinarith


private lemma mixed_smooth {d : ℕ} (i j : Fin d) (n m : ℕ) :
    Smooth (away d) (mixed i j n m) :=
  smooth_jet (smooth_jet (radial_smooth d) (unit i) n) (unit j) m

private lemma radial_triple {d : ℕ} (i j k : Fin d)
    (hij : i ≠ j) (hik : i ≠ k) (hjk : j ≠ k)
    (n m l : ℕ) (hn : 1 ≤ n) {p : Fin d → ℝ} (hp : p ∈ away d) :
    radiusSq p * J (unit k) l (mixed i j n m) p =
      -(2 * (n : ℝ) - 1) * p i * J (unit k) l (mixed i j (n - 1) m) p -
      ((n - 1 : ℕ) : ℝ) ^ 2 * J (unit k) l (mixed i j (n - 2) m) p -
      2 * (m : ℝ) * p j * J (unit k) l (mixed i j n (m - 1)) p -
      (m : ℝ) * ((m - 1 : ℕ) : ℝ) * J (unit k) l (mixed i j n (m - 2)) p -
      2 * (l : ℝ) * p k * J (unit k) (l - 1) (mixed i j n m) p -
      (l : ℝ) * ((l - 1 : ℕ) : ℝ) * J (unit k) (l - 2) (mixed i j n m) p := by
  let A : ℝ := -(2 * (n : ℝ) - 1)
  let B : ℝ := -((n - 1 : ℕ) : ℝ) ^ 2
  let C : ℝ := -2 * (m : ℝ)
  let D : ℝ := -(m : ℝ) * ((m - 1 : ℕ) : ℝ)
  let r1 := fun t : Fin d → ℝ =>
    A * (t i * mixed i j (n - 1) m t) + B * mixed i j (n - 2) m t
  let r2 := fun t : Fin d → ℝ =>
    C * (t j * mixed i j n (m - 1) t) + D * mixed i j n (m - 2) t
  have hs1 : Smooth (away d) r1 := by
    intro t ht
    exact (contDiffAt_const.mul ((coordinate_smooth i t ht).mul
      (mixed_smooth i j (n - 1) m t ht))).add
      (contDiffAt_const.mul (mixed_smooth i j (n - 2) m t ht))
  have hs2 : Smooth (away d) r2 := by
    intro t ht
    exact (contDiffAt_const.mul ((coordinate_smooth j t ht).mul
      (mixed_smooth i j n (m - 1) t ht))).add
      (contDiffAt_const.mul (mixed_smooth i j n (m - 2) t ht))
  have he : EqOn (fun t => radiusSq t * mixed i j n m t)
      (fun t => r1 t + r2 t) (away d) := by
    intro t ht
    have h := radial_mixed i j hij n m hn ht
    dsimp [r1, r2, A, B, C, D]
    nlinarith
  have ht := jet_congr (away_open d) he (unit k) l hp
  simp only [jet_quadratic (away_open d) (fun t ht => radiusSq_smooth t)
      (coordinate_smooth k) (mixed_smooth i j n m) (unit k)
      (fun t ht => radiusSq_d k t) (fun t ht => by simp [coordinate_d]) l hp,
    jet_add (away_open d) hs1 hs2 (unit k) l hp] at ht
  dsimp only [r1, r2] at ht
  simp only [jet_linear (away_open d) (coordinate_smooth i)
      (mixed_smooth i j (n - 1) m) (mixed_smooth i j (n - 2) m)
      (unit k) (fun t ht => by simp [coordinate_d, Ne.symm hik]) A B l hp,
    jet_linear (away_open d) (coordinate_smooth j)
      (mixed_smooth i j n (m - 1)) (mixed_smooth i j n (m - 2))
      (unit k) (fun t ht => by simp [coordinate_d, Ne.symm hjk]) C D l hp] at ht
  dsimp [A, B, C, D] at ht
  nlinarith

/-- Ordered x/y mixed partial of the three-dimensional kernel, with z held
fixed: take n x partials first, then m y partials, evaluated at (x,y,z).
The name refers to the two differentiation directions, not the kernel dimension. -/
noncomputable def invRadiusMixed2D (n m : ℕ) (x y z : ℝ) : ℝ :=
  iteratedDirectionalPartial (Pi.single (1 : Fin 3) 1) m
    (iteratedDirectionalPartial (Pi.single (0 : Fin 3) 1) n
      (fun p : Fin 3 → ℝ => invRadius (p 0) (p 1) (p 2))) ![x, y, z]

/-- Ordered mixed partial of the actual three-variable kernel: take n x
partials, then m y partials, then l z partials, evaluated at (x,y,z). -/
noncomputable def invRadiusMixed3D (n m l : ℕ) (x y z : ℝ) : ℝ :=
  iteratedDirectionalPartial (Pi.single (2 : Fin 3) 1) l
    (iteratedDirectionalPartial (Pi.single (1 : Fin 3) 1) m
      (iteratedDirectionalPartial (Pi.single (0 : Fin 3) 1) n
        (fun p : Fin 3 → ℝ => invRadius (p 0) (p 1) (p 2)))) ![x, y, z]

private lemma radial_eq3 : (radial : (Fin 3 → ℝ) → ℝ) =
    (fun p => invRadius (p 0) (p 1) (p 2)) := by
  funext p
  simp [radial, radiusSq, invRadius, Fin.sum_univ_succ, add_assoc]

/-- Arbitrary x/y mixed recurrence for the three-dimensional inverse radius,
off the origin, with z fixed and x differentiated before y.
Here n ≥ 1 and m is arbitrary. The n-2 term has zero coefficient at n=1; the
first lower y term has zero coefficient at m=0 and the second at m<2. These are
vanishing-coefficient statements, not claims that eligible terms are nonzero. -/
theorem invRadius_mixed_recurrence_2D (n m : ℕ) (hn : 1 ≤ n) (x y z : ℝ)
    (h : x ^ 2 + y ^ 2 + z ^ 2 ≠ 0) :
    (Real.sqrt (x ^ 2 + y ^ 2 + z ^ 2)) ^ 2 * invRadiusMixed2D n m x y z =
      -(2 * (n : ℝ) - 1) * x * invRadiusMixed2D (n - 1) m x y z -
      ((n - 1 : ℕ) : ℝ) ^ 2 * invRadiusMixed2D (n - 2) m x y z -
      2 * (m : ℝ) * y * invRadiusMixed2D n (m - 1) x y z -
      (m : ℝ) * ((m - 1 : ℕ) : ℝ) * invRadiusMixed2D n (m - 2) x y z := by
  have hp : ![x, y, z] ∈ away 3 := by
    simpa [away, radiusSq, Fin.sum_univ_succ, add_assoc] using h
  have hr := radial_mixed (0 : Fin 3) 1 (by decide) n m hn hp
  rw [Real.sq_sqrt (by positivity)]
  simpa [radiusSq, Fin.sum_univ_succ, add_assoc, mixed, unit, radial_eq3,
    invRadiusMixed2D] using hr

/-- Arbitrary three-variable inverse-radius mixed recurrence, equation
(eq:deriv_laplace_3d). The radius is genuinely
sqrt(x²+y²+z²), and the order is x, then y, then z. The origin is excluded;
n ≥ 1 while m,l are arbitrary naturals. All lower orders use natural
subtraction and carry coefficients that vanish at the boundary. Even when
a coefficient can be nonzero, the corresponding derivative term need not be.
No interchange theorem or operation-count bound is asserted. -/
theorem invRadius_mixed_recurrence_3D (n m l : ℕ) (hn : 1 ≤ n) (x y z : ℝ)
    (h : x ^ 2 + y ^ 2 + z ^ 2 ≠ 0) :
    (Real.sqrt (x ^ 2 + y ^ 2 + z ^ 2)) ^ 2 * invRadiusMixed3D n m l x y z =
      -(2 * (n : ℝ) - 1) * x * invRadiusMixed3D (n - 1) m l x y z -
      ((n - 1 : ℕ) : ℝ) ^ 2 * invRadiusMixed3D (n - 2) m l x y z -
      2 * (m : ℝ) * y * invRadiusMixed3D n (m - 1) l x y z -
      (m : ℝ) * ((m - 1 : ℕ) : ℝ) * invRadiusMixed3D n (m - 2) l x y z -
      2 * (l : ℝ) * z * invRadiusMixed3D n m (l - 1) x y z -
      (l : ℝ) * ((l - 1 : ℕ) : ℝ) * invRadiusMixed3D n m (l - 2) x y z := by
  have hp : ![x, y, z] ∈ away 3 := by
    simpa [away, radiusSq, Fin.sum_univ_succ, add_assoc] using h
  have hr := radial_triple (0 : Fin 3) 1 2 (by decide) (by decide) (by decide)
    n m l hn hp
  rw [Real.sq_sqrt (by positivity)]
  simpa [radiusSq, Fin.sum_univ_succ, add_assoc, mixed, unit, radial_eq3,
    invRadiusMixed3D] using hr


/-!
# Two-dimensional biharmonic derivative recurrences

The kernel is exactly `(x² + y²) * log (sqrt (x² + y²))`.
A quadratic Leibniz formula reduces the pure recurrence to the existing log-radius
recurrence. Differentiating that identity in y gives the ordered mixed recurrence.
All smoothness needed in the proofs is established locally away from the origin.
-/

/-- The actual two-dimensional biharmonic kernel `r² log r`, with y fixed.
`Real.log`, `Real.sqrt`, and derivatives are totalized; our identities are only
asserted off the origin, where they have their classical meaning. -/
noncomputable def biharmonicRadius (y x : ℝ) : ℝ :=
  (x ^ 2 + y ^ 2) * Real.log (Real.sqrt (x ^ 2 + y ^ 2))

private theorem quadratic_leibniz (f : ℝ → ℝ) (a b c x : ℝ) (m : ℕ)
    (hf : ContDiffAt ℝ ⊤ f x) :
    iteratedDeriv m (fun t => (a * t ^ 2 + b * t + c) * f t) x =
      (a * x ^ 2 + b * x + c) * iteratedDeriv m f x +
      (m : ℝ) * (2 * a * x + b) * iteratedDeriv (m - 1) f x +
      (m : ℝ) * ((m : ℝ) - 1) * a * iteratedDeriv (m - 2) f x := by
  have hq (j : ℕ) : iteratedDeriv j (fun t : ℝ => a * t ^ 2 + b * t + c) x =
      a * ((2 : ℕ).descFactorial j * x ^ (2 - j)) +
      b * (if j = 0 then x else if j = 1 then 1 else 0) +
      (if j = 0 then c else 0) := by
    rw [iteratedDeriv_fun_add (by fun_prop) (by fun_prop),
      iteratedDeriv_fun_add (by fun_prop) (by fun_prop)]
    rw [iteratedDeriv_const_mul_field a (fun t : ℝ => t ^ 2),
      iteratedDeriv_const_mul_field b (fun t : ℝ => t)]
    simp only [iteratedDeriv_pow, iteratedDeriv_fun_id, iteratedDeriv_const]

  rw [iteratedDeriv_fun_mul (by fun_prop) (hf.of_le (by simp))]
  cases m with
  | zero => simp [hq]
  | succ m =>
    cases m with
    | zero =>
      norm_num [Finset.sum_range_succ, hq]
      exact Or.inl (by ring)
    | succ k =>
      rw [show k + 1 + 1 + 1 = 3 + k by omega, Finset.sum_range_add]
      have hz : ∑ i ∈ Finset.range k,
          ((k + 1 + 1).choose (3 + i) : ℝ) *
            iteratedDeriv (3 + i) (fun t : ℝ => a * t ^ 2 + b * t + c) x *
            iteratedDeriv (k + 1 + 1 - (3 + i)) f x = 0 := by
        apply Finset.sum_eq_zero
        intro i hi
        rw [hq]
        simp [Nat.descFactorial_eq_zero_iff_lt.mpr (by omega : 2 < 3 + i),
          show 3 + i ≠ 1 by omega]
      rw [hz, add_zero]
      simp only [Finset.sum_range_succ, Finset.sum_range_zero, hq]
      rw [Nat.cast_choose_two]
      norm_num [Nat.cast_add, Nat.add_assoc]
      ring

private theorem log_smooth (x y : ℝ) (h : x ^ 2 + y ^ 2 ≠ 0) :
    ContDiffAt ℝ ⊤ (logRadius y) x := by
  exact ((contDiffAt_id.pow 2).add contDiffAt_const).sqrt h |>.log
    (Real.sqrt_ne_zero'.mpr (lt_of_le_of_ne (by positivity) (Ne.symm h)))

/-- The pure x recurrence in the Biharmonic 2D subsection, for `n ≥ 5`. -/
theorem biharmonicRadius_x_recurrence (n : ℕ) (hn : 5 ≤ n) (x y : ℝ)
    (h : x ^ 2 + y ^ 2 ≠ 0) :
    (Real.sqrt (x ^ 2 + y ^ 2)) ^ 2 * iteratedDeriv n (biharmonicRadius y) x =
      -2 * ((n : ℝ) - 2) * x * iteratedDeriv (n - 1) (biharmonicRadius y) x -
      ((n : ℝ) - 1) * ((n : ℝ) - 4) *
        iteratedDeriv (n - 2) (biharmonicRadius y) x := by
  have hb (j : ℕ) := quadratic_leibniz (logRadius y) 1 0 (y ^ 2) x j (log_smooth x y h)
  simp only [one_mul, zero_mul, add_zero] at hb
  change ∀ j, iteratedDeriv j (biharmonicRadius y) x = _ at hb
  have r0 := logRadius_x_recurrence n (by omega) x y h
  have r1 := logRadius_x_recurrence (n - 1) (by omega) x y h
  have r2 := logRadius_x_recurrence (n - 2) (by omega) x y h
  rw [Real.sq_sqrt (by positivity)] at r0 r1 r2 ⊢
  rw [hb n, hb (n - 1), hb (n - 2)]
  have c1 : ((n - 1 : ℕ) : ℝ) = (n : ℝ) - 1 := by
    rw [Nat.cast_sub (by omega)]; norm_num
  have c2 : ((n - 2 : ℕ) : ℝ) = (n : ℝ) - 2 := by
    rw [Nat.cast_sub (by omega)]; norm_num
  simp only [c1, c2, Nat.sub_sub] at r1 r2 ⊢
  linear_combination (x ^ 2 + y ^ 2) * r0 +
    (2 * ((n : ℝ) - 1) * x) * r1 + (((n : ℝ) - 1) * ((n : ℝ) - 2)) * r2

-- Smoothness in both variables is retained while taking the x derivatives.
-- This is parameter differentiation, not an interchange of x and y derivatives.
private theorem biharmonic_x_joint_smooth (j : ℕ) (p : ℝ × ℝ)
    (h : p.1 ^ 2 + p.2 ^ 2 ≠ 0) :
    ContDiffAt ℝ ⊤ (fun q : ℝ × ℝ => iteratedDeriv j (biharmonicRadius q.2) q.1) p := by
  induction j generalizing p with
  | zero =>
    simp only [iteratedDeriv_zero]
    have hq : ContDiffAt ℝ ⊤ (fun q : ℝ × ℝ => q.1 ^ 2 + q.2 ^ 2) p := by
      fun_prop
    exact hq.mul (hq.sqrt h |>.log
      (Real.sqrt_ne_zero'.mpr (lt_of_le_of_ne (by positivity) (Ne.symm h))))
  | succ j ih =>
    have hj : ContDiffAt ℝ ⊤
        (Function.uncurry (fun (q : ℝ × ℝ) (t : ℝ) =>
          iteratedDeriv j (biharmonicRadius q.2) t)) (p, p.1) := by
      exact (ih p h).comp (p, p.1) (show ContDiffAt ℝ ⊤
        (fun q : (ℝ × ℝ) × ℝ => (q.2, q.1.2)) (p, p.1) by fun_prop)
    have hd := hj.fderiv (show ContDiffAt ℝ ⊤ (fun q : ℝ × ℝ => q.1) p by
      fun_prop) (by simp)
    simpa only [iteratedDeriv_succ, deriv] using hd.clm_apply contDiffAt_const

private theorem biharmonic_x_y_smooth (j : ℕ) (x y : ℝ)
    (h : x ^ 2 + y ^ 2 ≠ 0) :
    ContDiffAt ℝ ⊤ (fun s => iteratedDeriv j (biharmonicRadius s) x) y :=
  (biharmonic_x_joint_smooth j (x, y) h).comp y (by fun_prop)

/-- Ordered mixed partial: first `n` x derivatives, then `m` y derivatives.
This choice lets us differentiate the pure identity directly; no separate
interchange theorem for partial derivatives is asserted or needed. -/
noncomputable def biharmonicMixed (n m : ℕ) (x y : ℝ) : ℝ :=
  iteratedDeriv m (fun s => iteratedDeriv n (biharmonicRadius s) x) y

/-- Equation `eq:deriv_biharmonic_2d`, for `n ≥ 5` and every natural `m ≥ 0`.
The nonorigin hypothesis is essential to the intended classical interpretation.
Natural subtraction in the last two derivative orders is truncated at low m.
At `m = 0` both correction coefficients vanish; at `m = 1` the last coefficient
vanishes. Thus every derivative value introduced by truncation is irrelevant. The paper's “nonzero” for these terms means *potentially*
nonzero, not that every allowed point and derivative order gives a nonzero value.
No operation-count claim is formalized. -/
theorem biharmonicRadius_mixed_recurrence (n m : ℕ) (hn : 5 ≤ n) (x y : ℝ)
    (h : x ^ 2 + y ^ 2 ≠ 0) :
    (Real.sqrt (x ^ 2 + y ^ 2)) ^ 2 * biharmonicMixed n m x y =
      -2 * ((n : ℝ) - 2) * x * biharmonicMixed (n - 1) m x y -
      ((n : ℝ) - 1) * ((n : ℝ) - 4) * biharmonicMixed (n - 2) m x y -
      2 * (m : ℝ) * y * biharmonicMixed n (m - 1) x y -
      (m : ℝ) * ((m : ℝ) - 1) * biharmonicMixed n (m - 2) x y := by
  have he : (fun s => (x ^ 2 + s ^ 2) * iteratedDeriv n (biharmonicRadius s) x) =ᶠ[𝓝 y]
      (fun s => -2 * ((n : ℝ) - 2) * x * iteratedDeriv (n - 1) (biharmonicRadius s) x -
        ((n : ℝ) - 1) * ((n : ℝ) - 4) * iteratedDeriv (n - 2) (biharmonicRadius s) x) := by
    have hh : ∀ᶠ s in 𝓝 y, x ^ 2 + s ^ 2 ≠ 0 :=
      (continuousAt_const.add (continuousAt_id.pow 2)).eventually_ne h
    filter_upwards [hh] with s hs
    simpa only [Real.sq_sqrt (show 0 ≤ x ^ 2 + s ^ 2 by positivity)] using
      biharmonicRadius_x_recurrence n hn x s hs
  have hd := he.iteratedDeriv_eq m
  have hq := quadratic_leibniz
    (fun s => iteratedDeriv n (biharmonicRadius s) x) 1 0 (x ^ 2) y m
    (biharmonic_x_y_smooth n x y h)
  simp only [one_mul, zero_mul, add_zero] at hq
  have hswap : (fun s => (x ^ 2 + s ^ 2) * iteratedDeriv n (biharmonicRadius s) x) =
      (fun s => (s ^ 2 + x ^ 2) * iteratedDeriv n (biharmonicRadius s) x) := by
    funext s; rw [add_comm]
  rw [hswap, hq] at hd
  rw [iteratedDeriv_fun_sub
    ((contDiffAt_const.mul (biharmonic_x_y_smooth (n - 1) x y h)).of_le (by simp))
    ((contDiffAt_const.mul (biharmonic_x_y_smooth (n - 2) x y h)).of_le (by simp))] at hd
  simp only [iteratedDeriv_const_mul_field] at hd
  rw [Real.sq_sqrt (by positivity)]
  dsimp only [biharmonicMixed]
  linear_combination hd

end PDETranslations.Recurrences
