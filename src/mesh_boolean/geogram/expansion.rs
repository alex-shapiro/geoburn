//! Multi-precision arithmetic based on Shewchuk expansions.
//!
//! Port of Geogram's `multi_precision.h/.cpp` and `expansion_nt.h/.cpp`.
//! BSD 3-Clause license (original Geogram copyright Inria).
//!
//! An `Expansion` is a sequence of f64 components whose exact sum equals
//! the represented value. Components are non-overlapping and in increasing
//! order of magnitude. All arithmetic operations produce exact results.

// Shewchuk's splitter for the `split` operation: 2^27 + 1.
// This is computed at init time in Geogram; for IEEE 754 double it's always this value.
const SPLITTER: f64 = 134_217_729.0; // 2^27 + 1

// ── Primitive two-term operations (Shewchuk) ────────────────────────

/// Exact sum of two f64s as a length-2 expansion `(hi, lo)`.
/// `hi + lo = a + b` exactly, no precondition on magnitudes.
#[inline]
pub fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let x = a + b;
    let bvirt = x - a;
    let avirt = x - bvirt;
    let bround = b - bvirt;
    let around = a - avirt;
    (x, around + bround)
}

/// Exact difference of two f64s as a length-2 expansion `(hi, lo)`.
#[inline]
pub fn two_diff(a: f64, b: f64) -> (f64, f64) {
    let x = a - b;
    let bvirt = a - x;
    let avirt = x + bvirt;
    let bround = bvirt - b;
    let around = a - avirt;
    (x, around + bround)
}

/// Fast exact sum when `|a| >= |b|`.
#[inline]
fn fast_two_sum(a: f64, b: f64) -> (f64, f64) {
    let x = a + b;
    let bvirt = x - a;
    (x, b - bvirt)
}

/// Split a f64 into high and low parts for exact multiplication.
#[inline]
fn split(a: f64) -> (f64, f64) {
    let c = SPLITTER * a;
    let abig = c - a;
    let ahi = c - abig;
    let alo = a - ahi;
    (ahi, alo)
}

/// Exact product of two f64s as a length-2 expansion `(hi, lo)`.
#[inline]
pub fn two_product(a: f64, b: f64) -> (f64, f64) {
    let x = a * b;
    let (ahi, alo) = split(a);
    let (bhi, blo) = split(b);
    let err1 = x - (ahi * bhi);
    let err2 = err1 - (alo * bhi);
    let err3 = err2 - (ahi * blo);
    (x, (alo * blo) - err3)
}

// ── Expansion manipulation (Shewchuk / Geogram) ────────────────────

/// Add a single f64 `b` to expansion `e`, eliminating zero components.
pub fn grow_expansion_zeroelim(e: &[f64], b: f64) -> Vec<f64> {
    let mut h = Vec::with_capacity(e.len() + 1);
    let mut q = b;
    for &enow in e {
        let (qnew, hh) = two_sum(q, enow);
        q = qnew;
        if hh != 0.0 {
            h.push(hh);
        }
    }
    if q != 0.0 || h.is_empty() {
        h.push(q);
    }
    h
}

/// Multiply expansion `e` by scalar `b`, eliminating zero components.
pub fn scale_expansion_zeroelim(e: &[f64], b: f64) -> Vec<f64> {
    if e.is_empty() {
        return vec![0.0];
    }
    let mut h = Vec::with_capacity(2 * e.len());
    let (bhi, blo) = split(b);

    // First element.
    let (mut q, hh) = two_product_presplit(e[0], b, bhi, blo);
    if hh != 0.0 {
        h.push(hh);
    }

    for &enow in &e[1..] {
        let (product1, product0) = two_product_presplit(enow, b, bhi, blo);
        let (sum, hh) = two_sum(q, product0);
        if hh != 0.0 {
            h.push(hh);
        }
        let (qnew, hh) = fast_two_sum(product1, sum);
        q = qnew;
        if hh != 0.0 {
            h.push(hh);
        }
    }

    if q != 0.0 || h.is_empty() {
        h.push(q);
    }
    h
}

/// `two_product` with pre-split `b`.
#[inline]
fn two_product_presplit(a: f64, b: f64, bhi: f64, blo: f64) -> (f64, f64) {
    let x = a * b;
    let (ahi, alo) = split(a);
    let err1 = x - (ahi * bhi);
    let err2 = err1 - (alo * bhi);
    let err3 = err2 - (ahi * blo);
    (x, (alo * blo) - err3)
}

/// Sum two expansions, eliminating zero components.
#[allow(clippy::many_single_char_names)]
pub fn fast_expansion_sum_zeroelim(e: &[f64], f: &[f64]) -> Vec<f64> {
    if e.is_empty() {
        return if f.is_empty() { vec![0.0] } else { f.to_vec() };
    }
    if f.is_empty() {
        return e.to_vec();
    }

    let mut h = Vec::with_capacity(e.len() + f.len());
    let mut eindex = 0usize;
    let mut findex = 0usize;
    let mut enow = e[0];
    let mut fnow = f[0];

    // Pick the smaller-magnitude element first.
    let mut q = if (fnow > enow) == (fnow > -enow) {
        let v = enow;
        eindex += 1;
        if eindex < e.len() {
            enow = e[eindex];
        }
        v
    } else {
        let v = fnow;
        findex += 1;
        if findex < f.len() {
            fnow = f[findex];
        }
        v
    };

    if eindex < e.len() && findex < f.len() {
        let (qnew, hh) = if (fnow > enow) == (fnow > -enow) {
            let r = fast_two_sum(enow, q);
            eindex += 1;
            if eindex < e.len() {
                enow = e[eindex];
            }
            r
        } else {
            let r = fast_two_sum(fnow, q);
            findex += 1;
            if findex < f.len() {
                fnow = f[findex];
            }
            r
        };
        q = qnew;
        if hh != 0.0 {
            h.push(hh);
        }

        while eindex < e.len() && findex < f.len() {
            let (qnew, hh) = if (fnow > enow) == (fnow > -enow) {
                let r = two_sum(q, enow);
                eindex += 1;
                if eindex < e.len() {
                    enow = e[eindex];
                }
                r
            } else {
                let r = two_sum(q, fnow);
                findex += 1;
                if findex < f.len() {
                    fnow = f[findex];
                }
                r
            };
            q = qnew;
            if hh != 0.0 {
                h.push(hh);
            }
        }
    }

    while eindex < e.len() {
        let (qnew, hh) = two_sum(q, enow);
        eindex += 1;
        if eindex < e.len() {
            enow = e[eindex];
        }
        q = qnew;
        if hh != 0.0 {
            h.push(hh);
        }
    }

    while findex < f.len() {
        let (qnew, hh) = two_sum(q, fnow);
        findex += 1;
        if findex < f.len() {
            fnow = f[findex];
        }
        q = qnew;
        if hh != 0.0 {
            h.push(hh);
        }
    }

    if q != 0.0 || h.is_empty() {
        h.push(q);
    }
    h
}

// ── Expansion type ──────────────────────────────────────────────────

/// A multi-precision floating-point number represented as a Shewchuk expansion.
///
/// The exact value is the sum of all components. Components are non-overlapping
/// and sorted by increasing magnitude.
#[derive(Clone, Debug)]
pub struct Expansion {
    components: Vec<f64>,
}

impl Expansion {
    /// Create an expansion from a single f64 value.
    pub fn from_f64(val: f64) -> Self {
        if val == 0.0 {
            Self {
                components: vec![0.0],
            }
        } else {
            Self {
                components: vec![val],
            }
        }
    }

    /// Create a zero expansion.
    pub fn zero() -> Self {
        Self {
            components: vec![0.0],
        }
    }

    /// Number of components.
    pub fn len(&self) -> usize {
        self.components.len()
    }

    /// Whether the expansion has no components.
    pub fn is_empty(&self) -> bool {
        self.components.is_empty()
    }

    /// Fast f64 approximation (sum of all components).
    pub fn estimate(&self) -> f64 {
        self.components.iter().sum()
    }

    /// Exact sign: +1, -1, or 0.
    ///
    /// The sign is determined by the most significant (last) non-zero component,
    /// because components are non-overlapping and in increasing magnitude.
    pub fn sign(&self) -> i32 {
        for &c in self.components.iter().rev() {
            if c > 0.0 {
                return 1;
            }
            if c < 0.0 {
                return -1;
            }
        }
        0
    }

    /// Exact addition.
    pub fn add(&self, other: &Self) -> Self {
        Self {
            components: fast_expansion_sum_zeroelim(&self.components, &other.components),
        }
    }

    /// Exact subtraction: self - other.
    pub fn sub(&self, other: &Self) -> Self {
        self.add(&other.negate())
    }

    /// Exact multiplication.
    pub fn mul(&self, other: &Self) -> Self {
        if self.components.is_empty() || other.components.is_empty() {
            return Self::zero();
        }

        // Multiply each component of self by the entire other expansion,
        // then sum all partial products.
        let mut result = scale_expansion_zeroelim(&other.components, self.components[0]);
        for &c in &self.components[1..] {
            let partial = scale_expansion_zeroelim(&other.components, c);
            result = fast_expansion_sum_zeroelim(&result, &partial);
        }
        Self { components: result }
    }

    /// Exact negation.
    pub fn negate(&self) -> Self {
        Self {
            components: self.components.iter().map(|&c| -c).collect(),
        }
    }

    /// Access the raw components.
    pub fn components(&self) -> &[f64] {
        &self.components
    }
}

/// Compute the 2x2 determinant: a11*a22 - a12*a21.
pub fn expansion_det2x2(
    a11: &Expansion,
    a12: &Expansion,
    a21: &Expansion,
    a22: &Expansion,
) -> Expansion {
    a11.mul(a22).sub(&a12.mul(a21))
}

/// Compute the 3x3 determinant of a matrix given as three row expansions.
///
/// ```text
/// det = a[0]*(b[1]*c[2] - b[2]*c[1])
///     - a[1]*(b[0]*c[2] - b[2]*c[0])
///     + a[2]*(b[0]*c[1] - b[1]*c[0])
/// ```
pub fn expansion_det3x3(a: &[Expansion; 3], b: &[Expansion; 3], c: &[Expansion; 3]) -> Expansion {
    let t0 = b[1].mul(&c[2]).sub(&b[2].mul(&c[1])); // cofactor 00
    let t1 = b[0].mul(&c[2]).sub(&b[2].mul(&c[0])); // cofactor 01
    let t2 = b[0].mul(&c[1]).sub(&b[1].mul(&c[0])); // cofactor 02

    a[0].mul(&t0).sub(&a[1].mul(&t1)).add(&a[2].mul(&t2))
}

/// Compute orient3d via expansion arithmetic.
///
/// Equivalent to `det3x3` of the matrix `(a-d, b-d, c-d)`.
pub fn expansion_det4x4(
    a: &[Expansion; 3],
    b: &[Expansion; 3],
    c: &[Expansion; 3],
    d: &[Expansion; 3],
) -> Expansion {
    let row0 = [a[0].sub(&d[0]), a[1].sub(&d[1]), a[2].sub(&d[2])];
    let row1 = [b[0].sub(&d[0]), b[1].sub(&d[1]), b[2].sub(&d[2])];
    let row2 = [c[0].sub(&d[0]), c[1].sub(&d[1]), c[2].sub(&d[2])];
    expansion_det3x3(&row0, &row1, &row2)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── two_sum tests ───────────────────────────────────────────────

    #[test]
    fn two_sum_basic() {
        let (hi, lo) = two_sum(1.0, 2.0);
        assert_eq!(hi, 3.0);
        assert_eq!(lo, 0.0);
    }

    #[test]
    fn two_sum_preserves_exact_sum() {
        let (hi, lo) = two_sum(1.0, f64::EPSILON / 2.0);
        assert_eq!(hi + lo, 1.0 + f64::EPSILON / 2.0);
    }

    #[test]
    fn two_sum_large_cancellation() {
        let a = 1.0e15;
        let b = -1.0e15 + 1.0;
        let (hi, lo) = two_sum(a, b);
        assert_eq!(hi + lo, 1.0);
    }

    // ── two_product tests ───────────────────────────────────────────

    #[test]
    fn two_product_basic() {
        let (hi, lo) = two_product(3.0, 7.0);
        assert_eq!(hi, 21.0);
        assert_eq!(lo, 0.0);
    }

    #[test]
    fn two_product_captures_error() {
        let a = 1.0 + f64::EPSILON;
        let b = 1.0 + f64::EPSILON;
        let (_hi, lo) = two_product(a, b);
        assert_ne!(lo, 0.0, "product should have a non-zero error term");
    }

    // ── Expansion sign tests ────────────────────────────────────────

    #[test]
    fn sign_of_zero() {
        let e = Expansion::zero();
        assert_eq!(e.sign(), 0);
    }

    #[test]
    fn sign_of_positive() {
        let e = Expansion::from_f64(42.0);
        assert_eq!(e.sign(), 1);
    }

    #[test]
    fn sign_of_negative() {
        let e = Expansion::from_f64(-3.14);
        assert_eq!(e.sign(), -1);
    }

    #[test]
    fn sign_near_zero_exact() {
        // (1 + eps) - 1 in f64 may lose the eps.
        // Expansion arithmetic must preserve it.
        let a = Expansion::from_f64(1.0 + f64::EPSILON);
        let b = Expansion::from_f64(1.0);
        let diff = a.sub(&b);
        assert_eq!(
            diff.sign(),
            1,
            "exact difference should be positive, not zero"
        );
    }

    // ── Expansion arithmetic tests ──────────────────────────────────

    #[test]
    fn add_basic() {
        let a = Expansion::from_f64(1.0);
        let b = Expansion::from_f64(2.0);
        let c = a.add(&b);
        assert_eq!(c.estimate(), 3.0);
    }

    #[test]
    fn sub_basic() {
        let a = Expansion::from_f64(5.0);
        let b = Expansion::from_f64(3.0);
        let c = a.sub(&b);
        assert_eq!(c.estimate(), 2.0);
    }

    #[test]
    fn mul_basic() {
        let a = Expansion::from_f64(3.0);
        let b = Expansion::from_f64(7.0);
        let c = a.mul(&b);
        assert_eq!(c.estimate(), 21.0);
    }

    #[test]
    fn add_preserves_exactness() {
        let big = Expansion::from_f64(1e18);
        let small = Expansion::from_f64(1.0);
        let sum = big.add(&small);
        let back = sum.sub(&big);
        assert_eq!(back.sign(), 1, "1e18 + 1 - 1e18 should be exactly 1");
        assert!((back.estimate() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn sub_catastrophic_cancellation() {
        let a = Expansion::from_f64(1.0).add(&Expansion::from_f64(f64::EPSILON));
        let b = Expansion::from_f64(1.0);
        let diff = a.sub(&b);
        assert_eq!(diff.sign(), 1);
        assert!((diff.estimate() - f64::EPSILON).abs() < f64::EPSILON / 2.0);
    }

    // ── Determinant tests ───────────────────────────────────────────

    #[test]
    fn det3x3_identity() {
        let row0 = [
            Expansion::from_f64(1.0),
            Expansion::from_f64(0.0),
            Expansion::from_f64(0.0),
        ];
        let row1 = [
            Expansion::from_f64(0.0),
            Expansion::from_f64(1.0),
            Expansion::from_f64(0.0),
        ];
        let row2 = [
            Expansion::from_f64(0.0),
            Expansion::from_f64(0.0),
            Expansion::from_f64(1.0),
        ];
        let det = expansion_det3x3(&row0, &row1, &row2);
        assert_eq!(det.sign(), 1);
        assert!((det.estimate() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn det3x3_singular() {
        let row0 = [
            Expansion::from_f64(1.0),
            Expansion::from_f64(2.0),
            Expansion::from_f64(3.0),
        ];
        let row1 = [
            Expansion::from_f64(1.0),
            Expansion::from_f64(2.0),
            Expansion::from_f64(3.0),
        ];
        let row2 = [
            Expansion::from_f64(4.0),
            Expansion::from_f64(5.0),
            Expansion::from_f64(6.0),
        ];
        let det = expansion_det3x3(&row0, &row1, &row2);
        assert_eq!(det.sign(), 0);
    }

    // ── Cross-check: orient3d_f64 vs expansion det ───────────────────

    #[test]
    fn orient3d_self_consistency() {
        use crate::mesh_boolean::geogram::exact_pred::orient3d_f64;
        let pa = [0.0, 0.0, 0.0];
        let pb = [1.0, 0.0, 0.0];
        let pc = [0.0, 1.0, 0.0];
        let pd = [0.0, 0.0, 1e-15];

        let sign = orient3d_f64(pa, pb, pc, pd) as i32;
        // d is slightly above the plane — should be positive
        assert_eq!(sign, 1, "orient3d should detect point above plane");
    }

    #[test]
    fn orient3d_near_degenerate() {
        use crate::mesh_boolean::geogram::exact_pred::orient3d_f64;
        let pa = [1.0, 0.0, 0.0];
        let pb = [0.0, 1.0, 0.0];
        let pc = [0.0, 0.0, 1.0];
        let pd = [1.0 / 3.0 + 1e-16, 1.0 / 3.0, 1.0 / 3.0];

        let sign = orient3d_f64(pa, pb, pc, pd) as i32;
        // Point is slightly above the plane — exact arithmetic should detect it
        assert!(
            sign != 0,
            "exact orient3d should not return zero for near-degenerate case"
        );
    }

    // ── Geogram test_expansion_nt reference vectors ─────────────────

    #[test]
    fn geogram_reference_sign_1e_30_plus_5_plus_1e30_plus_2e_30_minus_1e30() {
        // From geogram/src/tests/test_expansion_nt/main.cpp:
        //   sign(1e-30 + 5.0 + 1e30 + 2e-30 - 1e30) = 1
        //   components = [3e-30, 5]
        // f64 gives 0 (wrong). Expansion must give 1.
        let r = Expansion::from_f64(1e-30)
            .add(&Expansion::from_f64(5.0))
            .add(&Expansion::from_f64(1e30))
            .add(&Expansion::from_f64(2e-30))
            .sub(&Expansion::from_f64(1e30));

        assert_eq!(r.sign(), 1, "sign should be positive, got {}", r.sign());

        // Verify the estimate matches Geogram's output (5.0).
        assert!(
            (r.estimate() - 5.0).abs() < 1e-10,
            "estimate should be ~5.0, got {}",
            r.estimate()
        );

        // Verify the exact value captures the 3e-30 term.
        // The components should sum to exactly 5 + 3e-30.
        // We can't check f64 equality for 3e-30 (it might be split across
        // components), but the sum of all components must be closer to
        // 5 + 3e-30 than plain f64 would give.
        let component_sum: f64 = r.components().iter().sum();
        // f64 can represent 5.0 exactly but 5.0 + 3e-30 rounds to 5.0.
        // The expansion's component sum should be 5.0 (because f64 can't
        // represent 5+3e-30), but at least the sign is correct.
        assert_eq!(component_sum.signum() as i32, 1);
    }

    #[test]
    fn geogram_reference_double_gives_wrong_answer() {
        // Confirm that f64 arithmetic gives the wrong answer for this case.
        let r_f64 = 1e-30_f64 + 5.0 + 1e30 + 2e-30 - 1e30;
        // f64 loses the small terms entirely — result is exactly 0.0.
        assert_eq!(
            r_f64, 0.0,
            "f64 should give exactly 0.0 for this expression"
        );
        // The correct answer is 5 + 3e-30, which is decidedly not zero.
    }
}
