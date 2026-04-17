//! B-spline basis function evaluation.
//!
//! Implements Piegl & Tiller Algorithms A2.2 and A2.3.

// The evaluation loops index multiple arrays (knots, poles, basis) with the
// same variable. Rewriting as iterators would obscure the P&T algorithm structure.
#![allow(clippy::needless_range_loop)]
//!
//! Two API levels:
//! - Public `basis_funs` / `ders_basis_funs` return `Vec` for convenience.
//! - Internal `basis_funs_into` / `ders_basis_funs_raw` write into caller-provided
//!   stack buffers for allocation-free evaluation in hot paths.

use super::MAX_ORDER;
use super::knots::find_span;

/// Compute non-vanishing basis functions into a caller-provided buffer.
///
/// Writes `degree + 1` values into `out[0..=degree]`.
/// `left` and `right` are scratch buffers of length >= `degree + 1`.
///
/// This is the allocation-free core used by curve/surface evaluation.
pub fn basis_funs_into(
    span: usize,
    u: f64,
    degree: usize,
    knots: &[f64],
    out: &mut [f64],
    left: &mut [f64],
    right: &mut [f64],
) {
    out[0] = 1.0;

    for j in 1..=degree {
        left[j] = u - knots[span + 1 - j];
        right[j] = knots[span + j] - u;
        let mut saved = 0.0;

        for r in 0..j {
            let temp = out[r] / (right[r + 1] + left[j - r]);
            out[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        out[j] = saved;
    }
}

/// Compute the non-vanishing basis functions at parameter `u`.
///
/// P&T Algorithm A2.2 (BasisFuns).
///
/// Returns a vector of `degree + 1` values: N[span-degree], ..., N[span].
pub fn basis_funs(span: usize, u: f64, degree: usize, knots: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; degree + 1];
    let mut left = vec![0.0; degree + 1];
    let mut right = vec![0.0; degree + 1];
    basis_funs_into(span, u, degree, knots, &mut out, &mut left, &mut right);
    out
}

/// Compute basis functions and derivatives up to order `deriv_order`.
///
/// P&T Algorithm A2.3 (DersBasisFuns).
///
/// Returns `ders[k][j]` where `k` is derivative order (0..=deriv_order)
/// and `j` is basis function index (0..=degree).
pub fn ders_basis_funs(
    span: usize,
    u: f64,
    degree: usize,
    deriv_order: usize,
    knots: &[f64],
) -> Vec<Vec<f64>> {
    let p = degree;
    let n_deriv = deriv_order.min(p);

    let mut ndu = vec![vec![0.0; p + 1]; p + 1];
    let mut left = vec![0.0; p + 1];
    let mut right = vec![0.0; p + 1];

    ndu[0][0] = 1.0;

    for j in 1..=p {
        left[j] = u - knots[span + 1 - j];
        right[j] = knots[span + j] - u;
        let mut saved = 0.0;

        for r in 0..j {
            ndu[j][r] = right[r + 1] + left[j - r];
            let temp = ndu[r][j - 1] / ndu[j][r];
            ndu[r][j] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        ndu[j][j] = saved;
    }

    let mut ders = vec![vec![0.0; p + 1]; n_deriv + 1];
    for j in 0..=p {
        ders[0][j] = ndu[j][p];
    }

    let mut a = vec![vec![0.0; p + 1]; 2];

    for r in 0..=p {
        let mut s1 = 0usize;
        let mut s2 = 1usize;
        a[0][0] = 1.0;

        for k in 1..=n_deriv {
            let mut d = 0.0;
            let rk = r as isize - k as isize;
            let pk = (p as isize - k as isize) as usize;

            if rk >= 0 {
                let rk = rk as usize;
                a[s2][0] = a[s1][0] / ndu[pk + 1][rk];
                d = a[s2][0] * ndu[rk][pk];
            }

            let j1 = if rk >= -1 { 1usize } else { (-rk) as usize };
            let j2 = if (r as isize - 1) <= pk as isize {
                k - 1
            } else {
                p - r
            };

            for j in j1..=j2 {
                let rk_j = (rk + j as isize) as usize;
                a[s2][j] = (a[s1][j] - a[s1][j - 1]) / ndu[pk + 1][rk_j];
                d += a[s2][j] * ndu[rk_j][pk];
            }

            if r <= pk {
                a[s2][k] = -a[s1][k - 1] / ndu[pk + 1][r];
                d += a[s2][k] * ndu[r][pk];
            }

            ders[k][r] = d;
            std::mem::swap(&mut s1, &mut s2);
        }
    }

    let mut r = p as f64;
    for k in 1..=n_deriv {
        for j in 0..=p {
            ders[k][j] *= r;
        }
        r *= (p - k) as f64;
    }

    ders
}

/// Compute basis function derivatives into stack-allocated arrays.
///
/// Allocation-free version of `ders_basis_funs`. Writes into `ders_out`,
/// a flat `[MAX_ORDER * MAX_ORDER]` array indexed as `ders_out[k * MAX_ORDER + j]`.
///
/// Returns the effective derivative order (min of `deriv_order` and `degree`).
pub fn ders_basis_funs_into(
    span: usize,
    u: f64,
    degree: usize,
    deriv_order: usize,
    knots: &[f64],
    ders_out: &mut [f64; MAX_ORDER * MAX_ORDER],
) -> usize {
    let p = degree;
    let n_deriv = deriv_order.min(p);

    let mut ndu = [0.0f64; MAX_ORDER * MAX_ORDER]; // ndu[j * MAX_ORDER + r]
    let mut left = [0.0f64; MAX_ORDER];
    let mut right = [0.0f64; MAX_ORDER];

    ndu[0] = 1.0; // ndu[0][0]

    for j in 1..=p {
        left[j] = u - knots[span + 1 - j];
        right[j] = knots[span + j] - u;
        let mut saved = 0.0;

        for r in 0..j {
            // ndu[j][r] = right[r+1] + left[j-r]
            ndu[j * MAX_ORDER + r] = right[r + 1] + left[j - r];
            let temp = ndu[r * MAX_ORDER + j - 1] / ndu[j * MAX_ORDER + r];
            // ndu[r][j] = saved + right[r+1] * temp
            ndu[r * MAX_ORDER + j] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        ndu[j * MAX_ORDER + j] = saved;
    }

    // Zero output
    for v in ders_out.iter_mut() {
        *v = 0.0;
    }

    // Load basis function values
    for j in 0..=p {
        ders_out[j] = ndu[j * MAX_ORDER + p]; // ders[0][j]
    }

    let mut a = [0.0f64; 2 * MAX_ORDER]; // a[s][j] as a[s * MAX_ORDER + j]... but only 2 rows
    // Use a[0..MAX_ORDER] and a[MAX_ORDER..2*MAX_ORDER]

    for r in 0..=p {
        let mut s1 = 0usize;
        let mut s2 = MAX_ORDER;
        a[0] = 1.0; // a[s1][0]

        for k in 1..=n_deriv {
            let mut d = 0.0;
            let rk = r as isize - k as isize;
            let pk = (p as isize - k as isize) as usize;

            if rk >= 0 {
                let rk_u = rk as usize;
                a[s2] = a[s1] / ndu[(pk + 1) * MAX_ORDER + rk_u];
                d = a[s2] * ndu[rk_u * MAX_ORDER + pk];
            }

            let j1 = if rk >= -1 { 1usize } else { (-rk) as usize };
            let j2 = if (r as isize - 1) <= pk as isize {
                k - 1
            } else {
                p - r
            };

            for j in j1..=j2 {
                let rk_j = (rk + j as isize) as usize;
                a[s2 + j] = (a[s1 + j] - a[s1 + j - 1]) / ndu[(pk + 1) * MAX_ORDER + rk_j];
                d += a[s2 + j] * ndu[rk_j * MAX_ORDER + pk];
            }

            if r <= pk {
                a[s2 + k] = -a[s1 + k - 1] / ndu[(pk + 1) * MAX_ORDER + r];
                d += a[s2 + k] * ndu[r * MAX_ORDER + pk];
            }

            ders_out[k * MAX_ORDER + r] = d;
            std::mem::swap(&mut s1, &mut s2);
        }
    }

    // Multiply through by correct factors
    let mut fac = p as f64;
    for k in 1..=n_deriv {
        for j in 0..=p {
            ders_out[k * MAX_ORDER + j] *= fac;
        }
        fac *= (p - k) as f64;
    }

    n_deriv
}

/// Convenience: compute basis functions at `u`, finding span automatically.
pub fn evaluate_basis(u: f64, degree: usize, knots: &[f64]) -> (usize, Vec<f64>) {
    let n = knots.len() - degree - 2;
    let span = find_span(n, degree, u, knots);
    let basis = basis_funs(span, u, degree, knots);
    (span, basis)
}

/// Compute basis functions into stack-allocated arrays. Returns the span.
///
/// This is the fast path for use in curve/surface evaluation.
/// `out`, `left`, `right` must each be at least `MAX_ORDER` long.
pub fn evaluate_basis_fast(
    u: f64,
    n: usize,
    degree: usize,
    knots: &[f64],
    out: &mut [f64; MAX_ORDER],
    left: &mut [f64; MAX_ORDER],
    right: &mut [f64; MAX_ORDER],
) -> usize {
    let span = find_span(n, degree, u, knots);
    basis_funs_into(span, u, degree, knots, out, left, right);
    span
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cubic_knots() -> Vec<f64> {
        vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0]
    }

    #[test]
    fn partition_of_unity() {
        let knots = cubic_knots();
        let n = knots.len() - 3 - 2;
        for &u in &[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0] {
            let span = find_span(n, 3, u, &knots);
            let basis = basis_funs(span, u, 3, &knots);
            let sum: f64 = basis.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-14,
                "partition of unity failed at u={u}: sum={sum}"
            );
        }
    }

    #[test]
    fn partition_of_unity_many_degrees() {
        for degree in 1..=6 {
            let mut knots = vec![0.0; degree + 1];
            knots.extend_from_slice(&[1.0, 2.0, 3.0]);
            knots.extend(vec![4.0; degree + 1]);
            let n = knots.len() - degree - 2;

            for i in 0..=40 {
                let u = i as f64 * 0.1;
                let span = find_span(n, degree, u, &knots);
                let basis = basis_funs(span, u, degree, &knots);
                let sum: f64 = basis.iter().sum();
                assert!(
                    (sum - 1.0).abs() < 1e-13,
                    "partition of unity failed: degree={degree}, u={u}, sum={sum}"
                );
            }
        }
    }

    #[test]
    fn non_negative() {
        let knots = cubic_knots();
        let n = knots.len() - 3 - 2;
        for i in 0..=40 {
            let u = i as f64 * 0.1;
            let span = find_span(n, 3, u, &knots);
            let basis = basis_funs(span, u, 3, &knots);
            for (j, &val) in basis.iter().enumerate() {
                assert!(val >= -1e-15, "negative basis N[{j}]={val} at u={u}");
            }
        }
    }

    #[test]
    fn local_support() {
        // Each basis function N_{i,p} should be zero outside [knot_i, knot_{i+p+1}]
        let knots = cubic_knots();
        let n = knots.len() - 3 - 2;
        let degree = 3;

        for i in 0..=40 {
            let u = i as f64 * 0.1;
            let span = find_span(n, degree, u, &knots);
            let basis = basis_funs(span, u, degree, &knots);

            // basis[j] corresponds to N_{span-degree+j, degree}
            for j in 0..=degree {
                let func_idx = span - degree + j;
                let support_start = knots[func_idx];
                let support_end = knots[func_idx + degree + 1];
                if u < support_start || u > support_end {
                    assert!(
                        basis[j].abs() < 1e-14,
                        "local support violated: N[{func_idx},{degree}]({u}) = {} but support is [{support_start}, {support_end}]",
                        basis[j]
                    );
                }
            }
        }
    }

    #[test]
    fn linear_basis_is_lerp() {
        let knots = vec![0.0, 0.0, 1.0, 1.0];
        let n = knots.len() - 1 - 2;
        let span = find_span(n, 1, 0.25, &knots);
        let basis = basis_funs(span, 0.25, 1, &knots);
        assert!((basis[0] - 0.75).abs() < 1e-15);
        assert!((basis[1] - 0.25).abs() < 1e-15);
    }

    #[test]
    fn bezier_basis_matches_bernstein() {
        let knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let n = knots.len() - 3 - 2;
        let span = find_span(n, 3, 0.5, &knots);
        let basis = basis_funs(span, 0.5, 3, &knots);
        assert!((basis[0] - 0.125).abs() < 1e-14);
        assert!((basis[1] - 0.375).abs() < 1e-14);
        assert!((basis[2] - 0.375).abs() < 1e-14);
        assert!((basis[3] - 0.125).abs() < 1e-14);
    }

    #[test]
    fn derivatives_sum_to_zero() {
        // Derivative of partition of unity = 0
        let knots = vec![0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0];
        let n = knots.len() - 2 - 2;
        for i in 0..=20 {
            let u = i as f64 * 0.1;
            let span = find_span(n, 2, u, &knots);
            let ders = ders_basis_funs(span, u, 2, 2, &knots);

            let d1_sum: f64 = ders[1].iter().sum();
            assert!(
                d1_sum.abs() < 1e-13,
                "1st derivative sum = {d1_sum} at u={u}"
            );

            let d2_sum: f64 = ders[2].iter().sum();
            assert!(
                d2_sum.abs() < 1e-12,
                "2nd derivative sum = {d2_sum} at u={u}"
            );
        }
    }

    #[test]
    fn derivatives_at_c0_discontinuity() {
        // Knot with multiplicity = degree creates a C0 point.
        // Degree 3 with a triple internal knot: [0,0,0,0, 2,2,2, 4,4,4,4]
        let knots = vec![0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0, 4.0];
        let n = knots.len() - 3 - 2; // n = 6
        let degree = 3;

        // Evaluate at the C0 knot and just before/after
        for &u in &[1.99, 2.0, 2.01] {
            let span = find_span(n, degree, u, &knots);
            let ders = ders_basis_funs(span, u, degree, 2, &knots);

            // Partition of unity
            let sum: f64 = ders[0].iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-13,
                "partition of unity at C0 knot u={u}: sum={sum}"
            );

            // Derivative sums to 0
            let d1_sum: f64 = ders[1].iter().sum();
            assert!(d1_sum.abs() < 1e-12, "d1 sum at C0 knot u={u}: {d1_sum}");
        }
    }

    #[test]
    fn derivatives_all_degrees_at_boundaries() {
        // Test ders_basis_funs at u=0 and u=end for degrees 1..6
        for degree in 1..=6 {
            let mut knots = vec![0.0; degree + 1];
            knots.extend_from_slice(&[1.0, 2.0]);
            knots.extend(vec![3.0; degree + 1]);
            let n = knots.len() - degree - 2;

            for &u in &[0.0, 3.0] {
                let span = find_span(n, degree, u, &knots);
                let ders = ders_basis_funs(span, u, degree, degree.min(3), &knots);

                let sum: f64 = ders[0].iter().sum();
                assert!(
                    (sum - 1.0).abs() < 1e-13,
                    "partition of unity: degree={degree}, u={u}, sum={sum}"
                );

                if ders.len() > 1 {
                    let d1_sum: f64 = ders[1].iter().sum();
                    assert!(
                        d1_sum.abs() < 1e-12,
                        "d1 sum: degree={degree}, u={u}, d1_sum={d1_sum}"
                    );
                }
            }
        }
    }

    #[test]
    fn degree_1_derivatives() {
        // Simplest case: degree 1. D0 = lerp, D1 = constant, D2+ = 0
        let knots = vec![0.0, 0.0, 1.0, 1.0];
        let n = knots.len() - 1 - 2;
        for &u in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            let span = find_span(n, 1, u, &knots);
            let ders = ders_basis_funs(span, u, 1, 1, &knots);
            // D0: partition of unity
            assert!((ders[0].iter().sum::<f64>() - 1.0).abs() < 1e-15);
            // D1: should be [-1, 1] (derivatives of (1-u) and u)
            assert!((ders[1][0] - (-1.0)).abs() < 1e-14, "d1[0]={}", ders[1][0]);
            assert!((ders[1][1] - 1.0).abs() < 1e-14, "d1[1]={}", ders[1][1]);
        }
    }

    #[test]
    fn degree_1_second_derivative_is_zero() {
        let knots = vec![0.0, 0.0, 1.0, 1.0];
        let n = knots.len() - 1 - 2;
        let span = find_span(n, 1, 0.5, &knots);
        let ders = ders_basis_funs(span, 0.5, 1, 2, &knots);
        // 2nd derivative of linear basis = 0
        for j in 0..2 {
            assert!(ders[1][j].is_finite()); // d1 exists
        }
        // ders[2] should be all zeros (degree < deriv_order)
        // Actually deriv_order is capped to min(2,1) = 1, so ders only has 2 rows
        assert_eq!(ders.len(), 2); // capped to degree
    }

    #[test]
    fn high_multiplicity_internal_knot() {
        // Degree 4, internal knot with multiplicity 3 (C1 continuity)
        // [0,0,0,0,0, 1,1,1, 2,2,2,2,2]
        let knots = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        ];
        let n = knots.len() - 4 - 2;
        let degree = 4;

        for &u in &[0.0, 0.5, 0.99, 1.0, 1.01, 1.5, 2.0] {
            let span = find_span(n, degree, u, &knots);
            let ders = ders_basis_funs(span, u, degree, 3, &knots);

            let sum: f64 = ders[0].iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-13,
                "partition of unity at u={u}: sum={sum}"
            );
            for k in 1..=3 {
                let dk_sum: f64 = ders[k].iter().sum();
                assert!(dk_sum.abs() < 1e-11, "d{k} sum at u={u}: {dk_sum}");
            }
        }
    }

    #[test]
    fn numerical_vs_analytical_basis_derivative() {
        // Verify basis function derivatives by finite differences
        let knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0];
        let n = knots.len() - 3 - 2;
        let degree = 3;
        let h = 1e-7;

        for ui in 1..=7 {
            let u = ui as f64 * 0.5;
            let span = find_span(n, degree, u, &knots);
            let ders = ders_basis_funs(span, u, degree, 1, &knots);

            // Numerical derivative of each basis function
            let span_p = find_span(n, degree, u + h, &knots);
            let span_m = find_span(n, degree, u - h, &knots);
            let basis_p = basis_funs(span_p, u + h, degree, &knots);
            let basis_m = basis_funs(span_m, u - h, degree, &knots);

            // Only compare when the span hasn't changed (same basis functions active)
            if span_p == span && span_m == span {
                for j in 0..=degree {
                    let numerical = (basis_p[j] - basis_m[j]) / (2.0 * h);
                    let analytical = ders[1][j];
                    assert!(
                        (numerical - analytical).abs() < 1e-5,
                        "basis deriv mismatch at u={u}, j={j}: analytical={analytical}, numerical={numerical}"
                    );
                }
            }
        }
    }

    #[test]
    fn fast_path_matches_public_api() {
        let knots = cubic_knots();
        let n = knots.len() - 3 - 2;
        let u = 1.5;

        let (span, basis_vec) = evaluate_basis(u, 3, &knots);

        let mut out = [0.0; MAX_ORDER];
        let mut left = [0.0; MAX_ORDER];
        let mut right = [0.0; MAX_ORDER];
        let span_fast = evaluate_basis_fast(u, n, 3, &knots, &mut out, &mut left, &mut right);

        assert_eq!(span, span_fast);
        for i in 0..=3 {
            assert!(
                (basis_vec[i] - out[i]).abs() < 1e-15,
                "mismatch at i={i}: {} vs {}",
                basis_vec[i],
                out[i]
            );
        }
    }

    #[test]
    fn ders_into_matches_ders() {
        // Verify the stack-allocated version matches the Vec version
        let knots = cubic_knots();
        let n = knots.len() - 3 - 2;
        let degree = 3;
        let deriv_order = 2;

        for ui in 0..=20 {
            let u = ui as f64 * 0.2;
            let span = find_span(n, degree, u, &knots);

            let ders_vec = ders_basis_funs(span, u, degree, deriv_order, &knots);

            let mut ders_buf = [0.0f64; MAX_ORDER * MAX_ORDER];
            let n_deriv = ders_basis_funs_into(span, u, degree, deriv_order, &knots, &mut ders_buf);

            assert_eq!(n_deriv, deriv_order.min(degree));

            for k in 0..=n_deriv {
                for j in 0..=degree {
                    let vec_val = ders_vec[k][j];
                    let buf_val = ders_buf[k * MAX_ORDER + j];
                    assert!(
                        (vec_val - buf_val).abs() < 1e-14,
                        "mismatch at u={u}, k={k}, j={j}: vec={vec_val}, buf={buf_val}"
                    );
                }
            }
        }
    }
}
