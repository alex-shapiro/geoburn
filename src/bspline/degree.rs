//! Degree elevation for B-spline curves.

#![allow(clippy::needless_range_loop)]

use super::ControlPoint;
use super::knots::validate_knots;

/// Elevate the degree of a B-spline curve from `degree` to `new_degree`.
///
/// Returns `(new_knots, new_poles)`. The curve shape is unchanged.
///
/// Strategy: decompose into Bezier segments, elevate each independently
/// using the binomial coefficient formula, then reassemble.
pub fn elevate_degree<P: ControlPoint>(
    degree: usize,
    new_degree: usize,
    knots: &[f64],
    poles: &[P],
) -> (Vec<f64>, Vec<P>) {
    validate_knots(degree, knots);
    assert!(
        new_degree > degree,
        "elevate_degree: new_degree {new_degree} must exceed current degree {degree}"
    );

    let t = new_degree - degree;

    // Compute elevation coefficients
    let bezalfs = bezier_elevation_coeffs(degree, t);

    // Extract Bezier segments
    let segments = super::insert::extract_bezier(degree, knots, poles);
    if segments.is_empty() {
        return (knots.to_vec(), poles.to_vec());
    }

    // Elevate each segment
    let new_order = new_degree + 1;
    let mut elevated: Vec<Vec<P>> = Vec::with_capacity(segments.len());
    for (_seg_knots, seg_poles) in &segments {
        elevated.push(elevate_bezier(seg_poles, degree, new_degree, &bezalfs));
    }

    // Reassemble: new knot vector and poles
    // First segment contributes all its poles
    let mut new_poles = elevated[0].clone();

    // Subsequent segments share the first pole with the previous segment's last
    for seg in &elevated[1..] {
        // seg[0] should equal the last pole of new_poles (shared boundary)
        new_poles.extend_from_slice(&seg[1..]);
    }

    // Build new knot vector.
    //
    // Because we decomposed to Bezier segments before elevating, all
    // internal knots were raised to multiplicity = degree (full Bezier
    // boundaries). Elevation by t gives each internal knot multiplicity
    // = degree + t = new_degree. This is deterministic — no heuristic needed.
    //
    // End knots: new_degree + 1 (clamped)
    // Internal knots: new_degree each
    let (distinct, _) = super::knots::distinct_knots(knots, 0.0);

    let mut new_knots = Vec::new();

    // First knot
    for _ in 0..new_order {
        new_knots.push(distinct[0]);
    }

    // Internal knots: each at multiplicity new_degree
    for i in 1..distinct.len() - 1 {
        for _ in 0..new_degree {
            new_knots.push(distinct[i]);
        }
    }

    // Last knot
    for _ in 0..new_order {
        new_knots.push(*distinct.last().unwrap());
    }

    debug_assert_eq!(
        new_knots.len(),
        new_poles.len() + new_degree + 1,
        "knot/pole count mismatch after degree elevation: {} knots, {} poles, degree {}",
        new_knots.len(),
        new_poles.len(),
        new_degree,
    );

    (new_knots, new_poles)
}

/// Elevate a single Bezier segment from degree `p` to degree `p + t`.
fn elevate_bezier<P: ControlPoint>(
    poles: &[P],
    p: usize,
    new_p: usize,
    bezalfs: &[Vec<f64>],
) -> Vec<P> {
    let mut new_poles = vec![P::origin(); new_p + 1];
    for i in 0..=new_p {
        let j_min = i.saturating_sub(new_p - p);
        let j_max = i.min(p);
        for j in j_min..=j_max {
            new_poles[i] = new_poles[i].add(&poles[j].scaled(bezalfs[i][j]));
        }
    }
    new_poles
}

/// Bezier degree elevation coefficients.
///
/// `bezalfs[i][j]` = C(p,j) * C(t,i-j) / C(p+t,i)
fn bezier_elevation_coeffs(p: usize, t: usize) -> Vec<Vec<f64>> {
    let new_p = p + t;
    let mut bezalfs = vec![vec![0.0; p + 1]; new_p + 1];
    for i in 0..=new_p {
        let j_min = i.saturating_sub(t);
        let j_max = i.min(p);
        for j in j_min..=j_max {
            bezalfs[i][j] = binom(p, j) as f64 * binom(t, i - j) as f64 / binom(new_p, i) as f64;
        }
    }
    bezalfs
}

fn binom(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    let k = k.min(n - k);
    let mut result = 1usize;
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gp::Pnt;
    use nalgebra::Point3;

    fn pnt(x: f64, y: f64, z: f64) -> Pnt {
        Point3::new(x, y, z)
    }

    fn assert_pnt_near(a: &Pnt, b: &Pnt, tol: f64) {
        assert!((a - b).norm() < tol, "{a:?} vs {b:?}");
    }

    fn assert_curves_equal(
        d1: usize,
        k1: &[f64],
        p1: &[Pnt],
        d2: usize,
        k2: &[f64],
        p2: &[Pnt],
        n_samples: usize,
        tol: f64,
    ) {
        let u_min = k1[d1];
        let u_max = k1[k1.len() - d1 - 1];
        for i in 0..=n_samples {
            let u = u_min + (u_max - u_min) * i as f64 / n_samples as f64;
            let a: Pnt = super::super::curves::curve_point(d1, k1, p1, u);
            let b: Pnt = super::super::curves::curve_point(d2, k2, p2, u);
            assert_pnt_near(&a, &b, tol);
        }
    }

    #[test]
    fn elevate_linear_to_quadratic() {
        let knots = vec![0.0, 0.0, 1.0, 1.0];
        let poles = vec![pnt(0.0, 0.0, 0.0), pnt(1.0, 0.0, 0.0)];
        let (nk, np) = elevate_degree(1, 2, &knots, &poles);
        assert_eq!(np.len(), 3);
        assert_curves_equal(1, &knots, &poles, 2, &nk, &np, 20, 1e-13);
    }

    #[test]
    fn elevate_quadratic_to_cubic() {
        let knots = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let poles = vec![pnt(0.0, 0.0, 0.0), pnt(0.5, 1.0, 0.0), pnt(1.0, 0.0, 0.0)];
        let (nk, np) = elevate_degree(2, 3, &knots, &poles);
        assert_curves_equal(2, &knots, &poles, 3, &nk, &np, 20, 1e-12);
    }

    #[test]
    fn elevate_preserves_multi_span_curve() {
        let knots = vec![0.0, 0.0, 0.0, 0.0, 2.0, 4.0, 4.0, 4.0, 4.0];
        let poles: Vec<Pnt> = (0..5)
            .map(|i| pnt(i as f64, (i as f64).sin(), (i as f64).cos()))
            .collect();
        let (nk, np) = elevate_degree(3, 4, &knots, &poles);
        assert_curves_equal(3, &knots, &poles, 4, &nk, &np, 40, 1e-11);
    }

    #[test]
    fn elevate_by_two() {
        let knots = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let poles = vec![pnt(0.0, 0.0, 0.0), pnt(0.5, 1.0, 0.0), pnt(1.0, 0.0, 0.0)];
        let (nk, np) = elevate_degree(2, 4, &knots, &poles);
        assert_curves_equal(2, &knots, &poles, 4, &nk, &np, 20, 1e-12);
    }
}
