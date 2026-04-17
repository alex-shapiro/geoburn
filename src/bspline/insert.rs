//! Knot insertion and Bezier extraction for B-spline curves.

use super::ControlPoint;
use super::knots::{find_span, validate_knots, validate_poles};

/// Insert knot `u` with multiplicity `r` into a B-spline curve.
///
/// P&T Algorithm A5.1. Generic over control point type.
/// Returns `(new_knots, new_poles)`. The curve shape is unchanged.
pub fn insert_knot<P: ControlPoint>(
    degree: usize,
    knots: &[f64],
    poles: &[P],
    u: f64,
    r: usize,
) -> (Vec<f64>, Vec<P>) {
    validate_knots(degree, knots);
    validate_poles(degree, knots, poles.len());
    assert!(r > 0, "insert_knot: r must be positive");

    let n = poles.len() - 1;
    let p = degree;
    let k = find_span(n, p, u, knots);
    let s = knots.iter().filter(|&&v| v == u).count();

    assert!(
        s + r <= p,
        "insert_knot: final multiplicity {} would exceed degree {p}",
        s + r
    );

    let mut new_knots = vec![0.0; knots.len() + r];
    let mut new_poles = vec![P::origin(); poles.len() + r];

    new_knots[..=k].copy_from_slice(&knots[..=k]);
    for i in 1..=r {
        new_knots[k + i] = u;
    }
    new_knots[k + 1 + r..].copy_from_slice(&knots[k + 1..]);

    new_poles[..=k - p].copy_from_slice(&poles[..=k - p]);
    new_poles[k - s + r..=n + r].copy_from_slice(&poles[k - s..=n]);

    let mut rw = vec![P::origin(); p - s + 1];
    rw[..=p - s].copy_from_slice(&poles[k - p..=k - s]);

    for j in 1..=r {
        let l = k - p + j;
        for i in 0..=(p - j - s) {
            let alpha = (u - knots[l + i]) / (knots[i + k + 1] - knots[l + i]);
            rw[i] = rw[i].scaled(1.0 - alpha).add(&rw[i + 1].scaled(alpha));
        }
        new_poles[l] = rw[0];
        new_poles[k + r - j - s] = rw[p - j - s];
    }

    let l = k - p + r;
    if l + 1 < k - s {
        new_poles[l + 1..k - s].copy_from_slice(&rw[1..k - s - l]);
    }

    (new_knots, new_poles)
}

/// Insert knot into a rational B-spline (NURBS) curve.
///
/// Converts to weighted control points, inserts, converts back.
pub fn insert_knot_rational<P: ControlPoint>(
    degree: usize,
    knots: &[f64],
    poles: &[P],
    weights: &[f64],
    u: f64,
    r: usize,
) -> (Vec<f64>, Vec<P>, Vec<f64>) {
    assert_eq!(
        weights.len(),
        poles.len(),
        "weights length must match poles"
    );

    let weighted: Vec<P> = poles
        .iter()
        .zip(weights)
        .map(|(p, &w)| p.scaled(w))
        .collect();
    let w_vals: Vec<f64> = weights.to_vec();

    let (new_knots, new_weighted) = insert_knot(degree, knots, &weighted, u, r);
    let (_, new_weights) = insert_knot(degree, knots, &w_vals, u, r);

    let new_poles: Vec<P> = new_weighted
        .iter()
        .zip(new_weights.iter())
        .map(|(p, &w)| p.div(w))
        .collect();

    (new_knots, new_poles, new_weights)
}

/// Extract Bezier segments from a B-spline curve by inserting all internal
/// knots to full multiplicity (= degree).
///
/// Returns a vector of `(knots, poles)` pairs, one per Bezier segment.
/// Each segment has `degree + 1` poles and knots `[a, a, ..., a, b, b, ..., b]`.
pub fn extract_bezier<P: ControlPoint>(
    degree: usize,
    knots: &[f64],
    poles: &[P],
) -> Vec<(Vec<f64>, Vec<P>)> {
    validate_knots(degree, knots);
    validate_poles(degree, knots, poles.len());

    // Insert all internal knots to full multiplicity
    let mut cur_knots = knots.to_vec();
    let mut cur_poles = poles.to_vec();

    // Collect distinct internal knots and their current multiplicities
    let _u_first = knots[degree];
    let _u_last = knots[knots.len() - degree - 1];

    let mut internal_knots: Vec<(f64, usize)> = Vec::new();
    let mut i = degree + 1;
    while i < knots.len() - degree - 1 {
        let u = knots[i];
        let mut mult = 1;
        while i + mult < knots.len() && knots[i + mult] == u {
            mult += 1;
        }
        let needed = degree - mult;
        if needed > 0 {
            internal_knots.push((u, needed));
        }
        i += mult;
    }

    // Insert each (in reverse order to keep indices stable isn't needed
    // since insert_knot recomputes spans, but inserting from left is fine)
    for (u, r) in internal_knots {
        let (nk, np) = insert_knot(degree, &cur_knots, &cur_poles, u, r);
        cur_knots = nk;
        cur_poles = np;
    }

    // Now split into Bezier segments.
    // After full insertion, find spans where the knot value changes.
    // Each span [knots[i], knots[i+1]) where knots[i] < knots[i+1]
    // corresponds to one Bezier segment of degree+1 poles starting
    // at pole index i - degree.
    let order = degree + 1;
    let mut segments = Vec::new();

    let mut i = degree;
    while i < cur_knots.len() - degree - 1 {
        if cur_knots[i] < cur_knots[i + 1] {
            let u_start = cur_knots[i];
            let u_end = cur_knots[i + 1];
            let pole_start = i - degree;
            let pole_end = pole_start + order;

            let mut seg_knots = vec![u_start; order];
            seg_knots.extend(vec![u_end; order]);
            let seg_poles = cur_poles[pole_start..pole_end].to_vec();
            segments.push((seg_knots, seg_poles));
        }
        i += 1;
    }

    segments
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gp::{Pnt, Pnt2d};
    use nalgebra::{Point2, Point3};

    fn pnt(x: f64, y: f64, z: f64) -> Pnt {
        Point3::new(x, y, z)
    }

    fn assert_pnt_near(a: &Pnt, b: &Pnt, tol: f64) {
        assert!((a - b).norm() < tol, "{a:?} vs {b:?}");
    }

    fn assert_curves_equal<P: ControlPoint + std::fmt::Debug>(
        degree: usize,
        knots1: &[f64],
        poles1: &[P],
        knots2: &[f64],
        poles2: &[P],
        n_samples: usize,
    ) where
        P: std::ops::Sub<P, Output = nalgebra::Vector3<f64>>,
    {
        let u_min = knots1[degree];
        let u_max = knots1[knots1.len() - degree - 1];
        for i in 0..=n_samples {
            let u = u_min + (u_max - u_min) * i as f64 / n_samples as f64;
            let p1: P = super::super::curves::curve_point(degree, knots1, poles1, u);
            let p2: P = super::super::curves::curve_point(degree, knots2, poles2, u);
            let diff = p1 - p2;
            assert!(diff.norm() < 1e-12, "mismatch at u={u}: {p1:?} vs {p2:?}");
        }
    }

    // -- 3D insertion --

    #[test]
    fn insert_preserves_curve_shape() {
        let knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0];
        let poles: Vec<Pnt> = (0..7)
            .map(|i| pnt(i as f64, (i as f64).sin(), (i as f64).cos()))
            .collect();
        let (nk, np) = insert_knot(3, &knots, &poles, 1.5, 1);
        assert_curves_equal(3, &knots, &poles, &nk, &np, 40);
    }

    #[test]
    fn insert_increases_pole_count() {
        let knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0];
        let poles: Vec<Pnt> = (0..7).map(|i| pnt(i as f64, 0.0, 0.0)).collect();
        let (nk, np) = insert_knot(3, &knots, &poles, 1.5, 1);
        assert_eq!(np.len(), poles.len() + 1);
        assert_eq!(nk.len(), knots.len() + 1);
    }

    #[test]
    fn insert_multiple_times() {
        let knots = vec![0.0, 0.0, 0.0, 0.0, 2.0, 4.0, 4.0, 4.0, 4.0];
        let poles: Vec<Pnt> = (0..5).map(|i| pnt(i as f64, (i * i) as f64, 0.0)).collect();
        let (nk, np) = insert_knot(3, &knots, &poles, 1.0, 2);
        assert_eq!(np.len(), poles.len() + 2);
        assert_curves_equal(3, &knots, &poles, &nk, &np, 40);
    }

    #[test]
    fn insert_at_existing_knot() {
        let knots = vec![0.0, 0.0, 0.0, 0.0, 2.0, 4.0, 4.0, 4.0, 4.0];
        let poles: Vec<Pnt> = (0..5).map(|i| pnt(i as f64, (i * i) as f64, 0.0)).collect();
        let (nk, np) = insert_knot(3, &knots, &poles, 2.0, 1);
        assert_eq!(nk.iter().filter(|&&v| v == 2.0).count(), 2);
        assert_curves_equal(3, &knots, &poles, &nk, &np, 40);
    }

    #[test]
    fn insert_into_bezier() {
        let knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let poles = vec![
            pnt(0.0, 0.0, 0.0),
            pnt(0.0, 1.0, 0.0),
            pnt(1.0, 1.0, 0.0),
            pnt(1.0, 0.0, 0.0),
        ];
        let (nk, np) = insert_knot(3, &knots, &poles, 0.5, 1);
        assert_curves_equal(3, &knots, &poles, &nk, &np, 20);
    }

    // -- 2D insertion --

    #[test]
    fn insert_2d() {
        let knots = vec![0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0];
        let poles: Vec<Pnt2d> = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 2.0),
            Point2::new(2.0, 1.0),
            Point2::new(3.0, 0.0),
        ];
        let (nk, np) = insert_knot(2, &knots, &poles, 0.5, 1);
        assert_eq!(np.len(), poles.len() + 1);
        // Verify shape preservation at several points
        for i in 0..=20 {
            let u = i as f64 * 0.1;
            let p1: Pnt2d = super::super::curves::curve_point(2, &knots, &poles, u);
            let p2: Pnt2d = super::super::curves::curve_point(2, &nk, &np, u);
            assert!((p1 - p2).norm() < 1e-12, "2D mismatch at u={u}");
        }
    }

    // -- Rational --

    #[test]
    fn rational_insert_preserves_circle() {
        let knots = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let poles = vec![pnt(1.0, 0.0, 0.0), pnt(1.0, 1.0, 0.0), pnt(0.0, 1.0, 0.0)];
        let w = std::f64::consts::FRAC_1_SQRT_2;
        let weights = vec![1.0, w, 1.0];

        let (nk, np, nw) = insert_knot_rational(2, &knots, &poles, &weights, 0.5, 1);

        for i in 0..=20 {
            let u = i as f64 / 20.0;
            let p: Pnt = super::super::curves::rational_curve_point(2, &nk, &np, &nw, u);
            let radius = (p.x * p.x + p.y * p.y).sqrt();
            assert!((radius - 1.0).abs() < 1e-12, "u={u}: r={radius}");
        }
    }

    // -- Bezier extraction --

    #[test]
    fn extract_bezier_from_single_bezier() {
        let knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let poles = vec![
            pnt(0.0, 0.0, 0.0),
            pnt(0.0, 1.0, 0.0),
            pnt(1.0, 1.0, 0.0),
            pnt(1.0, 0.0, 0.0),
        ];
        let segments = extract_bezier(3, &knots, &poles);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].1.len(), 4);
    }

    #[test]
    fn extract_bezier_multi_span() {
        let knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0];
        let poles: Vec<Pnt> = (0..7)
            .map(|i| pnt(i as f64, (i as f64).sin(), 0.0))
            .collect();

        let segments = extract_bezier(3, &knots, &poles);
        // 4 spans: [0,1], [1,2], [2,3], [3,4]
        assert_eq!(segments.len(), 4, "expected 4 Bezier segments");

        // Each segment evaluated should match the original curve
        for (seg_knots, seg_poles) in &segments {
            let u_start = seg_knots[0];
            let u_end = *seg_knots.last().unwrap();
            for j in 0..=10 {
                let u = u_start + (u_end - u_start) * j as f64 / 10.0;
                let p_orig: Pnt = super::super::curves::curve_point(3, &knots, &poles, u);
                let p_bez: Pnt = super::super::curves::curve_point(3, seg_knots, seg_poles, u);
                assert_pnt_near(&p_orig, &p_bez, 1e-11);
            }
        }
    }
}
