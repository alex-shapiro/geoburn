//! B-spline curve evaluation: point and derivatives.
//!
//! All functions are generic over `ControlPoint`, supporting both 2D and 3D.

#![allow(clippy::needless_range_loop)]
//! Each function has `_at_span` variants for allocation-free hot loops.

use super::basis::{basis_funs_into, ders_basis_funs_into};
use super::knots::{find_span, validate_knots, validate_poles};
use super::{ControlPoint, MAX_ORDER};

/// Evaluate a non-rational B-spline curve at parameter `u`.
///
/// P&T Algorithm A3.1.
pub fn curve_point<P: ControlPoint>(degree: usize, knots: &[f64], poles: &[P], u: f64) -> P {
    validate_knots(degree, knots);
    validate_poles(degree, knots, poles.len());
    let n = poles.len() - 1;
    let span = find_span(n, degree, u, knots);
    curve_point_at_span(degree, knots, poles, u, span)
}

/// Evaluate at a precomputed span (no validation, no span search).
pub fn curve_point_at_span<P: ControlPoint>(
    degree: usize,
    knots: &[f64],
    poles: &[P],
    u: f64,
    span: usize,
) -> P {
    let mut basis = [0.0f64; MAX_ORDER];
    let mut left = [0.0f64; MAX_ORDER];
    let mut right = [0.0f64; MAX_ORDER];
    basis_funs_into(span, u, degree, knots, &mut basis, &mut left, &mut right);

    let mut point = P::origin();
    for i in 0..=degree {
        point = point.add(&poles[span - degree + i].scaled(basis[i]));
    }
    point
}

/// Evaluate curve and derivatives up to `deriv_order`.
///
/// P&T Algorithm A3.2.
///
/// Returns `deriv_order + 1` values. `[0]` is the point, `[k]` is the kth derivative.
pub fn curve_derivs<P: ControlPoint>(
    degree: usize,
    knots: &[f64],
    poles: &[P],
    u: f64,
    deriv_order: usize,
) -> Vec<P> {
    validate_knots(degree, knots);
    validate_poles(degree, knots, poles.len());
    let n = poles.len() - 1;
    let span = find_span(n, degree, u, knots);
    curve_derivs_at_span(degree, knots, poles, u, span, deriv_order)
}

/// Derivatives at a precomputed span.
///
/// Uses stack-allocated basis derivative computation (allocation-free).
pub fn curve_derivs_at_span<P: ControlPoint>(
    degree: usize,
    knots: &[f64],
    poles: &[P],
    u: f64,
    span: usize,
    deriv_order: usize,
) -> Vec<P> {
    let mut ders_buf = [0.0f64; MAX_ORDER * MAX_ORDER];
    let n_deriv = ders_basis_funs_into(span, u, degree, deriv_order, knots, &mut ders_buf);
    let mut ck = vec![P::origin(); deriv_order + 1];

    for k in 0..=n_deriv {
        for j in 0..=degree {
            ck[k] = ck[k].add(&poles[span - degree + j].scaled(ders_buf[k * MAX_ORDER + j]));
        }
    }
    ck
}

/// Evaluate a rational B-spline (NURBS) curve at parameter `u`.
///
/// P&T Algorithm A4.1.
pub fn rational_curve_point<P: ControlPoint>(
    degree: usize,
    knots: &[f64],
    poles: &[P],
    weights: &[f64],
    u: f64,
) -> P {
    validate_knots(degree, knots);
    validate_poles(degree, knots, poles.len());
    assert_eq!(
        weights.len(),
        poles.len(),
        "weights length must match poles"
    );
    let n = poles.len() - 1;
    let span = find_span(n, degree, u, knots);
    rational_curve_point_at_span(degree, knots, poles, weights, u, span)
}

/// Rational curve point at a precomputed span.
pub fn rational_curve_point_at_span<P: ControlPoint>(
    degree: usize,
    knots: &[f64],
    poles: &[P],
    weights: &[f64],
    u: f64,
    span: usize,
) -> P {
    let mut basis = [0.0f64; MAX_ORDER];
    let mut left = [0.0f64; MAX_ORDER];
    let mut right = [0.0f64; MAX_ORDER];
    basis_funs_into(span, u, degree, knots, &mut basis, &mut left, &mut right);

    let mut cw = P::origin();
    let mut w = 0.0;

    for i in 0..=degree {
        let idx = span - degree + i;
        let wi = weights[idx] * basis[i];
        cw = cw.add(&poles[idx].scaled(wi));
        w += wi;
    }

    cw.div(w)
}

/// Evaluate a rational B-spline curve and its derivatives.
///
/// P&T Eq. 4.8.
pub fn rational_curve_derivs<P: ControlPoint>(
    degree: usize,
    knots: &[f64],
    poles: &[P],
    weights: &[f64],
    u: f64,
    deriv_order: usize,
) -> Vec<P> {
    validate_knots(degree, knots);
    validate_poles(degree, knots, poles.len());
    assert_eq!(
        weights.len(),
        poles.len(),
        "weights length must match poles"
    );
    let n = poles.len() - 1;
    let span = find_span(n, degree, u, knots);
    rational_curve_derivs_at_span(degree, knots, poles, weights, u, span, deriv_order)
}

/// Rational derivatives at a precomputed span.
pub fn rational_curve_derivs_at_span<P: ControlPoint>(
    degree: usize,
    knots: &[f64],
    poles: &[P],
    weights: &[f64],
    u: f64,
    span: usize,
    deriv_order: usize,
) -> Vec<P> {
    let mut ders_buf = [0.0f64; MAX_ORDER * MAX_ORDER];
    let n_deriv = ders_basis_funs_into(span, u, degree, deriv_order, knots, &mut ders_buf);

    let mut a_ders = vec![P::origin(); deriv_order + 1];
    let mut w_ders = vec![0.0f64; deriv_order + 1];

    for k in 0..=n_deriv {
        for j in 0..=degree {
            let idx = span - degree + j;
            let nw = ders_buf[k * MAX_ORDER + j] * weights[idx];
            a_ders[k] = a_ders[k].add(&poles[idx].scaled(nw));
            w_ders[k] += nw;
        }
    }

    let mut ck = vec![P::origin(); deriv_order + 1];

    for k in 0..=deriv_order {
        let mut v = a_ders[k];
        for i in 1..=k {
            v = v.add(&ck[k - i].scaled(-w_ders[i] * binomial(k, i) as f64));
        }
        ck[k] = v.div(w_ders[0]);
    }

    ck
}

fn binomial(n: usize, k: usize) -> usize {
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
    use crate::gp::{Pnt, Pnt2d, Vec3};
    use nalgebra::{Point2, Point3};

    fn pnt(x: f64, y: f64, z: f64) -> Pnt {
        Point3::new(x, y, z)
    }

    fn pnt2d(x: f64, y: f64) -> Pnt2d {
        Point2::new(x, y)
    }

    fn assert_pnt_near(a: &Pnt, b: &Pnt, tol: f64) {
        assert!((a - b).norm() < tol, "{a:?} vs {b:?}");
    }

    // -- 3D: Linear --

    #[test]
    fn linear_bspline() {
        let knots = vec![0.0, 0.0, 1.0, 1.0];
        let poles = vec![pnt(0.0, 0.0, 0.0), pnt(10.0, 0.0, 0.0)];
        assert_pnt_near(
            &curve_point(1, &knots, &poles, 0.0),
            &pnt(0.0, 0.0, 0.0),
            1e-15,
        );
        assert_pnt_near(
            &curve_point(1, &knots, &poles, 0.5),
            &pnt(5.0, 0.0, 0.0),
            1e-15,
        );
        assert_pnt_near(
            &curve_point(1, &knots, &poles, 1.0),
            &pnt(10.0, 0.0, 0.0),
            1e-15,
        );
    }

    // -- 3D: Bezier --

    #[test]
    fn cubic_bezier_midpoint() {
        let knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let poles = vec![
            pnt(0.0, 0.0, 0.0),
            pnt(0.0, 1.0, 0.0),
            pnt(1.0, 1.0, 0.0),
            pnt(1.0, 0.0, 0.0),
        ];
        assert_pnt_near(
            &curve_point(3, &knots, &poles, 0.5),
            &pnt(0.5, 0.75, 0.0),
            1e-14,
        );
    }

    // -- 2D: works with same code --

    #[test]
    fn linear_2d() {
        let knots = vec![0.0, 0.0, 1.0, 1.0];
        let poles = vec![pnt2d(0.0, 0.0), pnt2d(10.0, 5.0)];
        let mid: Pnt2d = curve_point(1, &knots, &poles, 0.5);
        assert!((mid.x - 5.0).abs() < 1e-15);
        assert!((mid.y - 2.5).abs() < 1e-15);
    }

    #[test]
    fn quadratic_2d() {
        let knots = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let poles = vec![pnt2d(0.0, 0.0), pnt2d(0.5, 1.0), pnt2d(1.0, 0.0)];
        let mid: Pnt2d = curve_point(2, &knots, &poles, 0.5);
        // Quadratic Bezier at t=0.5: (0.5, 0.5)
        assert!((mid.x - 0.5).abs() < 1e-14);
        assert!((mid.y - 0.5).abs() < 1e-14);
    }

    // -- Affine invariance --

    #[test]
    fn affine_invariance() {
        let knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0];
        let poles: Vec<Pnt> = (0..7).map(|i| pnt(i as f64, (i * i) as f64, 0.0)).collect();
        let offset = Vec3::new(100.0, 200.0, 300.0);
        let u = 1.7;

        let p1 = Pnt::from(curve_point(3, &knots, &poles, u).coords + offset);
        let shifted: Vec<Pnt> = poles.iter().map(|p| Pnt::from(p.coords + offset)).collect();
        let p2 = curve_point(3, &knots, &shifted, u);
        assert_pnt_near(&p1, &p2, 1e-12);
    }

    // -- Derivatives --

    #[test]
    fn linear_derivative_is_constant() {
        let knots = vec![0.0, 0.0, 1.0, 1.0];
        let poles = vec![pnt(0.0, 0.0, 0.0), pnt(10.0, 0.0, 0.0)];
        let ders: Vec<Pnt> = curve_derivs(1, &knots, &poles, 0.5, 1);
        assert!((ders[0].coords - Vec3::new(5.0, 0.0, 0.0)).norm() < 1e-14);
        assert!((ders[1].coords - Vec3::new(10.0, 0.0, 0.0)).norm() < 1e-14);
    }

    #[test]
    fn second_derivative_of_linear_is_zero() {
        let knots = vec![0.0, 0.0, 1.0, 1.0];
        let poles = vec![pnt(0.0, 0.0, 0.0), pnt(10.0, 0.0, 0.0)];
        let ders: Vec<Pnt> = curve_derivs(1, &knots, &poles, 0.5, 2);
        assert!(ders[2].coords.norm() < 1e-14);
    }

    #[test]
    fn numerical_vs_analytical_derivative() {
        let knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0];
        let poles: Vec<Pnt> = (0..7)
            .map(|i| pnt(i as f64, (i as f64).sin(), (i as f64).cos()))
            .collect();
        let u = 1.5;
        let h = 1e-7;
        let ders: Vec<Pnt> = curve_derivs(3, &knots, &poles, u, 1);

        let p_plus: Pnt = curve_point(3, &knots, &poles, u + h);
        let p_minus: Pnt = curve_point(3, &knots, &poles, u - h);
        let numerical_d1 = (p_plus - p_minus) / (2.0 * h);

        assert!((ders[1].coords - numerical_d1).norm() < 1e-5);
    }

    // -- Rational: quarter circle --

    #[test]
    fn rational_quarter_circle() {
        let knots = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let poles = vec![pnt(1.0, 0.0, 0.0), pnt(1.0, 1.0, 0.0), pnt(0.0, 1.0, 0.0)];
        let w = std::f64::consts::FRAC_1_SQRT_2;
        let weights = vec![1.0, w, 1.0];

        for i in 0..=20 {
            let u = i as f64 / 20.0;
            let p: Pnt = rational_curve_point(2, &knots, &poles, &weights, u);
            let radius = (p.x * p.x + p.y * p.y).sqrt();
            assert!((radius - 1.0).abs() < 1e-14, "u={u}: r={radius}");
        }
    }

    #[test]
    fn rational_unit_weights_equals_nonrational() {
        let knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0];
        let poles: Vec<Pnt> = (0..7).map(|i| pnt(i as f64, (i * i) as f64, 0.0)).collect();
        let weights = vec![1.0; 7];

        for i in 0..=20 {
            let u = i as f64 * 0.2;
            let p1: Pnt = curve_point(3, &knots, &poles, u);
            let p2: Pnt = rational_curve_point(3, &knots, &poles, &weights, u);
            assert_pnt_near(&p1, &p2, 1e-14);
        }
    }

    // -- Rational curvature --

    #[test]
    fn rational_circle_curvature() {
        let knots = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let poles = vec![pnt(1.0, 0.0, 0.0), pnt(1.0, 1.0, 0.0), pnt(0.0, 1.0, 0.0)];
        let w = std::f64::consts::FRAC_1_SQRT_2;
        let weights = vec![1.0, w, 1.0];

        for i in 1..=9 {
            let u = i as f64 / 10.0;
            let ders: Vec<Pnt> = rational_curve_derivs(2, &knots, &poles, &weights, u, 2);
            let d1 = ders[1].coords;
            let d2 = ders[2].coords;
            let kappa = d1.cross(&d2).norm() / d1.norm().powi(3);
            assert!((kappa - 1.0).abs() < 1e-10, "u={u}: kappa={kappa}");
        }
    }

    #[test]
    fn rational_numerical_vs_analytical_2nd_derivative() {
        let knots = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let poles = vec![pnt(1.0, 0.0, 0.0), pnt(1.0, 1.0, 0.0), pnt(0.0, 1.0, 0.0)];
        let w = std::f64::consts::FRAC_1_SQRT_2;
        let weights = vec![1.0, w, 1.0];
        let u = 0.5;
        let h = 1e-4;

        let ders: Vec<Pnt> = rational_curve_derivs(2, &knots, &poles, &weights, u, 2);
        let eval =
            |t: f64| -> Vec3 { rational_curve_point::<Pnt>(2, &knots, &poles, &weights, t).coords };
        let numerical_d2 = (eval(u + h) - eval(u) * 2.0 + eval(u - h)) / (h * h);

        assert!((ders[2].coords - numerical_d2).norm() < 1e-7);
    }

    // -- Validation --

    #[test]
    #[should_panic(expected = "expected")]
    fn wrong_pole_count_panics() {
        let knots = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let poles = vec![pnt(0.0, 0.0, 0.0), pnt(1.0, 0.0, 0.0)];
        curve_point(2, &knots, &poles, 0.5);
    }

    #[test]
    fn at_span_matches_standard() {
        let knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0];
        let poles: Vec<Pnt> = (0..7).map(|i| pnt(i as f64, (i * i) as f64, 0.0)).collect();

        for i in 0..=40 {
            let u = i as f64 * 0.1;
            let p1: Pnt = curve_point(3, &knots, &poles, u);
            let n = poles.len() - 1;
            let span = find_span(n, 3, u, &knots);
            let p2: Pnt = curve_point_at_span(3, &knots, &poles, u, span);
            assert_pnt_near(&p1, &p2, 1e-15);
        }
    }
}
