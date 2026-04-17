//! B-spline surface evaluation: tensor product surfaces.
//!
//! Poles are stored in row-major order: `poles[u_index * n_v + v_index]`.

#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

use super::basis::{basis_funs_into, ders_basis_funs_into};
use super::knots::{find_span, validate_knots};
use super::{ControlPoint, MAX_ORDER};

/// Evaluate a non-rational B-spline surface at parameters `(u, v)`.
///
/// Tensor product evaluation: compute U and V basis functions independently,
/// then blend the control net.
///
/// - `poles` is row-major: `poles[i * n_v + j]`
/// - `n_v` is the number of poles in the V direction
pub fn surface_point<P: ControlPoint>(
    u_degree: usize,
    v_degree: usize,
    u_knots: &[f64],
    v_knots: &[f64],
    poles: &[P],
    n_v: usize,
    u: f64,
    v: f64,
) -> P {
    let n_u = validate_knots(u_degree, u_knots);
    let n_v_check = validate_knots(v_degree, v_knots);
    assert_eq!(n_v, n_v_check, "n_v mismatch with v_knots");
    assert_eq!(poles.len(), n_u * n_v, "poles size mismatch");

    let u_n = n_u - 1;
    let v_n = n_v - 1;
    let u_span = find_span(u_n, u_degree, u, u_knots);
    let v_span = find_span(v_n, v_degree, v, v_knots);

    surface_point_at_span(
        u_degree, v_degree, u_knots, v_knots, poles, n_v, u, v, u_span, v_span,
    )
}

/// Evaluate at precomputed spans (allocation-free hot path).
pub fn surface_point_at_span<P: ControlPoint>(
    u_degree: usize,
    v_degree: usize,
    u_knots: &[f64],
    v_knots: &[f64],
    poles: &[P],
    n_v: usize,
    u: f64,
    v: f64,
    u_span: usize,
    v_span: usize,
) -> P {
    let mut u_basis = [0.0f64; MAX_ORDER];
    let mut u_left = [0.0f64; MAX_ORDER];
    let mut u_right = [0.0f64; MAX_ORDER];
    basis_funs_into(
        u_span,
        u,
        u_degree,
        u_knots,
        &mut u_basis,
        &mut u_left,
        &mut u_right,
    );

    let mut v_basis = [0.0f64; MAX_ORDER];
    let mut v_left = [0.0f64; MAX_ORDER];
    let mut v_right = [0.0f64; MAX_ORDER];
    basis_funs_into(
        v_span,
        v,
        v_degree,
        v_knots,
        &mut v_basis,
        &mut v_left,
        &mut v_right,
    );

    let mut result = P::origin();
    for i in 0..=u_degree {
        let u_idx = u_span - u_degree + i;
        let mut temp = P::origin();
        for j in 0..=v_degree {
            let v_idx = v_span - v_degree + j;
            temp = temp.add(&poles[u_idx * n_v + v_idx].scaled(v_basis[j]));
        }
        result = result.add(&temp.scaled(u_basis[i]));
    }
    result
}

/// Evaluate a non-rational B-spline surface and its first partial derivatives.
///
/// Returns `(point, du, dv)`.
pub fn surface_derivs<P: ControlPoint>(
    u_degree: usize,
    v_degree: usize,
    u_knots: &[f64],
    v_knots: &[f64],
    poles: &[P],
    n_v: usize,
    u: f64,
    v: f64,
) -> (P, P, P) {
    let n_u = validate_knots(u_degree, u_knots);
    let n_v_check = validate_knots(v_degree, v_knots);
    assert_eq!(n_v, n_v_check, "n_v mismatch with v_knots");
    assert_eq!(poles.len(), n_u * n_v, "poles size mismatch");

    let u_n = n_u - 1;
    let v_n = n_v - 1;
    let u_span = find_span(u_n, u_degree, u, u_knots);
    let v_span = find_span(v_n, v_degree, v, v_knots);

    surface_derivs_at_span(
        u_degree, v_degree, u_knots, v_knots, poles, n_v, u, v, u_span, v_span,
    )
}

/// Surface first derivatives at precomputed spans (allocation-free).
pub fn surface_derivs_at_span<P: ControlPoint>(
    u_degree: usize,
    v_degree: usize,
    u_knots: &[f64],
    v_knots: &[f64],
    poles: &[P],
    n_v: usize,
    u: f64,
    v: f64,
    u_span: usize,
    v_span: usize,
) -> (P, P, P) {
    // U basis + 1st derivative
    let mut u_ders = [0.0f64; MAX_ORDER * MAX_ORDER];
    ders_basis_funs_into(u_span, u, u_degree, 1, u_knots, &mut u_ders);

    // V basis + 1st derivative
    let mut v_ders = [0.0f64; MAX_ORDER * MAX_ORDER];
    ders_basis_funs_into(v_span, v, v_degree, 1, v_knots, &mut v_ders);

    let mut point = P::origin();
    let mut du = P::origin();
    let mut dv = P::origin();

    for i in 0..=u_degree {
        let u_idx = u_span - u_degree + i;
        let n0_u = u_ders[i]; // N_i(u)
        let n1_u = u_ders[MAX_ORDER + i]; // dN_i/du

        let mut temp_p = P::origin();
        let mut temp_dv = P::origin();

        for j in 0..=v_degree {
            let v_idx = v_span - v_degree + j;
            let pole = poles[u_idx * n_v + v_idx];
            let n0_v = v_ders[j]; // N_j(v)
            let n1_v = v_ders[MAX_ORDER + j]; // dN_j/dv

            temp_p = temp_p.add(&pole.scaled(n0_v));
            temp_dv = temp_dv.add(&pole.scaled(n1_v));
        }

        point = point.add(&temp_p.scaled(n0_u));
        du = du.add(&temp_p.scaled(n1_u));
        dv = dv.add(&temp_dv.scaled(n0_u));
    }

    (point, du, dv)
}

/// Evaluate a non-rational B-spline surface with first and second partial derivatives.
///
/// Returns `(point, du, dv, duu, dvv, duv)`.
pub fn surface_d2<P: ControlPoint>(
    u_degree: usize,
    v_degree: usize,
    u_knots: &[f64],
    v_knots: &[f64],
    poles: &[P],
    n_v: usize,
    u: f64,
    v: f64,
) -> (P, P, P, P, P, P) {
    let n_u = validate_knots(u_degree, u_knots);
    let n_v_check = validate_knots(v_degree, v_knots);
    assert_eq!(n_v, n_v_check);
    assert_eq!(poles.len(), n_u * n_v);

    let u_n = n_u - 1;
    let v_n = n_v - 1;
    let u_span = find_span(u_n, u_degree, u, u_knots);
    let v_span = find_span(v_n, v_degree, v, v_knots);

    // U and V basis + derivatives up to 2nd order
    let mut u_ders = [0.0f64; MAX_ORDER * MAX_ORDER];
    ders_basis_funs_into(u_span, u, u_degree, 2, u_knots, &mut u_ders);

    let mut v_ders = [0.0f64; MAX_ORDER * MAX_ORDER];
    ders_basis_funs_into(v_span, v, v_degree, 2, v_knots, &mut v_ders);

    let mut point = P::origin();
    let mut du = P::origin();
    let mut dv = P::origin();
    let mut duu = P::origin();
    let mut dvv = P::origin();
    let mut duv = P::origin();

    for i in 0..=u_degree {
        let u_idx = u_span - u_degree + i;
        let n0_u = u_ders[i];
        let n1_u = u_ders[MAX_ORDER + i];
        let n2_u = u_ders[2 * MAX_ORDER + i];

        let mut tp = P::origin();
        let mut tdv = P::origin();
        let mut tdvv = P::origin();

        for j in 0..=v_degree {
            let v_idx = v_span - v_degree + j;
            let pole = poles[u_idx * n_v + v_idx];
            let n0_v = v_ders[j];
            let n1_v = v_ders[MAX_ORDER + j];
            let n2_v = v_ders[2 * MAX_ORDER + j];

            tp = tp.add(&pole.scaled(n0_v));
            tdv = tdv.add(&pole.scaled(n1_v));
            tdvv = tdvv.add(&pole.scaled(n2_v));
        }

        point = point.add(&tp.scaled(n0_u));
        du = du.add(&tp.scaled(n1_u));
        dv = dv.add(&tdv.scaled(n0_u));
        duu = duu.add(&tp.scaled(n2_u));
        dvv = dvv.add(&tdvv.scaled(n0_u));
        duv = duv.add(&tdv.scaled(n1_u));
    }

    (point, du, dv, duu, dvv, duv)
}

/// Evaluate a rational B-spline (NURBS) surface at `(u, v)`.
pub fn rational_surface_point<P: ControlPoint>(
    u_degree: usize,
    v_degree: usize,
    u_knots: &[f64],
    v_knots: &[f64],
    poles: &[P],
    weights: &[f64],
    n_v: usize,
    u: f64,
    v: f64,
) -> P {
    let n_u = validate_knots(u_degree, u_knots);
    let n_v_check = validate_knots(v_degree, v_knots);
    assert_eq!(n_v, n_v_check);
    assert_eq!(poles.len(), n_u * n_v);
    assert_eq!(
        weights.len(),
        poles.len(),
        "weights length must match poles"
    );

    let u_n = n_u - 1;
    let v_n = n_v - 1;
    let u_span = find_span(u_n, u_degree, u, u_knots);
    let v_span = find_span(v_n, v_degree, v, v_knots);

    rational_surface_point_at_span(
        u_degree, v_degree, u_knots, v_knots, poles, weights, n_v, u, v, u_span, v_span,
    )
}

/// Rational surface point at precomputed spans.
pub fn rational_surface_point_at_span<P: ControlPoint>(
    u_degree: usize,
    v_degree: usize,
    u_knots: &[f64],
    v_knots: &[f64],
    poles: &[P],
    weights: &[f64],
    n_v: usize,
    u: f64,
    v: f64,
    u_span: usize,
    v_span: usize,
) -> P {
    let mut u_basis = [0.0f64; MAX_ORDER];
    let mut u_left = [0.0f64; MAX_ORDER];
    let mut u_right = [0.0f64; MAX_ORDER];
    basis_funs_into(
        u_span,
        u,
        u_degree,
        u_knots,
        &mut u_basis,
        &mut u_left,
        &mut u_right,
    );

    let mut v_basis = [0.0f64; MAX_ORDER];
    let mut v_left = [0.0f64; MAX_ORDER];
    let mut v_right = [0.0f64; MAX_ORDER];
    basis_funs_into(
        v_span,
        v,
        v_degree,
        v_knots,
        &mut v_basis,
        &mut v_left,
        &mut v_right,
    );

    let mut result = P::origin();
    let mut w_sum = 0.0;

    for i in 0..=u_degree {
        let u_idx = u_span - u_degree + i;
        let mut temp = P::origin();
        let mut w_temp = 0.0;
        for j in 0..=v_degree {
            let v_idx = v_span - v_degree + j;
            let idx = u_idx * n_v + v_idx;
            let wv = weights[idx] * v_basis[j];
            temp = temp.add(&poles[idx].scaled(wv));
            w_temp += wv;
        }
        result = result.add(&temp.scaled(u_basis[i]));
        w_sum += w_temp * u_basis[i];
    }

    result.div(w_sum)
}

/// Evaluate a rational B-spline surface and its first partial derivatives.
///
/// Uses P&T Eq. 4.20: compute weighted (homogeneous) derivatives, then
/// apply the rational formula to recover the Cartesian derivatives.
///
/// Returns `(point, du, dv)`.
pub fn rational_surface_derivs<P: ControlPoint>(
    u_degree: usize,
    v_degree: usize,
    u_knots: &[f64],
    v_knots: &[f64],
    poles: &[P],
    weights: &[f64],
    n_v: usize,
    u: f64,
    v: f64,
) -> (P, P, P) {
    let n_u = validate_knots(u_degree, u_knots);
    let n_v_check = validate_knots(v_degree, v_knots);
    assert_eq!(n_v, n_v_check);
    assert_eq!(poles.len(), n_u * n_v);
    assert_eq!(
        weights.len(),
        poles.len(),
        "weights length must match poles"
    );

    let u_n = n_u - 1;
    let v_n = n_v - 1;
    let u_span = find_span(u_n, u_degree, u, u_knots);
    let v_span = find_span(v_n, v_degree, v, v_knots);

    rational_surface_derivs_at_span(
        u_degree, v_degree, u_knots, v_knots, poles, weights, n_v, u, v, u_span, v_span,
    )
}

/// Rational surface first derivatives at precomputed spans.
pub fn rational_surface_derivs_at_span<P: ControlPoint>(
    u_degree: usize,
    v_degree: usize,
    u_knots: &[f64],
    v_knots: &[f64],
    poles: &[P],
    weights: &[f64],
    n_v: usize,
    u: f64,
    v: f64,
    u_span: usize,
    v_span: usize,
) -> (P, P, P) {
    // U and V basis functions + 1st derivatives
    let mut u_ders = [0.0f64; MAX_ORDER * MAX_ORDER];
    ders_basis_funs_into(u_span, u, u_degree, 1, u_knots, &mut u_ders);

    let mut v_ders = [0.0f64; MAX_ORDER * MAX_ORDER];
    ders_basis_funs_into(v_span, v, v_degree, 1, v_knots, &mut v_ders);

    // Compute weighted derivatives: A (point*weight) and w (weight only)
    // a_00 = S(u,v)*w, a_10 = dS/du*w component, a_01 = dS/dv*w component
    let mut a_00 = P::origin();
    let mut a_10 = P::origin();
    let mut a_01 = P::origin();
    let mut w_00 = 0.0f64;
    let mut w_10 = 0.0f64;
    let mut w_01 = 0.0f64;

    for i in 0..=u_degree {
        let u_idx = u_span - u_degree + i;
        let n0_u = u_ders[i];
        let n1_u = u_ders[MAX_ORDER + i];

        let mut temp_p = P::origin();
        let mut temp_dv = P::origin();
        let mut wt_p = 0.0f64;
        let mut wt_dv = 0.0f64;

        for j in 0..=v_degree {
            let v_idx = v_span - v_degree + j;
            let idx = u_idx * n_v + v_idx;
            let w = weights[idx];
            let n0_v = v_ders[j];
            let n1_v = v_ders[MAX_ORDER + j];

            // Weighted pole
            let wp = poles[idx].scaled(w);

            temp_p = temp_p.add(&wp.scaled(n0_v));
            temp_dv = temp_dv.add(&wp.scaled(n1_v));
            wt_p += w * n0_v;
            wt_dv += w * n1_v;
        }

        a_00 = a_00.add(&temp_p.scaled(n0_u));
        a_10 = a_10.add(&temp_p.scaled(n1_u));
        a_01 = a_01.add(&temp_dv.scaled(n0_u));
        w_00 += wt_p * n0_u;
        w_10 += wt_p * n1_u;
        w_01 += wt_dv * n0_u;
    }

    // Rational formula (P&T Eq. 4.20, first derivatives):
    // S = A_00 / w_00
    // dS/du = (A_10 - w_10 * S) / w_00
    // dS/dv = (A_01 - w_01 * S) / w_00
    let point = a_00.div(w_00);
    let du = a_10.add(&point.scaled(-w_10)).div(w_00);
    let dv = a_01.add(&point.scaled(-w_01)).div(w_00);

    (point, du, dv)
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

    // -- Bilinear patch (degree 1×1) --

    #[test]
    fn bilinear_patch() {
        let u_knots = vec![0.0, 0.0, 1.0, 1.0];
        let v_knots = vec![0.0, 0.0, 1.0, 1.0];
        // 2×2 grid: corners of unit square at z=0
        let poles = vec![
            pnt(0.0, 0.0, 0.0), // (0,0)
            pnt(1.0, 0.0, 0.0), // (0,1)
            pnt(0.0, 1.0, 0.0), // (1,0)
            pnt(1.0, 1.0, 0.0), // (1,1)
        ];

        assert_pnt_near(
            &surface_point(1, 1, &u_knots, &v_knots, &poles, 2, 0.0, 0.0),
            &pnt(0.0, 0.0, 0.0),
            1e-15,
        );
        assert_pnt_near(
            &surface_point(1, 1, &u_knots, &v_knots, &poles, 2, 1.0, 1.0),
            &pnt(1.0, 1.0, 0.0),
            1e-15,
        );
        assert_pnt_near(
            &surface_point(1, 1, &u_knots, &v_knots, &poles, 2, 0.5, 0.5),
            &pnt(0.5, 0.5, 0.0),
            1e-15,
        );
    }

    // -- Bicubic Bezier patch --

    #[test]
    fn bicubic_bezier_corners() {
        let u_knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let v_knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        // 4×4 grid
        let mut poles = Vec::new();
        for i in 0..4 {
            for j in 0..4 {
                poles.push(pnt(j as f64, i as f64, 0.0));
            }
        }

        // Corners should interpolate
        assert_pnt_near(
            &surface_point(3, 3, &u_knots, &v_knots, &poles, 4, 0.0, 0.0),
            &pnt(0.0, 0.0, 0.0),
            1e-14,
        );
        assert_pnt_near(
            &surface_point(3, 3, &u_knots, &v_knots, &poles, 4, 1.0, 1.0),
            &pnt(3.0, 3.0, 0.0),
            1e-14,
        );
    }

    // -- Flat plane: u,v map linearly to x,y --

    #[test]
    fn flat_plane_interior() {
        let u_knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let v_knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        // poles[i * n_v + j]: row i = U direction, col j = V direction
        // Map U → x, V → y for a clean identity test
        let mut poles = Vec::new();
        for i in 0..4 {
            for j in 0..4 {
                let x = i as f64 / 3.0; // U direction → x
                let y = j as f64 / 3.0; // V direction → y
                poles.push(pnt(x, y, 0.0));
            }
        }

        for ui in 0..=10 {
            for vi in 0..=10 {
                let u = ui as f64 / 10.0;
                let v = vi as f64 / 10.0;
                let p: Pnt = surface_point(3, 3, &u_knots, &v_knots, &poles, 4, u, v);
                assert_pnt_near(&p, &pnt(u, v, 0.0), 1e-13);
            }
        }
    }

    // -- 2D surface (for completeness) --

    #[test]
    fn bilinear_2d() {
        let u_knots = vec![0.0, 0.0, 1.0, 1.0];
        let v_knots = vec![0.0, 0.0, 1.0, 1.0];
        let poles: Vec<Pnt2d> = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(0.0, 1.0),
            Point2::new(1.0, 1.0),
        ];
        let p: Pnt2d = surface_point(1, 1, &u_knots, &v_knots, &poles, 2, 0.5, 0.5);
        assert!((p.x - 0.5).abs() < 1e-15);
        assert!((p.y - 0.5).abs() < 1e-15);
    }

    // -- Rational: sphere patch --

    #[test]
    fn rational_surface_unit_weights() {
        let u_knots = vec![0.0, 0.0, 1.0, 1.0];
        let v_knots = vec![0.0, 0.0, 1.0, 1.0];
        let poles = vec![
            pnt(0.0, 0.0, 0.0),
            pnt(1.0, 0.0, 0.0),
            pnt(0.0, 1.0, 0.0),
            pnt(1.0, 1.0, 0.0),
        ];
        let weights = vec![1.0; 4];

        for ui in 0..=5 {
            for vi in 0..=5 {
                let u = ui as f64 / 5.0;
                let v = vi as f64 / 5.0;
                let p1: Pnt = surface_point(1, 1, &u_knots, &v_knots, &poles, 2, u, v);
                let p2: Pnt =
                    rational_surface_point(1, 1, &u_knots, &v_knots, &poles, &weights, 2, u, v);
                assert_pnt_near(&p1, &p2, 1e-14);
            }
        }
    }

    // -- Rational surface derivatives --

    #[test]
    fn rational_derivs_unit_weights_match_nonrational() {
        let u_knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let v_knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let mut poles = Vec::new();
        for i in 0..4 {
            for j in 0..4 {
                let x = i as f64 / 3.0;
                let y = j as f64 / 3.0;
                let z = (x * 2.0).sin() * (y * 3.0).cos();
                poles.push(pnt(x, y, z));
            }
        }
        let weights = vec![1.0; 16];

        for ui in 0..=5 {
            for vi in 0..=5 {
                let u = ui as f64 / 5.0;
                let v = vi as f64 / 5.0;
                let (p1, du1, dv1) = surface_derivs(3, 3, &u_knots, &v_knots, &poles, 4, u, v);
                let (p2, du2, dv2) =
                    rational_surface_derivs(3, 3, &u_knots, &v_knots, &poles, &weights, 4, u, v);
                assert_pnt_near(&p1, &p2, 1e-13);
                assert!((du1.coords - du2.coords).norm() < 1e-12, "du at ({u},{v})");
                assert!((dv1.coords - dv2.coords).norm() < 1e-12, "dv at ({u},{v})");
            }
        }
    }

    #[test]
    fn rational_derivs_numerical_check() {
        // Non-uniform weights: verify with finite differences
        let u_knots = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let v_knots = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let mut poles = Vec::new();
        for i in 0..3 {
            for j in 0..3 {
                poles.push(pnt(i as f64, j as f64, (i + j) as f64));
            }
        }
        let weights = vec![1.0, 2.0, 1.0, 0.5, 3.0, 0.5, 1.0, 2.0, 1.0];

        let h = 1e-7;
        for ui in 1..=4 {
            for vi in 1..=4 {
                let u = ui as f64 / 5.0;
                let v = vi as f64 / 5.0;

                let (p, du, dv) =
                    rational_surface_derivs(2, 2, &u_knots, &v_knots, &poles, &weights, 3, u, v);

                let p_check: Pnt =
                    rational_surface_point(2, 2, &u_knots, &v_knots, &poles, &weights, 3, u, v);
                assert_pnt_near(&p, &p_check, 1e-13);

                let num_du = (rational_surface_point::<Pnt>(
                    2,
                    2,
                    &u_knots,
                    &v_knots,
                    &poles,
                    &weights,
                    3,
                    u + h,
                    v,
                )
                .coords
                    - rational_surface_point::<Pnt>(
                        2,
                        2,
                        &u_knots,
                        &v_knots,
                        &poles,
                        &weights,
                        3,
                        u - h,
                        v,
                    )
                    .coords)
                    / (2.0 * h);
                let num_dv = (rational_surface_point::<Pnt>(
                    2,
                    2,
                    &u_knots,
                    &v_knots,
                    &poles,
                    &weights,
                    3,
                    u,
                    v + h,
                )
                .coords
                    - rational_surface_point::<Pnt>(
                        2,
                        2,
                        &u_knots,
                        &v_knots,
                        &poles,
                        &weights,
                        3,
                        u,
                        v - h,
                    )
                    .coords)
                    / (2.0 * h);

                assert!(
                    (du.coords - num_du).norm() < 1e-5,
                    "du mismatch at ({u},{v}): analytical={:?}, numerical={:?}",
                    du.coords,
                    num_du
                );
                assert!(
                    (dv.coords - num_dv).norm() < 1e-5,
                    "dv mismatch at ({u},{v}): analytical={:?}, numerical={:?}",
                    dv.coords,
                    num_dv
                );
            }
        }
    }

    // -- Non-rational surface derivatives --

    #[test]
    fn flat_plane_derivs() {
        // Bilinear patch mapping (u,v) → (u, v, 0): du=(1,0,0), dv=(0,1,0)
        let u_knots = vec![0.0, 0.0, 1.0, 1.0];
        let v_knots = vec![0.0, 0.0, 1.0, 1.0];
        let poles = vec![
            pnt(0.0, 0.0, 0.0),
            pnt(0.0, 1.0, 0.0),
            pnt(1.0, 0.0, 0.0),
            pnt(1.0, 1.0, 0.0),
        ];

        let (p, du, dv) = surface_derivs(1, 1, &u_knots, &v_knots, &poles, 2, 0.5, 0.5);
        assert_pnt_near(&p, &pnt(0.5, 0.5, 0.0), 1e-14);
        // du should be (1, 0, 0)
        assert!((du.coords - nalgebra::Vector3::new(1.0, 0.0, 0.0)).norm() < 1e-14);
        // dv should be (0, 1, 0)
        assert!((dv.coords - nalgebra::Vector3::new(0.0, 1.0, 0.0)).norm() < 1e-14);
    }

    #[test]
    fn derivs_numerical_vs_analytical() {
        // Bicubic surface, compare analytical derivatives with finite differences
        let u_knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let v_knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let mut poles = Vec::new();
        for i in 0..4 {
            for j in 0..4 {
                let x = i as f64 / 3.0;
                let y = j as f64 / 3.0;
                let z = (x * 3.0).sin() * (y * 2.0).cos();
                poles.push(pnt(x, y, z));
            }
        }

        let h = 1e-7;
        for ui in 1..=9 {
            for vi in 1..=9 {
                let u = ui as f64 / 10.0;
                let v = vi as f64 / 10.0;

                let (p, du, dv) = surface_derivs(3, 3, &u_knots, &v_knots, &poles, 4, u, v);

                let p_check: Pnt = surface_point(3, 3, &u_knots, &v_knots, &poles, 4, u, v);
                assert_pnt_near(&p, &p_check, 1e-14);

                let num_du = (surface_point::<Pnt>(3, 3, &u_knots, &v_knots, &poles, 4, u + h, v)
                    .coords
                    - surface_point::<Pnt>(3, 3, &u_knots, &v_knots, &poles, 4, u - h, v).coords)
                    / (2.0 * h);
                let num_dv = (surface_point::<Pnt>(3, 3, &u_knots, &v_knots, &poles, 4, u, v + h)
                    .coords
                    - surface_point::<Pnt>(3, 3, &u_knots, &v_knots, &poles, 4, u, v - h).coords)
                    / (2.0 * h);

                assert!(
                    (du.coords - num_du).norm() < 1e-5,
                    "du mismatch at ({u},{v})"
                );
                assert!(
                    (dv.coords - num_dv).norm() < 1e-5,
                    "dv mismatch at ({u},{v})"
                );
            }
        }
    }

    #[test]
    fn derivs_point_matches_value() {
        // The point from surface_derivs should exactly match surface_point
        let u_knots = vec![0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0];
        let v_knots = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let n_u = 4;
        let n_v = 3;
        let mut poles = Vec::new();
        for i in 0..n_u {
            for j in 0..n_v {
                poles.push(pnt(i as f64, j as f64, (i + j) as f64));
            }
        }

        for ui in 0..=20 {
            for vi in 0..=10 {
                let u = ui as f64 * 0.1;
                let v = vi as f64 * 0.1;
                let p1: Pnt = surface_point(2, 2, &u_knots, &v_knots, &poles, n_v, u, v);
                let (p2, _, _) = surface_derivs(2, 2, &u_knots, &v_knots, &poles, n_v, u, v);
                assert_pnt_near(&p1, &p2, 1e-14);
            }
        }
    }
}
