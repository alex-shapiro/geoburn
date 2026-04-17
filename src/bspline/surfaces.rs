//! B-spline surface evaluation: tensor product surfaces.
//!
//! Poles are stored in row-major order: `poles[u_index * n_v + v_index]`.

#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

use super::basis::basis_funs_into;
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
}
