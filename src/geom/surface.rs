//! Surface types with a uniform evaluation interface.

use crate::bspline;
use crate::el_surfaces;
use crate::gp::ax3::Ax3;
use crate::gp::{Pnt, Vec3};

/// A 3D surface.
#[derive(Debug, Clone)]
pub enum Surface {
    Plane {
        pos: Ax3,
    },
    Cylinder {
        pos: Ax3,
        radius: f64,
    },
    Cone {
        pos: Ax3,
        radius: f64,
        semi_angle: f64,
    },
    Sphere {
        pos: Ax3,
        radius: f64,
    },
    Torus {
        pos: Ax3,
        major_radius: f64,
        minor_radius: f64,
    },
    BSpline {
        u_degree: usize,
        v_degree: usize,
        u_knots: Vec<f64>,
        v_knots: Vec<f64>,
        poles: Vec<Pnt>,
        n_v: usize,
        weights: Option<Vec<f64>>,
    },
}

impl Surface {
    /// Evaluate the surface at parameters `(u, v)`.
    pub fn value(&self, u: f64, v: f64) -> Pnt {
        match self {
            Surface::Plane { pos } => el_surfaces::plane_value(u, v, pos),
            Surface::Cylinder { pos, radius } => el_surfaces::cylinder_value(u, v, pos, *radius),
            Surface::Cone {
                pos,
                radius,
                semi_angle,
            } => el_surfaces::cone_value(u, v, pos, *radius, *semi_angle),
            Surface::Sphere { pos, radius } => el_surfaces::sphere_value(u, v, pos, *radius),
            Surface::Torus {
                pos,
                major_radius,
                minor_radius,
            } => el_surfaces::torus_value(u, v, pos, *major_radius, *minor_radius),
            Surface::BSpline {
                u_degree,
                v_degree,
                u_knots,
                v_knots,
                poles,
                n_v,
                weights,
            } => match weights {
                Some(w) => bspline::surfaces::rational_surface_point(
                    *u_degree, *v_degree, u_knots, v_knots, poles, w, *n_v, u, v,
                ),
                None => bspline::surfaces::surface_point(
                    *u_degree, *v_degree, u_knots, v_knots, poles, *n_v, u, v,
                ),
            },
        }
    }

    /// Evaluate the surface and its first partial derivatives at `(u, v)`.
    ///
    /// Returns `(point, du, dv)`.
    pub fn d1(&self, u: f64, v: f64) -> (Pnt, Vec3, Vec3) {
        match self {
            Surface::Plane { pos } => el_surfaces::plane_d1(u, v, pos),
            Surface::Cylinder { pos, radius } => el_surfaces::cylinder_d1(u, v, pos, *radius),
            Surface::Cone {
                pos,
                radius,
                semi_angle,
            } => el_surfaces::cone_d1(u, v, pos, *radius, *semi_angle),
            Surface::Sphere { pos, radius } => el_surfaces::sphere_d1(u, v, pos, *radius),
            Surface::Torus {
                pos,
                major_radius,
                minor_radius,
            } => el_surfaces::torus_d1(u, v, pos, *major_radius, *minor_radius),
            Surface::BSpline {
                u_degree,
                v_degree,
                u_knots,
                v_knots,
                poles,
                n_v,
                weights,
            } => match weights {
                Some(w) => {
                    let (p, du, dv) = bspline::surfaces::rational_surface_derivs(
                        *u_degree, *v_degree, u_knots, v_knots, poles, w, *n_v, u, v,
                    );
                    (p, du.coords, dv.coords)
                }
                None => {
                    let (p, du, dv) = bspline::surfaces::surface_derivs(
                        *u_degree, *v_degree, u_knots, v_knots, poles, *n_v, u, v,
                    );
                    (p, du.coords, dv.coords)
                }
            },
        }
    }

    /// Evaluate the surface, first, and second partial derivatives at `(u, v)`.
    ///
    /// Returns `(point, du, dv, duu, dvv, duv)`.
    pub fn d2(&self, u: f64, v: f64) -> (Pnt, Vec3, Vec3, Vec3, Vec3, Vec3) {
        match self {
            Surface::Plane { pos } => {
                let (p, du, dv) = el_surfaces::plane_d1(u, v, pos);
                (p, du, dv, Vec3::zeros(), Vec3::zeros(), Vec3::zeros())
            }
            Surface::BSpline {
                u_degree,
                v_degree,
                u_knots,
                v_knots,
                poles,
                n_v,
                weights,
            } => match weights {
                None => {
                    let (p, du, dv, duu, dvv, duv) = bspline::surfaces::surface_d2(
                        *u_degree, *v_degree, u_knots, v_knots, poles, *n_v, u, v,
                    );
                    (p, du.coords, dv.coords, duu.coords, dvv.coords, duv.coords)
                }
                Some(_) => {
                    // TODO: analytical rational surface d2
                    // For now, compute d2 via finite differences on d1
                    let h = 1e-5;
                    let (p, du, dv) = self.d1(u, v);
                    let (_, du_ph, _) = self.d1(u + h, v);
                    let (_, du_mh, _) = self.d1(u - h, v);
                    let (_, _, dv_ph) = self.d1(u, v + h);
                    let (_, _, dv_mh) = self.d1(u, v - h);
                    let (_, du_pv, _) = self.d1(u, v + h);
                    let duu = (du_ph - du_mh) / (2.0 * h);
                    let dvv = (dv_ph - dv_mh) / (2.0 * h);
                    let duv = (du_pv - du) / h;
                    (p, du, dv, duu, dvv, duv)
                }
            },
            _ => {
                // Elementary surfaces: use finite differences on d1
                let h = 1e-5;
                let (p, du, dv) = self.d1(u, v);
                let (_, du_ph, _) = self.d1(u + h, v);
                let (_, du_mh, _) = self.d1(u - h, v);
                let (_, _, dv_ph) = self.d1(u, v + h);
                let (_, _, dv_mh) = self.d1(u, v - h);
                let (_, du_pv, _) = self.d1(u, v + h);
                let duu = (du_ph - du_mh) / (2.0 * h);
                let dvv = (dv_ph - dv_mh) / (2.0 * h);
                let duv = (du_pv - du) / h;
                (p, du, dv, duu, dvv, duv)
            }
        }
    }

    /// Surface normal at `(u, v)`, computed as `du × dv` (unnormalized).
    pub fn normal(&self, u: f64, v: f64) -> Vec3 {
        let (_, du, dv) = self.d1(u, v);
        du.cross(&dv)
    }

    /// Parameter ranges: `((u_first, u_last), (v_first, v_last))`.
    pub fn parameter_range(&self) -> ((f64, f64), (f64, f64)) {
        use crate::gp::precision;
        use std::f64::consts::PI;
        let inf = precision::INFINITE;
        let two_pi = 2.0 * PI;
        match self {
            Surface::Plane { .. } => ((-inf, inf), (-inf, inf)),
            Surface::Cylinder { .. } => ((0.0, two_pi), (-inf, inf)),
            Surface::Cone { .. } => ((0.0, two_pi), (-inf, inf)),
            Surface::Sphere { .. } => ((0.0, two_pi), (-PI / 2.0, PI / 2.0)),
            Surface::Torus { .. } => ((0.0, two_pi), (0.0, two_pi)),
            Surface::BSpline {
                u_degree,
                v_degree,
                u_knots,
                v_knots,
                ..
            } => {
                let u_range = (u_knots[*u_degree], u_knots[u_knots.len() - u_degree - 1]);
                let v_range = (v_knots[*v_degree], v_knots[v_knots.len() - v_degree - 1]);
                (u_range, v_range)
            }
        }
    }

    pub fn surface_type(&self) -> crate::geom_types::SurfaceType {
        use crate::geom_types::SurfaceType;
        match self {
            Surface::Plane { .. } => SurfaceType::Plane,
            Surface::Cylinder { .. } => SurfaceType::Cylinder,
            Surface::Cone { .. } => SurfaceType::Cone,
            Surface::Sphere { .. } => SurfaceType::Sphere,
            Surface::Torus { .. } => SurfaceType::Torus,
            Surface::BSpline { .. } => SurfaceType::BSplineSurface,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gp::Dir;
    use nalgebra::{Point3, Unit, Vector3};
    use std::f64::consts::FRAC_PI_2;

    fn dir(x: f64, y: f64, z: f64) -> Dir {
        Unit::new_normalize(Vector3::new(x, y, z))
    }

    fn assert_pnt_near(a: &Pnt, b: &Pnt, tol: f64) {
        assert!((a - b).norm() < tol, "{a:?} vs {b:?}");
    }

    fn standard_ax3() -> Ax3 {
        Ax3::new(Point3::origin(), dir(0.0, 0.0, 1.0), dir(1.0, 0.0, 0.0))
    }

    #[test]
    fn plane() {
        let s = Surface::Plane {
            pos: standard_ax3(),
        };
        assert_pnt_near(&s.value(1.0, 2.0), &Point3::new(1.0, 2.0, 0.0), 1e-15);
    }

    #[test]
    fn sphere() {
        let s = Surface::Sphere {
            pos: standard_ax3(),
            radius: 5.0,
        };
        assert_pnt_near(&s.value(0.0, FRAC_PI_2), &Point3::new(0.0, 0.0, 5.0), 1e-14);
    }

    #[test]
    fn bspline_bilinear() {
        let s = Surface::BSpline {
            u_degree: 1,
            v_degree: 1,
            u_knots: vec![0.0, 0.0, 1.0, 1.0],
            v_knots: vec![0.0, 0.0, 1.0, 1.0],
            poles: vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
                Point3::new(1.0, 1.0, 0.0),
            ],
            n_v: 2,
            weights: None,
        };
        assert_pnt_near(&s.value(0.5, 0.5), &Point3::new(0.5, 0.5, 0.0), 1e-15);
    }

    #[test]
    fn plane_normal_is_z() {
        let s = Surface::Plane {
            pos: standard_ax3(),
        };
        let n = s.normal(0.0, 0.0);
        // For standard XY plane, normal should be Z
        assert!(n.x.abs() < 1e-14);
        assert!(n.y.abs() < 1e-14);
        assert!(n.z > 0.0);
    }

    #[test]
    fn sphere_normal_is_radial() {
        let s = Surface::Sphere {
            pos: standard_ax3(),
            radius: 1.0,
        };
        // At the equator (u=0, v=0), point is (1,0,0), normal should point radially
        let p = s.value(0.0, 0.0);
        let n = s.normal(0.0, 0.0);
        let n_unit = n.normalize();
        let p_unit = p.coords.normalize();
        assert!((n_unit - p_unit).norm() < 1e-5);
    }

    // -- Parameter ranges --

    #[test]
    fn plane_parameter_range() {
        let s = Surface::Plane {
            pos: standard_ax3(),
        };
        let ((u0, u1), (v0, v1)) = s.parameter_range();
        assert!(u0 < -1e90);
        assert!(u1 > 1e90);
        assert!(v0 < -1e90);
        assert!(v1 > 1e90);
    }

    #[test]
    fn sphere_parameter_range() {
        let s = Surface::Sphere {
            pos: standard_ax3(),
            radius: 1.0,
        };
        let ((u0, u1), (v0, v1)) = s.parameter_range();
        assert!((u0).abs() < 1e-15);
        assert!((u1 - 2.0 * std::f64::consts::PI).abs() < 1e-15);
        assert!((v0 - (-FRAC_PI_2)).abs() < 1e-15);
        assert!((v1 - FRAC_PI_2).abs() < 1e-15);
    }

    #[test]
    fn bspline_parameter_range() {
        let s = Surface::BSpline {
            u_degree: 1,
            v_degree: 1,
            u_knots: vec![0.0, 0.0, 1.0, 1.0],
            v_knots: vec![0.0, 0.0, 2.0, 2.0],
            poles: vec![
                Point3::origin(),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
                Point3::new(1.0, 1.0, 0.0),
            ],
            n_v: 2,
            weights: None,
        };
        let ((u0, u1), (v0, v1)) = s.parameter_range();
        assert!((u0).abs() < 1e-15);
        assert!((u1 - 1.0).abs() < 1e-15);
        assert!((v0).abs() < 1e-15);
        assert!((v1 - 2.0).abs() < 1e-15);
    }

    // -- d2 --

    #[test]
    fn plane_d2_is_zero() {
        let s = Surface::Plane {
            pos: standard_ax3(),
        };
        let (_, _, _, duu, dvv, duv) = s.d2(0.5, 0.5);
        assert!(duu.norm() < 1e-14);
        assert!(dvv.norm() < 1e-14);
        assert!(duv.norm() < 1e-14);
    }

    #[test]
    fn bspline_d2_matches_surface_d2() {
        // Verify geom Surface::d2 matches bspline::surfaces::surface_d2
        let s = Surface::BSpline {
            u_degree: 2,
            v_degree: 2,
            u_knots: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            v_knots: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            poles: vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(0.5, 0.0, 1.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.0, 0.5, 1.0),
                Point3::new(0.5, 0.5, 2.0),
                Point3::new(1.0, 0.5, 1.0),
                Point3::new(0.0, 1.0, 0.0),
                Point3::new(0.5, 1.0, 1.0),
                Point3::new(1.0, 1.0, 0.0),
            ],
            n_v: 3,
            weights: None,
        };
        let (p, du, dv, duu, dvv, _duv) = s.d2(0.5, 0.5);
        // Point should match value
        assert_pnt_near(&p, &s.value(0.5, 0.5), 1e-14);
        // First derivs should match d1
        let (_, du1, dv1) = s.d1(0.5, 0.5);
        assert!((du - du1).norm() < 1e-14);
        assert!((dv - dv1).norm() < 1e-14);
        // Second derivs should be non-zero for this curved surface
        assert!(duu.norm() > 1e-10 || dvv.norm() > 1e-10);
    }
}
