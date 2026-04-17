//! 3D curve types with a uniform evaluation interface.

use crate::bspline;
use crate::el_curves;
use crate::gp::ax1::Ax1;
use crate::gp::ax2::Ax2;
use crate::gp::{Pnt, Vec3};

/// A 3D curve.
///
/// Closed enum covering all fundamental curve types. Each variant
/// stores the minimal geometric data needed for evaluation.
#[derive(Debug, Clone)]
pub enum Curve3d {
    Line {
        origin: Pnt,
        dir: crate::gp::Dir,
    },
    Circle {
        pos: Ax2,
        radius: f64,
    },
    Ellipse {
        pos: Ax2,
        major_radius: f64,
        minor_radius: f64,
    },
    Hyperbola {
        pos: Ax2,
        major_radius: f64,
        minor_radius: f64,
    },
    Parabola {
        pos: Ax2,
        focal_length: f64,
    },
    BSpline {
        degree: usize,
        knots: Vec<f64>,
        poles: Vec<Pnt>,
        weights: Option<Vec<f64>>,
    },
}

impl Curve3d {
    /// Evaluate the curve at parameter `u`.
    pub fn value(&self, u: f64) -> Pnt {
        match self {
            Curve3d::Line { origin, dir } => {
                let ax = Ax1::new(*origin, *dir);
                el_curves::line_value(u, &ax)
            }
            Curve3d::Circle { pos, radius } => el_curves::circle_value(u, pos, *radius),
            Curve3d::Ellipse {
                pos,
                major_radius,
                minor_radius,
            } => el_curves::ellipse_value(u, pos, *major_radius, *minor_radius),
            Curve3d::Hyperbola {
                pos,
                major_radius,
                minor_radius,
            } => el_curves::hyperbola_value(u, pos, *major_radius, *minor_radius),
            Curve3d::Parabola { pos, focal_length } => {
                el_curves::parabola_value(u, pos, *focal_length)
            }
            Curve3d::BSpline {
                degree,
                knots,
                poles,
                weights,
            } => match weights {
                Some(w) => bspline::curves::rational_curve_point(*degree, knots, poles, w, u),
                None => bspline::curves::curve_point(*degree, knots, poles, u),
            },
        }
    }

    /// Evaluate the curve and its first derivative at `u`.
    ///
    /// Returns `(point, tangent_vector)`.
    pub fn d1(&self, u: f64) -> (Pnt, Vec3) {
        match self {
            Curve3d::Line { origin, dir } => {
                let ax = Ax1::new(*origin, *dir);
                el_curves::line_d1(u, &ax)
            }
            Curve3d::Circle { pos, radius } => {
                let (p, v) = el_curves::circle_d1(u, pos, *radius);
                (p, v)
            }
            Curve3d::Ellipse {
                pos,
                major_radius,
                minor_radius,
            } => el_curves::ellipse_d1(u, pos, *major_radius, *minor_radius),
            Curve3d::Hyperbola {
                pos,
                major_radius,
                minor_radius,
            } => el_curves::hyperbola_d1(u, pos, *major_radius, *minor_radius),
            Curve3d::Parabola { pos, focal_length } => {
                el_curves::parabola_d1(u, pos, *focal_length)
            }
            Curve3d::BSpline {
                degree,
                knots,
                poles,
                weights,
            } => {
                let ders = match weights {
                    Some(w) => {
                        bspline::curves::rational_curve_derivs(*degree, knots, poles, w, u, 1)
                    }
                    None => bspline::curves::curve_derivs(*degree, knots, poles, u, 1),
                };
                (ders[0], ders[1].coords)
            }
        }
    }

    /// Evaluate the curve, first, and second derivatives at `u`.
    ///
    /// Returns `(point, d1, d2)`.
    pub fn d2(&self, u: f64) -> (Pnt, Vec3, Vec3) {
        match self {
            Curve3d::Circle { pos, radius } => el_curves::circle_d2(u, pos, *radius),
            Curve3d::Ellipse {
                pos,
                major_radius,
                minor_radius,
            } => el_curves::ellipse_d2(u, pos, *major_radius, *minor_radius),
            Curve3d::Hyperbola {
                pos,
                major_radius,
                minor_radius,
            } => el_curves::hyperbola_d2(u, pos, *major_radius, *minor_radius),
            Curve3d::Parabola { pos, focal_length } => {
                el_curves::parabola_d2(u, pos, *focal_length)
            }
            Curve3d::Line { origin, dir } => {
                let ax = Ax1::new(*origin, *dir);
                let (p, v1) = el_curves::line_d1(u, &ax);
                (p, v1, Vec3::zeros())
            }
            Curve3d::BSpline {
                degree,
                knots,
                poles,
                weights,
            } => {
                let ders = match weights {
                    Some(w) => {
                        bspline::curves::rational_curve_derivs(*degree, knots, poles, w, u, 2)
                    }
                    None => bspline::curves::curve_derivs(*degree, knots, poles, u, 2),
                };
                (ders[0], ders[1].coords, ders[2].coords)
            }
        }
    }

    /// Parameter range `(first, last)`.
    ///
    /// For unbounded curves (line, hyperbola, parabola), returns `(-1e100, 1e100)`.
    /// For periodic curves (circle, ellipse), returns `(0, 2π)`.
    /// For B-splines, returns `(knots[degree], knots[n+1])`.
    pub fn parameter_range(&self) -> (f64, f64) {
        use crate::gp::precision;
        match self {
            Curve3d::Line { .. } | Curve3d::Hyperbola { .. } | Curve3d::Parabola { .. } => {
                (-precision::INFINITE, precision::INFINITE)
            }
            Curve3d::Circle { .. } | Curve3d::Ellipse { .. } => (0.0, 2.0 * std::f64::consts::PI),
            Curve3d::BSpline { degree, knots, .. } => {
                (knots[*degree], knots[knots.len() - degree - 1])
            }
        }
    }

    /// Whether the curve is rational (has non-uniform weights).
    pub fn is_rational(&self) -> bool {
        matches!(
            self,
            Curve3d::BSpline {
                weights: Some(_),
                ..
            } | Curve3d::Circle { .. }
                | Curve3d::Ellipse { .. }
                | Curve3d::Hyperbola { .. }
                | Curve3d::Parabola { .. }
        )
    }

    /// The curve type classification.
    pub fn curve_type(&self) -> crate::geom_types::CurveType {
        use crate::geom_types::CurveType;
        match self {
            Curve3d::Line { .. } => CurveType::Line,
            Curve3d::Circle { .. } => CurveType::Circle,
            Curve3d::Ellipse { .. } => CurveType::Ellipse,
            Curve3d::Hyperbola { .. } => CurveType::Hyperbola,
            Curve3d::Parabola { .. } => CurveType::Parabola,
            Curve3d::BSpline { .. } => CurveType::BSplineCurve,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Point3, Unit, Vector3};
    use std::f64::consts::FRAC_PI_2;

    fn dir(x: f64, y: f64, z: f64) -> crate::gp::Dir {
        Unit::new_normalize(Vector3::new(x, y, z))
    }

    fn assert_pnt_near(a: &Pnt, b: &Pnt, tol: f64) {
        assert!((a - b).norm() < tol, "{a:?} vs {b:?}");
    }

    #[test]
    fn line_value() {
        let c = Curve3d::Line {
            origin: Point3::origin(),
            dir: dir(1.0, 0.0, 0.0),
        };
        assert_pnt_near(&c.value(5.0), &Point3::new(5.0, 0.0, 0.0), 1e-15);
    }

    #[test]
    fn circle_value() {
        let c = Curve3d::Circle {
            pos: Ax2::new(Point3::origin(), dir(0.0, 0.0, 1.0), dir(1.0, 0.0, 0.0)),
            radius: 2.0,
        };
        assert_pnt_near(&c.value(0.0), &Point3::new(2.0, 0.0, 0.0), 1e-15);
        assert_pnt_near(&c.value(FRAC_PI_2), &Point3::new(0.0, 2.0, 0.0), 1e-14);
    }

    #[test]
    fn bspline_nonrational() {
        let c = Curve3d::BSpline {
            degree: 1,
            knots: vec![0.0, 0.0, 1.0, 1.0],
            poles: vec![Point3::origin(), Point3::new(10.0, 0.0, 0.0)],
            weights: None,
        };
        assert_pnt_near(&c.value(0.5), &Point3::new(5.0, 0.0, 0.0), 1e-15);
    }

    #[test]
    fn bspline_rational_circle() {
        let w = std::f64::consts::FRAC_1_SQRT_2;
        let c = Curve3d::BSpline {
            degree: 2,
            knots: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            poles: vec![
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(1.0, 1.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
            ],
            weights: Some(vec![1.0, w, 1.0]),
        };
        // All points should lie on the unit circle
        for i in 0..=20 {
            let u = i as f64 / 20.0;
            let p = c.value(u);
            let r = (p.x * p.x + p.y * p.y).sqrt();
            assert!((r - 1.0).abs() < 1e-14, "u={u}: r={r}");
        }
    }

    #[test]
    fn d1_line_tangent() {
        let c = Curve3d::Line {
            origin: Point3::origin(),
            dir: dir(1.0, 0.0, 0.0),
        };
        let (_, tangent) = c.d1(0.0);
        assert!((tangent - Vec3::new(1.0, 0.0, 0.0)).norm() < 1e-15);
    }

    #[test]
    fn d2_circle_curvature() {
        let c = Curve3d::Circle {
            pos: Ax2::new(Point3::origin(), dir(0.0, 0.0, 1.0), dir(1.0, 0.0, 0.0)),
            radius: 1.0,
        };
        let (_, d1, d2) = c.d2(0.5);
        let kappa = d1.cross(&d2).norm() / d1.norm().powi(3);
        assert!((kappa - 1.0).abs() < 1e-14);
    }

    #[test]
    fn curve_type_classification() {
        use crate::geom_types::CurveType;
        let c = Curve3d::Line {
            origin: Point3::origin(),
            dir: dir(1.0, 0.0, 0.0),
        };
        assert_eq!(c.curve_type(), CurveType::Line);
    }

    #[test]
    fn parameter_range_line() {
        let c = Curve3d::Line {
            origin: Point3::origin(),
            dir: dir(1.0, 0.0, 0.0),
        };
        let (a, b) = c.parameter_range();
        assert!(a < -1e90);
        assert!(b > 1e90);
    }

    #[test]
    fn parameter_range_circle() {
        let c = Curve3d::Circle {
            pos: Ax2::new(Point3::origin(), dir(0.0, 0.0, 1.0), dir(1.0, 0.0, 0.0)),
            radius: 1.0,
        };
        let (a, b) = c.parameter_range();
        assert!(a.abs() < 1e-15);
        assert!((b - 2.0 * std::f64::consts::PI).abs() < 1e-15);
    }

    #[test]
    fn parameter_range_bspline() {
        let c = Curve3d::BSpline {
            degree: 2,
            knots: vec![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
            poles: vec![
                Point3::origin(),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(2.0, 1.0, 0.0),
                Point3::new(3.0, 0.0, 0.0),
            ],
            weights: None,
        };
        let (a, b) = c.parameter_range();
        assert!(a.abs() < 1e-15);
        assert!((b - 1.0).abs() < 1e-15);
    }
}
