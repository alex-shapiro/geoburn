//! 2D curve types with a uniform evaluation interface.

use crate::bspline;
use crate::gp::ax22d::Ax22d;
use crate::gp::{Pnt2d, Vec2d};

/// A 2D curve (used for trim curves in UV space).
#[derive(Debug, Clone)]
pub enum Curve2d {
    Line {
        origin: Pnt2d,
        dir: crate::gp::Dir2d,
    },
    Circle {
        pos: Ax22d,
        radius: f64,
    },
    Ellipse {
        pos: Ax22d,
        major_radius: f64,
        minor_radius: f64,
    },
    BSpline {
        degree: usize,
        knots: Vec<f64>,
        poles: Vec<Pnt2d>,
        weights: Option<Vec<f64>>,
    },
}

impl Curve2d {
    /// Evaluate the curve at parameter `u`.
    pub fn value(&self, u: f64) -> Pnt2d {
        match self {
            Curve2d::Line { origin, dir } => Pnt2d::from(origin.coords + dir.as_ref() * u),
            Curve2d::Circle { pos, radius } => {
                let (cu, su) = (u.cos(), u.sin());
                Pnt2d::from(
                    pos.origin().coords
                        + pos.x_direction().as_ref() * (*radius * cu)
                        + pos.y_direction().as_ref() * (*radius * su),
                )
            }
            Curve2d::Ellipse {
                pos,
                major_radius,
                minor_radius,
            } => {
                let (cu, su) = (u.cos(), u.sin());
                Pnt2d::from(
                    pos.origin().coords
                        + pos.x_direction().as_ref() * (*major_radius * cu)
                        + pos.y_direction().as_ref() * (*minor_radius * su),
                )
            }
            Curve2d::BSpline {
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

    /// Evaluate the curve and its first derivative.
    pub fn d1(&self, u: f64) -> (Pnt2d, Vec2d) {
        match self {
            Curve2d::Line { origin, dir } => {
                let p = Pnt2d::from(origin.coords + dir.as_ref() * u);
                (p, **dir)
            }
            Curve2d::Circle { pos, radius } => {
                let (cu, su) = (u.cos(), u.sin());
                let x: &Vec2d = pos.x_direction().as_ref();
                let y: &Vec2d = pos.y_direction().as_ref();
                let r = *radius;
                let p = Pnt2d::from(pos.origin().coords + x * (r * cu) + y * (r * su));
                let v = x * (-r * su) + y * (r * cu);
                (p, v)
            }
            Curve2d::Ellipse {
                pos,
                major_radius,
                minor_radius,
            } => {
                let (cu, su) = (u.cos(), u.sin());
                let x: &Vec2d = pos.x_direction().as_ref();
                let y: &Vec2d = pos.y_direction().as_ref();
                let a = *major_radius;
                let b = *minor_radius;
                let p = Pnt2d::from(pos.origin().coords + x * (a * cu) + y * (b * su));
                let v = x * (-a * su) + y * (b * cu);
                (p, v)
            }
            Curve2d::BSpline {
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

    pub fn parameter_range(&self) -> (f64, f64) {
        use crate::gp::precision;
        match self {
            Curve2d::Line { .. } => (-precision::INFINITE, precision::INFINITE),
            Curve2d::Circle { .. } | Curve2d::Ellipse { .. } => (0.0, 2.0 * std::f64::consts::PI),
            Curve2d::BSpline { degree, knots, .. } => {
                (knots[*degree], knots[knots.len() - degree - 1])
            }
        }
    }

    pub fn curve_type(&self) -> crate::geom_types::CurveType {
        use crate::geom_types::CurveType;
        match self {
            Curve2d::Line { .. } => CurveType::Line,
            Curve2d::Circle { .. } => CurveType::Circle,
            Curve2d::Ellipse { .. } => CurveType::Ellipse,
            Curve2d::BSpline { .. } => CurveType::BSplineCurve,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Point2, Unit, Vector2};

    fn dir2d(x: f64, y: f64) -> crate::gp::Dir2d {
        Unit::new_normalize(Vector2::new(x, y))
    }

    #[test]
    fn line_2d() {
        let c = Curve2d::Line {
            origin: Point2::origin(),
            dir: dir2d(1.0, 0.0),
        };
        let p = c.value(5.0);
        assert!((p.x - 5.0).abs() < 1e-15);
        assert!(p.y.abs() < 1e-15);
    }

    #[test]
    fn bspline_2d_linear() {
        let c = Curve2d::BSpline {
            degree: 1,
            knots: vec![0.0, 0.0, 1.0, 1.0],
            poles: vec![Point2::new(0.0, 0.0), Point2::new(1.0, 1.0)],
            weights: None,
        };
        let p = c.value(0.5);
        assert!((p.x - 0.5).abs() < 1e-15);
        assert!((p.y - 0.5).abs() < 1e-15);
    }

    #[test]
    fn circle_2d() {
        let c = Curve2d::Circle {
            pos: Ax22d::new(Point2::origin(), dir2d(1.0, 0.0), true),
            radius: 3.0,
        };
        let p = c.value(0.0);
        assert!((p.x - 3.0).abs() < 1e-15);
        assert!(p.y.abs() < 1e-15);
    }
}
