use nalgebra::{Matrix2, Rotation2};

use super::ax2d::Ax2d;
use super::transform::TransformKind;
use super::{Dir2d, Pnt2d, RESOLUTION, Vec2d};

use TransformKind::*;

/// A rigid transformation in 2D space with uniform scale.
///
/// Applies as: `p' = scale * matrix * p + loc`. The 2D analog of `Transform`.
#[derive(Debug, Clone, Copy)]
pub struct Transform2d {
    scale: f64,
    kind: TransformKind,
    matrix: Matrix2<f64>,
    loc: Vec2d,
}

impl Transform2d {
    pub fn identity() -> Self {
        Self {
            scale: 1.0,
            kind: Identity,
            matrix: Matrix2::identity(),
            loc: Vec2d::zeros(),
        }
    }

    pub fn translation(v: &Vec2d) -> Self {
        Self {
            scale: 1.0,
            kind: Translation,
            matrix: Matrix2::identity(),
            loc: *v,
        }
    }

    /// Rotation about a center point by `angle` radians.
    pub fn rotation(center: &Pnt2d, angle: f64) -> Self {
        let rot = Rotation2::new(angle);
        let matrix = *rot.matrix();
        let origin = center.coords;
        Self {
            scale: 1.0,
            kind: Rotation,
            matrix,
            loc: origin - matrix * origin,
        }
    }

    /// Uniform scale about a center point.
    pub fn scale(center: &Pnt2d, scale: f64) -> Self {
        assert!(
            scale.abs() > RESOLUTION,
            "Transform2d::scale() - scale factor is zero"
        );
        Self {
            scale,
            kind: Scale,
            matrix: Matrix2::identity(),
            loc: center.coords * (1.0 - scale),
        }
    }

    /// Point mirror (central symmetry): p' = 2*center - p.
    pub fn mirror_point(center: &Pnt2d) -> Self {
        Self {
            scale: -1.0,
            kind: PointMirror,
            matrix: Matrix2::identity(),
            loc: center.coords * 2.0,
        }
    }

    /// Line mirror (axial symmetry about a 2D axis).
    /// Points on the line are fixed; perpendicular components are negated.
    pub fn mirror_axis(ax: &Ax2d) -> Self {
        let vx = ax.dir.x;
        let vy = ax.dir.y;
        let x0 = ax.origin.x;
        let y0 = ax.origin.y;

        // Householder-like reflection matrix for 2D line mirror
        // matrix = I - 2*n*nT where n is the normal to the line
        let matrix = Matrix2::new(
            1.0 - 2.0 * vx * vx,
            -2.0 * vx * vy,
            -2.0 * vx * vy,
            1.0 - 2.0 * vy * vy,
        );

        let loc = Vec2d::new(
            -2.0 * ((vx * vx - 1.0) * x0 + vx * vy * y0),
            -2.0 * (vx * vy * x0 + (vy * vy - 1.0) * y0),
        );

        Self {
            scale: -1.0,
            kind: AxisMirror,
            matrix,
            loc,
        }
    }

    // -- Accessors --

    pub fn kind(&self) -> TransformKind {
        self.kind
    }

    pub fn scale_factor(&self) -> f64 {
        self.scale
    }

    pub fn is_negative(&self) -> bool {
        (self.matrix.determinant() * self.scale) < 0.0
    }

    pub fn homogeneous_vectorial_part(&self) -> &Matrix2<f64> {
        &self.matrix
    }

    pub fn vectorial_part(&self) -> Matrix2<f64> {
        if self.scale == 1.0 {
            self.matrix
        } else if self.kind == Scale || self.kind == PointMirror {
            let mut m = self.matrix;
            m[(0, 0)] *= self.scale;
            m[(1, 1)] *= self.scale;
            m
        } else {
            self.matrix * self.scale
        }
    }

    pub fn translation_part(&self) -> &Vec2d {
        &self.loc
    }

    pub fn rotation_angle(&self) -> f64 {
        self.matrix[(1, 0)].atan2(self.matrix[(0, 0)])
    }

    // -- Application --

    pub fn transform_point(&self, p: &Pnt2d) -> Pnt2d {
        match self.kind {
            Identity => *p,
            Translation => Pnt2d::from(p.coords + self.loc),
            Scale => Pnt2d::from(p.coords * self.scale + self.loc),
            PointMirror => Pnt2d::from(-p.coords + self.loc),
            _ => {
                let mut coords = self.matrix * p.coords;
                if self.scale != 1.0 {
                    coords *= self.scale;
                }
                coords += self.loc;
                Pnt2d::from(coords)
            }
        }
    }

    pub fn transform_vector(&self, v: &Vec2d) -> Vec2d {
        match self.kind {
            Identity | Translation => *v,
            PointMirror => -*v,
            Scale => *v * self.scale,
            _ => {
                let mut result = self.matrix * v;
                if self.scale != 1.0 {
                    result *= self.scale;
                }
                result
            }
        }
    }

    pub fn transform_dir(&self, d: &Dir2d) -> Dir2d {
        match self.kind {
            Identity | Translation => *d,
            PointMirror => -(*d),
            Scale => {
                if self.scale < 0.0 {
                    -(*d)
                } else {
                    *d
                }
            }
            _ => {
                let v = self.matrix * d.as_ref();
                let result = nalgebra::Unit::new_normalize(v);
                if self.scale < 0.0 { -result } else { result }
            }
        }
    }

    // -- Inverse --

    pub fn inverse(&self) -> Self {
        match self.kind {
            Identity => *self,
            Translation | PointMirror => Self {
                kind: self.kind,
                scale: self.scale,
                matrix: self.matrix,
                loc: -self.loc,
            },
            Scale => {
                assert!(
                    self.scale.abs() > RESOLUTION,
                    "Transform2d::inverse() - zero scale"
                );
                let inv_scale = 1.0 / self.scale;
                Self {
                    kind: Scale,
                    scale: inv_scale,
                    matrix: Matrix2::identity(),
                    loc: self.loc * (-inv_scale),
                }
            }
            _ => {
                assert!(
                    self.scale.abs() > RESOLUTION,
                    "Transform2d::inverse() - zero scale"
                );
                let inv_scale = 1.0 / self.scale;
                let inv_matrix = self.matrix.transpose();
                Self {
                    kind: self.kind,
                    scale: inv_scale,
                    matrix: inv_matrix,
                    loc: inv_matrix * self.loc * (-inv_scale),
                }
            }
        }
    }

    // -- Composition --

    /// Compose: `self * other` (apply `other` first, then `self`).
    #[allow(clippy::cognitive_complexity)]
    pub fn compose(&self, other: &Transform2d) -> Transform2d {
        match (self.kind, other.kind) {
            (_, Identity) => *self,
            (Identity, _) => *other,

            (Rotation, Rotation) => {
                let new_loc = if other.loc.norm_squared() > 0.0 {
                    self.loc + self.matrix * other.loc
                } else {
                    self.loc
                };
                Transform2d {
                    kind: Rotation,
                    scale: 1.0,
                    matrix: self.matrix * other.matrix,
                    loc: new_loc,
                }
            }

            (Translation, Translation) => Transform2d {
                kind: Translation,
                scale: 1.0,
                matrix: Matrix2::identity(),
                loc: self.loc + other.loc,
            },

            (Scale, Scale) => Transform2d {
                kind: Scale,
                scale: self.scale * other.scale,
                matrix: Matrix2::identity(),
                loc: self.loc + other.loc * self.scale,
            },

            (PointMirror, PointMirror) => Transform2d {
                kind: Translation,
                scale: 1.0,
                matrix: Matrix2::identity(),
                loc: self.loc - other.loc,
            },

            (AxisMirror, AxisMirror) => {
                let mut tloc = self.matrix * other.loc;
                tloc *= self.scale;
                let new_scale = self.scale * other.scale;
                Transform2d {
                    kind: Rotation,
                    scale: new_scale,
                    matrix: self.matrix * other.matrix,
                    loc: self.loc + tloc,
                }
            }

            (Compound | Rotation | AxisMirror, Translation) => {
                let mut tloc = self.matrix * other.loc;
                if self.scale != 1.0 {
                    tloc *= self.scale;
                }
                Transform2d {
                    kind: self.kind,
                    scale: self.scale,
                    matrix: self.matrix,
                    loc: self.loc + tloc,
                }
            }

            (Scale | PointMirror, Translation) => Transform2d {
                kind: self.kind,
                scale: self.scale,
                matrix: self.matrix,
                loc: self.loc + other.loc * self.scale,
            },

            (Translation, Compound | Rotation | AxisMirror) => Transform2d {
                kind: Compound,
                scale: other.scale,
                matrix: other.matrix,
                loc: self.loc + other.loc,
            },

            (Translation, Scale | PointMirror) => Transform2d {
                kind: other.kind,
                scale: other.scale,
                matrix: other.matrix,
                loc: self.loc + other.loc,
            },

            (PointMirror | Scale, PointMirror | Scale) => Transform2d {
                kind: Compound,
                scale: self.scale * other.scale,
                matrix: Matrix2::identity(),
                loc: self.loc + other.loc * self.scale,
            },

            (Compound | Rotation | AxisMirror, Scale | PointMirror) => {
                let mut tloc = self.matrix * other.loc;
                let new_scale = if self.scale == 1.0 {
                    other.scale
                } else {
                    tloc *= self.scale;
                    self.scale * other.scale
                };
                Transform2d {
                    kind: Compound,
                    scale: new_scale,
                    matrix: self.matrix,
                    loc: self.loc + tloc,
                }
            }

            (Scale | PointMirror, Compound | Rotation | AxisMirror) => Transform2d {
                kind: Compound,
                scale: self.scale * other.scale,
                matrix: other.matrix,
                loc: self.loc + other.loc * self.scale,
            },

            _ => {
                let mut tloc = self.matrix * other.loc;
                let new_scale = if self.scale != 1.0 {
                    tloc *= self.scale;
                    self.scale * other.scale
                } else {
                    other.scale
                };
                Transform2d {
                    kind: Compound,
                    scale: new_scale,
                    matrix: self.matrix * other.matrix,
                    loc: self.loc + tloc,
                }
            }
        }
    }

    /// Construct from raw parts (used by AffineTransform2d).
    pub(crate) fn from_parts(
        kind: TransformKind,
        scale: f64,
        matrix: Matrix2<f64>,
        loc: Vec2d,
    ) -> Self {
        Self {
            scale,
            kind,
            matrix,
            loc,
        }
    }
}

impl std::ops::Mul for Transform2d {
    type Output = Transform2d;

    fn mul(self, rhs: Transform2d) -> Transform2d {
        self.compose(&rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Point2, Unit, Vector2};
    use std::f64::consts::{FRAC_PI_2, PI};

    fn assert_point_near(a: &Pnt2d, b: &Pnt2d, tol: f64) {
        assert!(
            (a - b).norm() < tol,
            "points differ: {a:?} vs {b:?} (dist={})",
            (a - b).norm()
        );
    }

    fn dir(x: f64, y: f64) -> Dir2d {
        Unit::new_normalize(Vector2::new(x, y))
    }

    fn pnt(x: f64, y: f64) -> Pnt2d {
        Point2::new(x, y)
    }

    #[test]
    fn identity_preserves_point() {
        let t = Transform2d::identity();
        let p = pnt(1.0, 2.0);
        assert_point_near(&t.transform_point(&p), &p, 1e-15);
    }

    #[test]
    fn translation() {
        let t = Transform2d::translation(&Vector2::new(3.0, 4.0));
        assert_point_near(&t.transform_point(&pnt(0.0, 0.0)), &pnt(3.0, 4.0), 1e-15);
    }

    #[test]
    fn rotation_90() {
        let t = Transform2d::rotation(&Point2::origin(), FRAC_PI_2);
        assert_point_near(&t.transform_point(&pnt(1.0, 0.0)), &pnt(0.0, 1.0), 1e-14);
    }

    #[test]
    fn rotation_around_offset() {
        let t = Transform2d::rotation(&pnt(1.0, 0.0), PI);
        assert_point_near(&t.transform_point(&pnt(0.0, 0.0)), &pnt(2.0, 0.0), 1e-14);
    }

    #[test]
    fn scale_from_center() {
        let t = Transform2d::scale(&pnt(1.0, 0.0), 3.0);
        assert_point_near(&t.transform_point(&pnt(0.0, 0.0)), &pnt(-2.0, 0.0), 1e-15);
    }

    #[test]
    fn mirror_point() {
        let t = Transform2d::mirror_point(&Point2::origin());
        assert_point_near(&t.transform_point(&pnt(1.0, 2.0)), &pnt(-1.0, -2.0), 1e-15);
    }

    #[test]
    fn mirror_axis_x() {
        let ax = Ax2d::new(Point2::origin(), dir(1.0, 0.0));
        let t = Transform2d::mirror_axis(&ax);
        // Reflect across X axis: (1, 2) → (1, -2)
        assert_point_near(&t.transform_point(&pnt(1.0, 2.0)), &pnt(1.0, -2.0), 1e-14);
    }

    #[test]
    fn mirror_axis_preserves_points_on_line() {
        let ax = Ax2d::new(pnt(1.0, 1.0), dir(1.0, 1.0));
        let t = Transform2d::mirror_axis(&ax);
        let p = pnt(3.0, 3.0); // on the line y=x
        assert_point_near(&t.transform_point(&p), &p, 1e-13);
    }

    #[test]
    fn inverse_roundtrip() {
        let t = Transform2d::rotation(&pnt(1.0, 2.0), 1.23);
        let p = pnt(4.0, 5.0);
        let roundtrip = t.inverse().transform_point(&t.transform_point(&p));
        assert_point_near(&roundtrip, &p, 1e-13);
    }

    #[test]
    fn compose_matches_sequential() {
        let t1 = Transform2d::rotation(&pnt(1.0, 0.0), 0.7);
        let t2 = Transform2d::scale(&pnt(0.0, 1.0), 2.5);
        let p = pnt(3.0, -1.0);

        let sequential = t2.transform_point(&t1.transform_point(&p));
        let composed = t2.compose(&t1).transform_point(&p);
        assert_point_near(&sequential, &composed, 1e-13);
    }

    #[test]
    fn compose_translations() {
        let t1 = Transform2d::translation(&Vector2::new(1.0, 0.0));
        let t2 = Transform2d::translation(&Vector2::new(0.0, 2.0));
        let t = t1.compose(&t2);
        assert_eq!(t.kind(), Translation);
        assert_point_near(&t.transform_point(&Point2::origin()), &pnt(1.0, 2.0), 1e-15);
    }
}
