use nalgebra::{Matrix3, Rotation3};

use super::ax1::Ax1;
use super::ax2::Ax2;
use super::{Dir, Pnt, RESOLUTION, Vec3};
use TransformKind::*;

/// Classifies a transform for fast-path dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformKind {
    Identity,
    Rotation,
    Translation,
    PointMirror,
    AxisMirror,
    PlaneMirror,
    Scale,
    Compound,
}

/// A rigid transformation in 3D space with uniform scale.
///
/// Applies as: `p' = scale * matrix * p + loc`
///
/// The matrix is always orthogonal (det = ±1). The scale factor is stored
/// separately.
#[derive(Debug, Clone)]
pub struct Transform {
    scale: f64,
    kind: TransformKind,
    matrix: Matrix3<f64>,
    loc: Vec3,
}

impl Transform {
    pub fn identity() -> Self {
        Self {
            scale: 1.0,
            kind: Identity,
            matrix: Matrix3::identity(),
            loc: Vec3::zeros(),
        }
    }

    pub fn translation(v: &Vec3) -> Self {
        Self {
            scale: 1.0,
            kind: Translation,
            matrix: Matrix3::identity(),
            loc: *v,
        }
    }

    /// Rotation about an axis by `angle` radians.
    pub fn rotation(ax: &Ax1, angle: f64) -> Self {
        let rot = Rotation3::from_axis_angle(&ax.dir, angle);
        let matrix = *rot.matrix();
        let origin = ax.origin.coords;
        Self {
            scale: 1.0,
            kind: Rotation,
            matrix,
            loc: origin - matrix * origin,
        }
    }

    /// Uniform scale about a center point.
    ///
    /// Panics if `scale` is zero.
    pub fn scale(center: &Pnt, scale: f64) -> Self {
        assert!(
            scale.abs() > RESOLUTION,
            "Transform::scale() - scale factor is zero"
        );
        Self {
            scale,
            kind: Scale,
            matrix: Matrix3::identity(),
            loc: center.coords * (1.0 - scale),
        }
    }

    /// Point mirror (central symmetry): p' = 2*center - p.
    pub fn mirror_point(center: &Pnt) -> Self {
        Self {
            scale: -1.0,
            kind: PointMirror,
            matrix: Matrix3::identity(),
            loc: center.coords * 2.0,
        }
    }

    /// Axis mirror (rotational symmetry about a line).
    /// Points on the axis are fixed; perpendicular components are negated.
    pub fn mirror_axis(ax: &Ax1) -> Self {
        let n: &Vec3 = ax.dir.as_ref();
        let proj = n * n.transpose();
        // I - 2*n*nT (Householder)
        let householder = Matrix3::identity() - proj * 2.0;
        let origin = ax.origin.coords;
        let loc = householder * origin + origin;
        Self {
            scale: 1.0,
            kind: AxisMirror,
            // Store negated: 2*n*nT - I (scale=1, so effective = 2nnT - I)
            matrix: -householder,
            loc,
        }
    }

    /// Plane mirror (bilateral symmetry).
    /// The plane is defined by Ax2's origin and Z direction (normal).
    pub fn mirror_plane(ax: &Ax2) -> Self {
        let n: &Vec3 = ax.z_direction().as_ref();
        let proj = n * n.transpose();
        // 2*n*nT - I (stored matrix; scale=-1, so effective = I - 2nnT)
        let matrix = proj * 2.0 - Matrix3::identity();
        let origin = ax.origin().coords;
        let loc = matrix * origin + origin;
        Self {
            scale: -1.0,
            kind: PlaneMirror,
            matrix,
            loc,
        }
    }

    /// Construct from raw parts. Used internally (e.g. by AffineTransform).
    pub(crate) fn from_parts(
        kind: TransformKind,
        scale: f64,
        matrix: Matrix3<f64>,
        loc: Vec3,
    ) -> Self {
        Self {
            scale,
            kind,
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
        self.scale < 0.0
    }

    /// The 3×3 orthogonal matrix (without scale).
    pub fn homogeneous_vectorial_part(&self) -> &Matrix3<f64> {
        &self.matrix
    }

    /// The 3×3 matrix including scale: `scale * matrix`.
    pub fn vectorial_part(&self) -> Matrix3<f64> {
        if self.scale == 1.0 {
            self.matrix
        } else if self.kind == Scale || self.kind == PointMirror {
            let mut m = self.matrix;
            m[(0, 0)] *= self.scale;
            m[(1, 1)] *= self.scale;
            m[(2, 2)] *= self.scale;
            m
        } else {
            self.matrix * self.scale
        }
    }

    pub fn translation_part(&self) -> &Vec3 {
        &self.loc
    }

    // -- Application --

    /// Transform a point: `p' = scale * matrix * p + loc`.
    pub fn transform_point(&self, p: &Pnt) -> Pnt {
        match self.kind {
            Identity => *p,
            Translation => Pnt::from(p.coords + self.loc),
            Scale => Pnt::from(p.coords * self.scale + self.loc),
            PointMirror => Pnt::from(-p.coords + self.loc),
            _ => {
                let mut coords = self.matrix * p.coords;
                if self.scale != 1.0 {
                    coords *= self.scale;
                }
                coords += self.loc;
                Pnt::from(coords)
            }
        }
    }

    /// Transform a free vector (translation is not applied).
    pub fn transform_vector(&self, v: &Vec3) -> Vec3 {
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

    /// Transform a unit direction. The result is re-normalized.
    /// Translation has no effect; negative scale reverses direction.
    pub fn transform_dir(&self, d: &Dir) -> Dir {
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

    /// Compute the inverse transformation.
    ///
    /// Panics if the scale factor is zero.
    pub fn inverse(&self) -> Self {
        match self.kind {
            Identity => self.clone(),
            Translation | PointMirror => Self {
                kind: self.kind,
                scale: self.scale,
                matrix: self.matrix,
                loc: -self.loc,
            },
            Scale => {
                assert!(
                    self.scale.abs() > RESOLUTION,
                    "Transform::inverse() - zero scale"
                );
                let inv_scale = 1.0 / self.scale;
                Self {
                    kind: Scale,
                    scale: inv_scale,
                    matrix: Matrix3::identity(),
                    loc: self.loc * (-inv_scale),
                }
            }
            _ => {
                assert!(
                    self.scale.abs() > RESOLUTION,
                    "Transform::inverse() - zero scale"
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

    /// Compose: `self * other` (apply `other` first, then `self`)
    #[allow(clippy::cognitive_complexity)]
    pub fn compose(&self, other: &Transform) -> Transform {
        match (self.kind, other.kind) {
            // Trivial: either side is identity
            (_, Identity) => self.clone(),
            (Identity, _) => other.clone(),

            // Rotation * Rotation (both scale=1)
            (Rotation, Rotation) => {
                let new_loc = if other.loc.norm_squared() > 0.0 {
                    self.loc + self.matrix * other.loc
                } else {
                    self.loc
                };
                Transform {
                    kind: Rotation,
                    scale: 1.0,
                    matrix: self.matrix * other.matrix,
                    loc: new_loc,
                }
            }

            // Translation * Translation
            (Translation, Translation) => Transform {
                kind: Translation,
                scale: 1.0,
                matrix: Matrix3::identity(),
                loc: self.loc + other.loc,
            },

            // Scale * Scale (both have identity matrices)
            (Scale, Scale) => Transform {
                kind: Scale,
                scale: self.scale * other.scale,
                matrix: Matrix3::identity(),
                loc: self.loc + other.loc * self.scale,
            },

            // PointMirror * PointMirror → Translation
            (PointMirror, PointMirror) => Transform {
                kind: Translation,
                scale: 1.0,
                matrix: Matrix3::identity(),
                loc: self.loc - other.loc,
            },

            // AxisMirror * AxisMirror → Rotation
            (AxisMirror, AxisMirror) => Transform {
                kind: Rotation,
                scale: 1.0,
                matrix: self.matrix * other.matrix,
                loc: self.loc + self.matrix * other.loc,
            },

            // Self has matrix, other is pure translation
            (Compound | Rotation | AxisMirror | PlaneMirror, Translation) => {
                let mut tloc = self.matrix * other.loc;
                if self.scale != 1.0 {
                    tloc *= self.scale;
                }
                Transform {
                    kind: self.kind,
                    scale: self.scale,
                    matrix: self.matrix,
                    loc: self.loc + tloc,
                }
            }

            // Self is scale/mirror, other is pure translation
            (Scale | PointMirror, Translation) => Transform {
                kind: self.kind,
                scale: self.scale,
                matrix: self.matrix,
                loc: self.loc + other.loc * self.scale,
            },

            // Self is pure translation, other has matrix
            (Translation, Compound | Rotation | AxisMirror | PlaneMirror) => Transform {
                kind: Compound,
                scale: other.scale,
                matrix: other.matrix,
                loc: self.loc + other.loc,
            },

            // Self is pure translation, other is scale/mirror
            (Translation, Scale | PointMirror) => Transform {
                kind: other.kind,
                scale: other.scale,
                matrix: other.matrix,
                loc: self.loc + other.loc,
            },

            // Cross: (PointMirror|Scale) * (PointMirror|Scale)
            (PointMirror | Scale, PointMirror | Scale) => Transform {
                kind: Compound,
                scale: self.scale * other.scale,
                matrix: Matrix3::identity(),
                loc: self.loc + other.loc * self.scale,
            },

            // Self has matrix, other is scale/mirror (identity matrix)
            (Compound | Rotation | AxisMirror | PlaneMirror, Scale | PointMirror) => {
                let mut tloc = self.matrix * other.loc;
                let new_scale = if self.scale == 1.0 {
                    other.scale
                } else {
                    tloc *= self.scale;
                    self.scale * other.scale
                };
                Transform {
                    kind: Compound,
                    scale: new_scale,
                    matrix: self.matrix,
                    loc: self.loc + tloc,
                }
            }

            // Self is scale/mirror, other has matrix
            (Scale | PointMirror, Compound | Rotation | AxisMirror | PlaneMirror) => Transform {
                kind: Compound,
                scale: self.scale * other.scale,
                matrix: other.matrix,
                loc: self.loc + other.loc * self.scale,
            },

            // General case
            _ => {
                let mut tloc = self.matrix * other.loc;
                let new_scale = if self.scale != 1.0 {
                    tloc *= self.scale;
                    self.scale * other.scale
                } else {
                    other.scale
                };
                Transform {
                    kind: Compound,
                    scale: new_scale,
                    matrix: self.matrix * other.matrix,
                    loc: self.loc + tloc,
                }
            }
        }
    }
}

impl std::ops::Mul for Transform {
    type Output = Transform;

    fn mul(self, rhs: Transform) -> Transform {
        self.compose(&rhs)
    }
}

impl std::ops::Mul<&Transform> for &Transform {
    type Output = Transform;

    fn mul(self, rhs: &Transform) -> Transform {
        self.compose(rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Point3, Unit, Vector3};
    use std::f64::consts::{FRAC_PI_2, PI};

    fn assert_point_near(a: &Pnt, b: &Pnt, tol: f64) {
        assert!(
            (a - b).norm() < tol,
            "points differ: {a:?} vs {b:?} (dist={})",
            (a - b).norm()
        );
    }

    fn dir(x: f64, y: f64, z: f64) -> Dir {
        Unit::new_normalize(Vector3::new(x, y, z))
    }

    fn pnt(x: f64, y: f64, z: f64) -> Pnt {
        Point3::new(x, y, z)
    }

    // -- Identity --

    #[test]
    fn identity_preserves_point() {
        let t = Transform::identity();
        let p = pnt(1.0, 2.0, 3.0);
        assert_point_near(&t.transform_point(&p), &p, 1e-15);
    }

    // -- Translation --

    #[test]
    fn translation_shifts_point() {
        let t = Transform::translation(&Vector3::new(1.0, 0.0, 0.0));
        assert_point_near(
            &t.transform_point(&pnt(0.0, 0.0, 0.0)),
            &pnt(1.0, 0.0, 0.0),
            1e-15,
        );
    }

    #[test]
    fn translation_does_not_affect_vector() {
        let t = Transform::translation(&Vector3::new(10.0, 20.0, 30.0));
        let v = Vector3::new(1.0, 2.0, 3.0);
        assert!((t.transform_vector(&v) - v).norm() < 1e-15);
    }

    // -- Rotation --

    #[test]
    fn rotation_90_around_z() {
        let ax = Ax1::new(Point3::origin(), dir(0.0, 0.0, 1.0));
        let t = Transform::rotation(&ax, FRAC_PI_2);
        let p = pnt(1.0, 0.0, 0.0);
        assert_point_near(&t.transform_point(&p), &pnt(0.0, 1.0, 0.0), 1e-14);
    }

    #[test]
    fn rotation_around_offset_axis() {
        // Rotate 180° around Z axis at (1,0,0): (0,0,0) → (2,0,0)
        let ax = Ax1::new(pnt(1.0, 0.0, 0.0), dir(0.0, 0.0, 1.0));
        let t = Transform::rotation(&ax, PI);
        assert_point_near(
            &t.transform_point(&pnt(0.0, 0.0, 0.0)),
            &pnt(2.0, 0.0, 0.0),
            1e-14,
        );
    }

    // -- Scale --

    #[test]
    fn scale_from_origin() {
        let t = Transform::scale(&Point3::origin(), 2.0);
        assert_point_near(
            &t.transform_point(&pnt(1.0, 2.0, 3.0)),
            &pnt(2.0, 4.0, 6.0),
            1e-15,
        );
    }

    #[test]
    fn scale_from_center() {
        let t = Transform::scale(&pnt(1.0, 0.0, 0.0), 3.0);
        // (0,0,0) is 1 unit from center, scaled to 3 units away: (-2,0,0)
        assert_point_near(
            &t.transform_point(&pnt(0.0, 0.0, 0.0)),
            &pnt(-2.0, 0.0, 0.0),
            1e-15,
        );
    }

    // -- Point mirror --

    #[test]
    fn mirror_point_through_origin() {
        let t = Transform::mirror_point(&Point3::origin());
        assert_point_near(
            &t.transform_point(&pnt(1.0, 2.0, 3.0)),
            &pnt(-1.0, -2.0, -3.0),
            1e-15,
        );
    }

    #[test]
    fn mirror_point_center_is_fixed() {
        let center = pnt(5.0, 5.0, 5.0);
        let t = Transform::mirror_point(&center);
        assert_point_near(&t.transform_point(&center), &center, 1e-15);
    }

    // -- Axis mirror --

    #[test]
    fn mirror_axis_preserves_points_on_axis() {
        let ax = Ax1::new(Point3::origin(), dir(0.0, 0.0, 1.0));
        let t = Transform::mirror_axis(&ax);
        let p = pnt(0.0, 0.0, 7.0);
        assert_point_near(&t.transform_point(&p), &p, 1e-14);
    }

    #[test]
    fn mirror_axis_negates_perpendicular() {
        let ax = Ax1::new(Point3::origin(), dir(0.0, 0.0, 1.0));
        let t = Transform::mirror_axis(&ax);
        assert_point_near(
            &t.transform_point(&pnt(1.0, 0.0, 0.0)),
            &pnt(-1.0, 0.0, 0.0),
            1e-14,
        );
    }

    // -- Plane mirror --

    #[test]
    fn mirror_plane_reflects_across_xy() {
        let ax = Ax2::new(Point3::origin(), dir(0.0, 0.0, 1.0), dir(1.0, 0.0, 0.0));
        let t = Transform::mirror_plane(&ax);
        assert_point_near(
            &t.transform_point(&pnt(1.0, 2.0, 3.0)),
            &pnt(1.0, 2.0, -3.0),
            1e-14,
        );
    }

    #[test]
    fn mirror_plane_preserves_points_in_plane() {
        let ax = Ax2::new(Point3::origin(), dir(0.0, 0.0, 1.0), dir(1.0, 0.0, 0.0));
        let t = Transform::mirror_plane(&ax);
        let p = pnt(5.0, 3.0, 0.0);
        assert_point_near(&t.transform_point(&p), &p, 1e-14);
    }

    // -- Inverse --

    #[test]
    fn inverse_of_rotation() {
        let ax = Ax1::new(pnt(1.0, 2.0, 3.0), dir(1.0, 1.0, 1.0));
        let t = Transform::rotation(&ax, 1.23);
        let p = pnt(4.0, 5.0, 6.0);
        let roundtrip = t.inverse().transform_point(&t.transform_point(&p));
        assert_point_near(&roundtrip, &p, 1e-13);
    }

    #[test]
    fn inverse_of_scale() {
        let t = Transform::scale(&pnt(1.0, 0.0, 0.0), 3.0);
        let p = pnt(4.0, 5.0, 6.0);
        let roundtrip = t.inverse().transform_point(&t.transform_point(&p));
        assert_point_near(&roundtrip, &p, 1e-13);
    }

    #[test]
    fn inverse_of_mirror_is_self() {
        let ax = Ax1::new(Point3::origin(), dir(0.0, 0.0, 1.0));
        let t = Transform::mirror_axis(&ax);
        let p = pnt(1.0, 2.0, 3.0);
        let roundtrip = t.compose(&t).transform_point(&p);
        assert_point_near(&roundtrip, &p, 1e-13);
    }

    // -- Composition --

    #[test]
    fn compose_translations() {
        let t1 = Transform::translation(&Vector3::new(1.0, 0.0, 0.0));
        let t2 = Transform::translation(&Vector3::new(0.0, 2.0, 0.0));
        let t = t1.compose(&t2);
        assert_eq!(t.kind(), Translation);
        assert_point_near(
            &t.transform_point(&Point3::origin()),
            &pnt(1.0, 2.0, 0.0),
            1e-15,
        );
    }

    #[test]
    fn compose_rotations() {
        let ax = Ax1::new(Point3::origin(), dir(0.0, 0.0, 1.0));
        let t1 = Transform::rotation(&ax, FRAC_PI_2);
        let t2 = Transform::rotation(&ax, FRAC_PI_2);
        let t = t1.compose(&t2);
        assert_eq!(t.kind(), Rotation);
        // Two 90° rotations = 180°
        assert_point_near(
            &t.transform_point(&pnt(1.0, 0.0, 0.0)),
            &pnt(-1.0, 0.0, 0.0),
            1e-14,
        );
    }

    #[test]
    fn compose_matches_sequential_application() {
        // General test: T2(T1(p)) == (T2 * T1)(p)
        let ax = Ax1::new(pnt(1.0, 0.0, 0.0), dir(0.0, 1.0, 0.0));
        let t1 = Transform::rotation(&ax, 0.7);
        let t2 = Transform::scale(&pnt(0.0, 1.0, 0.0), 2.5);
        let p = pnt(3.0, -1.0, 4.0);

        let sequential = t2.transform_point(&t1.transform_point(&p));
        let composed = t2.compose(&t1).transform_point(&p);
        assert_point_near(&sequential, &composed, 1e-13);
    }

    #[test]
    fn compose_point_mirrors_gives_translation() {
        let t1 = Transform::mirror_point(&pnt(0.0, 0.0, 0.0));
        let t2 = Transform::mirror_point(&pnt(3.0, 0.0, 0.0));
        let t = t2.compose(&t1);
        assert_eq!(t.kind(), Translation);
    }

    #[test]
    fn operator_mul_equals_compose() {
        let t1 = Transform::translation(&Vector3::new(1.0, 2.0, 3.0));
        let t2 = Transform::scale(&Point3::origin(), 2.0);
        let p = pnt(1.0, 1.0, 1.0);
        let a = t1.compose(&t2).transform_point(&p);
        let b = (t1 * t2).transform_point(&p);
        assert_point_near(&a, &b, 1e-15);
    }

    // -- Dir transform --

    #[test]
    fn transform_dir_rotation() {
        let ax = Ax1::new(Point3::origin(), dir(0.0, 0.0, 1.0));
        let t = Transform::rotation(&ax, FRAC_PI_2);
        let d = dir(1.0, 0.0, 0.0);
        let result = t.transform_dir(&d);
        assert!((result.as_ref() - Vector3::new(0.0, 1.0, 0.0)).norm() < 1e-14);
    }

    #[test]
    fn transform_dir_negative_scale_reverses() {
        let t = Transform::mirror_point(&Point3::origin());
        let d = dir(1.0, 0.0, 0.0);
        let result = t.transform_dir(&d);
        assert!((result.as_ref() - Vector3::new(-1.0, 0.0, 0.0)).norm() < 1e-15);
    }
}
