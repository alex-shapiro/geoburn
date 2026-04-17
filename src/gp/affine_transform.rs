use nalgebra::Matrix3;

use super::transform::{Transform, TransformKind};
use super::{Pnt, Vec3};

/// A general affine transformation in 3D space.
///
/// Unlike `Transform`, the 3×3 matrix can be non-orthogonal (shear,
/// non-uniform scale). Applies as: `p' = matrix * p + loc` when the form
/// is `Other`, or `p' = scale * matrix * p + loc` for forms inherited
/// from `Transform`.
#[derive(Debug, Clone)]
pub struct AffineTransform {
    matrix: Matrix3<f64>,
    loc: Vec3,
    kind: TransformKind,
    scale: f64,
}

impl AffineTransform {
    /// Identity.
    pub fn identity() -> Self {
        Self {
            matrix: Matrix3::identity(),
            loc: Vec3::zeros(),
            kind: TransformKind::Identity,
            scale: 1.0,
        }
    }

    /// Create from a rigid `Transform`.
    pub fn from_transform(t: &Transform) -> Self {
        Self {
            matrix: *t.homogeneous_vectorial_part(),
            loc: *t.translation_part(),
            kind: t.kind(),
            scale: t.scale_factor(),
        }
    }

    /// Create from an explicit matrix and translation.
    /// The form is set to `Other` (non-orthogonal).
    pub fn from_matrix(matrix: Matrix3<f64>, loc: Vec3) -> Self {
        Self {
            matrix,
            loc,
            kind: TransformKind::Compound,
            scale: 0.0, // scale is embedded in matrix for Other form
        }
    }

    /// Convert back to a rigid `Transform`.
    ///
    /// This assumes the matrix is orthogonal. For truly non-orthogonal
    /// transforms (created via `from_matrix`), the result will be incorrect.
    pub fn to_transform(&self) -> Transform {
        Transform::from_parts(self.kind, self.scale, self.matrix, self.loc)
    }

    pub fn kind(&self) -> TransformKind {
        self.kind
    }

    pub fn is_negative(&self) -> bool {
        self.matrix.determinant() < 0.0
    }

    pub fn is_singular(&self) -> bool {
        self.matrix.determinant().abs() <= super::RESOLUTION
    }

    pub fn vectorial_part(&self) -> &Matrix3<f64> {
        &self.matrix
    }

    pub fn translation_part(&self) -> &Vec3 {
        &self.loc
    }

    /// Apply to coordinates: `p' = matrix * p + loc` (or `scale * matrix * p + loc`).
    pub fn transform_coords(&self, coords: &Vec3) -> Vec3 {
        let mut result = self.matrix * coords;
        if self.kind != TransformKind::Compound && self.scale != 1.0 {
            result *= self.scale;
        }
        result + self.loc
    }

    pub fn transform_point(&self, p: &Pnt) -> Pnt {
        Pnt::from(self.transform_coords(&p.coords))
    }

    /// Compute the inverse. Panics if singular.
    pub fn inverse(&self) -> Self {
        if self.kind == TransformKind::Compound {
            // Non-orthogonal: full matrix inverse
            let inv = self
                .matrix
                .try_inverse()
                .expect("AffineTransform::inverse() - singular matrix");
            Self {
                matrix: inv,
                loc: -(inv * self.loc),
                kind: TransformKind::Compound,
                scale: 0.0,
            }
        } else {
            // Orthogonal: delegate to Transform
            let t = self.to_transform();
            Self::from_transform(&t.inverse())
        }
    }

    /// Compose: `self * other` (apply `other` first, then `self`).
    pub fn compose(&self, other: &AffineTransform) -> AffineTransform {
        if self.kind == TransformKind::Compound || other.kind == TransformKind::Compound {
            // At least one side is non-orthogonal: raw matrix math
            let new_loc = self.loc + self.matrix * other.loc;
            AffineTransform {
                kind: TransformKind::Compound,
                matrix: self.matrix * other.matrix,
                loc: new_loc,
                scale: 0.0,
            }
        } else {
            // Both sides are orthogonal: use Transform's fast paths
            let t1 = self.to_transform();
            let t2 = other.to_transform();
            Self::from_transform(&t1.compose(&t2))
        }
    }
}

impl std::ops::Mul for AffineTransform {
    type Output = AffineTransform;

    fn mul(self, rhs: AffineTransform) -> AffineTransform {
        self.compose(&rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Point3;

    fn pnt(x: f64, y: f64, z: f64) -> Pnt {
        Point3::new(x, y, z)
    }

    fn assert_point_near(a: &Pnt, b: &Pnt, tol: f64) {
        assert!(
            (a - b).norm() < tol,
            "points differ: {a:?} vs {b:?} (dist={})",
            (a - b).norm()
        );
    }

    #[test]
    fn identity() {
        let t = AffineTransform::identity();
        let p = pnt(1.0, 2.0, 3.0);
        assert_point_near(&t.transform_point(&p), &p, 1e-15);
    }

    #[test]
    fn non_uniform_scale() {
        let matrix = Matrix3::new(2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0);
        let t = AffineTransform::from_matrix(matrix, Vec3::zeros());
        assert_point_near(
            &t.transform_point(&pnt(1.0, 1.0, 1.0)),
            &pnt(2.0, 3.0, 4.0),
            1e-15,
        );
    }

    #[test]
    fn shear() {
        // Shear X by Y
        let matrix = Matrix3::new(1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        let t = AffineTransform::from_matrix(matrix, Vec3::zeros());
        assert_point_near(
            &t.transform_point(&pnt(0.0, 1.0, 0.0)),
            &pnt(1.0, 1.0, 0.0),
            1e-15,
        );
    }

    #[test]
    fn inverse_roundtrip() {
        let matrix = Matrix3::new(2.0, 1.0, 0.0, 0.0, 3.0, 0.5, 0.0, 0.0, 1.0);
        let loc = Vec3::new(1.0, 2.0, 3.0);
        let t = AffineTransform::from_matrix(matrix, loc);
        let p = pnt(4.0, 5.0, 6.0);
        let roundtrip = t.inverse().transform_point(&t.transform_point(&p));
        assert_point_near(&roundtrip, &p, 1e-12);
    }

    #[test]
    fn compose_matches_sequential() {
        let m1 = Matrix3::new(2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        let m2 = Matrix3::new(1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        let t1 = AffineTransform::from_matrix(m1, Vec3::new(1.0, 0.0, 0.0));
        let t2 = AffineTransform::from_matrix(m2, Vec3::new(0.0, 1.0, 0.0));
        let p = pnt(1.0, 2.0, 3.0);

        let sequential = t1.transform_point(&t2.transform_point(&p));
        let composed = t1.compose(&t2).transform_point(&p);
        assert_point_near(&sequential, &composed, 1e-14);
    }
}
