use nalgebra::Matrix2;

use super::transform::TransformKind;
use super::transform2d::Transform2d;
use super::{Pnt2d, Vec2d};

/// A general affine transformation in 2D space.
///
/// Unlike `Transform2d`, the 2×2 matrix can be non-orthogonal (shear, non-uniform scale).
#[derive(Debug, Clone, Copy)]
pub struct AffineTransform2d {
    matrix: Matrix2<f64>,
    loc: Vec2d,
    kind: TransformKind,
    scale: f64,
}

impl AffineTransform2d {
    pub fn identity() -> Self {
        Self {
            matrix: Matrix2::identity(),
            loc: Vec2d::zeros(),
            kind: TransformKind::Identity,
            scale: 1.0,
        }
    }

    pub fn from_transform(t: &Transform2d) -> Self {
        Self {
            matrix: *t.homogeneous_vectorial_part(),
            loc: *t.translation_part(),
            kind: t.kind(),
            scale: t.scale_factor(),
        }
    }

    pub fn from_matrix(matrix: Matrix2<f64>, loc: Vec2d) -> Self {
        Self {
            matrix,
            loc,
            kind: TransformKind::Compound,
            scale: 0.0,
        }
    }

    pub fn to_transform(&self) -> Transform2d {
        Transform2d::from_parts(self.kind, self.scale, self.matrix, self.loc)
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

    pub fn vectorial_part(&self) -> &Matrix2<f64> {
        &self.matrix
    }

    pub fn translation_part(&self) -> &Vec2d {
        &self.loc
    }

    pub fn transform_coords(&self, coords: &Vec2d) -> Vec2d {
        let mut result = self.matrix * coords;
        if self.kind != TransformKind::Compound && self.scale != 1.0 {
            result *= self.scale;
        }
        result + self.loc
    }

    pub fn transform_point(&self, p: &Pnt2d) -> Pnt2d {
        Pnt2d::from(self.transform_coords(&p.coords))
    }

    pub fn inverse(&self) -> Self {
        if self.kind == TransformKind::Compound {
            let inv = self
                .matrix
                .try_inverse()
                .expect("AffineTransform2d::inverse() - singular matrix");
            Self {
                matrix: inv,
                loc: -(inv * self.loc),
                kind: TransformKind::Compound,
                scale: 0.0,
            }
        } else {
            let t = self.to_transform();
            Self::from_transform(&t.inverse())
        }
    }

    pub fn compose(&self, other: &AffineTransform2d) -> AffineTransform2d {
        if self.kind == TransformKind::Compound || other.kind == TransformKind::Compound {
            let new_loc = self.loc + self.matrix * other.loc;
            AffineTransform2d {
                kind: TransformKind::Compound,
                matrix: self.matrix * other.matrix,
                loc: new_loc,
                scale: 0.0,
            }
        } else {
            let t1 = self.to_transform();
            let t2 = other.to_transform();
            Self::from_transform(&t1.compose(&t2))
        }
    }
}

impl std::ops::Mul for AffineTransform2d {
    type Output = AffineTransform2d;

    fn mul(self, rhs: AffineTransform2d) -> AffineTransform2d {
        self.compose(&rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Point2;

    fn pnt(x: f64, y: f64) -> Pnt2d {
        Point2::new(x, y)
    }

    fn assert_point_near(a: &Pnt2d, b: &Pnt2d, tol: f64) {
        assert!(
            (a - b).norm() < tol,
            "points differ: {a:?} vs {b:?} (dist={})",
            (a - b).norm()
        );
    }

    #[test]
    fn non_uniform_scale() {
        let matrix = Matrix2::new(2.0, 0.0, 0.0, 3.0);
        let t = AffineTransform2d::from_matrix(matrix, Vec2d::zeros());
        assert_point_near(&t.transform_point(&pnt(1.0, 1.0)), &pnt(2.0, 3.0), 1e-15);
    }

    #[test]
    fn shear() {
        let matrix = Matrix2::new(1.0, 1.0, 0.0, 1.0);
        let t = AffineTransform2d::from_matrix(matrix, Vec2d::zeros());
        assert_point_near(&t.transform_point(&pnt(0.0, 1.0)), &pnt(1.0, 1.0), 1e-15);
    }

    #[test]
    fn inverse_roundtrip() {
        let matrix = Matrix2::new(2.0, 1.0, 0.5, 3.0);
        let loc = Vec2d::new(1.0, 2.0);
        let t = AffineTransform2d::from_matrix(matrix, loc);
        let p = pnt(4.0, 5.0);
        let roundtrip = t.inverse().transform_point(&t.transform_point(&p));
        assert_point_near(&roundtrip, &p, 1e-12);
    }

    #[test]
    fn compose_matches_sequential() {
        let m1 = Matrix2::new(2.0, 0.0, 0.0, 1.0);
        let m2 = Matrix2::new(1.0, 1.0, 0.0, 1.0);
        let t1 = AffineTransform2d::from_matrix(m1, Vec2d::new(1.0, 0.0));
        let t2 = AffineTransform2d::from_matrix(m2, Vec2d::new(0.0, 1.0));
        let p = pnt(1.0, 2.0);

        let sequential = t1.transform_point(&t2.transform_point(&p));
        let composed = t1.compose(&t2).transform_point(&p);
        assert_point_near(&sequential, &composed, 1e-14);
    }
}
