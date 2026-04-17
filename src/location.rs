//! A located transformation for positioning shapes in space.

use crate::gp::Pnt;
use crate::gp::transform::Transform;

/// A location in 3D space, wrapping a `Transform`.
///
/// Used to attach a position/orientation to shapes without modifying
/// the shape's intrinsic geometry. Locations compose via multiplication
/// and can be inverted.
#[derive(Debug, Clone)]
pub struct Location {
    trsf: Transform,
}

impl Location {
    /// The identity location (no transformation).
    pub fn identity() -> Self {
        Self {
            trsf: Transform::identity(),
        }
    }

    /// Create a location from a transform.
    pub fn from_transform(trsf: Transform) -> Self {
        Self { trsf }
    }

    /// Returns true if this is the identity location.
    pub fn is_identity(&self) -> bool {
        self.trsf.kind() == crate::gp::transform::TransformKind::Identity
    }

    /// The underlying transformation.
    pub fn transformation(&self) -> &Transform {
        &self.trsf
    }

    /// Apply this location to a point.
    pub fn transform_point(&self, p: &Pnt) -> Pnt {
        self.trsf.transform_point(p)
    }

    /// Compose: `self * other` (apply `other` first, then `self`).
    pub fn composed(&self, other: &Location) -> Location {
        Location {
            trsf: self.trsf.compose(&other.trsf),
        }
    }

    /// Inverse location.
    pub fn inverted(&self) -> Location {
        Location {
            trsf: self.trsf.inverse(),
        }
    }

    /// `self * other.inverse()`
    pub fn divided(&self, other: &Location) -> Location {
        self.composed(&other.inverted())
    }

    /// `other.inverse() * self`
    pub fn predivided(&self, other: &Location) -> Location {
        other.inverted().composed(self)
    }

    /// Repeated composition: `self^n`.
    /// - n > 0: self * self * ... * self
    /// - n = 0: identity
    /// - n < 0: inverse composed |n| times
    pub fn powered(&self, n: i32) -> Location {
        if n == 0 {
            return Location::identity();
        }
        let base = if n > 0 {
            self.trsf.clone()
        } else {
            self.trsf.inverse()
        };
        let count = n.unsigned_abs();
        let mut result = base.clone();
        for _ in 1..count {
            result = result.compose(&base);
        }
        Location { trsf: result }
    }
}

impl std::ops::Mul for Location {
    type Output = Location;

    fn mul(self, rhs: Location) -> Location {
        self.composed(&rhs)
    }
}

impl std::ops::Mul<&Location> for &Location {
    type Output = Location;

    fn mul(self, rhs: &Location) -> Location {
        self.composed(rhs)
    }
}

impl PartialEq for Location {
    fn eq(&self, other: &Self) -> bool {
        if self.is_identity() && other.is_identity() {
            return true;
        }
        // Compare by transforming a few test points and checking closeness
        let test_points = [
            Pnt::new(0.0, 0.0, 0.0),
            Pnt::new(1.0, 0.0, 0.0),
            Pnt::new(0.0, 1.0, 0.0),
            Pnt::new(0.0, 0.0, 1.0),
        ];
        let tol = 1e-14;
        test_points.iter().all(|p| {
            let a = self.trsf.transform_point(p);
            let b = other.trsf.transform_point(p);
            (a - b).norm() < tol
        })
    }
}

impl Eq for Location {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gp::Dir;
    use crate::gp::ax1::Ax1;
    use nalgebra::{Point3, Unit, Vector3};
    use std::f64::consts::FRAC_PI_2;

    fn dir(x: f64, y: f64, z: f64) -> Dir {
        Unit::new_normalize(Vector3::new(x, y, z))
    }

    fn pnt(x: f64, y: f64, z: f64) -> Pnt {
        Point3::new(x, y, z)
    }

    fn assert_pnt_near(a: &Pnt, b: &Pnt, tol: f64) {
        assert!((a - b).norm() < tol, "{a:?} vs {b:?}");
    }

    #[test]
    fn identity() {
        let loc = Location::identity();
        assert!(loc.is_identity());
        assert_pnt_near(
            &loc.transform_point(&pnt(1.0, 2.0, 3.0)),
            &pnt(1.0, 2.0, 3.0),
            1e-15,
        );
    }

    #[test]
    fn from_translation() {
        let t = Transform::translation(&Vector3::new(10.0, 0.0, 0.0));
        let loc = Location::from_transform(t);
        assert_pnt_near(
            &loc.transform_point(&pnt(0.0, 0.0, 0.0)),
            &pnt(10.0, 0.0, 0.0),
            1e-15,
        );
    }

    #[test]
    fn compose() {
        let t1 = Transform::translation(&Vector3::new(1.0, 0.0, 0.0));
        let t2 = Transform::translation(&Vector3::new(0.0, 2.0, 0.0));
        let loc = Location::from_transform(t1).composed(&Location::from_transform(t2));
        assert_pnt_near(
            &loc.transform_point(&pnt(0.0, 0.0, 0.0)),
            &pnt(1.0, 2.0, 0.0),
            1e-15,
        );
    }

    #[test]
    fn inverse_roundtrip() {
        let ax = Ax1::new(pnt(1.0, 2.0, 3.0), dir(0.0, 0.0, 1.0));
        let t = Transform::rotation(&ax, 1.23);
        let loc = Location::from_transform(t);
        let p = pnt(5.0, 6.0, 7.0);
        let roundtrip = loc.inverted().transform_point(&loc.transform_point(&p));
        assert_pnt_near(&roundtrip, &p, 1e-13);
    }

    #[test]
    fn powered_zero_is_identity() {
        let t = Transform::translation(&Vector3::new(10.0, 0.0, 0.0));
        let loc = Location::from_transform(t).powered(0);
        assert!(loc.is_identity());
    }

    #[test]
    fn powered_positive() {
        let t = Transform::translation(&Vector3::new(1.0, 0.0, 0.0));
        let loc = Location::from_transform(t).powered(3);
        assert_pnt_near(
            &loc.transform_point(&pnt(0.0, 0.0, 0.0)),
            &pnt(3.0, 0.0, 0.0),
            1e-14,
        );
    }

    #[test]
    fn equality() {
        let t = Transform::rotation(&Ax1::new(Point3::origin(), dir(0.0, 0.0, 1.0)), FRAC_PI_2);
        let a = Location::from_transform(t.clone());
        let b = Location::from_transform(t);
        assert_eq!(a, b);
    }

    #[test]
    fn mul_operator() {
        let t1 = Location::from_transform(Transform::translation(&Vector3::new(1.0, 0.0, 0.0)));
        let t2 = Location::from_transform(Transform::translation(&Vector3::new(0.0, 1.0, 0.0)));
        let loc = t1 * t2;
        assert_pnt_near(
            &loc.transform_point(&pnt(0.0, 0.0, 0.0)),
            &pnt(1.0, 1.0, 0.0),
            1e-15,
        );
    }
}
