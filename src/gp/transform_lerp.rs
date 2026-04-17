use nalgebra::UnitQuaternion;

use super::precision;
use super::transform::Transform;
use super::{Pnt, Vec3};

/// Linear interpolation of transforms.
///
/// Decomposes each transform into translation, rotation (quaternion),
/// and scale, then interpolates each independently:
/// - Translation: linear interpolation
/// - Rotation: normalized linear interpolation (nlerp)
/// - Scale: linear interpolation
///
/// This is approximate. The result may not match what a specific
/// application expects for large interpolation intervals.
pub struct TransformLerp {
    start: Transform,
    end: Transform,
    loc_start: Vec3,
    loc_end: Vec3,
    rot_start: UnitQuaternion<f64>,
    rot_end: UnitQuaternion<f64>,
    scale_start: f64,
    scale_end: f64,
}

impl TransformLerp {
    pub fn new(start: Transform, end: Transform) -> Self {
        let rot_start = UnitQuaternion::from_rotation_matrix(
            &nalgebra::Rotation3::from_matrix_unchecked(*start.homogeneous_vectorial_part()),
        );
        let rot_end = UnitQuaternion::from_rotation_matrix(
            &nalgebra::Rotation3::from_matrix_unchecked(*end.homogeneous_vectorial_part()),
        );

        Self {
            loc_start: *start.translation_part(),
            loc_end: *end.translation_part(),
            rot_start,
            rot_end,
            scale_start: start.scale_factor(),
            scale_end: end.scale_factor(),
            start,
            end,
        }
    }

    /// Interpolate at parameter `t` in [0, 1].
    ///
    /// Returns the start transform at t=0 and end transform at t=1.
    pub fn interpolate(&self, t: f64) -> Transform {
        if t.abs() < precision::CONFUSION {
            return self.start.clone();
        }
        if (t - 1.0).abs() < precision::CONFUSION {
            return self.end.clone();
        }

        let loc = self.loc_start.lerp(&self.loc_end, t);
        let rot = self.rot_start.nlerp(&self.rot_end, t);
        let scale = self.scale_start + (self.scale_end - self.scale_start) * t;

        let matrix = *rot.to_rotation_matrix().matrix();
        let origin = Pnt::origin();
        // Reconstruct: set rotation, then translation, then scale
        // p' = scale * matrix * p + loc
        Transform::from_parts(
            super::transform::TransformKind::Compound,
            scale,
            matrix,
            loc - origin.coords, // loc is already the translation vector
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gp::Dir;
    use crate::gp::ax1::Ax1;
    use nalgebra::{Point3, Unit, Vector3};
    use std::f64::consts::PI;

    fn dir(x: f64, y: f64, z: f64) -> Dir {
        Unit::new_normalize(Vector3::new(x, y, z))
    }

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
    fn endpoints_are_exact() {
        let t1 = Transform::translation(&Vector3::new(1.0, 0.0, 0.0));
        let t2 = Transform::translation(&Vector3::new(5.0, 0.0, 0.0));
        let lerp = TransformLerp::new(t1, t2);

        let p = pnt(0.0, 0.0, 0.0);
        assert_point_near(
            &lerp.interpolate(0.0).transform_point(&p),
            &pnt(1.0, 0.0, 0.0),
            1e-14,
        );
        assert_point_near(
            &lerp.interpolate(1.0).transform_point(&p),
            &pnt(5.0, 0.0, 0.0),
            1e-14,
        );
    }

    #[test]
    fn midpoint_translation() {
        let t1 = Transform::translation(&Vector3::new(0.0, 0.0, 0.0));
        let t2 = Transform::translation(&Vector3::new(10.0, 0.0, 0.0));
        let lerp = TransformLerp::new(t1, t2);

        let p = pnt(0.0, 0.0, 0.0);
        assert_point_near(
            &lerp.interpolate(0.5).transform_point(&p),
            &pnt(5.0, 0.0, 0.0),
            1e-14,
        );
    }

    #[test]
    fn midpoint_rotation() {
        let ax = Ax1::new(Point3::origin(), dir(0.0, 0.0, 1.0));
        let t1 = Transform::identity();
        let t2 = Transform::rotation(&ax, PI);
        let lerp = TransformLerp::new(t1, t2);

        let p = pnt(1.0, 0.0, 0.0);
        let mid = lerp.interpolate(0.5).transform_point(&p);
        // At t=0.5, ~90° rotation: (1,0,0) → approximately (0,1,0)
        assert!(mid.x.abs() < 0.1);
        assert!(mid.y > 0.9);
    }

    #[test]
    fn scale_interpolation() {
        let t1 = Transform::scale(&Point3::origin(), 1.0);
        let t2 = Transform::scale(&Point3::origin(), 3.0);
        let lerp = TransformLerp::new(t1, t2);

        let p = pnt(1.0, 0.0, 0.0);
        let mid = lerp.interpolate(0.5).transform_point(&p);
        assert_point_near(&mid, &pnt(2.0, 0.0, 0.0), 1e-13);
    }
}
