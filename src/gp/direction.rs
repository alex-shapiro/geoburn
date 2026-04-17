use super::Dir;
use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_2, PI};

/// Computes the angle between two unit vectors in [0, PI].
///
/// Switches between acos/asin for better precision near 0 and PI/2.
pub fn angle(a: &Dir, b: &Dir) -> f64 {
    let cos = a.dot(b);
    if cos > -FRAC_1_SQRT_2 && cos < FRAC_1_SQRT_2 {
        cos.acos()
    } else {
        let sin = a.cross(b).norm();
        if cos < 0.0 {
            PI - sin.asin()
        } else {
            sin.asin()
        }
    }
}

/// Computes the signed angle between two unit vectors in [-PI, PI].
///
/// `ref_dir` defines the positive rotation sense: the result is positive
/// when the cross product `a × b` has the same orientation as `ref_dir`.
pub fn angle_with_ref(a: &Dir, b: &Dir, ref_dir: &Dir) -> f64 {
    let cross = a.cross(b);
    let ang = angle(a, b);
    if cross.dot(ref_dir.as_ref()) >= 0.0 {
        ang
    } else {
        -ang
    }
}

/// Returns true if the angle between `a` and `b` is within `tolerance` of PI/2.
pub fn is_normal(a: &Dir, b: &Dir, tolerance: f64) -> bool {
    (FRAC_PI_2 - angle(a, b)).abs() <= tolerance
}

/// Returns true if the angle between `a` and `b` is within `tolerance` of PI.
pub fn is_opposite(a: &Dir, b: &Dir, tolerance: f64) -> bool {
    PI - angle(a, b) <= tolerance
}

/// Returns true if the angle between `a` and `b` is within `tolerance` of 0 or PI.
pub fn is_parallel(a: &Dir, b: &Dir, tolerance: f64) -> bool {
    let ang = angle(a, b);
    ang <= tolerance || PI - ang <= tolerance
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Unit;
    use std::f64::consts::FRAC_PI_4;

    fn dir(x: f64, y: f64, z: f64) -> Dir {
        Unit::new_normalize(nalgebra::Vector3::new(x, y, z))
    }

    #[test]
    fn angle_orthogonal() {
        let a = dir(1.0, 0.0, 0.0);
        let b = dir(0.0, 1.0, 0.0);
        assert!((angle(&a, &b) - FRAC_PI_2).abs() < 1e-15);
    }

    #[test]
    fn angle_same() {
        let a = dir(1.0, 0.0, 0.0);
        assert!(angle(&a, &a) < 1e-15);
    }

    #[test]
    fn angle_opposite() {
        let a = dir(1.0, 0.0, 0.0);
        let b = dir(-1.0, 0.0, 0.0);
        assert!((angle(&a, &b) - PI).abs() < 1e-15);
    }

    #[test]
    fn angle_45_degrees() {
        let a = dir(1.0, 0.0, 0.0);
        let b = dir(1.0, 1.0, 0.0);
        assert!((angle(&a, &b) - FRAC_PI_4).abs() < 1e-15);
    }

    #[test]
    fn angle_with_ref_sign() {
        let x = dir(1.0, 0.0, 0.0);
        let y = dir(0.0, 1.0, 0.0);
        let z = dir(0.0, 0.0, 1.0);
        assert!(angle_with_ref(&x, &y, &z) > 0.0);
        assert!(angle_with_ref(&y, &x, &z) < 0.0);
    }

    #[test]
    fn test_is_parallel() {
        let a = dir(1.0, 0.0, 0.0);
        let b = dir(-1.0, 0.0, 0.0);
        assert!(is_parallel(&a, &b, 1e-10));
        assert!(!is_parallel(&a, &dir(0.0, 1.0, 0.0), 1e-10));
    }

    #[test]
    fn test_is_normal() {
        let a = dir(1.0, 0.0, 0.0);
        let b = dir(0.0, 1.0, 0.0);
        assert!(is_normal(&a, &b, 1e-10));
        assert!(!is_normal(&a, &a, 1e-10));
    }

    #[test]
    fn test_is_opposite() {
        let a = dir(1.0, 0.0, 0.0);
        let b = dir(-1.0, 0.0, 0.0);
        assert!(is_opposite(&a, &b, 1e-10));
        assert!(!is_opposite(&a, &a, 1e-10));
    }
}
