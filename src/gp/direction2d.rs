use super::Dir2d;
use std::f64::consts::PI;

/// Computes the angle between two 2D unit vectors in [-PI, PI].
///
/// Positive = counterclockwise from `a` to `b`.
pub fn angle(a: &Dir2d, b: &Dir2d) -> f64 {
    let sin = a.x * b.y - a.y * b.x; // cross product (scalar in 2D)
    let cos = a.dot(b);
    sin.atan2(cos)
}

/// Returns true if the angle between `a` and `b` is within `tolerance` of PI/2.
pub fn is_normal(a: &Dir2d, b: &Dir2d, tolerance: f64) -> bool {
    (std::f64::consts::FRAC_PI_2 - angle(a, b).abs()).abs() <= tolerance
}

/// Returns true if the angle between `a` and `b` is within `tolerance` of PI.
pub fn is_opposite(a: &Dir2d, b: &Dir2d, tolerance: f64) -> bool {
    PI - angle(a, b).abs() <= tolerance
}

/// Returns true if the angle between `a` and `b` is within `tolerance` of 0 or PI.
pub fn is_parallel(a: &Dir2d, b: &Dir2d, tolerance: f64) -> bool {
    let ang = angle(a, b).abs();
    ang <= tolerance || PI - ang <= tolerance
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Unit, Vector2};

    fn dir(x: f64, y: f64) -> Dir2d {
        Unit::new_normalize(Vector2::new(x, y))
    }

    #[test]
    fn angle_orthogonal() {
        let a = dir(1.0, 0.0);
        let b = dir(0.0, 1.0);
        assert!((angle(&a, &b) - std::f64::consts::FRAC_PI_2).abs() < 1e-15);
    }

    #[test]
    fn angle_is_signed() {
        let a = dir(1.0, 0.0);
        let b = dir(0.0, 1.0);
        assert!(angle(&a, &b) > 0.0); // CCW
        assert!(angle(&b, &a) < 0.0); // CW
    }

    #[test]
    fn is_parallel_same_and_opposite() {
        let a = dir(1.0, 0.0);
        assert!(is_parallel(&a, &dir(1.0, 0.0), 1e-10));
        assert!(is_parallel(&a, &dir(-1.0, 0.0), 1e-10));
        assert!(!is_parallel(&a, &dir(0.0, 1.0), 1e-10));
    }
}
