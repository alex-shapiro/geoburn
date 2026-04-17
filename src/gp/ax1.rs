use super::{Dir, Pnt, direction};

/// An axis in 3D space: a point and a direction.
/// Used to describe axes of revolution, rotation, symmetry, etc.
#[derive(Debug, Clone, Copy)]
pub struct Ax1 {
    pub origin: Pnt,
    pub dir: Dir,
}

impl Ax1 {
    pub fn new(origin: Pnt, dir: Dir) -> Self {
        Self { origin, dir }
    }

    pub fn reversed(&self) -> Self {
        Self {
            origin: self.origin,
            dir: -self.dir,
        }
    }

    pub fn angle(&self, other: &Ax1) -> f64 {
        direction::angle(&self.dir, &other.dir)
    }

    pub fn is_normal(&self, other: &Ax1, tolerance: f64) -> bool {
        direction::is_normal(&self.dir, &other.dir, tolerance)
    }

    pub fn is_opposite(&self, other: &Ax1, tolerance: f64) -> bool {
        direction::is_opposite(&self.dir, &other.dir, tolerance)
    }

    pub fn is_parallel(&self, other: &Ax1, tolerance: f64) -> bool {
        direction::is_parallel(&self.dir, &other.dir, tolerance)
    }

    /// Returns true if two axes are coaxial: parallel directions and both
    /// origins lie on the other's line, within the given tolerances.
    pub fn is_coaxial(&self, other: &Ax1, angular_tolerance: f64, linear_tolerance: f64) -> bool {
        if !self.is_parallel(other, angular_tolerance) {
            return false;
        }
        let d = other.origin - self.origin;
        let dist_sq = d.norm_squared() - d.dot(&self.dir).powi(2);
        dist_sq <= linear_tolerance * linear_tolerance
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Point3, Unit, Vector3};

    fn ax(ox: f64, oy: f64, oz: f64, dx: f64, dy: f64, dz: f64) -> Ax1 {
        Ax1::new(
            Point3::new(ox, oy, oz),
            Unit::new_normalize(Vector3::new(dx, dy, dz)),
        )
    }

    #[test]
    fn reversed() {
        let a = ax(0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        let r = a.reversed();
        assert!((r.dir.z - (-1.0)).abs() < 1e-15);
        assert_eq!(r.origin, a.origin);
    }

    #[test]
    fn coaxial_same_axis() {
        let a = ax(0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        let b = ax(0.0, 0.0, 5.0, 0.0, 0.0, 1.0);
        assert!(a.is_coaxial(&b, 1e-10, 1e-10));
    }

    #[test]
    fn coaxial_opposite_direction() {
        let a = ax(0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        let b = ax(0.0, 0.0, 3.0, 0.0, 0.0, -1.0);
        assert!(a.is_coaxial(&b, 1e-10, 1e-10));
    }

    #[test]
    fn not_coaxial_offset() {
        let a = ax(0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        let b = ax(1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        assert!(!a.is_coaxial(&b, 1e-10, 1e-10));
    }

    #[test]
    fn not_coaxial_not_parallel() {
        let a = ax(0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        let b = ax(0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
        assert!(!a.is_coaxial(&b, 1e-10, 1e-10));
    }
}
