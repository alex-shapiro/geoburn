use crate::gp::{Pnt, Vec3};

/// A bounding sphere in 3D space.
#[derive(Debug, Clone, Copy)]
pub struct BoundingSphere {
    center: Vec3,
    radius: f64,
    valid: bool,
}

impl BoundingSphere {
    pub fn new(center: Pnt, radius: f64) -> Self {
        Self {
            center: center.coords,
            radius,
            valid: true,
        }
    }

    pub fn void() -> Self {
        Self {
            center: Vec3::zeros(),
            radius: 0.0,
            valid: false,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.valid
    }

    pub fn center(&self) -> Pnt {
        Pnt::from(self.center)
    }

    pub fn radius(&self) -> f64 {
        self.radius
    }

    /// Distance from the sphere surface to a point.
    pub fn distance(&self, p: &Pnt) -> f64 {
        let d = (p.coords - self.center).norm();
        (d - self.radius).abs()
    }

    /// Square distance from the sphere surface to a point.
    pub fn square_distance(&self, p: &Pnt) -> f64 {
        let d = self.distance(p);
        d * d
    }

    /// Check if a point is outside the sphere.
    pub fn is_out_point(&self, p: &Pnt) -> bool {
        if !self.valid {
            return true;
        }
        (p.coords - self.center).norm_squared() > self.radius * self.radius
    }

    /// Check if another sphere is completely outside.
    pub fn is_out(&self, other: &BoundingSphere) -> bool {
        if !self.valid || !other.valid {
            return true;
        }
        let dist_sq = (other.center - self.center).norm_squared();
        let r_sum = self.radius + other.radius;
        dist_sq > r_sum * r_sum
    }

    /// Merge with another sphere to create a bounding sphere enclosing both.
    pub fn add(&mut self, other: &BoundingSphere) {
        if !other.valid {
            return;
        }
        if !self.valid {
            *self = *other;
            return;
        }
        let d_vec = other.center - self.center;
        let dist = d_vec.norm();
        if dist + other.radius <= self.radius {
            return; // other is inside self
        }
        if dist + self.radius <= other.radius {
            *self = *other; // self is inside other
            return;
        }
        // New sphere enclosing both
        let new_radius = (dist + self.radius + other.radius) * 0.5;
        let shift = (new_radius - self.radius) / dist;
        self.center += d_vec * shift;
        self.radius = new_radius;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Point3;

    fn pnt(x: f64, y: f64, z: f64) -> Pnt {
        Point3::new(x, y, z)
    }

    #[test]
    fn point_inside() {
        let s = BoundingSphere::new(Point3::origin(), 5.0);
        assert!(!s.is_out_point(&pnt(1.0, 0.0, 0.0)));
    }

    #[test]
    fn point_outside() {
        let s = BoundingSphere::new(Point3::origin(), 5.0);
        assert!(s.is_out_point(&pnt(6.0, 0.0, 0.0)));
    }

    #[test]
    fn spheres_overlap() {
        let a = BoundingSphere::new(Point3::origin(), 5.0);
        let b = BoundingSphere::new(pnt(8.0, 0.0, 0.0), 5.0);
        assert!(!a.is_out(&b));
    }

    #[test]
    fn spheres_separated() {
        let a = BoundingSphere::new(Point3::origin(), 1.0);
        let b = BoundingSphere::new(pnt(10.0, 0.0, 0.0), 1.0);
        assert!(a.is_out(&b));
    }

    #[test]
    fn merge() {
        let mut a = BoundingSphere::new(Point3::origin(), 1.0);
        let b = BoundingSphere::new(pnt(10.0, 0.0, 0.0), 1.0);
        a.add(&b);
        // Merged sphere should contain both
        assert!(!a.is_out_point(&Point3::origin()));
        assert!(!a.is_out_point(&pnt(10.0, 0.0, 0.0)));
        assert!((a.radius - 6.0).abs() < 1e-14);
    }
}
