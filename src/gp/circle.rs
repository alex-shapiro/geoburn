use super::Pnt;
use super::ax1::Ax1;
use super::ax2::Ax2;
use std::f64::consts::PI;

/// A circle in 3D space.
///
/// Defined by a coordinate system (`Ax2`) whose origin is the center,
/// and a radius. The circle lies in the XY plane of the coordinate system.
#[derive(Debug, Clone)]
pub struct Circle {
    pos: Ax2,
    radius: f64,
}

impl Circle {
    /// Creates a circle. Panics if `radius < 0`.
    pub fn new(pos: Ax2, radius: f64) -> Self {
        assert!(radius >= 0.0, "Circle::new() - negative radius");
        Self { pos, radius }
    }

    pub fn position(&self) -> &Ax2 {
        &self.pos
    }

    pub fn center(&self) -> &Pnt {
        self.pos.origin()
    }

    pub fn radius(&self) -> f64 {
        self.radius
    }

    /// The normal axis (center + Z direction).
    pub fn axis(&self) -> Ax1 {
        Ax1::new(*self.pos.origin(), *self.pos.z_direction())
    }

    pub fn x_axis(&self) -> Ax1 {
        Ax1::new(*self.pos.origin(), *self.pos.x_direction())
    }

    pub fn y_axis(&self) -> Ax1 {
        Ax1::new(*self.pos.origin(), *self.pos.y_direction())
    }

    pub fn area(&self) -> f64 {
        PI * self.radius * self.radius
    }

    pub fn length(&self) -> f64 {
        2.0 * PI * self.radius
    }

    /// Minimum distance from a point to the circle circumference.
    pub fn distance_to_point(&self, p: &Pnt) -> f64 {
        self.square_distance_to_point(p).sqrt()
    }

    /// Square distance from a point to the circle circumference.
    pub fn square_distance_to_point(&self, p: &Pnt) -> f64 {
        let v = p - self.pos.origin();
        let x = v.dot(self.pos.x_direction().as_ref());
        let y = v.dot(self.pos.y_direction().as_ref());
        let z = v.dot(self.pos.z_direction().as_ref());
        let in_plane = (x * x + y * y).sqrt() - self.radius;
        in_plane * in_plane + z * z
    }

    /// Returns true if the point lies on the circumference within `tolerance`.
    pub fn contains(&self, p: &Pnt, tolerance: f64) -> bool {
        self.distance_to_point(p) <= tolerance
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gp::Dir;
    use nalgebra::{Point3, Unit, Vector3};

    fn dir(x: f64, y: f64, z: f64) -> Dir {
        Unit::new_normalize(Vector3::new(x, y, z))
    }

    fn pnt(x: f64, y: f64, z: f64) -> Pnt {
        Point3::new(x, y, z)
    }

    fn xy_circle(r: f64) -> Circle {
        Circle::new(
            Ax2::new(Point3::origin(), dir(0.0, 0.0, 1.0), dir(1.0, 0.0, 0.0)),
            r,
        )
    }

    #[test]
    fn point_on_circle() {
        let c = xy_circle(5.0);
        assert!(c.contains(&pnt(5.0, 0.0, 0.0), 1e-14));
        assert!(c.contains(&pnt(0.0, 5.0, 0.0), 1e-14));
    }

    #[test]
    fn center_distance_equals_radius() {
        let c = xy_circle(5.0);
        assert!((c.distance_to_point(&Point3::origin()) - 5.0).abs() < 1e-14);
    }

    #[test]
    fn point_above_circle() {
        let c = xy_circle(5.0);
        // Point at (5, 0, 3): on circumference projected, 3 above
        assert!((c.distance_to_point(&pnt(5.0, 0.0, 3.0)) - 3.0).abs() < 1e-14);
    }

    #[test]
    fn area_and_length() {
        let c = xy_circle(1.0);
        assert!((c.area() - PI).abs() < 1e-14);
        assert!((c.length() - 2.0 * PI).abs() < 1e-14);
    }
}
