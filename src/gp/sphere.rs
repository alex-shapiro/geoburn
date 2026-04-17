use super::Pnt;
use super::ax1::Ax1;
use super::ax3::Ax3;
use std::f64::consts::PI;

/// A sphere in 3D space.
///
/// Defined by a coordinate system (`Ax3`) whose origin is the center, and a radius.
#[derive(Debug, Clone)]
pub struct Sphere {
    pos: Ax3,
    radius: f64,
}

impl Sphere {
    pub fn new(pos: Ax3, radius: f64) -> Self {
        assert!(radius >= 0.0, "Sphere: negative radius");
        Self { pos, radius }
    }

    pub fn position(&self) -> &Ax3 {
        &self.pos
    }

    pub fn center(&self) -> &Pnt {
        self.pos.origin()
    }

    pub fn radius(&self) -> f64 {
        self.radius
    }

    pub fn is_direct(&self) -> bool {
        self.pos.is_direct()
    }

    pub fn area(&self) -> f64 {
        4.0 * PI * self.radius * self.radius
    }

    pub fn volume(&self) -> f64 {
        (4.0 / 3.0) * PI * self.radius * self.radius * self.radius
    }

    pub fn x_axis(&self) -> Ax1 {
        Ax1::new(*self.pos.origin(), *self.pos.x_direction())
    }

    pub fn y_axis(&self) -> Ax1 {
        Ax1::new(*self.pos.origin(), *self.pos.y_direction())
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

    fn unit_sphere() -> Sphere {
        Sphere::new(
            Ax3::new(Point3::origin(), dir(0.0, 0.0, 1.0), dir(1.0, 0.0, 0.0)),
            1.0,
        )
    }

    #[test]
    fn area() {
        assert!((unit_sphere().area() - 4.0 * PI).abs() < 1e-14);
    }

    #[test]
    fn volume() {
        assert!((unit_sphere().volume() - 4.0 / 3.0 * PI).abs() < 1e-14);
    }
}
