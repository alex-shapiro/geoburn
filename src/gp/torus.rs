use super::Pnt;
use super::ax1::Ax1;
use super::ax3::Ax3;
use std::f64::consts::PI;

/// A torus in 3D space.
///
/// Defined by a coordinate system (`Ax3`), a major radius (distance from
/// the center to the tube center), and a minor radius (tube radius).
/// The Z axis of the coordinate system is the torus axis.
#[derive(Debug, Clone)]
pub struct Torus {
    pos: Ax3,
    major_radius: f64,
    minor_radius: f64,
}

impl Torus {
    /// Creates a torus. Major radius must exceed minor radius.
    pub fn new(pos: Ax3, major_radius: f64, minor_radius: f64) -> Self {
        assert!(minor_radius >= 0.0, "Torus: negative minor radius");
        assert!(
            major_radius - minor_radius > super::RESOLUTION,
            "Torus: major radius must exceed minor radius"
        );
        Self {
            pos,
            major_radius,
            minor_radius,
        }
    }

    pub fn position(&self) -> &Ax3 {
        &self.pos
    }

    pub fn axis(&self) -> Ax1 {
        Ax1::new(*self.pos.origin(), *self.pos.z_direction())
    }

    pub fn location(&self) -> &Pnt {
        self.pos.origin()
    }

    pub fn major_radius(&self) -> f64 {
        self.major_radius
    }

    pub fn minor_radius(&self) -> f64 {
        self.minor_radius
    }

    pub fn is_direct(&self) -> bool {
        self.pos.is_direct()
    }

    pub fn area(&self) -> f64 {
        4.0 * PI * PI * self.major_radius * self.minor_radius
    }

    pub fn volume(&self) -> f64 {
        2.0 * PI * PI * self.major_radius * self.minor_radius * self.minor_radius
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

    fn standard_torus() -> Torus {
        Torus::new(
            Ax3::new(Point3::origin(), dir(0.0, 0.0, 1.0), dir(1.0, 0.0, 0.0)),
            5.0,
            1.0,
        )
    }

    #[test]
    fn area() {
        let t = standard_torus();
        assert!((t.area() - 4.0 * PI * PI * 5.0).abs() < 1e-12);
    }

    #[test]
    fn volume() {
        let t = standard_torus();
        assert!((t.volume() - 2.0 * PI * PI * 5.0).abs() < 1e-12);
    }
}
