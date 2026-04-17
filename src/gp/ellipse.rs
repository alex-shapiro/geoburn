use super::Pnt;
use super::ax1::Ax1;
use super::ax2::Ax2;
use super::line::Line;
use std::f64::consts::PI;

/// An ellipse in 3D space.
///
/// Defined by a coordinate system (`Ax2`) whose origin is the center,
/// a major radius, and a minor radius. The ellipse lies in the XY plane
/// of the coordinate system, with the major axis along X.
#[derive(Debug, Clone, Copy)]
pub struct Ellipse {
    pos: Ax2,
    major_radius: f64,
    minor_radius: f64,
}

impl Ellipse {
    /// Creates an ellipse. Panics if radii are negative or minor > major.
    pub fn new(pos: Ax2, major_radius: f64, minor_radius: f64) -> Self {
        assert!(minor_radius >= 0.0, "Ellipse: negative minor radius");
        assert!(
            major_radius >= minor_radius,
            "Ellipse: minor radius exceeds major"
        );
        Self {
            pos,
            major_radius,
            minor_radius,
        }
    }

    pub fn position(&self) -> &Ax2 {
        &self.pos
    }

    pub fn center(&self) -> &Pnt {
        self.pos.origin()
    }

    pub fn major_radius(&self) -> f64 {
        self.major_radius
    }

    pub fn minor_radius(&self) -> f64 {
        self.minor_radius
    }

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
        PI * self.major_radius * self.minor_radius
    }

    /// Linear eccentricity: distance from center to focus.
    fn linear_eccentricity(&self) -> f64 {
        (self.major_radius * self.major_radius - self.minor_radius * self.minor_radius).sqrt()
    }

    /// Eccentricity e = c / a, where c = sqrt(a² - b²).
    pub fn eccentricity(&self) -> f64 {
        if self.major_radius == 0.0 {
            return 0.0;
        }
        self.linear_eccentricity() / self.major_radius
    }

    /// Distance between the two foci: 2c.
    pub fn focal(&self) -> f64 {
        2.0 * self.linear_eccentricity()
    }

    /// First focus (center + c along X).
    pub fn focus1(&self) -> Pnt {
        let c = self.linear_eccentricity();
        Pnt::from(self.pos.origin().coords + self.pos.x_direction().as_ref() * c)
    }

    /// Second focus (center - c along X).
    pub fn focus2(&self) -> Pnt {
        let c = self.linear_eccentricity();
        Pnt::from(self.pos.origin().coords - self.pos.x_direction().as_ref() * c)
    }

    /// Semi-latus rectum: b²/a.
    pub fn parameter(&self) -> f64 {
        if self.major_radius == 0.0 {
            return 0.0;
        }
        self.minor_radius * self.minor_radius / self.major_radius
    }

    /// First directrix (line perpendicular to major axis at x = a/e).
    pub fn directrix1(&self) -> Line {
        let e = self.eccentricity();
        assert!(e > 0.0, "Ellipse::directrix1() - circle has no directrix");
        let offset = self.major_radius / e;
        let origin = Pnt::from(self.pos.origin().coords + self.pos.x_direction().as_ref() * offset);
        Line::new(origin, *self.pos.y_direction())
    }

    /// Second directrix (line perpendicular to major axis at x = -a/e).
    pub fn directrix2(&self) -> Line {
        let e = self.eccentricity();
        assert!(e > 0.0, "Ellipse::directrix2() - circle has no directrix");
        let offset = self.major_radius / e;
        let origin = Pnt::from(self.pos.origin().coords - self.pos.x_direction().as_ref() * offset);
        Line::new(origin, *self.pos.y_direction())
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

    fn standard_ellipse() -> Ellipse {
        Ellipse::new(
            Ax2::new(Point3::origin(), dir(0.0, 0.0, 1.0), dir(1.0, 0.0, 0.0)),
            5.0,
            3.0,
        )
    }

    #[test]
    fn area() {
        let e = standard_ellipse();
        assert!((e.area() - PI * 15.0).abs() < 1e-14);
    }

    #[test]
    fn eccentricity() {
        let e = standard_ellipse();
        let expected = 4.0 / 5.0; // c=4, a=5
        assert!((e.eccentricity() - expected).abs() < 1e-14);
    }

    #[test]
    fn foci_on_major_axis() {
        let e = standard_ellipse();
        let f1 = e.focus1();
        let f2 = e.focus2();
        assert!((f1.x - 4.0).abs() < 1e-14);
        assert!((f2.x - (-4.0)).abs() < 1e-14);
        assert!(f1.y.abs() < 1e-15);
        assert!(f2.y.abs() < 1e-15);
    }

    #[test]
    fn circle_eccentricity_is_zero() {
        let c = Ellipse::new(
            Ax2::new(Point3::origin(), dir(0.0, 0.0, 1.0), dir(1.0, 0.0, 0.0)),
            5.0,
            5.0,
        );
        assert!(c.eccentricity() < 1e-15);
    }
}
