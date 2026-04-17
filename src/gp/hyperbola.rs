use super::Pnt;
use super::ax1::Ax1;
use super::ax2::Ax2;
use super::line::Line;

/// A hyperbola in 3D space.
///
/// Defined by a coordinate system (`Ax2`) whose origin is the center,
/// a major radius (real semi-axis), and a minor radius (imaginary semi-axis).
/// The hyperbola lies in the XY plane with branches along the X axis.
#[derive(Debug, Clone)]
pub struct Hyperbola {
    pos: Ax2,
    major_radius: f64,
    minor_radius: f64,
}

impl Hyperbola {
    pub fn new(pos: Ax2, major_radius: f64, minor_radius: f64) -> Self {
        assert!(major_radius >= 0.0, "Hyperbola: negative major radius");
        assert!(minor_radius >= 0.0, "Hyperbola: negative minor radius");
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

    /// Linear eccentricity: c = sqrt(a² + b²).
    fn linear_eccentricity(&self) -> f64 {
        (self.major_radius * self.major_radius + self.minor_radius * self.minor_radius).sqrt()
    }

    /// Eccentricity e = c / a (always > 1 for a hyperbola).
    pub fn eccentricity(&self) -> f64 {
        assert!(
            self.major_radius > 0.0,
            "Hyperbola::eccentricity() - zero major radius"
        );
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
        assert!(
            self.major_radius > 0.0,
            "Hyperbola::parameter() - zero major radius"
        );
        self.minor_radius * self.minor_radius / self.major_radius
    }

    /// First directrix (at x = a/e from center).
    pub fn directrix1(&self) -> Line {
        let e = self.eccentricity();
        let offset = self.major_radius / e;
        let origin = Pnt::from(self.pos.origin().coords + self.pos.x_direction().as_ref() * offset);
        Line::new(origin, *self.pos.y_direction())
    }

    /// Second directrix (at x = -a/e from center).
    pub fn directrix2(&self) -> Line {
        let e = self.eccentricity();
        let offset = self.major_radius / e;
        let origin = Pnt::from(self.pos.origin().coords - self.pos.x_direction().as_ref() * offset);
        Line::new(origin, *self.pos.y_direction())
    }

    /// First asymptote line through center.
    pub fn asymptote1(&self) -> Line {
        assert!(
            self.major_radius > 0.0,
            "Hyperbola::asymptote1() - zero major radius"
        );
        let ratio = self.minor_radius / self.major_radius;
        let dir_vec = self.pos.x_direction().as_ref() + self.pos.y_direction().as_ref() * ratio;
        Line::new(*self.pos.origin(), nalgebra::Unit::new_normalize(dir_vec))
    }

    /// Second asymptote line through center.
    pub fn asymptote2(&self) -> Line {
        assert!(
            self.major_radius > 0.0,
            "Hyperbola::asymptote2() - zero major radius"
        );
        let ratio = self.minor_radius / self.major_radius;
        let dir_vec = self.pos.x_direction().as_ref() - self.pos.y_direction().as_ref() * ratio;
        Line::new(*self.pos.origin(), nalgebra::Unit::new_normalize(dir_vec))
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

    fn standard_hypr() -> Hyperbola {
        Hyperbola::new(
            Ax2::new(Point3::origin(), dir(0.0, 0.0, 1.0), dir(1.0, 0.0, 0.0)),
            3.0,
            4.0,
        )
    }

    #[test]
    fn eccentricity_gt_one() {
        let h = standard_hypr();
        assert!(h.eccentricity() > 1.0);
        // c = sqrt(9+16) = 5, e = 5/3
        assert!((h.eccentricity() - 5.0 / 3.0).abs() < 1e-14);
    }

    #[test]
    fn foci() {
        let h = standard_hypr();
        assert!((h.focus1().x - 5.0).abs() < 1e-14);
        assert!((h.focus2().x - (-5.0)).abs() < 1e-14);
    }

    #[test]
    fn asymptotes_are_symmetric() {
        let h = standard_hypr();
        let a1 = h.asymptote1();
        let a2 = h.asymptote2();
        // Both pass through origin
        assert!((a1.origin() - Point3::origin()).norm() < 1e-15);
        assert!((a2.origin() - Point3::origin()).norm() < 1e-15);
        // Y components have opposite sign
        assert!((a1.direction().y + a2.direction().y).abs() < 1e-14);
    }
}
