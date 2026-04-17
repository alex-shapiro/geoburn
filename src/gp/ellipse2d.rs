use super::Pnt2d;
use super::ax22d::Ax22d;
use super::line2d::Line2d;
use std::f64::consts::PI;

/// An ellipse in 2D space
#[derive(Debug, Clone, Copy)]
pub struct Ellipse2d {
    pos: Ax22d,
    major_radius: f64,
    minor_radius: f64,
}

impl Ellipse2d {
    pub fn new(pos: Ax22d, major_radius: f64, minor_radius: f64) -> Self {
        assert!(minor_radius >= 0.0, "Ellipse2d: negative minor radius");
        assert!(
            major_radius >= minor_radius,
            "Ellipse2d: minor radius exceeds major"
        );
        Self {
            pos,
            major_radius,
            minor_radius,
        }
    }

    pub fn position(&self) -> &Ax22d {
        &self.pos
    }

    pub fn center(&self) -> &Pnt2d {
        self.pos.origin()
    }

    pub fn major_radius(&self) -> f64 {
        self.major_radius
    }

    pub fn minor_radius(&self) -> f64 {
        self.minor_radius
    }

    pub fn area(&self) -> f64 {
        PI * self.major_radius * self.minor_radius
    }

    fn linear_eccentricity(&self) -> f64 {
        (self.major_radius * self.major_radius - self.minor_radius * self.minor_radius).sqrt()
    }

    pub fn eccentricity(&self) -> f64 {
        if self.major_radius == 0.0 {
            return 0.0;
        }
        self.linear_eccentricity() / self.major_radius
    }

    pub fn focal(&self) -> f64 {
        2.0 * self.linear_eccentricity()
    }

    pub fn focus1(&self) -> Pnt2d {
        let c = self.linear_eccentricity();
        Pnt2d::from(self.pos.origin().coords + self.pos.x_direction().as_ref() * c)
    }

    pub fn focus2(&self) -> Pnt2d {
        let c = self.linear_eccentricity();
        Pnt2d::from(self.pos.origin().coords - self.pos.x_direction().as_ref() * c)
    }

    pub fn parameter(&self) -> f64 {
        if self.major_radius == 0.0 {
            return 0.0;
        }
        self.minor_radius * self.minor_radius / self.major_radius
    }

    pub fn directrix1(&self) -> Line2d {
        let e = self.eccentricity();
        assert!(e > 0.0, "Ellipse2d::directrix1() - circle has no directrix");
        let offset = self.major_radius / e;
        let origin =
            Pnt2d::from(self.pos.origin().coords + self.pos.x_direction().as_ref() * offset);
        Line2d::new(origin, *self.pos.y_direction())
    }

    pub fn directrix2(&self) -> Line2d {
        let e = self.eccentricity();
        assert!(e > 0.0, "Ellipse2d::directrix2() - circle has no directrix");
        let offset = self.major_radius / e;
        let origin =
            Pnt2d::from(self.pos.origin().coords - self.pos.x_direction().as_ref() * offset);
        Line2d::new(origin, *self.pos.y_direction())
    }
}
