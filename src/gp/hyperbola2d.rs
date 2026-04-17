use super::Pnt2d;
use super::ax22d::Ax22d;
use super::line2d::Line2d;

/// A hyperbola in 2D space
#[derive(Debug, Clone, Copy)]
pub struct Hyperbola2d {
    pos: Ax22d,
    major_radius: f64,
    minor_radius: f64,
}

impl Hyperbola2d {
    pub fn new(pos: Ax22d, major_radius: f64, minor_radius: f64) -> Self {
        assert!(major_radius >= 0.0, "Hyperbola2d: negative major radius");
        assert!(minor_radius >= 0.0, "Hyperbola2d: negative minor radius");
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

    fn linear_eccentricity(&self) -> f64 {
        (self.major_radius * self.major_radius + self.minor_radius * self.minor_radius).sqrt()
    }

    pub fn eccentricity(&self) -> f64 {
        assert!(
            self.major_radius > 0.0,
            "Hyperbola2d::eccentricity() - zero major radius"
        );
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
        assert!(
            self.major_radius > 0.0,
            "Hyperbola2d::parameter() - zero major radius"
        );
        self.minor_radius * self.minor_radius / self.major_radius
    }

    pub fn directrix1(&self) -> Line2d {
        let e = self.eccentricity();
        let offset = self.major_radius / e;
        let origin =
            Pnt2d::from(self.pos.origin().coords + self.pos.x_direction().as_ref() * offset);
        Line2d::new(origin, *self.pos.y_direction())
    }

    pub fn directrix2(&self) -> Line2d {
        let e = self.eccentricity();
        let offset = self.major_radius / e;
        let origin =
            Pnt2d::from(self.pos.origin().coords - self.pos.x_direction().as_ref() * offset);
        Line2d::new(origin, *self.pos.y_direction())
    }

    pub fn asymptote1(&self) -> Line2d {
        assert!(
            self.major_radius > 0.0,
            "Hyperbola2d::asymptote1() - zero major radius"
        );
        let ratio = self.minor_radius / self.major_radius;
        let dir_vec = self.pos.x_direction().as_ref() + self.pos.y_direction().as_ref() * ratio;
        Line2d::new(*self.pos.origin(), nalgebra::Unit::new_normalize(dir_vec))
    }

    pub fn asymptote2(&self) -> Line2d {
        assert!(
            self.major_radius > 0.0,
            "Hyperbola2d::asymptote2() - zero major radius"
        );
        let ratio = self.minor_radius / self.major_radius;
        let dir_vec = self.pos.x_direction().as_ref() - self.pos.y_direction().as_ref() * ratio;
        Line2d::new(*self.pos.origin(), nalgebra::Unit::new_normalize(dir_vec))
    }
}
