use super::Pnt2d;
use super::ax22d::Ax22d;
use super::line2d::Line2d;

/// A parabola in 2D space
#[derive(Debug, Clone, Copy)]
pub struct Parabola2d {
    pos: Ax22d,
    focal_length: f64,
}

impl Parabola2d {
    pub fn new(pos: Ax22d, focal_length: f64) -> Self {
        assert!(focal_length >= 0.0, "Parabola2d: negative focal length");
        Self { pos, focal_length }
    }

    pub fn position(&self) -> &Ax22d {
        &self.pos
    }

    pub fn vertex(&self) -> &Pnt2d {
        self.pos.origin()
    }

    pub fn focal_length(&self) -> f64 {
        self.focal_length
    }

    pub fn focus(&self) -> Pnt2d {
        Pnt2d::from(self.pos.origin().coords + self.pos.x_direction().as_ref() * self.focal_length)
    }

    pub fn directrix(&self) -> Line2d {
        let origin = Pnt2d::from(
            self.pos.origin().coords - self.pos.x_direction().as_ref() * self.focal_length,
        );
        Line2d::new(origin, *self.pos.y_direction())
    }

    pub fn parameter(&self) -> f64 {
        2.0 * self.focal_length
    }
}
