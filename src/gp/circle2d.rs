use super::Pnt2d;
use super::ax22d::Ax22d;
use std::f64::consts::PI;

/// A circle in 2D space.
///
/// Defined by a coordinate system (`Ax22d`) whose origin is the center, and a radius.
#[derive(Debug, Clone, Copy)]
pub struct Circle2d {
    pos: Ax22d,
    radius: f64,
}

impl Circle2d {
    pub fn new(pos: Ax22d, radius: f64) -> Self {
        assert!(radius >= 0.0, "Circle2d: negative radius");
        Self { pos, radius }
    }

    pub fn position(&self) -> &Ax22d {
        &self.pos
    }

    pub fn center(&self) -> &Pnt2d {
        self.pos.origin()
    }

    pub fn radius(&self) -> f64 {
        self.radius
    }

    pub fn area(&self) -> f64 {
        PI * self.radius * self.radius
    }

    pub fn length(&self) -> f64 {
        2.0 * PI * self.radius
    }

    pub fn distance(&self, p: &Pnt2d) -> f64 {
        let d = (p - self.pos.origin()).norm();
        (d - self.radius).abs()
    }

    pub fn contains(&self, p: &Pnt2d, tolerance: f64) -> bool {
        self.distance(p) <= tolerance
    }
}
