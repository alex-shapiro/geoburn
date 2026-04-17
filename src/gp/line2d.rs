use super::ax2d::Ax2d;
use super::{Dir2d, Pnt2d, RESOLUTION};

/// An infinite line in 2D space
#[derive(Debug, Clone)]
pub struct Line2d {
    pub pos: Ax2d,
}

impl Line2d {
    pub fn new(origin: Pnt2d, dir: Dir2d) -> Self {
        Self {
            pos: Ax2d::new(origin, dir),
        }
    }

    pub fn from_ax2d(ax: Ax2d) -> Self {
        Self { pos: ax }
    }

    /// Create from the equation `a*x + b*y + c = 0`.
    pub fn from_equation(a: f64, b: f64, c: f64) -> Self {
        let norm = (a * a + b * b).sqrt();
        assert!(norm > RESOLUTION, "Line2d::from_equation() - null normal");
        // Normal is (a, b), direction is perpendicular: (-b, a)
        let dir = nalgebra::Unit::new_normalize(nalgebra::Vector2::new(-b, a));
        let origin = Pnt2d::new(-a * c / (norm * norm), -b * c / (norm * norm));
        Self::new(origin, dir)
    }

    pub fn origin(&self) -> &Pnt2d {
        &self.pos.origin
    }

    pub fn direction(&self) -> &Dir2d {
        &self.pos.dir
    }

    pub fn reversed(&self) -> Self {
        Self {
            pos: self.pos.reversed(),
        }
    }

    /// Signed distance from a point to this line.
    /// Positive = point is on the left side (counterclockwise).
    pub fn signed_distance(&self, p: &Pnt2d) -> f64 {
        let v = p - self.pos.origin;
        // cross product: dir × v (2D scalar cross)
        self.pos.dir.x * v.y - self.pos.dir.y * v.x
    }

    pub fn distance(&self, p: &Pnt2d) -> f64 {
        self.signed_distance(p).abs()
    }

    pub fn square_distance(&self, p: &Pnt2d) -> f64 {
        let d = self.signed_distance(p);
        d * d
    }

    pub fn contains(&self, p: &Pnt2d, tolerance: f64) -> bool {
        self.distance(p) <= tolerance
    }

    /// Coefficients `(a, b, c)` such that `a*x + b*y + c = 0`.
    pub fn coefficients(&self) -> (f64, f64, f64) {
        // Normal direction = (-dir.y, dir.x)
        let a = -self.pos.dir.y;
        let b = self.pos.dir.x;
        let c = -(a * self.pos.origin.x + b * self.pos.origin.y);
        (a, b, c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Point2, Unit, Vector2};

    fn dir(x: f64, y: f64) -> Dir2d {
        Unit::new_normalize(Vector2::new(x, y))
    }

    fn pnt(x: f64, y: f64) -> Pnt2d {
        Point2::new(x, y)
    }

    #[test]
    fn distance_to_point() {
        let l = Line2d::new(Point2::origin(), dir(1.0, 0.0));
        assert!((l.distance(&pnt(0.0, 5.0)) - 5.0).abs() < 1e-15);
    }

    #[test]
    fn signed_distance() {
        let l = Line2d::new(Point2::origin(), dir(1.0, 0.0));
        assert!(l.signed_distance(&pnt(0.0, 1.0)) > 0.0); // left
        assert!(l.signed_distance(&pnt(0.0, -1.0)) < 0.0); // right
    }

    #[test]
    fn coefficients_x_axis() {
        let l = Line2d::new(Point2::origin(), dir(1.0, 0.0));
        let (a, b, c) = l.coefficients();
        // y = 0 → 0*x + 1*y + 0 = 0
        assert!(a.abs() < 1e-15);
        assert!((b - 1.0).abs() < 1e-15);
        assert!(c.abs() < 1e-15);
    }
}
