use nalgebra::{Unit, Vector2};

use super::{Dir2d, Pnt2d};

/// A coordinate system in 2D space.
///
/// Defined by an origin and two orthogonal unit vectors (X, Y).
/// Can be right-handed (direct) or left-handed. The 2D analog of `Ax3`.
#[derive(Debug, Clone, Copy)]
pub struct Ax22d {
    origin: Pnt2d,
    vx: Dir2d,
    vy: Dir2d,
}

impl Ax22d {
    /// Creates a right-handed coordinate system from an origin and X direction.
    pub fn new(origin: Pnt2d, x_dir: Dir2d, is_direct: bool) -> Self {
        let vy = if is_direct {
            Unit::new_unchecked(Vector2::new(-x_dir.y, x_dir.x))
        } else {
            Unit::new_unchecked(Vector2::new(x_dir.y, -x_dir.x))
        };
        Self {
            origin,
            vx: x_dir,
            vy,
        }
    }

    /// Creates from explicit X and Y directions. Y is recomputed to be
    /// orthogonal to X, preserving the sense (sign of X cross Y).
    pub fn from_directions(origin: Pnt2d, vx: Dir2d, vy: Dir2d) -> Self {
        let cross = vx.x * vy.y - vx.y * vy.x;
        let vy = if cross >= 0.0 {
            Unit::new_unchecked(Vector2::new(-vx.y, vx.x))
        } else {
            Unit::new_unchecked(Vector2::new(vx.y, -vx.x))
        };
        Self { origin, vx, vy }
    }

    pub fn origin(&self) -> &Pnt2d {
        &self.origin
    }

    pub fn x_direction(&self) -> &Dir2d {
        &self.vx
    }

    pub fn y_direction(&self) -> &Dir2d {
        &self.vy
    }

    /// Returns true if the coordinate system is right-handed (X cross Y > 0).
    pub fn is_direct(&self) -> bool {
        self.vx.x * self.vy.y - self.vx.y * self.vy.x > 0.0
    }

    pub fn set_origin(&mut self, p: Pnt2d) {
        self.origin = p;
    }

    /// Sets the X direction. Y is recomputed, preserving handedness.
    pub fn set_x_direction(&mut self, vx: Dir2d) {
        let direct = self.is_direct();
        self.vx = vx;
        self.vy = if direct {
            Unit::new_unchecked(Vector2::new(-vx.y, vx.x))
        } else {
            Unit::new_unchecked(Vector2::new(vx.y, -vx.x))
        };
    }

    /// Sets the Y direction. X is recomputed, preserving handedness.
    pub fn set_y_direction(&mut self, vy: Dir2d) {
        let direct = self.is_direct();
        self.vy = vy;
        self.vx = if direct {
            Unit::new_unchecked(Vector2::new(vy.y, -vy.x))
        } else {
            Unit::new_unchecked(Vector2::new(-vy.y, vy.x))
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Point2, Unit, Vector2};

    fn dir(x: f64, y: f64) -> Dir2d {
        Unit::new_normalize(Vector2::new(x, y))
    }

    #[test]
    fn default_is_direct() {
        let a = Ax22d::new(Point2::origin(), dir(1.0, 0.0), true);
        assert!(a.is_direct());
        assert!((a.vx.x - 1.0).abs() < 1e-15);
        assert!((a.vy.y - 1.0).abs() < 1e-15);
    }

    #[test]
    fn indirect() {
        let a = Ax22d::new(Point2::origin(), dir(1.0, 0.0), false);
        assert!(!a.is_direct());
        assert!((a.vy.y - (-1.0)).abs() < 1e-15);
    }

    #[test]
    fn set_x_preserves_handedness() {
        let mut a = Ax22d::new(Point2::origin(), dir(1.0, 0.0), true);
        a.set_x_direction(dir(0.0, 1.0));
        assert!(a.is_direct());
        assert!((a.vx.y - 1.0).abs() < 1e-15);
        // Y should be (-1, 0) for right-handed with X = (0, 1)
        assert!((a.vy.x - (-1.0)).abs() < 1e-15);
    }
}
