use super::{Dir2d, Pnt2d, direction2d};

/// An axis in 2D space: a point and a direction. The 2D analog of `Ax1`.
#[derive(Debug, Clone, Copy)]
pub struct Ax2d {
    pub origin: Pnt2d,
    pub dir: Dir2d,
}

impl Ax2d {
    pub fn new(origin: Pnt2d, dir: Dir2d) -> Self {
        Self { origin, dir }
    }

    pub fn reversed(&self) -> Self {
        Self {
            origin: self.origin,
            dir: -self.dir,
        }
    }

    pub fn angle(&self, other: &Ax2d) -> f64 {
        direction2d::angle(&self.dir, &other.dir)
    }

    pub fn is_normal(&self, other: &Ax2d, tolerance: f64) -> bool {
        direction2d::is_normal(&self.dir, &other.dir, tolerance)
    }

    pub fn is_opposite(&self, other: &Ax2d, tolerance: f64) -> bool {
        direction2d::is_opposite(&self.dir, &other.dir, tolerance)
    }

    pub fn is_parallel(&self, other: &Ax2d, tolerance: f64) -> bool {
        direction2d::is_parallel(&self.dir, &other.dir, tolerance)
    }
}
