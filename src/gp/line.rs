use nalgebra::Unit;

use super::ax1::Ax1;
use super::{Dir, Pnt, Vec3};

/// An infinite line in 3D space, defined by a point and a direction
#[derive(Debug, Clone)]
pub struct Line {
    pub pos: Ax1,
}

impl Line {
    pub fn new(origin: Pnt, dir: Dir) -> Self {
        Self {
            pos: Ax1::new(origin, dir),
        }
    }

    pub fn from_ax1(ax: Ax1) -> Self {
        Self { pos: ax }
    }

    pub fn origin(&self) -> &Pnt {
        &self.pos.origin
    }

    pub fn direction(&self) -> &Dir {
        &self.pos.dir
    }

    pub fn reversed(&self) -> Self {
        Self {
            pos: self.pos.reversed(),
        }
    }

    /// Distance from a point to this line.
    pub fn distance_to_point(&self, p: &Pnt) -> f64 {
        self.square_distance_to_point(p).sqrt()
    }

    /// Square distance from a point to this line.
    pub fn square_distance_to_point(&self, p: &Pnt) -> f64 {
        let v = p - self.pos.origin;
        let cross = v.cross(self.pos.dir.as_ref());
        cross.norm_squared()
    }

    /// Distance between two lines (minimum distance between any two points).
    pub fn distance_to_line(&self, other: &Line) -> f64 {
        let cross = self.pos.dir.cross(&*other.pos.dir);
        let norm = cross.norm();
        if norm < super::RESOLUTION {
            // Parallel: distance is the distance from other's origin to self
            self.distance_to_point(&other.pos.origin)
        } else {
            let d = other.pos.origin - self.pos.origin;
            d.dot(&cross).abs() / norm
        }
    }

    /// Returns true if the point lies on this line within `tolerance`.
    pub fn contains(&self, p: &Pnt, tolerance: f64) -> bool {
        self.distance_to_point(p) <= tolerance
    }

    /// Line through `p` perpendicular to `self`, in the plane containing
    /// `self` and `p`. Panics if `p` is on the line.
    pub fn normal(&self, p: &Pnt) -> Line {
        let v = p - self.pos.origin;
        // Project v perpendicular to self.dir: v - (v·d)*d, then normalize
        let d: &Vec3 = self.pos.dir.as_ref();
        let proj = v - d * v.dot(d);
        let dir = Unit::new_normalize(proj);
        Line::new(*p, dir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Point3, Unit, Vector3};

    fn dir(x: f64, y: f64, z: f64) -> Dir {
        Unit::new_normalize(Vector3::new(x, y, z))
    }

    fn pnt(x: f64, y: f64, z: f64) -> Pnt {
        Point3::new(x, y, z)
    }

    #[test]
    fn distance_to_point_on_line() {
        let l = Line::new(Point3::origin(), dir(1.0, 0.0, 0.0));
        assert!(l.distance_to_point(&pnt(5.0, 0.0, 0.0)) < 1e-15);
    }

    #[test]
    fn distance_to_point_off_line() {
        let l = Line::new(Point3::origin(), dir(1.0, 0.0, 0.0));
        assert!((l.distance_to_point(&pnt(0.0, 3.0, 4.0)) - 5.0).abs() < 1e-14);
    }

    #[test]
    fn distance_between_parallel_lines() {
        let l1 = Line::new(Point3::origin(), dir(1.0, 0.0, 0.0));
        let l2 = Line::new(pnt(0.0, 3.0, 0.0), dir(1.0, 0.0, 0.0));
        assert!((l1.distance_to_line(&l2) - 3.0).abs() < 1e-14);
    }

    #[test]
    fn distance_between_skew_lines() {
        // Z axis and a line along X offset by 2 in Y
        let l1 = Line::new(Point3::origin(), dir(0.0, 0.0, 1.0));
        let l2 = Line::new(pnt(0.0, 2.0, 0.0), dir(1.0, 0.0, 0.0));
        assert!((l1.distance_to_line(&l2) - 2.0).abs() < 1e-14);
    }

    #[test]
    fn contains() {
        let l = Line::new(Point3::origin(), dir(1.0, 0.0, 0.0));
        assert!(l.contains(&pnt(100.0, 0.0, 0.0), 1e-10));
        assert!(!l.contains(&pnt(0.0, 1.0, 0.0), 1e-10));
    }
}
