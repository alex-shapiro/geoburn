use super::ax1::Ax1;
use super::ax3::Ax3;
use super::direction;
use super::line::Line;
use super::{Dir, Pnt, RESOLUTION};

/// An infinite plane in 3D space.
///
/// Defined by an `Ax3` coordinate system: the origin and Z direction
/// define the plane, while X and Y define the parameterization.
#[derive(Debug, Clone, Copy)]
pub struct Plane {
    pos: Ax3,
}

impl Plane {
    pub fn new(origin: Pnt, normal: Dir) -> Self {
        Self {
            pos: Ax3::from_origin_z(origin, normal),
        }
    }

    pub fn from_ax3(ax: Ax3) -> Self {
        Self { pos: ax }
    }

    /// Create from the cartesian equation `a*x + b*y + c*z + d = 0`.
    ///
    /// Panics if `(a, b, c)` is zero.
    pub fn from_equation(a: f64, b: f64, c: f64, d: f64) -> Self {
        let norm = (a * a + b * b + c * c).sqrt();
        assert!(norm > RESOLUTION, "Plane::from_equation() - null normal");
        let normal = nalgebra::Unit::new_normalize(nalgebra::Vector3::new(a, b, c));
        let origin = Pnt::from(normal.as_ref() * (-d / norm));
        Self::new(origin, normal)
    }

    pub fn position(&self) -> &Ax3 {
        &self.pos
    }

    pub fn origin(&self) -> &Pnt {
        self.pos.origin()
    }

    pub fn normal(&self) -> &Dir {
        self.pos.z_direction()
    }

    pub fn x_axis(&self) -> Ax1 {
        Ax1::new(*self.pos.origin(), *self.pos.x_direction())
    }

    pub fn y_axis(&self) -> Ax1 {
        Ax1::new(*self.pos.origin(), *self.pos.y_direction())
    }

    pub fn is_direct(&self) -> bool {
        self.pos.is_direct()
    }

    /// Cartesian coefficients `(a, b, c, d)` such that `a*x + b*y + c*z + d = 0`.
    pub fn coefficients(&self) -> (f64, f64, f64, f64) {
        let n = self.pos.z_direction();
        let (a, b, c) = if self.pos.is_direct() {
            (n.x, n.y, n.z)
        } else {
            (-n.x, -n.y, -n.z)
        };
        let p = self.pos.origin();
        let d = -(a * p.x + b * p.y + c * p.z);
        (a, b, c, d)
    }

    /// Signed distance from a point to this plane.
    /// Positive = point is on the normal side.
    pub fn signed_distance_to_point(&self, p: &Pnt) -> f64 {
        let d = p - self.pos.origin();
        self.pos.z_direction().dot(&d)
    }

    /// Distance from a point to this plane.
    pub fn distance_to_point(&self, p: &Pnt) -> f64 {
        self.signed_distance_to_point(p).abs()
    }

    /// Signed distance from this plane to a line.
    /// Returns 0 if the line intersects the plane.
    pub fn signed_distance_to_line(&self, l: &Line) -> f64 {
        if !direction::is_normal(self.pos.z_direction(), l.direction(), RESOLUTION) {
            return 0.0;
        }
        self.signed_distance_to_point(l.origin())
    }

    /// Distance from this plane to a line.
    pub fn distance_to_line(&self, l: &Line) -> f64 {
        self.signed_distance_to_line(l).abs()
    }

    /// Signed distance between two parallel planes.
    /// Returns 0 if the planes intersect.
    pub fn signed_distance_to_plane(&self, other: &Plane) -> f64 {
        if !direction::is_parallel(self.pos.z_direction(), other.pos.z_direction(), RESOLUTION) {
            return 0.0;
        }
        self.signed_distance_to_point(other.pos.origin())
    }

    /// Distance between two planes.
    pub fn distance_to_plane(&self, other: &Plane) -> f64 {
        self.signed_distance_to_plane(other).abs()
    }

    /// Returns true if the point lies on this plane within `tolerance`.
    pub fn contains_point(&self, p: &Pnt, tolerance: f64) -> bool {
        self.distance_to_point(p) <= tolerance
    }

    /// Returns true if the line lies in this plane within the given tolerances.
    pub fn contains_line(&self, l: &Line, linear_tolerance: f64, angular_tolerance: f64) -> bool {
        self.contains_point(l.origin(), linear_tolerance)
            && direction::is_normal(self.pos.z_direction(), l.direction(), angular_tolerance)
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
    fn distance_to_point() {
        let pl = Plane::new(Point3::origin(), dir(0.0, 0.0, 1.0));
        assert!((pl.distance_to_point(&pnt(3.0, 4.0, 5.0)) - 5.0).abs() < 1e-15);
    }

    #[test]
    fn signed_distance_positive() {
        let pl = Plane::new(Point3::origin(), dir(0.0, 0.0, 1.0));
        assert!(pl.signed_distance_to_point(&pnt(0.0, 0.0, 3.0)) > 0.0);
    }

    #[test]
    fn signed_distance_negative() {
        let pl = Plane::new(Point3::origin(), dir(0.0, 0.0, 1.0));
        assert!(pl.signed_distance_to_point(&pnt(0.0, 0.0, -3.0)) < 0.0);
    }

    #[test]
    fn contains_point_in_plane() {
        let pl = Plane::new(Point3::origin(), dir(0.0, 0.0, 1.0));
        assert!(pl.contains_point(&pnt(100.0, 200.0, 0.0), 1e-10));
    }

    #[test]
    fn coefficients_xy_plane() {
        let pl = Plane::new(Point3::origin(), dir(0.0, 0.0, 1.0));
        let (a, _b, c, d) = pl.coefficients();
        // z = 0 plane: 0x + 0y + 1z + 0 = 0
        assert!(a.abs() < 1e-15);
        assert!((c - 1.0).abs() < 1e-15);
        assert!(d.abs() < 1e-15);
    }

    #[test]
    fn coefficients_offset_plane() {
        let pl = Plane::new(pnt(0.0, 0.0, 5.0), dir(0.0, 0.0, 1.0));
        let (a, _b, c, d) = pl.coefficients();
        // z = 5 plane: 0x + 0y + 1z - 5 = 0
        assert!((c - 1.0).abs() < 1e-15);
        assert!((d - (-5.0)).abs() < 1e-15);
        assert!(a.abs() < 1e-15);
    }

    #[test]
    fn distance_parallel_planes() {
        let p1 = Plane::new(Point3::origin(), dir(0.0, 0.0, 1.0));
        let p2 = Plane::new(pnt(0.0, 0.0, 7.0), dir(0.0, 0.0, 1.0));
        assert!((p1.distance_to_plane(&p2) - 7.0).abs() < 1e-14);
    }

    #[test]
    fn distance_intersecting_planes_is_zero() {
        let p1 = Plane::new(Point3::origin(), dir(0.0, 0.0, 1.0));
        let p2 = Plane::new(Point3::origin(), dir(1.0, 0.0, 0.0));
        assert!(p1.distance_to_plane(&p2).abs() < 1e-15);
    }
}
