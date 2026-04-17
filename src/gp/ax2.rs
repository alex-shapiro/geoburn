use nalgebra::Unit;

use super::{Dir, Pnt, Vec3, direction, precision};

/// A right-handed coordinate system in 3D space.
///
/// Defined by an origin and three mutually orthogonal unit vectors (X, Y, Z) satisfying Z = X × Y.
///
/// When the main direction (Z) is changed, X and Y are recomputed.
/// When X or Y is changed, the other is recomputed but Z stays fixed.
#[derive(Debug, Clone, Copy)]
pub struct Ax2 {
    origin: Pnt,
    vx: Dir,
    vy: Dir,
    vz: Dir,
}

impl Ax2 {
    /// Creates a right-handed coordinate system from an origin, a Z direction,
    /// and a reference X direction. The actual X direction is computed as the
    /// component of `ref_x` perpendicular to `z`, and Y = Z × X.
    ///
    /// Panics if `z` and `ref_x` are parallel.
    pub fn new(origin: Pnt, z: Dir, ref_x: Dir) -> Self {
        let vx = cross_cross(&z, &ref_x, &z);
        let vy = Unit::new_normalize(z.cross(&vx));
        Self {
            origin,
            vx,
            vy,
            vz: z,
        }
    }

    /// Creates a coordinate system from an origin and Z direction.
    /// X and Y are chosen automatically.
    pub fn from_origin_z(origin: Pnt, z: Dir) -> Self {
        let ref_x = pick_perpendicular(&z);
        let vx = cross_cross(&z, &ref_x, &z);
        let vy = Unit::new_normalize(z.cross(&vx));
        Self {
            origin,
            vx,
            vy,
            vz: z,
        }
    }

    pub fn origin(&self) -> &Pnt {
        &self.origin
    }

    pub fn x_direction(&self) -> &Dir {
        &self.vx
    }

    pub fn y_direction(&self) -> &Dir {
        &self.vy
    }

    pub fn z_direction(&self) -> &Dir {
        &self.vz
    }

    pub fn set_origin(&mut self, p: Pnt) {
        self.origin = p;
    }

    /// Sets the main direction (Z). Recomputes X and Y.
    /// If the new Z is parallel to the current X, the axes are cycled
    /// rather than failing.
    pub fn set_z_direction(&mut self, z: Dir) {
        let dot = z.dot(&self.vx);
        if (dot.abs() - 1.0).abs() <= precision::ANGULAR {
            // Z is parallel to current X — cycle the axes
            if dot > 0.0 {
                self.vx = self.vy;
                self.vy = self.vz;
            } else {
                self.vx = self.vz;
            }
            self.vz = z;
        } else {
            self.vz = z;
            self.vx = cross_cross(&z, &self.vx, &z);
            self.vy = Unit::new_normalize(z.cross(&self.vx));
        }
    }

    /// Sets the X direction. Z stays fixed, Y is recomputed.
    /// The actual X is the component of `ref_x` perpendicular to Z.
    pub fn set_x_direction(&mut self, ref_x: Dir) {
        self.vx = cross_cross(&self.vz, &ref_x, &self.vz);
        self.vy = Unit::new_normalize(self.vz.cross(&self.vx));
    }

    /// Sets the Y direction. Z stays fixed, X is recomputed.
    pub fn set_y_direction(&mut self, ref_y: Dir) {
        self.vx = Unit::new_normalize(ref_y.cross(&*self.vz));
        self.vy = Unit::new_normalize(self.vz.cross(&self.vx));
    }

    pub fn angle(&self, other: &Ax2) -> f64 {
        direction::angle(&self.vz, &other.vz)
    }

    /// Returns true if two coordinate systems are coplanar: their Z axes
    /// are parallel and both origins lie in the same plane.
    pub fn is_coplanar(&self, other: &Ax2, linear_tolerance: f64, angular_tolerance: f64) -> bool {
        let d = other.origin - self.origin;
        let dist = self.vz.dot(&d).abs();
        dist <= linear_tolerance && direction::is_parallel(&self.vz, &other.vz, angular_tolerance)
    }
}

/// Computes the triple cross product: a × (b × c), normalized.
/// Used to project `b` perpendicular to the plane defined by `a` and `c`.
fn cross_cross(a: &Dir, b: &Dir, c: &Dir) -> Dir {
    let bc = b.cross(&**c);
    Unit::new_normalize(a.cross(&bc))
}

/// Picks an arbitrary direction perpendicular to `d`.
fn pick_perpendicular(d: &Dir) -> Dir {
    if d.x.abs() > d.z.abs() {
        Unit::new_normalize(Vec3::new(-d.y, d.x, 0.0))
    } else {
        Unit::new_normalize(Vec3::new(0.0, -d.z, d.y))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Point3, Unit, Vector3};
    use std::f64::consts::FRAC_PI_2;

    fn dir(x: f64, y: f64, z: f64) -> Dir {
        Unit::new_normalize(Vector3::new(x, y, z))
    }

    fn standard_frame() -> Ax2 {
        Ax2::new(Point3::origin(), dir(0.0, 0.0, 1.0), dir(1.0, 0.0, 0.0))
    }

    #[test]
    fn standard_frame_is_identity() {
        let f = standard_frame();
        assert!((f.vx.x - 1.0).abs() < 1e-15);
        assert!((f.vy.y - 1.0).abs() < 1e-15);
        assert!((f.vz.z - 1.0).abs() < 1e-15);
    }

    #[test]
    fn right_handed() {
        let f = standard_frame();
        let cross = f.vx.cross(&*f.vy);
        assert!((cross - *f.vz).norm() < 1e-15);
    }

    #[test]
    fn oblique_ref_x_is_orthogonalized() {
        let f = Ax2::new(
            Point3::origin(),
            dir(0.0, 0.0, 1.0),
            dir(1.0, 1.0, 1.0), // not perpendicular to Z
        );
        // X should be in the XY plane
        assert!(f.vx.z.abs() < 1e-15);
        // still right-handed
        let cross = f.vx.cross(&*f.vy);
        assert!((cross - *f.vz).norm() < 1e-15);
    }

    #[test]
    fn from_origin_z_is_right_handed() {
        let f = Ax2::from_origin_z(Point3::origin(), dir(1.0, 1.0, 1.0));
        let cross = f.vx.cross(&*f.vy);
        assert!((cross - *f.vz).norm() < 1e-10);
    }

    #[test]
    fn set_z_direction_recomputes() {
        let mut f = standard_frame();
        f.set_z_direction(dir(0.0, 1.0, 0.0));
        assert!((f.vz.y - 1.0).abs() < 1e-15);
        // still right-handed
        let cross = f.vx.cross(&*f.vy);
        assert!((cross - *f.vz).norm() < 1e-10);
    }

    #[test]
    fn set_x_direction_keeps_z() {
        let mut f = standard_frame();
        let old_z = *f.vz;
        f.set_x_direction(dir(0.0, 1.0, 0.0));
        assert!((*f.vz - old_z).norm() < 1e-15);
        // X should now be along Y
        assert!((f.vx.y - 1.0).abs() < 1e-15);
        // still right-handed
        let cross = f.vx.cross(&*f.vy);
        assert!((cross - *f.vz).norm() < 1e-10);
    }

    #[test]
    fn angle_between_frames() {
        let a = standard_frame();
        let b = Ax2::new(Point3::origin(), dir(1.0, 0.0, 0.0), dir(0.0, 1.0, 0.0));
        assert!((a.angle(&b) - FRAC_PI_2).abs() < 1e-15);
    }
}
