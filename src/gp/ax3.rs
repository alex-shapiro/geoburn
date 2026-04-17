use nalgebra::Unit;

use super::ax2::Ax2;
use super::{Dir, Pnt, Vec3, direction, precision};

/// A coordinate system in 3D space that can be right-handed or left-handed.
///
/// Unlike `Ax2`, which is always right-handed, `Ax3` preserves its handedness through mutations.
///
/// Right-handed: Z = X × Y
/// Left-handed:  Z = -(X × Y)
#[derive(Debug, Clone)]
pub struct Ax3 {
    origin: Pnt,
    vx: Dir,
    vy: Dir,
    vz: Dir,
}

impl Ax3 {
    /// Creates a right-handed coordinate system.
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
    /// X and Y are chosen automatically. Right-handed.
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

    pub fn from_ax2(ax2: &Ax2) -> Self {
        Self {
            origin: *ax2.origin(),
            vx: *ax2.x_direction(),
            vy: *ax2.y_direction(),
            vz: *ax2.z_direction(),
        }
    }

    /// Returns the equivalent right-handed `Ax2`.
    /// If this system is left-handed, the Z direction is reversed.
    pub fn to_ax2(&self) -> Ax2 {
        let z = if self.is_direct() { self.vz } else { -self.vz };
        Ax2::new(self.origin, z, self.vx)
    }

    /// Returns true if the coordinate system is right-handed.
    pub fn is_direct(&self) -> bool {
        self.vx.cross(&*self.vy).dot(&self.vz) > 0.0
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

    pub fn x_reverse(&mut self) {
        self.vx = -self.vx;
    }

    pub fn y_reverse(&mut self) {
        self.vy = -self.vy;
    }

    pub fn z_reverse(&mut self) {
        self.vz = -self.vz;
    }

    /// Sets the main direction (Z). Recomputes X and Y while preserving handedness.
    pub fn set_z_direction(&mut self, z: Dir) {
        let dot = z.dot(&self.vx);
        if (dot.abs() - 1.0).abs() <= precision::ANGULAR {
            if dot > 0.0 {
                self.vx = self.vy;
                self.vy = self.vz;
            } else {
                self.vx = self.vz;
            }
            self.vz = z;
        } else {
            let direct = self.is_direct();
            self.vz = z;
            self.vx = cross_cross(&z, &self.vx, &z);
            if direct {
                self.vy = Unit::new_normalize(z.cross(&self.vx));
            } else {
                self.vy = Unit::new_normalize(self.vx.cross(&*z));
            }
        }
    }

    /// Sets the X direction. Z stays fixed, Y is recomputed preserving handedness.
    pub fn set_x_direction(&mut self, ref_x: Dir) {
        let dot = ref_x.dot(&self.vz);
        if (dot.abs() - 1.0).abs() <= precision::ANGULAR {
            if dot > 0.0 {
                self.vz = self.vx;
                self.vy = -self.vy;
            } else {
                self.vz = self.vx;
            }
            self.vx = ref_x;
        } else {
            let direct = self.is_direct();
            self.vx = cross_cross(&self.vz, &ref_x, &self.vz);
            if direct {
                self.vy = Unit::new_normalize(self.vz.cross(&self.vx));
            } else {
                self.vy = Unit::new_normalize(self.vx.cross(&*self.vz));
            }
        }
    }

    /// Sets the Y direction. Z stays fixed, X is recomputed preserving handedness.
    pub fn set_y_direction(&mut self, ref_y: Dir) {
        let dot = ref_y.dot(&self.vz);
        if (dot.abs() - 1.0).abs() <= precision::ANGULAR {
            if dot > 0.0 {
                self.vz = self.vy;
                self.vx = -self.vx;
            } else {
                self.vz = self.vy;
            }
            self.vy = ref_y;
        } else {
            let direct = self.is_direct();
            self.vx = Unit::new_normalize(ref_y.cross(&*self.vz));
            self.vy = Unit::new_normalize(self.vz.cross(&self.vx));
            if !direct {
                self.vx = -self.vx;
            }
        }
    }

    pub fn angle(&self, other: &Ax3) -> f64 {
        direction::angle(&self.vz, &other.vz)
    }

    /// Returns true if two coordinate systems are coplanar.
    pub fn is_coplanar(&self, other: &Ax3, linear_tolerance: f64, angular_tolerance: f64) -> bool {
        let d = other.origin - self.origin;
        let d1 = self.vz.dot(&d).abs();
        let d2 = other.vz.dot(&d).abs();
        d1 <= linear_tolerance
            && d2 <= linear_tolerance
            && direction::is_parallel(&self.vz, &other.vz, angular_tolerance)
    }
}

fn cross_cross(a: &Dir, b: &Dir, c: &Dir) -> Dir {
    let bc = b.cross(&**c);
    Unit::new_normalize(a.cross(&bc))
}

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

    fn dir(x: f64, y: f64, z: f64) -> Dir {
        Unit::new_normalize(Vector3::new(x, y, z))
    }

    fn standard() -> Ax3 {
        Ax3::new(Point3::origin(), dir(0.0, 0.0, 1.0), dir(1.0, 0.0, 0.0))
    }

    #[test]
    fn default_is_direct() {
        assert!(standard().is_direct());
    }

    #[test]
    fn z_reverse_makes_indirect() {
        let mut f = standard();
        f.z_reverse();
        assert!(!f.is_direct());
    }

    #[test]
    fn to_ax2_direct() {
        let f = standard();
        let a2 = f.to_ax2();
        assert!((a2.x_direction().x - 1.0).abs() < 1e-15);
        assert!((a2.z_direction().z - 1.0).abs() < 1e-15);
    }

    #[test]
    fn to_ax2_indirect_flips_z() {
        let mut f = standard();
        f.z_reverse();
        let a2 = f.to_ax2();
        // Ax2 is always right-handed, so Z should be flipped back
        assert!((a2.z_direction().z - 1.0).abs() < 1e-15);
    }

    #[test]
    fn set_z_preserves_handedness_direct() {
        let mut f = standard();
        assert!(f.is_direct());
        f.set_z_direction(dir(0.0, 1.0, 0.0));
        assert!(f.is_direct());
        let cross = f.vx.cross(&*f.vy);
        assert!(cross.dot(&f.vz) > 0.0);
    }

    #[test]
    fn set_z_preserves_handedness_indirect() {
        let mut f = standard();
        f.z_reverse();
        assert!(!f.is_direct());
        f.set_z_direction(dir(0.0, 1.0, 0.0));
        assert!(!f.is_direct());
    }

    #[test]
    fn set_x_preserves_handedness() {
        let mut f = standard();
        f.z_reverse();
        assert!(!f.is_direct());
        f.set_x_direction(dir(0.0, 1.0, 0.0));
        assert!(!f.is_direct());
    }

    #[test]
    fn set_y_preserves_handedness() {
        let mut f = standard();
        f.z_reverse();
        assert!(!f.is_direct());
        f.set_y_direction(dir(1.0, 0.0, 0.0));
        assert!(!f.is_direct());
    }

    #[test]
    fn from_ax2_roundtrip() {
        let a2 = Ax2::new(
            Point3::new(1.0, 2.0, 3.0),
            dir(0.0, 0.0, 1.0),
            dir(1.0, 0.0, 0.0),
        );
        let a3 = Ax3::from_ax2(&a2);
        let a2_back = a3.to_ax2();
        assert!((a2.x_direction().as_ref() - a2_back.x_direction().as_ref()).norm() < 1e-15);
        assert!((a2.y_direction().as_ref() - a2_back.y_direction().as_ref()).norm() < 1e-15);
        assert!((a2.z_direction().as_ref() - a2_back.z_direction().as_ref()).norm() < 1e-15);
    }
}
