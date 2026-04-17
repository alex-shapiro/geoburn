use super::Pnt;
use super::ax1::Ax1;
use super::ax3::Ax3;

/// An infinite cone in 3D space.
///
/// Defined by a coordinate system (`Ax3`), a half-angle, and a reference
/// radius (the radius at the origin of the coordinate system).
/// The Z axis of the coordinate system is the cone's axis.
#[derive(Debug, Clone)]
pub struct Cone {
    pos: Ax3,
    radius: f64,
    semi_angle: f64,
}

impl Cone {
    /// Creates a cone. `semi_angle` is in radians, must be in (0, PI/2).
    pub fn new(pos: Ax3, semi_angle: f64, radius: f64) -> Self {
        assert!(
            semi_angle > 0.0 && semi_angle < std::f64::consts::FRAC_PI_2,
            "Cone: semi_angle must be in (0, PI/2)"
        );
        assert!(radius >= 0.0, "Cone: negative radius");
        Self {
            pos,
            radius,
            semi_angle,
        }
    }

    pub fn position(&self) -> &Ax3 {
        &self.pos
    }

    pub fn axis(&self) -> Ax1 {
        Ax1::new(*self.pos.origin(), *self.pos.z_direction())
    }

    pub fn location(&self) -> &Pnt {
        self.pos.origin()
    }

    pub fn ref_radius(&self) -> f64 {
        self.radius
    }

    pub fn semi_angle(&self) -> f64 {
        self.semi_angle
    }

    pub fn is_direct(&self) -> bool {
        self.pos.is_direct()
    }

    /// The apex of the cone: the point where radius is zero.
    pub fn apex(&self) -> Pnt {
        let offset = self.radius / self.semi_angle.tan();
        Pnt::from(self.pos.origin().coords - self.pos.z_direction().as_ref() * offset)
    }

    pub fn x_axis(&self) -> Ax1 {
        Ax1::new(*self.pos.origin(), *self.pos.x_direction())
    }

    pub fn y_axis(&self) -> Ax1 {
        Ax1::new(*self.pos.origin(), *self.pos.y_direction())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gp::Dir;
    use nalgebra::{Point3, Unit, Vector3};

    fn dir(x: f64, y: f64, z: f64) -> Dir {
        Unit::new_normalize(Vector3::new(x, y, z))
    }

    #[test]
    fn apex_45_degrees() {
        let c = Cone::new(
            Ax3::new(Point3::origin(), dir(0.0, 0.0, 1.0), dir(1.0, 0.0, 0.0)),
            std::f64::consts::FRAC_PI_4, // 45°
            5.0,
        );
        // tan(45°) = 1, offset = 5, apex at (0,0,-5)
        let apex = c.apex();
        assert!((apex.z - (-5.0)).abs() < 1e-14);
    }
}
