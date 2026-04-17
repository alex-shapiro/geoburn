use super::Pnt;
use super::ax1::Ax1;
use super::ax3::Ax3;

/// An infinite cylinder in 3D space.
///
/// Defined by a coordinate system (`Ax3`) whose Z axis is the cylinder
/// axis, and a radius.
#[derive(Debug, Clone, Copy)]
pub struct Cylinder {
    pos: Ax3,
    radius: f64,
}

impl Cylinder {
    pub fn new(pos: Ax3, radius: f64) -> Self {
        assert!(radius >= 0.0, "Cylinder: negative radius");
        Self { pos, radius }
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

    pub fn radius(&self) -> f64 {
        self.radius
    }

    pub fn is_direct(&self) -> bool {
        self.pos.is_direct()
    }

    pub fn x_axis(&self) -> Ax1 {
        Ax1::new(*self.pos.origin(), *self.pos.x_direction())
    }

    pub fn y_axis(&self) -> Ax1 {
        Ax1::new(*self.pos.origin(), *self.pos.y_direction())
    }
}
