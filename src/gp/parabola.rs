use super::Pnt;
use super::ax1::Ax1;
use super::ax2::Ax2;
use super::line::Line;

/// A parabola in 3D space.
///
/// Defined by a coordinate system (`Ax2`) whose origin is the vertex,
/// and a focal length. The parabola lies in the XY plane with the axis
/// of symmetry along X, opening in the +X direction.
#[derive(Debug, Clone)]
pub struct Parabola {
    pos: Ax2,
    focal_length: f64,
}

impl Parabola {
    pub fn new(pos: Ax2, focal_length: f64) -> Self {
        assert!(focal_length >= 0.0, "Parabola: negative focal length");
        Self { pos, focal_length }
    }

    pub fn position(&self) -> &Ax2 {
        &self.pos
    }

    pub fn vertex(&self) -> &Pnt {
        self.pos.origin()
    }

    pub fn focal_length(&self) -> f64 {
        self.focal_length
    }

    pub fn axis(&self) -> Ax1 {
        Ax1::new(*self.pos.origin(), *self.pos.z_direction())
    }

    /// Focus: vertex + focal_length along X.
    pub fn focus(&self) -> Pnt {
        Pnt::from(self.pos.origin().coords + self.pos.x_direction().as_ref() * self.focal_length)
    }

    /// Directrix: line at -focal_length along X from vertex, perpendicular to X.
    pub fn directrix(&self) -> Line {
        let origin = Pnt::from(
            self.pos.origin().coords - self.pos.x_direction().as_ref() * self.focal_length,
        );
        Line::new(origin, *self.pos.y_direction())
    }

    /// Parameter (semi-latus rectum): 2 * focal_length.
    pub fn parameter(&self) -> f64 {
        2.0 * self.focal_length
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
    fn focus_and_directrix() {
        let p = Parabola::new(
            Ax2::new(Point3::origin(), dir(0.0, 0.0, 1.0), dir(1.0, 0.0, 0.0)),
            5.0,
        );
        assert!((p.focus().x - 5.0).abs() < 1e-15);
        assert!((p.directrix().origin().x - (-5.0)).abs() < 1e-15);
    }

    #[test]
    fn parameter() {
        let p = Parabola::new(
            Ax2::new(Point3::origin(), dir(0.0, 0.0, 1.0), dir(1.0, 0.0, 0.0)),
            3.0,
        );
        assert!((p.parameter() - 6.0).abs() < 1e-15);
    }
}
