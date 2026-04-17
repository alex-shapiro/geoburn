pub mod basis;
pub mod curves;
pub mod degree;
pub mod insert;
pub mod knots;
pub mod surfaces;

// TODO: Add `cache` module for span-cached Horner evaluation.
// Precompute polynomial coefficients per knot span (O(p²) one-time),
// then evaluate via Horner's method (O(p) per point). This is the SOTA
// optimization for tessellation, where hundreds of points are evaluated
// on the same span. Add this when a tessellator exists to benchmark against.

/// Maximum supported B-spline degree.
pub const MAX_DEGREE: usize = 25;

/// Maximum order (degree + 1).
pub const MAX_ORDER: usize = MAX_DEGREE + 1;

/// Trait for types that can be used as B-spline control points.
///
/// Abstracts over point dimension (2D, 3D) and scalars (weights).
/// The only operations needed are zero, scalar multiply, addition,
/// and scalar division — the building blocks of de Boor evaluation.
pub trait ControlPoint: Copy {
    fn origin() -> Self;
    fn scaled(&self, s: f64) -> Self;
    fn add(&self, other: &Self) -> Self;
    fn div(&self, s: f64) -> Self;
}

impl ControlPoint for crate::gp::Pnt {
    fn origin() -> Self {
        nalgebra::Point3::origin()
    }
    fn scaled(&self, s: f64) -> Self {
        Self::from(self.coords * s)
    }
    fn add(&self, other: &Self) -> Self {
        Self::from(self.coords + other.coords)
    }
    fn div(&self, s: f64) -> Self {
        Self::from(self.coords / s)
    }
}

impl ControlPoint for crate::gp::Pnt2d {
    fn origin() -> Self {
        nalgebra::Point2::origin()
    }
    fn scaled(&self, s: f64) -> Self {
        Self::from(self.coords * s)
    }
    fn add(&self, other: &Self) -> Self {
        Self::from(self.coords + other.coords)
    }
    fn div(&self, s: f64) -> Self {
        Self::from(self.coords / s)
    }
}

/// Weights can be inserted/elevated using the same algorithms as points.
impl ControlPoint for f64 {
    fn origin() -> Self {
        0.0
    }
    fn scaled(&self, s: f64) -> Self {
        self * s
    }
    fn add(&self, other: &Self) -> Self {
        self + other
    }
    fn div(&self, s: f64) -> Self {
        self / s
    }
}
