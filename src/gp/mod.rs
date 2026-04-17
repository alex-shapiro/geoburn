use nalgebra::{Matrix2, Point2, Point3, Unit, Vector2, Vector3};

pub mod affine_transform;
pub mod affine_transform2d;
pub mod ax1;
pub mod ax2;
pub mod ax22d;
pub mod ax2d;
pub mod ax3;
pub mod circle;
pub mod circle2d;
pub mod cone;
pub mod cylinder;
pub mod direction;
pub mod direction2d;
pub mod ellipse;
pub mod ellipse2d;
pub mod euler_sequence;
pub mod hyperbola;
pub mod hyperbola2d;
pub mod line;
pub mod line2d;
pub mod parabola;
pub mod parabola2d;
pub mod plane;
pub mod precision;
pub mod sphere;
pub mod torus;
pub mod transform;
pub mod transform2d;
pub mod transform_lerp;

// 3D type aliases
pub type Dir = Unit<Vector3<f64>>;
pub type Pnt = Point3<f64>;
pub type Vec3 = Vector3<f64>;

// 2D type aliases
pub type Dir2d = Unit<Vector2<f64>>;
pub type Pnt2d = Point2<f64>;
pub type Vec2d = Vector2<f64>;
pub type Mat2d = Matrix2<f64>;

/// The smallest positive f64. Used as the zero-magnitude guard.
pub const RESOLUTION: f64 = f64::MIN_POSITIVE;
