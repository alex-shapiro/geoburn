use crate::gp::{Pnt, Vec3};

/// An oriented bounding box in 3D space.
///
/// Defined by a center, three orthogonal axes, and half-extents along each.
/// Uses the separating axis theorem (SAT) for intersection tests.
#[derive(Debug, Clone)]
pub struct Obb {
    center: Vec3,
    axes: [Vec3; 3],
    half_dims: [f64; 3],
}

impl Obb {
    /// Create an OBB from center, axes, and half-extents.
    pub fn new(center: Pnt, axes: [Vec3; 3], half_dims: [f64; 3]) -> Self {
        Self {
            center: center.coords,
            axes,
            half_dims,
        }
    }

    /// Create an axis-aligned OBB from min/max corners.
    pub fn from_aabb(min: Pnt, max: Pnt) -> Self {
        let center = (min.coords + max.coords) * 0.5;
        let half = (max.coords - min.coords) * 0.5;
        Self {
            center,
            axes: [
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
            ],
            half_dims: [half.x, half.y, half.z],
        }
    }

    pub fn center(&self) -> Pnt {
        Pnt::from(self.center)
    }

    pub fn axes(&self) -> &[Vec3; 3] {
        &self.axes
    }

    pub fn half_dims(&self) -> &[f64; 3] {
        &self.half_dims
    }

    /// Get the 8 corner vertices.
    pub fn vertices(&self) -> [Pnt; 8] {
        let mut result = [Pnt::origin(); 8];
        for (i, vertex) in result.iter_mut().enumerate() {
            let sx = if i & 1 == 0 { 1.0 } else { -1.0 };
            let sy = if i & 2 == 0 { 1.0 } else { -1.0 };
            let sz = if i & 4 == 0 { 1.0 } else { -1.0 };
            *vertex = Pnt::from(
                self.center
                    + self.axes[0] * (sx * self.half_dims[0])
                    + self.axes[1] * (sy * self.half_dims[1])
                    + self.axes[2] * (sz * self.half_dims[2]),
            );
        }
        result
    }

    /// Check if a point is outside using projection onto each axis.
    pub fn is_out_point(&self, p: &Pnt) -> bool {
        let d = p.coords - self.center;
        for i in 0..3 {
            if d.dot(&self.axes[i]).abs() > self.half_dims[i] {
                return true;
            }
        }
        false
    }

    /// Check if another OBB is completely outside using the separating axis theorem.
    /// Tests 15 axes: 3 from self, 3 from other, 9 cross products.
    pub fn is_out(&self, other: &Obb) -> bool {
        let d = other.center - self.center;

        // Test 6 face normals (3 from each box)
        for i in 0..3 {
            if separated_on_axis(&d, &self.axes[i], self, other) {
                return true;
            }
        }
        for i in 0..3 {
            if separated_on_axis(&d, &other.axes[i], self, other) {
                return true;
            }
        }

        // Test 9 edge cross products
        for i in 0..3 {
            for j in 0..3 {
                let axis = self.axes[i].cross(&other.axes[j]);
                if axis.norm_squared() < 1e-30 {
                    continue; // parallel axes, skip
                }
                if separated_on_axis(&d, &axis, self, other) {
                    return true;
                }
            }
        }

        false
    }
}

/// Check if two OBBs are separated along a given axis.
fn separated_on_axis(d: &Vec3, axis: &Vec3, a: &Obb, b: &Obb) -> bool {
    let dist = d.dot(axis).abs();
    let ra = a.half_dims[0] * a.axes[0].dot(axis).abs()
        + a.half_dims[1] * a.axes[1].dot(axis).abs()
        + a.half_dims[2] * a.axes[2].dot(axis).abs();
    let rb = b.half_dims[0] * b.axes[0].dot(axis).abs()
        + b.half_dims[1] * b.axes[1].dot(axis).abs()
        + b.half_dims[2] * b.axes[2].dot(axis).abs();
    dist > ra + rb
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Point3;

    fn pnt(x: f64, y: f64, z: f64) -> Pnt {
        Point3::new(x, y, z)
    }

    #[test]
    fn aabb_contains_center() {
        let b = Obb::from_aabb(pnt(0.0, 0.0, 0.0), pnt(2.0, 2.0, 2.0));
        assert!(!b.is_out_point(&pnt(1.0, 1.0, 1.0)));
    }

    #[test]
    fn aabb_excludes_outside() {
        let b = Obb::from_aabb(pnt(0.0, 0.0, 0.0), pnt(2.0, 2.0, 2.0));
        assert!(b.is_out_point(&pnt(3.0, 1.0, 1.0)));
    }

    #[test]
    fn overlapping_obbs() {
        let a = Obb::from_aabb(pnt(0.0, 0.0, 0.0), pnt(2.0, 2.0, 2.0));
        let b = Obb::from_aabb(pnt(1.0, 1.0, 1.0), pnt(3.0, 3.0, 3.0));
        assert!(!a.is_out(&b));
    }

    #[test]
    fn separated_obbs() {
        let a = Obb::from_aabb(pnt(0.0, 0.0, 0.0), pnt(1.0, 1.0, 1.0));
        let b = Obb::from_aabb(pnt(5.0, 5.0, 5.0), pnt(6.0, 6.0, 6.0));
        assert!(a.is_out(&b));
    }

    #[test]
    fn vertices_count() {
        let b = Obb::from_aabb(pnt(0.0, 0.0, 0.0), pnt(2.0, 2.0, 2.0));
        assert_eq!(b.vertices().len(), 8);
    }
}
