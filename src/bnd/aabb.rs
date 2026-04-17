use crate::gp::transform::{Transform, TransformKind};
use crate::gp::{Dir, Pnt, Vec3, precision};

const INFINITE: f64 = precision::INFINITE;

bitflags::bitflags! {
    /// Tracks void, open, and whole-space states for a bounding box.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct BoxFlags: u8 {
        const VOID     = 0b0100_0000;
        const OPEN_XMIN = 0b0000_0001;
        const OPEN_XMAX = 0b0000_0010;
        const OPEN_YMIN = 0b0000_0100;
        const OPEN_YMAX = 0b0000_1000;
        const OPEN_ZMIN = 0b0001_0000;
        const OPEN_ZMAX = 0b0010_0000;
        const WHOLE = Self::OPEN_XMIN.bits() | Self::OPEN_XMAX.bits()
                    | Self::OPEN_YMIN.bits() | Self::OPEN_YMAX.bits()
                    | Self::OPEN_ZMIN.bits() | Self::OPEN_ZMAX.bits();
    }
}

/// An axis-aligned bounding box in 3D space.
///
/// - **Gap**: a tolerance applied to both sides of all bounds during queries
/// - **Open directions**: the box can extend to infinity in any of the 6 directions
/// - **Transform**: correctly transforms corners and open directions
///
/// The raw bounds are stored without gap; `get()` and intersection tests
/// apply the gap automatically.
#[derive(Debug, Clone)]
pub struct Aabb {
    xmin: f64,
    xmax: f64,
    ymin: f64,
    ymax: f64,
    zmin: f64,
    zmax: f64,
    gap: f64,
    flags: BoxFlags,
}

impl Aabb {
    /// Creates an empty (void) bounding box.
    pub fn void() -> Self {
        Self {
            xmin: f64::MAX,
            xmax: f64::MIN,
            ymin: f64::MAX,
            ymax: f64::MIN,
            zmin: f64::MAX,
            zmax: f64::MIN,
            gap: 0.0,
            flags: BoxFlags::VOID,
        }
    }

    /// Creates a bounding box from min/max corners.
    pub fn new(min: Pnt, max: Pnt) -> Self {
        Self {
            xmin: min.x,
            xmax: max.x,
            ymin: min.y,
            ymax: max.y,
            zmin: min.z,
            zmax: max.z,
            gap: 0.0,
            flags: BoxFlags::empty(),
        }
    }

    /// Makes the box infinite in all directions.
    pub fn set_whole(&mut self) {
        self.flags = BoxFlags::WHOLE;
    }

    /// Makes the box empty.
    pub fn set_void(&mut self) {
        *self = Self::void();
    }

    // -- Gap --

    pub fn gap(&self) -> f64 {
        self.gap
    }

    pub fn set_gap(&mut self, tol: f64) {
        self.gap = tol.abs();
    }

    /// Enlarge the gap to be at least `tol`.
    pub fn enlarge(&mut self, tol: f64) {
        self.gap = self.gap.max(tol.abs());
    }

    // -- Flags --

    pub fn is_void(&self) -> bool {
        self.flags.contains(BoxFlags::VOID)
    }

    pub fn is_whole(&self) -> bool {
        self.flags.contains(BoxFlags::WHOLE)
    }

    pub fn is_open(&self) -> bool {
        self.flags.intersects(BoxFlags::WHOLE)
    }

    pub fn is_open_xmin(&self) -> bool {
        self.flags.contains(BoxFlags::OPEN_XMIN)
    }
    pub fn is_open_xmax(&self) -> bool {
        self.flags.contains(BoxFlags::OPEN_XMAX)
    }
    pub fn is_open_ymin(&self) -> bool {
        self.flags.contains(BoxFlags::OPEN_YMIN)
    }
    pub fn is_open_ymax(&self) -> bool {
        self.flags.contains(BoxFlags::OPEN_YMAX)
    }
    pub fn is_open_zmin(&self) -> bool {
        self.flags.contains(BoxFlags::OPEN_ZMIN)
    }
    pub fn is_open_zmax(&self) -> bool {
        self.flags.contains(BoxFlags::OPEN_ZMAX)
    }

    pub fn open_xmin(&mut self) {
        self.flags.insert(BoxFlags::OPEN_XMIN);
    }
    pub fn open_xmax(&mut self) {
        self.flags.insert(BoxFlags::OPEN_XMAX);
    }
    pub fn open_ymin(&mut self) {
        self.flags.insert(BoxFlags::OPEN_YMIN);
    }
    pub fn open_ymax(&mut self) {
        self.flags.insert(BoxFlags::OPEN_YMAX);
    }
    pub fn open_zmin(&mut self) {
        self.flags.insert(BoxFlags::OPEN_ZMIN);
    }
    pub fn open_zmax(&mut self) {
        self.flags.insert(BoxFlags::OPEN_ZMAX);
    }

    // -- Bounds (with gap and open direction handling) --

    fn effective_xmin(&self) -> f64 {
        if self.is_open_xmin() {
            -INFINITE
        } else {
            self.xmin - self.gap
        }
    }
    fn effective_xmax(&self) -> f64 {
        if self.is_open_xmax() {
            INFINITE
        } else {
            self.xmax + self.gap
        }
    }
    fn effective_ymin(&self) -> f64 {
        if self.is_open_ymin() {
            -INFINITE
        } else {
            self.ymin - self.gap
        }
    }
    fn effective_ymax(&self) -> f64 {
        if self.is_open_ymax() {
            INFINITE
        } else {
            self.ymax + self.gap
        }
    }
    fn effective_zmin(&self) -> f64 {
        if self.is_open_zmin() {
            -INFINITE
        } else {
            self.zmin - self.gap
        }
    }
    fn effective_zmax(&self) -> f64 {
        if self.is_open_zmax() {
            INFINITE
        } else {
            self.zmax + self.gap
        }
    }

    /// Returns the effective bounds including gap. Returns `None` if void.
    pub fn get(&self) -> Option<(f64, f64, f64, f64, f64, f64)> {
        if self.is_void() {
            return None;
        }
        Some((
            self.effective_xmin(),
            self.effective_ymin(),
            self.effective_zmin(),
            self.effective_xmax(),
            self.effective_ymax(),
            self.effective_zmax(),
        ))
    }

    /// Returns the corners including gap. Returns `None` if void.
    pub fn bounds(&self) -> Option<(Pnt, Pnt)> {
        let (xn, yn, zn, xx, yx, zx) = self.get()?;
        Some((Pnt::new(xn, yn, zn), Pnt::new(xx, yx, zx)))
    }

    /// Center of the bounding box (including gap). Returns `None` if void.
    pub fn center(&self) -> Option<Pnt> {
        let (xn, yn, zn, xx, yx, zx) = self.get()?;
        Some(Pnt::new((xn + xx) * 0.5, (yn + yx) * 0.5, (zn + zx) * 0.5))
    }

    /// Returns true if the box has finite bounds (not void, and at least
    /// one direction is not open on both sides).
    fn has_finite_part(&self) -> bool {
        !self.is_void() && !self.is_whole()
    }

    // -- Adding geometry --

    /// Add a point to the bounding box.
    pub fn add_point(&mut self, p: &Pnt) {
        if self.is_void() {
            self.xmin = p.x;
            self.xmax = p.x;
            self.ymin = p.y;
            self.ymax = p.y;
            self.zmin = p.z;
            self.zmax = p.z;
            self.flags.remove(BoxFlags::VOID);
        } else {
            self.xmin = self.xmin.min(p.x);
            self.xmax = self.xmax.max(p.x);
            self.ymin = self.ymin.min(p.y);
            self.ymax = self.ymax.max(p.y);
            self.zmin = self.zmin.min(p.z);
            self.zmax = self.zmax.max(p.z);
        }
    }

    /// Extend the box in a direction (makes it open in the appropriate directions).
    pub fn add_dir(&mut self, d: &Dir) {
        let v: &Vec3 = d.as_ref();
        if v.x > 0.0 {
            self.open_xmax();
        } else if v.x < 0.0 {
            self.open_xmin();
        }
        if v.y > 0.0 {
            self.open_ymax();
        } else if v.y < 0.0 {
            self.open_ymin();
        }
        if v.z > 0.0 {
            self.open_zmax();
        } else if v.z < 0.0 {
            self.open_zmin();
        }
    }

    /// Merge with another bounding box.
    pub fn add(&mut self, other: &Aabb) {
        if other.is_void() {
            return;
        }
        if self.is_void() {
            *self = other.clone();
            return;
        }
        self.xmin = self.xmin.min(other.xmin);
        self.xmax = self.xmax.max(other.xmax);
        self.ymin = self.ymin.min(other.ymin);
        self.ymax = self.ymax.max(other.ymax);
        self.zmin = self.zmin.min(other.zmin);
        self.zmax = self.zmax.max(other.zmax);
        self.flags |= other.flags & BoxFlags::WHOLE;
        self.gap = self.gap.max(other.gap);
    }

    // -- Thinness checks --

    pub fn is_x_thin(&self, tol: f64) -> bool {
        !self.is_open_xmin() && !self.is_open_xmax() && (self.xmax - self.xmin) < tol
    }

    pub fn is_y_thin(&self, tol: f64) -> bool {
        !self.is_open_ymin() && !self.is_open_ymax() && (self.ymax - self.ymin) < tol
    }

    pub fn is_z_thin(&self, tol: f64) -> bool {
        !self.is_open_zmin() && !self.is_open_zmax() && (self.zmax - self.zmin) < tol
    }

    pub fn is_thin(&self, tol: f64) -> bool {
        self.is_x_thin(tol) && self.is_y_thin(tol) && self.is_z_thin(tol)
    }

    // -- Intersection tests --

    /// Returns true if the point is outside the box (including gap).
    pub fn is_out_point(&self, p: &Pnt) -> bool {
        if self.is_whole() {
            return false;
        }
        if self.is_void() {
            return true;
        }
        p.x < self.effective_xmin()
            || p.x > self.effective_xmax()
            || p.y < self.effective_ymin()
            || p.y > self.effective_ymax()
            || p.z < self.effective_zmin()
            || p.z > self.effective_zmax()
    }

    /// Returns true if another box is completely outside this one (including gaps).
    pub fn is_out(&self, other: &Aabb) -> bool {
        if self.is_whole() || other.is_whole() {
            return false;
        }
        if self.is_void() || other.is_void() {
            return true;
        }
        self.effective_xmax() < other.effective_xmin()
            || self.effective_xmin() > other.effective_xmax()
            || self.effective_ymax() < other.effective_ymin()
            || self.effective_ymin() > other.effective_ymax()
            || self.effective_zmax() < other.effective_zmin()
            || self.effective_zmin() > other.effective_zmax()
    }

    /// Minimum distance between two boxes (accounting for gaps). Zero if they overlap.
    pub fn distance(&self, other: &Aabb) -> f64 {
        self.square_distance(other).sqrt()
    }

    /// Square of minimum distance between two boxes.
    pub fn square_distance(&self, other: &Aabb) -> f64 {
        if self.is_void() || other.is_void() {
            return f64::MAX;
        }
        let dx = axis_distance(
            self.effective_xmin(),
            self.effective_xmax(),
            other.effective_xmin(),
            other.effective_xmax(),
        );
        let dy = axis_distance(
            self.effective_ymin(),
            self.effective_ymax(),
            other.effective_ymin(),
            other.effective_ymax(),
        );
        let dz = axis_distance(
            self.effective_zmin(),
            self.effective_zmax(),
            other.effective_zmin(),
            other.effective_zmax(),
        );
        dx * dx + dy * dy + dz * dz
    }

    /// Diagonal extent (not including gap).
    pub fn diagonal(&self) -> f64 {
        if self.is_void() {
            return 0.0;
        }
        let dx = self.xmax - self.xmin;
        let dy = self.ymax - self.ymin;
        let dz = self.zmax - self.zmin;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    // -- Transform --

    /// Returns a new bounding box transformed by `t`.
    ///
    /// For identity/translation, applies directly. For general transforms,
    /// transforms all 8 corners and rebuilds the AABB. Open directions
    /// are transformed as direction vectors.
    pub fn transformed(&self, t: &Transform) -> Aabb {
        if self.is_void() {
            return Aabb::void();
        }
        if t.kind() == TransformKind::Identity {
            return self.clone();
        }
        if t.kind() == TransformKind::Translation && self.has_finite_part() {
            let delta = t.translation_part();
            let mut result = self.clone();
            result.xmin += delta.x;
            result.xmax += delta.x;
            result.ymin += delta.y;
            result.ymax += delta.y;
            result.zmin += delta.z;
            result.zmax += delta.z;
            return result;
        }

        let mut result = Aabb::void();

        // Transform finite corners
        if self.has_finite_part() {
            let corners = [
                Pnt::new(self.xmin, self.ymin, self.zmin),
                Pnt::new(self.xmax, self.ymin, self.zmin),
                Pnt::new(self.xmin, self.ymax, self.zmin),
                Pnt::new(self.xmax, self.ymax, self.zmin),
                Pnt::new(self.xmin, self.ymin, self.zmax),
                Pnt::new(self.xmax, self.ymin, self.zmax),
                Pnt::new(self.xmin, self.ymax, self.zmax),
                Pnt::new(self.xmax, self.ymax, self.zmax),
            ];
            for corner in &corners {
                result.add_point(&t.transform_point(corner));
            }
        }

        result.gap = self.gap;

        // Transform open directions
        if self.is_open() {
            let open_dirs: [(BoxFlags, Vec3); 6] = [
                (BoxFlags::OPEN_XMIN, Vec3::new(-1.0, 0.0, 0.0)),
                (BoxFlags::OPEN_XMAX, Vec3::new(1.0, 0.0, 0.0)),
                (BoxFlags::OPEN_YMIN, Vec3::new(0.0, -1.0, 0.0)),
                (BoxFlags::OPEN_YMAX, Vec3::new(0.0, 1.0, 0.0)),
                (BoxFlags::OPEN_ZMIN, Vec3::new(0.0, 0.0, -1.0)),
                (BoxFlags::OPEN_ZMAX, Vec3::new(0.0, 0.0, 1.0)),
            ];
            for (flag, dir_vec) in &open_dirs {
                if self.flags.contains(*flag) {
                    let transformed = t.transform_vector(dir_vec);
                    if let Some(d) = nalgebra::Unit::try_new(transformed, crate::gp::RESOLUTION) {
                        result.add_dir(&d);
                    }
                }
            }
        }

        result
    }
}

/// Gap between two 1D intervals. Zero if they overlap.
fn axis_distance(min1: f64, max1: f64, min2: f64, max2: f64) -> f64 {
    if max1 < min2 {
        min2 - max1
    } else if max2 < min1 {
        min1 - max2
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gp::ax1::Ax1;
    use nalgebra::{Point3, Unit, Vector3};
    use std::f64::consts::FRAC_PI_2;

    fn pnt(x: f64, y: f64, z: f64) -> Pnt {
        Point3::new(x, y, z)
    }

    fn dir(x: f64, y: f64, z: f64) -> Dir {
        Unit::new_normalize(Vector3::new(x, y, z))
    }

    #[test]
    fn void_box() {
        let b = Aabb::void();
        assert!(b.is_void());
        assert!(b.bounds().is_none());
    }

    #[test]
    fn add_points() {
        let mut b = Aabb::void();
        b.add_point(&pnt(1.0, 2.0, 3.0));
        b.add_point(&pnt(-1.0, -2.0, -3.0));
        let (mn, mx) = b.bounds().unwrap();
        assert_eq!(mn, pnt(-1.0, -2.0, -3.0));
        assert_eq!(mx, pnt(1.0, 2.0, 3.0));
    }

    #[test]
    fn gap_expands_bounds() {
        let mut b = Aabb::new(pnt(0.0, 0.0, 0.0), pnt(1.0, 1.0, 1.0));
        b.set_gap(0.5);
        let (mn, mx) = b.bounds().unwrap();
        assert_eq!(mn, pnt(-0.5, -0.5, -0.5));
        assert_eq!(mx, pnt(1.5, 1.5, 1.5));
    }

    #[test]
    fn gap_affects_is_out() {
        let mut b = Aabb::new(pnt(0.0, 0.0, 0.0), pnt(1.0, 1.0, 1.0));
        assert!(b.is_out_point(&pnt(1.3, 0.5, 0.5)));
        b.set_gap(0.5);
        assert!(!b.is_out_point(&pnt(1.3, 0.5, 0.5)));
    }

    #[test]
    fn open_direction() {
        let mut b = Aabb::new(pnt(0.0, 0.0, 0.0), pnt(1.0, 1.0, 1.0));
        b.open_xmax();
        assert!(!b.is_out_point(&pnt(1e50, 0.5, 0.5)));
        assert!(b.is_out_point(&pnt(-1.0, 0.5, 0.5))); // still bounded in -X
    }

    #[test]
    fn whole_space_contains_everything() {
        let mut b = Aabb::void();
        b.set_whole();
        assert!(!b.is_out_point(&pnt(1e99, -1e99, 42.0)));
    }

    #[test]
    fn add_dir_opens_direction() {
        let mut b = Aabb::new(pnt(0.0, 0.0, 0.0), pnt(1.0, 1.0, 1.0));
        b.add_dir(&dir(1.0, 0.0, 0.0));
        assert!(b.is_open_xmax());
        assert!(!b.is_open_xmin());
    }

    #[test]
    fn is_out_point() {
        let b = Aabb::new(pnt(0.0, 0.0, 0.0), pnt(1.0, 1.0, 1.0));
        assert!(!b.is_out_point(&pnt(0.5, 0.5, 0.5)));
        assert!(b.is_out_point(&pnt(2.0, 0.5, 0.5)));
    }

    #[test]
    fn is_out_boxes() {
        let a = Aabb::new(pnt(0.0, 0.0, 0.0), pnt(1.0, 1.0, 1.0));
        let b = Aabb::new(pnt(2.0, 0.0, 0.0), pnt(3.0, 1.0, 1.0));
        assert!(a.is_out(&b));
        let c = Aabb::new(pnt(0.5, 0.5, 0.5), pnt(1.5, 1.5, 1.5));
        assert!(!a.is_out(&c));
    }

    #[test]
    fn distance_separated() {
        let a = Aabb::new(pnt(0.0, 0.0, 0.0), pnt(1.0, 1.0, 1.0));
        let b = Aabb::new(pnt(3.0, 0.0, 0.0), pnt(4.0, 1.0, 1.0));
        assert!((a.distance(&b) - 2.0).abs() < 1e-15);
    }

    #[test]
    fn distance_with_gap() {
        let mut a = Aabb::new(pnt(0.0, 0.0, 0.0), pnt(1.0, 1.0, 1.0));
        let b = Aabb::new(pnt(3.0, 0.0, 0.0), pnt(4.0, 1.0, 1.0));
        a.set_gap(0.5);
        // gap on A shrinks the distance by 0.5
        assert!((a.distance(&b) - 1.5).abs() < 1e-15);
    }

    #[test]
    fn merge() {
        let mut a = Aabb::new(pnt(0.0, 0.0, 0.0), pnt(1.0, 1.0, 1.0));
        let b = Aabb::new(pnt(2.0, 2.0, 2.0), pnt(3.0, 3.0, 3.0));
        a.add(&b);
        let (mn, mx) = a.bounds().unwrap();
        assert_eq!(mn, pnt(0.0, 0.0, 0.0));
        assert_eq!(mx, pnt(3.0, 3.0, 3.0));
    }

    #[test]
    fn merge_preserves_open() {
        let mut a = Aabb::new(pnt(0.0, 0.0, 0.0), pnt(1.0, 1.0, 1.0));
        let mut b = Aabb::new(pnt(0.0, 0.0, 0.0), pnt(1.0, 1.0, 1.0));
        b.open_xmax();
        a.add(&b);
        assert!(a.is_open_xmax());
    }

    #[test]
    fn transformed_translation() {
        let b = Aabb::new(pnt(0.0, 0.0, 0.0), pnt(1.0, 1.0, 1.0));
        let t = Transform::translation(&Vector3::new(10.0, 0.0, 0.0));
        let b2 = b.transformed(&t);
        let (mn, mx) = b2.bounds().unwrap();
        assert_eq!(mn, pnt(10.0, 0.0, 0.0));
        assert_eq!(mx, pnt(11.0, 1.0, 1.0));
    }

    #[test]
    fn transformed_rotation() {
        let b = Aabb::new(pnt(0.0, 0.0, 0.0), pnt(1.0, 0.0, 0.0));
        let ax = Ax1::new(Point3::origin(), dir(0.0, 0.0, 1.0));
        let t = Transform::rotation(&ax, FRAC_PI_2);
        let b2 = b.transformed(&t);
        // (1,0,0) rotated 90° around Z → (0,1,0)
        // AABB of {(0,0,0), (0,1,0)} = [0,0,0] to [0,1,0]
        let (mn, mx) = b2.bounds().unwrap();
        assert!(mn.x.abs() < 1e-14);
        assert!((mx.y - 1.0).abs() < 1e-14);
    }

    #[test]
    fn transformed_preserves_gap() {
        let mut b = Aabb::new(pnt(0.0, 0.0, 0.0), pnt(1.0, 1.0, 1.0));
        b.set_gap(0.25);
        let t = Transform::translation(&Vector3::new(5.0, 0.0, 0.0));
        let b2 = b.transformed(&t);
        assert!((b2.gap() - 0.25).abs() < 1e-15);
    }

    #[test]
    fn is_thin() {
        let b = Aabb::new(pnt(0.0, 0.0, 0.0), pnt(0.001, 0.001, 0.001));
        assert!(b.is_thin(0.01));
        assert!(!b.is_thin(0.0001));
    }
}
