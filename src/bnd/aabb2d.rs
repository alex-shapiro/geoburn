use crate::gp::transform::TransformKind;
use crate::gp::transform2d::Transform2d;
use crate::gp::{Dir2d, Pnt2d, Vec2d, precision};

const INFINITE: f64 = precision::INFINITE;

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct BoxFlags2d: u8 {
        const VOID      = 0b0001_0000;
        const OPEN_XMIN = 0b0000_0001;
        const OPEN_XMAX = 0b0000_0010;
        const OPEN_YMIN = 0b0000_0100;
        const OPEN_YMAX = 0b0000_1000;
        const WHOLE = Self::OPEN_XMIN.bits() | Self::OPEN_XMAX.bits()
                    | Self::OPEN_YMIN.bits() | Self::OPEN_YMAX.bits();
    }
}

/// An axis-aligned bounding box in 2D space.
#[derive(Debug, Clone, Copy)]
pub struct Aabb2d {
    xmin: f64,
    xmax: f64,
    ymin: f64,
    ymax: f64,
    gap: f64,
    flags: BoxFlags2d,
}

impl Aabb2d {
    pub fn void() -> Self {
        Self {
            xmin: f64::MAX,
            xmax: f64::MIN,
            ymin: f64::MAX,
            ymax: f64::MIN,
            gap: 0.0,
            flags: BoxFlags2d::VOID,
        }
    }

    pub fn new(min: Pnt2d, max: Pnt2d) -> Self {
        Self {
            xmin: min.x,
            xmax: max.x,
            ymin: min.y,
            ymax: max.y,
            gap: 0.0,
            flags: BoxFlags2d::empty(),
        }
    }

    pub fn set_whole(&mut self) {
        self.flags = BoxFlags2d::WHOLE;
    }

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

    pub fn enlarge(&mut self, tol: f64) {
        self.gap = self.gap.max(tol.abs());
    }

    // -- Flags --

    pub fn is_void(&self) -> bool {
        self.flags.contains(BoxFlags2d::VOID)
    }

    pub fn is_whole(&self) -> bool {
        self.flags.contains(BoxFlags2d::WHOLE)
    }

    pub fn is_open(&self) -> bool {
        self.flags.intersects(BoxFlags2d::WHOLE)
    }

    pub fn is_open_xmin(&self) -> bool {
        self.flags.contains(BoxFlags2d::OPEN_XMIN)
    }
    pub fn is_open_xmax(&self) -> bool {
        self.flags.contains(BoxFlags2d::OPEN_XMAX)
    }
    pub fn is_open_ymin(&self) -> bool {
        self.flags.contains(BoxFlags2d::OPEN_YMIN)
    }
    pub fn is_open_ymax(&self) -> bool {
        self.flags.contains(BoxFlags2d::OPEN_YMAX)
    }

    pub fn open_xmin(&mut self) {
        self.flags.insert(BoxFlags2d::OPEN_XMIN);
    }
    pub fn open_xmax(&mut self) {
        self.flags.insert(BoxFlags2d::OPEN_XMAX);
    }
    pub fn open_ymin(&mut self) {
        self.flags.insert(BoxFlags2d::OPEN_YMIN);
    }
    pub fn open_ymax(&mut self) {
        self.flags.insert(BoxFlags2d::OPEN_YMAX);
    }

    // -- Effective bounds --

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

    pub fn get(&self) -> Option<(f64, f64, f64, f64)> {
        if self.is_void() {
            return None;
        }
        Some((
            self.effective_xmin(),
            self.effective_ymin(),
            self.effective_xmax(),
            self.effective_ymax(),
        ))
    }

    pub fn bounds(&self) -> Option<(Pnt2d, Pnt2d)> {
        let (xn, yn, xx, yx) = self.get()?;
        Some((Pnt2d::new(xn, yn), Pnt2d::new(xx, yx)))
    }

    pub fn center(&self) -> Option<Pnt2d> {
        let (xn, yn, xx, yx) = self.get()?;
        Some(Pnt2d::new((xn + xx) * 0.5, (yn + yx) * 0.5))
    }

    fn has_finite_part(&self) -> bool {
        !self.is_void() && !self.is_whole()
    }

    // -- Adding geometry --

    pub fn add_point(&mut self, p: &Pnt2d) {
        if self.is_void() {
            self.xmin = p.x;
            self.xmax = p.x;
            self.ymin = p.y;
            self.ymax = p.y;
            self.flags.remove(BoxFlags2d::VOID);
        } else {
            self.xmin = self.xmin.min(p.x);
            self.xmax = self.xmax.max(p.x);
            self.ymin = self.ymin.min(p.y);
            self.ymax = self.ymax.max(p.y);
        }
    }

    pub fn add_dir(&mut self, d: &Dir2d) {
        if d.x > 0.0 {
            self.open_xmax();
        } else if d.x < 0.0 {
            self.open_xmin();
        }
        if d.y > 0.0 {
            self.open_ymax();
        } else if d.y < 0.0 {
            self.open_ymin();
        }
    }

    pub fn add(&mut self, other: &Aabb2d) {
        if other.is_void() {
            return;
        }
        if self.is_void() {
            *self = *other;
            return;
        }
        self.xmin = self.xmin.min(other.xmin);
        self.xmax = self.xmax.max(other.xmax);
        self.ymin = self.ymin.min(other.ymin);
        self.ymax = self.ymax.max(other.ymax);
        self.flags |= other.flags & BoxFlags2d::WHOLE;
        self.gap = self.gap.max(other.gap);
    }

    // -- Thinness --

    pub fn is_x_thin(&self, tol: f64) -> bool {
        !self.is_open_xmin() && !self.is_open_xmax() && (self.xmax - self.xmin) < tol
    }

    pub fn is_y_thin(&self, tol: f64) -> bool {
        !self.is_open_ymin() && !self.is_open_ymax() && (self.ymax - self.ymin) < tol
    }

    pub fn is_thin(&self, tol: f64) -> bool {
        self.is_x_thin(tol) && self.is_y_thin(tol)
    }

    // -- Intersection tests --

    pub fn is_out_point(&self, p: &Pnt2d) -> bool {
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
    }

    pub fn is_out(&self, other: &Aabb2d) -> bool {
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
    }

    pub fn distance(&self, other: &Aabb2d) -> f64 {
        self.square_distance(other).sqrt()
    }

    pub fn square_distance(&self, other: &Aabb2d) -> f64 {
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
        dx * dx + dy * dy
    }

    pub fn diagonal(&self) -> f64 {
        if self.is_void() {
            return 0.0;
        }
        let dx = self.xmax - self.xmin;
        let dy = self.ymax - self.ymin;
        (dx * dx + dy * dy).sqrt()
    }

    // -- Transform --

    pub fn transformed(&self, t: &Transform2d) -> Aabb2d {
        if self.is_void() {
            return Aabb2d::void();
        }
        if t.kind() == TransformKind::Identity {
            return *self;
        }
        if t.kind() == TransformKind::Translation && self.has_finite_part() {
            let delta = t.translation_part();
            let mut result = *self;
            result.xmin += delta.x;
            result.xmax += delta.x;
            result.ymin += delta.y;
            result.ymax += delta.y;
            return result;
        }

        let mut result = Aabb2d::void();

        if self.has_finite_part() {
            let corners = [
                Pnt2d::new(self.xmin, self.ymin),
                Pnt2d::new(self.xmax, self.ymin),
                Pnt2d::new(self.xmin, self.ymax),
                Pnt2d::new(self.xmax, self.ymax),
            ];
            for corner in &corners {
                result.add_point(&t.transform_point(corner));
            }
        }

        result.gap = self.gap;

        if self.is_open() {
            let open_dirs: [(BoxFlags2d, Vec2d); 4] = [
                (BoxFlags2d::OPEN_XMIN, Vec2d::new(-1.0, 0.0)),
                (BoxFlags2d::OPEN_XMAX, Vec2d::new(1.0, 0.0)),
                (BoxFlags2d::OPEN_YMIN, Vec2d::new(0.0, -1.0)),
                (BoxFlags2d::OPEN_YMAX, Vec2d::new(0.0, 1.0)),
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
    use nalgebra::{Point2, Vector2};
    use std::f64::consts::FRAC_PI_2;

    fn pnt(x: f64, y: f64) -> Pnt2d {
        Point2::new(x, y)
    }

    #[test]
    fn void_box() {
        let b = Aabb2d::void();
        assert!(b.is_void());
        assert!(b.bounds().is_none());
    }

    #[test]
    fn add_points() {
        let mut b = Aabb2d::void();
        b.add_point(&pnt(1.0, 2.0));
        b.add_point(&pnt(-1.0, -2.0));
        let (mn, mx) = b.bounds().unwrap();
        assert_eq!(mn, pnt(-1.0, -2.0));
        assert_eq!(mx, pnt(1.0, 2.0));
    }

    #[test]
    fn gap_expands_bounds() {
        let mut b = Aabb2d::new(pnt(0.0, 0.0), pnt(1.0, 1.0));
        b.set_gap(0.5);
        let (mn, mx) = b.bounds().unwrap();
        assert_eq!(mn, pnt(-0.5, -0.5));
        assert_eq!(mx, pnt(1.5, 1.5));
    }

    #[test]
    fn gap_affects_is_out() {
        let mut b = Aabb2d::new(pnt(0.0, 0.0), pnt(1.0, 1.0));
        assert!(b.is_out_point(&pnt(1.3, 0.5)));
        b.set_gap(0.5);
        assert!(!b.is_out_point(&pnt(1.3, 0.5)));
    }

    #[test]
    fn open_direction() {
        let mut b = Aabb2d::new(pnt(0.0, 0.0), pnt(1.0, 1.0));
        b.open_xmax();
        assert!(!b.is_out_point(&pnt(1e50, 0.5)));
        assert!(b.is_out_point(&pnt(-1.0, 0.5)));
    }

    #[test]
    fn whole_space() {
        let mut b = Aabb2d::void();
        b.set_whole();
        assert!(!b.is_out_point(&pnt(1e99, -1e99)));
    }

    #[test]
    fn is_out_boxes() {
        let a = Aabb2d::new(pnt(0.0, 0.0), pnt(1.0, 1.0));
        let b = Aabb2d::new(pnt(2.0, 0.0), pnt(3.0, 1.0));
        assert!(a.is_out(&b));
        let c = Aabb2d::new(pnt(0.5, 0.5), pnt(1.5, 1.5));
        assert!(!a.is_out(&c));
    }

    #[test]
    fn distance_with_gap() {
        let mut a = Aabb2d::new(pnt(0.0, 0.0), pnt(1.0, 1.0));
        let b = Aabb2d::new(pnt(3.0, 0.0), pnt(4.0, 1.0));
        assert!((a.distance(&b) - 2.0).abs() < 1e-15);
        a.set_gap(0.5);
        assert!((a.distance(&b) - 1.5).abs() < 1e-15);
    }

    #[test]
    fn merge_preserves_open() {
        let mut a = Aabb2d::new(pnt(0.0, 0.0), pnt(1.0, 1.0));
        let mut b = Aabb2d::new(pnt(0.0, 0.0), pnt(1.0, 1.0));
        b.open_xmax();
        a.add(&b);
        assert!(a.is_open_xmax());
    }

    #[test]
    fn transformed_translation() {
        let b = Aabb2d::new(pnt(0.0, 0.0), pnt(1.0, 1.0));
        let t = Transform2d::translation(&Vector2::new(10.0, 0.0));
        let b2 = b.transformed(&t);
        let (mn, mx) = b2.bounds().unwrap();
        assert_eq!(mn, pnt(10.0, 0.0));
        assert_eq!(mx, pnt(11.0, 1.0));
    }

    #[test]
    fn transformed_rotation() {
        let b = Aabb2d::new(pnt(0.0, 0.0), pnt(1.0, 0.0));
        let t = Transform2d::rotation(&Point2::origin(), FRAC_PI_2);
        let b2 = b.transformed(&t);
        let (mn, mx) = b2.bounds().unwrap();
        assert!(mn.x.abs() < 1e-14);
        assert!((mx.y - 1.0).abs() < 1e-14);
    }

    #[test]
    fn transformed_preserves_gap() {
        let mut b = Aabb2d::new(pnt(0.0, 0.0), pnt(1.0, 1.0));
        b.set_gap(0.25);
        let t = Transform2d::translation(&Vector2::new(5.0, 0.0));
        let b2 = b.transformed(&t);
        assert!((b2.gap() - 0.25).abs() < 1e-15);
    }
}
