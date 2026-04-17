/// A 1D interval [min, max].
#[derive(Debug, Clone, Copy)]
pub struct Range {
    first: f64,
    last: f64,
}

impl Range {
    /// Creates a void range.
    pub fn void() -> Self {
        Self {
            first: 1.0,
            last: -1.0,
        }
    }

    /// Creates a range from bounds.
    pub fn new(first: f64, last: f64) -> Self {
        Self { first, last }
    }

    pub fn is_void(&self) -> bool {
        self.last < self.first
    }

    pub fn min(&self) -> Option<f64> {
        if self.is_void() {
            None
        } else {
            Some(self.first)
        }
    }

    pub fn max(&self) -> Option<f64> {
        if self.is_void() {
            None
        } else {
            Some(self.last)
        }
    }

    pub fn bounds(&self) -> Option<(f64, f64)> {
        if self.is_void() {
            None
        } else {
            Some((self.first, self.last))
        }
    }

    pub fn delta(&self) -> f64 {
        if self.is_void() {
            0.0
        } else {
            self.last - self.first
        }
    }

    /// Extend to include a value.
    pub fn add(&mut self, value: f64) {
        if self.is_void() {
            self.first = value;
            self.last = value;
        } else {
            if value < self.first {
                self.first = value;
            }
            if value > self.last {
                self.last = value;
            }
        }
    }

    /// Extend to include another range.
    pub fn add_range(&mut self, other: &Range) {
        if other.is_void() {
            return;
        }
        self.add(other.first);
        self.add(other.last);
    }

    /// Intersection with another range. Returns void if disjoint.
    pub fn common(&self, other: &Range) -> Range {
        if self.is_void() || other.is_void() {
            return Range::void();
        }
        let first = self.first.max(other.first);
        let last = self.last.min(other.last);
        Range { first, last }
    }

    /// Union of two overlapping ranges. Returns `None` if they don't overlap.
    pub fn union(&self, other: &Range) -> Option<Range> {
        if self.is_void() {
            return if other.is_void() { None } else { Some(*other) };
        }
        if other.is_void() {
            return Some(*self);
        }
        if self.last < other.first || other.last < self.first {
            return None;
        }
        Some(Range::new(
            self.first.min(other.first),
            self.last.max(other.last),
        ))
    }

    pub fn contains(&self, value: f64) -> bool {
        !self.is_void() && value >= self.first && value <= self.last
    }

    pub fn shift(&mut self, offset: f64) {
        self.first += offset;
        self.last += offset;
    }

    pub fn enlarge(&mut self, delta: f64) {
        self.first -= delta;
        self.last += delta;
    }

    pub fn trim_from(&mut self, value: f64) {
        if !self.is_void() && value > self.first {
            self.first = value;
        }
    }

    pub fn trim_to(&mut self, value: f64) {
        if !self.is_void() && value < self.last {
            self.last = value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn void_range() {
        let r = Range::void();
        assert!(r.is_void());
        assert!(r.bounds().is_none());
    }

    #[test]
    fn add_builds_range() {
        let mut r = Range::void();
        r.add(5.0);
        r.add(2.0);
        r.add(8.0);
        assert_eq!(r.bounds(), Some((2.0, 8.0)));
    }

    #[test]
    fn common_overlap() {
        let a = Range::new(0.0, 5.0);
        let b = Range::new(3.0, 8.0);
        let c = a.common(&b);
        assert_eq!(c.bounds(), Some((3.0, 5.0)));
    }

    #[test]
    fn common_disjoint_is_void() {
        let a = Range::new(0.0, 2.0);
        let b = Range::new(3.0, 5.0);
        assert!(a.common(&b).is_void());
    }

    #[test]
    fn union_overlap() {
        let a = Range::new(0.0, 5.0);
        let b = Range::new(3.0, 8.0);
        let u = a.union(&b).unwrap();
        assert_eq!(u.bounds(), Some((0.0, 8.0)));
    }

    #[test]
    fn union_disjoint_is_none() {
        let a = Range::new(0.0, 2.0);
        let b = Range::new(3.0, 5.0);
        assert!(a.union(&b).is_none());
    }

    #[test]
    fn contains() {
        let r = Range::new(1.0, 5.0);
        assert!(r.contains(3.0));
        assert!(!r.contains(0.0));
        assert!(!r.contains(6.0));
    }
}
