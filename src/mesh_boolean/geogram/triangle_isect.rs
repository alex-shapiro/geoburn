//! Symbolic computation of triangle-triangle intersection.
//!
//! Faithful port of Geogram's `triangle_intersection.h/.cpp`.
//! BSD 3-Clause license (original Geogram copyright Inria 2000-2022).

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::doc_markdown,
    clippy::redundant_else,
    clippy::if_not_else,
    clippy::enum_glob_use,
    clippy::float_cmp
)]

use std::cell::Cell;

use super::exact_pred::{
    Sign, pck_aligned_3d, pck_dot_3d, pck_orient_2d, pck_orient_3d, triangle_normal_axis,
};

// ---------------------------------------------------------------------------
// TriangleRegion enum
// ---------------------------------------------------------------------------

/// Encodes the location of a point within a triangle.
///
/// A point can be located in 6 different regions: three vertices, three edges,
/// and the interior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum TriangleRegion {
    T1RgnP0 = 0,
    T1RgnP1 = 1,
    T1RgnP2 = 2,

    T2RgnP0 = 3,
    T2RgnP1 = 4,
    T2RgnP2 = 5,

    T1RgnE0 = 6,
    T1RgnE1 = 7,
    T1RgnE2 = 8,

    T2RgnE0 = 9,
    T2RgnE1 = 10,
    T2RgnE2 = 11,

    T1RgnT = 12,
    T2RgnT = 13,

    TRgnNb = 14,
}

use TriangleRegion::*;

impl TriangleRegion {
    #[inline]
    fn idx(self) -> usize {
        self as usize
    }
}

/// Symbolic representation of a single triangle intersection point.
pub type TriangleIsect = (TriangleRegion, TriangleRegion);

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Tests whether a region belongs to triangle T1.
#[inline]
pub fn is_in_t1(r: TriangleRegion) -> bool {
    matches!(
        r,
        T1RgnP0 | T1RgnP1 | T1RgnP2 | T1RgnE0 | T1RgnE1 | T1RgnE2 | T1RgnT
    )
}

/// Replaces T1 with T2 or T2 with T1 in a region code.
pub fn swap_t1_t2(r: TriangleRegion) -> TriangleRegion {
    match r {
        T1RgnP0 => T2RgnP0,
        T1RgnP1 => T2RgnP1,
        T1RgnP2 => T2RgnP2,
        T2RgnP0 => T1RgnP0,
        T2RgnP1 => T1RgnP1,
        T2RgnP2 => T1RgnP2,
        T1RgnE0 => T2RgnE0,
        T1RgnE1 => T2RgnE1,
        T1RgnE2 => T2RgnE2,
        T2RgnE0 => T1RgnE0,
        T2RgnE1 => T1RgnE1,
        T2RgnE2 => T1RgnE2,
        T1RgnT => T2RgnT,
        T2RgnT => T1RgnT,
        TRgnNb => panic!("swap_t1_t2: TRgnNb"),
    }
}

/// Gets the dimension of a triangle region (0=vertex, 1=edge, 2=interior).
pub fn region_dim(r: TriangleRegion) -> u32 {
    const DIM: [u32; 14] = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2];
    debug_assert!((r as usize) < 14);
    DIM[r as usize]
}

/// Gets the three vertex regions of a triangle region.
pub fn get_triangle_vertices(
    t: TriangleRegion,
) -> (TriangleRegion, TriangleRegion, TriangleRegion) {
    debug_assert!(region_dim(t) == 2);
    match t {
        T1RgnT => (T1RgnP0, T1RgnP1, T1RgnP2),
        T2RgnT => (T2RgnP0, T2RgnP1, T2RgnP2),
        _ => panic!("get_triangle_vertices: not a triangle region"),
    }
}

/// Gets the three edge regions of a triangle region.
pub fn get_triangle_edges(t: TriangleRegion) -> (TriangleRegion, TriangleRegion, TriangleRegion) {
    debug_assert!(region_dim(t) == 2);
    match t {
        T1RgnT => (T1RgnE0, T1RgnE1, T1RgnE2),
        T2RgnT => (T2RgnE0, T2RgnE1, T2RgnE2),
        _ => panic!("get_triangle_edges: not a triangle region"),
    }
}

/// Gets the two vertex regions of an edge region.
pub fn get_edge_vertices(e: TriangleRegion) -> (TriangleRegion, TriangleRegion) {
    debug_assert!(region_dim(e) == 1);
    match e {
        T1RgnE0 => (T1RgnP1, T1RgnP2),
        T1RgnE1 => (T1RgnP2, T1RgnP0),
        T1RgnE2 => (T1RgnP0, T1RgnP1),
        T2RgnE0 => (T2RgnP1, T2RgnP2),
        T2RgnE1 => (T2RgnP2, T2RgnP0),
        T2RgnE2 => (T2RgnP0, T2RgnP1),
        _ => panic!("get_edge_vertices: not an edge region"),
    }
}

/// Computes the convex hull of two regions (purely combinatorial).
pub fn regions_convex_hull(r1: TriangleRegion, r2: TriangleRegion) -> TriangleRegion {
    debug_assert!(is_in_t1(r1) == is_in_t1(r2));
    if r1 == r2 {
        return r1;
    }

    let r = if is_in_t1(r1) { T1RgnT } else { T2RgnT };

    if region_dim(r1) == 1 && region_dim(r2) == 0 {
        let (v1, v2) = get_edge_vertices(r1);
        if r2 == v1 || r2 == v2 {
            return r1;
        }
    } else if region_dim(r2) == 1 && region_dim(r1) == 0 {
        let (v1, v2) = get_edge_vertices(r2);
        if r1 == v1 || r1 == v2 {
            return r2;
        }
    } else if region_dim(r1) == 0 && region_dim(r2) == 0 {
        for e in [T1RgnE0, T1RgnE1, T1RgnE2, T2RgnE0, T2RgnE1, T2RgnE2] {
            let (v1, v2) = get_edge_vertices(e);
            if (r1 == v1 && r2 == v2) || (r1 == v2 && r2 == v1) {
                return e;
            }
        }
    }

    r
}

/// Converts a triangle region code to a string.
pub fn region_to_string(rgn: TriangleRegion) -> &'static str {
    const STRS: [&str; 14] = [
        "T1.P0", "T1.P1", "T1.P2", "T2.P0", "T2.P1", "T2.P2", "T1.E0", "T1.E1", "T1.E2", "T2.E0",
        "T2.E1", "T2.E2", "T1.T", "T2.T",
    ];
    debug_assert!((rgn as usize) < 14);
    STRS[rgn as usize]
}

// ---------------------------------------------------------------------------
// TriangleTriangleIntersection
// ---------------------------------------------------------------------------

const CACHE_UNINITIALIZED: i8 = -2;

/// Internal implementation struct for triangle-triangle intersection.
struct TriangleTriangleIntersection<'a> {
    p: [[f64; 3]; 6],
    result: Option<&'a mut Vec<TriangleIsect>>,
    has_non_degenerate_intersection: bool,
    o3d_cache: [Cell<i8>; 64],
}

impl<'a> TriangleTriangleIntersection<'a> {
    fn new(
        p0: &[f64; 3],
        p1: &[f64; 3],
        p2: &[f64; 3],
        q0: &[f64; 3],
        q1: &[f64; 3],
        q2: &[f64; 3],
        result: Option<&'a mut Vec<TriangleIsect>>,
    ) -> Self {
        Self {
            p: [*p0, *p1, *p2, *q0, *q1, *q2],
            result,
            has_non_degenerate_intersection: false,
            o3d_cache: std::array::from_fn(|_| Cell::new(CACHE_UNINITIALIZED)),
        }
    }

    fn compute(&mut self) {
        if let Some(ref mut r) = self.result {
            r.clear();
        }

        // Test for degenerate triangles.
        if self.triangle_dim(T1RgnP0, T1RgnP1, T1RgnP2) != 2
            || self.triangle_dim(T2RgnP0, T2RgnP1, T2RgnP2) != 2
        {
            return;
        }

        // If T1 is strictly on one side of the supporting plane of T2,
        // there is no intersection.
        {
            let (p1, p2, p3) = get_triangle_vertices(T1RgnT);
            let (q1, q2, q3) = get_triangle_vertices(T2RgnT);
            let o1 = self.orient3d(q1, q2, q3, p1);
            let o2 = self.orient3d(q1, q2, q3, p2);
            let o3 = self.orient3d(q1, q2, q3, p3);
            if (o1 as i32) * (o2 as i32) == 1
                && (o2 as i32) * (o3 as i32) == 1
                && (o3 as i32) * (o1 as i32) == 1
            {
                return;
            }
        }

        self.intersect_edge_triangle(T1RgnE0, T2RgnT);
        if self.finished() {
            return;
        }
        self.intersect_edge_triangle(T1RgnE1, T2RgnT);
        if self.finished() {
            return;
        }
        self.intersect_edge_triangle(T1RgnE2, T2RgnT);
        if self.finished() {
            return;
        }

        self.intersect_edge_triangle(T2RgnE0, T1RgnT);
        if self.finished() {
            return;
        }
        self.intersect_edge_triangle(T2RgnE1, T1RgnT);
        if self.finished() {
            return;
        }
        self.intersect_edge_triangle(T2RgnE2, T1RgnT);
        if self.finished() {
            return;
        }

        // Remove duplicates.
        if let Some(ref mut r) = self.result {
            r.sort();
            r.dedup();
        }
    }

    #[inline]
    fn finished(&self) -> bool {
        self.result.is_none() && self.has_non_degenerate_intersection
    }

    fn intersect_edge_triangle(&mut self, e: TriangleRegion, t: TriangleRegion) {
        debug_assert!(region_dim(e) == 1);
        debug_assert!(region_dim(t) == 2);

        let mut r1 = e;
        let mut r2 = t;

        let (p1, p2, p3) = get_triangle_vertices(t);
        let (e1, e2, e3) = get_triangle_edges(t);
        let (q1, q2) = get_edge_vertices(e);

        let o1 = self.orient3d(p1, p2, p3, q1);
        let o2 = self.orient3d(p1, p2, p3, q2);

        // Both extremities on same side => no intersection.
        if (o1 as i32) * (o2 as i32) == 1 {
            return;
        }

        if o1 == Sign::Zero && o2 == Sign::Zero {
            // Coplanar case.
            let nax = self.normal_axis(p1, p2, p3);

            // Test whether extremities of the segment are in the triangle.
            {
                let a1 = self.orient2d(q1, p1, p2, nax);
                let a2 = self.orient2d(q1, p2, p3, nax);
                let a3 = self.orient2d(q1, p3, p1, nax);

                let b1 = self.orient2d(q2, p1, p2, nax);
                let b2 = self.orient2d(q2, p2, p3, nax);
                let b3 = self.orient2d(q2, p3, p1, nax);

                if (a1 as i32) * (a2 as i32) > 0
                    && (a2 as i32) * (a3 as i32) > 0
                    && (a3 as i32) * (a1 as i32) > 0
                {
                    self.add_intersection(q1, t);
                    if self.finished() {
                        return;
                    }
                }

                if (b1 as i32) * (b2 as i32) > 0
                    && (b2 as i32) * (b3 as i32) > 0
                    && (b3 as i32) * (b1 as i32) > 0
                {
                    self.add_intersection(q2, t);
                    if self.finished() {
                        return;
                    }
                }
            }

            self.intersect_edge_edge_2d(e, e1, nax);
            if self.finished() {
                return;
            }
            self.intersect_edge_edge_2d(e, e2, nax);
            if self.finished() {
                return;
            }
            self.intersect_edge_edge_2d(e, e3, nax);
        } else {
            // Update symbolic information of segment if one of the segment
            // vertices is on the triangle's supporting plane.
            if o1 == Sign::Zero {
                r1 = q1;
            } else if o2 == Sign::Zero {
                r1 = q2;
            }

            let oo1 = self.orient3d(p2, p3, q1, q2);
            let oo2 = self.orient3d(p3, p1, q1, q2);

            if (oo1 as i32) * (oo2 as i32) == -1 {
                return;
            }

            let oo3 = self.orient3d(p1, p2, q1, q2);

            // Update symbolic information of triangle.
            let nb_zeros = i32::from(oo1 == Sign::Zero)
                + i32::from(oo2 == Sign::Zero)
                + i32::from(oo3 == Sign::Zero);
            debug_assert!(nb_zeros != 3);
            if nb_zeros == 1 {
                if oo1 == Sign::Zero {
                    r2 = e1;
                } else if oo2 == Sign::Zero {
                    r2 = e2;
                } else {
                    r2 = e3;
                }
            } else if nb_zeros == 2 {
                if oo1 != Sign::Zero {
                    r2 = p1;
                } else if oo2 != Sign::Zero {
                    r2 = p2;
                } else {
                    r2 = p3;
                }
            }

            let outside = (oo1 as i32) * (oo2 as i32) == -1
                || (oo2 as i32) * (oo3 as i32) == -1
                || (oo3 as i32) * (oo1 as i32) == -1;

            if !outside {
                self.add_intersection(r1, r2);
            }
        }
    }

    fn intersect_edge_edge_2d(
        &mut self,
        e1_rgn: TriangleRegion,
        e2_rgn: TriangleRegion,
        nax: usize,
    ) {
        debug_assert!(region_dim(e1_rgn) == 1);
        debug_assert!(region_dim(e2_rgn) == 1);

        let mut r1 = e1_rgn;
        let mut r2 = e2_rgn;

        let (p1, p2) = get_edge_vertices(e1_rgn);
        let (q1, q2) = get_edge_vertices(e2_rgn);

        let a1 = self.orient2d(q1, q2, p1, nax);
        let a2 = self.orient2d(q1, q2, p2, nax);

        if a1 == Sign::Zero && a2 == Sign::Zero {
            // 1D case: all four edge extremities are collinear.
            self.intersect_edge_edge_1d(e1_rgn, e2_rgn);
        } else {
            if a1 == Sign::Zero {
                r1 = p1;
            } else if a2 == Sign::Zero {
                r1 = p2;
            }

            let b1 = self.orient2d(p1, p2, q1, nax);
            let b2 = self.orient2d(p1, p2, q2, nax);

            if b1 == Sign::Zero {
                r2 = q1;
            } else if b2 == Sign::Zero {
                r2 = q2;
            }

            if (a1 as i32) * (a2 as i32) != 1 && (b1 as i32) * (b2 as i32) != 1 {
                self.add_intersection(r1, r2);
            }
        }
    }

    fn intersect_edge_edge_1d(&mut self, e1_rgn: TriangleRegion, e2_rgn: TriangleRegion) {
        debug_assert!(region_dim(e1_rgn) == 1);
        debug_assert!(region_dim(e2_rgn) == 1);

        let (p1, p2) = get_edge_vertices(e1_rgn);
        let (q1, q2) = get_edge_vertices(e2_rgn);

        let d1 = self.dot3d(p1, q1, q2);
        let d2 = self.dot3d(p2, q1, q2);
        let d3 = self.dot3d(q1, p1, p2);
        let d4 = self.dot3d(q2, p1, p2);

        // Test for identical vertices.
        if d1 == Sign::Zero && d3 == Sign::Zero && self.points_are_identical(p1, q1) {
            self.add_intersection(p1, q1);
        }
        if d2 == Sign::Zero && d3 == Sign::Zero && self.points_are_identical(p2, q1) {
            self.add_intersection(p2, q1);
        }
        if d1 == Sign::Zero && d4 == Sign::Zero && self.points_are_identical(p1, q2) {
            self.add_intersection(p1, q2);
        }
        if d2 == Sign::Zero && d4 == Sign::Zero && self.points_are_identical(p2, q2) {
            self.add_intersection(p2, q2);
        }

        // Test for point in segment: c in [a,b] iff (c-a).(c-b) < 0.
        if d1 == Sign::Negative {
            self.add_intersection(p1, e2_rgn);
        }
        if d2 == Sign::Negative {
            self.add_intersection(p2, e2_rgn);
        }
        if d3 == Sign::Negative {
            self.add_intersection(e1_rgn, q1);
        }
        if d4 == Sign::Negative {
            self.add_intersection(e1_rgn, q2);
        }
    }

    fn add_intersection(&mut self, r1: TriangleRegion, r2: TriangleRegion) {
        if region_dim(r1) >= 1 || region_dim(r2) >= 1 {
            self.has_non_degenerate_intersection = true;
        }
        if is_in_t1(r1) {
            debug_assert!(!is_in_t1(r2));
            if let Some(ref mut r) = self.result {
                r.push((r1, r2));
            }
        } else {
            debug_assert!(is_in_t1(r2));
            if let Some(ref mut r) = self.result {
                r.push((r2, r1));
            }
        }
    }

    fn orient3d(
        &self,
        i: TriangleRegion,
        j: TriangleRegion,
        k: TriangleRegion,
        l: TriangleRegion,
    ) -> Sign {
        debug_assert!(region_dim(i) == 0);
        debug_assert!(region_dim(j) == 0);
        debug_assert!(region_dim(k) == 0);
        debug_assert!(region_dim(l) == 0);

        // Cache index: 1 bit set for each vertex.
        let o3d_idx =
            (1usize << i.idx()) | (1usize << j.idx()) | (1usize << k.idx()) | (1usize << l.idx());
        debug_assert!(o3d_idx < 64);

        // Flip if the argument order is an odd permutation of canonical order.
        let flip = Self::odd_order(i.idx(), j.idx(), k.idx(), l.idx());

        // If cache not initialized, compute and store.
        if self.o3d_cache[o3d_idx].get() == CACHE_UNINITIALIZED {
            let raw = pck_orient_3d(
                &self.p[i.idx()],
                &self.p[j.idx()],
                &self.p[k.idx()],
                &self.p[l.idx()],
            );
            let o = if flip { -(raw as i32) } else { raw as i32 };
            self.o3d_cache[o3d_idx].set(o as i8);
        }

        let cached = self.o3d_cache[o3d_idx].get();
        if flip {
            Sign::from_i32(-i32::from(cached))
        } else {
            Sign::from_i32(i32::from(cached))
        }
    }

    /// Tests the parity of the permutation of four distinct indices
    /// with respect to the canonical (sorted) order.
    fn odd_order(i: usize, j: usize, k: usize, l: usize) -> bool {
        let mut tab = [i, j, k, l];
        let mut result = false;
        for pass in 0..3 {
            for idx in 0..(3 - pass) {
                if tab[idx] > tab[idx + 1] {
                    tab.swap(idx, idx + 1);
                    result = !result;
                }
            }
        }
        result
    }

    fn orient2d(
        &self,
        i: TriangleRegion,
        j: TriangleRegion,
        k: TriangleRegion,
        normal_axis: usize,
    ) -> Sign {
        debug_assert!(region_dim(i) == 0);
        debug_assert!(region_dim(j) == 0);
        debug_assert!(region_dim(k) == 0);

        let mut pi = [0.0f64; 2];
        let mut pj = [0.0f64; 2];
        let mut pk = [0.0f64; 2];
        for c in 0..2usize {
            let coord = (normal_axis + 1 + c) % 3;
            pi[c] = self.p[i.idx()][coord];
            pj[c] = self.p[j.idx()][coord];
            pk[c] = self.p[k.idx()][coord];
        }
        pck_orient_2d(&pi, &pj, &pk)
    }

    fn dot3d(&self, i: TriangleRegion, j: TriangleRegion, k: TriangleRegion) -> Sign {
        debug_assert!(region_dim(i) == 0);
        debug_assert!(region_dim(j) == 0);
        debug_assert!(region_dim(k) == 0);
        pck_dot_3d(&self.p[i.idx()], &self.p[j.idx()], &self.p[k.idx()])
    }

    fn normal_axis(&self, v1: TriangleRegion, v2: TriangleRegion, v3: TriangleRegion) -> usize {
        debug_assert!(region_dim(v1) == 0);
        debug_assert!(region_dim(v2) == 0);
        debug_assert!(region_dim(v3) == 0);
        triangle_normal_axis(self.p[v1.idx()], self.p[v2.idx()], self.p[v3.idx()])
    }

    fn points_are_identical(&self, i: TriangleRegion, j: TriangleRegion) -> bool {
        debug_assert!(region_dim(i) == 0);
        debug_assert!(region_dim(j) == 0);
        let a = &self.p[i.idx()];
        let b = &self.p[j.idx()];
        a[0] == b[0] && a[1] == b[1] && a[2] == b[2]
    }

    /// Detects degenerate triangles: returns 0 if all vertices identical,
    /// 1 if collinear, 2 if a proper triangle.
    fn triangle_dim(&self, i: TriangleRegion, j: TriangleRegion, k: TriangleRegion) -> u32 {
        debug_assert!(region_dim(i) == 0);
        debug_assert!(region_dim(j) == 0);
        debug_assert!(region_dim(k) == 0);

        if !pck_aligned_3d(&self.p[i.idx()], &self.p[j.idx()], &self.p[k.idx()]) {
            return 2;
        }
        if self.points_are_identical(i, j) && self.points_are_identical(j, k) {
            return 0;
        }
        1
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Triangle-triangle intersection with symbolic information.
///
/// Returns `true` if there is a non-degenerate intersection. The `result`
/// vector is filled with `(TriangleRegion, TriangleRegion)` pairs describing
/// each intersection point symbolically.
pub fn triangles_intersections(
    p0: &[f64; 3],
    p1: &[f64; 3],
    p2: &[f64; 3],
    q0: &[f64; 3],
    q1: &[f64; 3],
    q2: &[f64; 3],
    result: &mut Vec<TriangleIsect>,
) -> bool {
    result.clear();
    let mut tti = TriangleTriangleIntersection::new(p0, p1, p2, q0, q1, q2, Some(result));
    tti.compute();
    tti.has_non_degenerate_intersection
}

/// Triangle-triangle intersection predicate (returns `true` if there is
/// a non-degenerate intersection, without computing symbolic information).
pub fn triangles_intersections_pred(
    p0: &[f64; 3],
    p1: &[f64; 3],
    p2: &[f64; 3],
    q0: &[f64; 3],
    q1: &[f64; 3],
    q2: &[f64; 3],
) -> bool {
    let mut tti = TriangleTriangleIntersection::new(p0, p1, p2, q0, q1, q2, None);
    tti.compute();
    tti.has_non_degenerate_intersection
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn run(
        p0: [f64; 3],
        p1: [f64; 3],
        p2: [f64; 3],
        q0: [f64; 3],
        q1: [f64; 3],
        q2: [f64; 3],
    ) -> (bool, Vec<TriangleIsect>) {
        let mut result = Vec::new();
        let has = triangles_intersections(&p0, &p1, &p2, &q0, &q1, &q2, &mut result);
        (has, result)
    }

    /// parallel: has=0 n=0
    #[test]
    fn test_parallel() {
        let (has, result) = run(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        );
        assert!(!has);
        assert_eq!(result.len(), 0);
    }

    /// coplanar_disjoint: has=0 n=0
    #[test]
    fn test_coplanar_disjoint() {
        let (has, result) = run(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
        );
        assert!(!has);
        assert_eq!(result.len(), 0);
    }

    /// crossing: has=true, n=2, pairs=(T1_RGN_E2,T2_RGN_T)(T1_RGN_T,T2_RGN_E2)
    #[test]
    fn test_crossing() {
        // T1 in z=0 plane, T2 in x=0 plane; they cross through each other.
        let (has, result) = run(
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.5, -1.0],
            [0.0, 0.5, 1.0],
            [0.0, -1.0, 0.0],
        );
        assert!(has);
        assert_eq!(result.len(), 2);
        assert!(result.contains(&(T1RgnE2, T2RgnT)));
        assert!(result.contains(&(T1RgnT, T2RgnE2)));
    }

    /// edge_on_edge: has=true, n=1, pair=(T1_RGN_E2,T2_RGN_P0)
    #[test]
    fn test_edge_on_edge() {
        // Q0=(0,0,0) lies on E2 of T1 (the P0-P1 segment).
        let (has, result) = run(
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, -1.0, 1.0],
        );
        assert!(has);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], (T1RgnE2, T2RgnP0));
    }

    /// vertex_in_tri: has=0 n=0 — degenerate triangle
    #[test]
    fn test_vertex_in_tri() {
        let (has, result) = run(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.25, 0.25, 0.0],
            [0.25, 0.25, 0.0],
            [0.25, 0.25, 0.0],
        );
        assert!(!has);
        assert_eq!(result.len(), 0);
    }

    /// shared_vertex: has=0 n=1 pair=(T1_RGN_P0,T2_RGN_P0)
    #[test]
    fn test_shared_vertex() {
        let (has, result) = run(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        );
        assert!(!has);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], (T1RgnP0, T2RgnP0));
    }

    /// perp_cross: has=true, n=1, pair=(T1_RGN_E2,T2_RGN_E1)
    #[test]
    fn test_perp_cross() {
        // E2 of T1 (P0-P1) and E1 of T2 (Q2-Q0) cross at a single point.
        // E2: (-1,0,0)-(1,0,0) along x-axis. E1: (0,1,1)-(0,-1,-1) through origin.
        let (has, result) = run(
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, -1.0, -1.0],
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 1.0],
        );
        assert!(has);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], (T1RgnE2, T2RgnE1));
    }

    // ── Additional Geogram reference tests ──────────────────────────

    #[test]
    fn test_coplanar_overlap() {
        // Geogram: has=1 n=3 (1,11)(6,10)(8,3)
        let (has, result) = run(
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 2.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
        );
        assert!(has);
        assert_eq!(result.len(), 3);
        assert!(result.contains(&(T1RgnP1, T2RgnE2)));
        assert!(result.contains(&(T1RgnE0, T2RgnE1)));
        assert!(result.contains(&(T1RgnE2, T2RgnP0)));
    }

    #[test]
    fn test_edge_thru_interior() {
        // Geogram: has=1 n=2 (8,13)(12,9)
        let (has, result) = run(
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
            [2.0, 0.0, 0.0],
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
        );
        assert!(has);
        assert_eq!(result.len(), 2);
        assert!(result.contains(&(T1RgnE2, T2RgnT)));
        assert!(result.contains(&(T1RgnT, T2RgnE0)));
    }

    #[test]
    fn test_vertex_on_edge() {
        // Geogram: has=1 n=2 (8,11)(12,4)
        let (has, result) = run(
            [0.5, 0.0, -1.0],
            [0.5, 0.0, 1.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
        );
        assert!(has);
        assert_eq!(result.len(), 2);
        assert!(result.contains(&(T1RgnE2, T2RgnE2)));
        assert!(result.contains(&(T1RgnT, T2RgnP1)));
    }

    #[test]
    fn test_shared_edge() {
        // Geogram: has=0 n=2 (0,3)(1,4) — degenerate (shared edge)
        let (has, result) = run(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, -1.0, 0.0],
        );
        assert!(!has);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_near_flat() {
        // Geogram: has=1 n=2 (7,11)(12,9)
        let (has, result) = run(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1e-15, 1.0],
            [0.25, -1.0, 0.5],
            [0.25, 1.0, 0.5],
            [0.75, 0.0, 0.5],
        );
        assert!(has);
        assert_eq!(result.len(), 2);
        assert!(result.contains(&(T1RgnE1, T2RgnE2)));
        assert!(result.contains(&(T1RgnT, T2RgnE0)));
    }

    #[test]
    fn test_contained_coplanar() {
        // Geogram: has=1 n=3 (12,3)(12,4)(12,5)
        let (has, result) = run(
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [5.0, 10.0, 0.0],
            [3.0, 1.0, 0.0],
            [5.0, 1.0, 0.0],
            [4.0, 3.0, 0.0],
        );
        assert!(has);
        assert_eq!(result.len(), 3);
        assert!(result.contains(&(T1RgnT, T2RgnP0)));
        assert!(result.contains(&(T1RgnT, T2RgnP1)));
        assert!(result.contains(&(T1RgnT, T2RgnP2)));
    }

    #[test]
    fn test_colinear_edges() {
        // Geogram: has=1 n=3 (1,11)(6,10)(8,3)
        let (has, result) = run(
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
        );
        assert!(has);
        assert_eq!(result.len(), 3);
        assert!(result.contains(&(T1RgnP1, T2RgnE2)));
        assert!(result.contains(&(T1RgnE0, T2RgnE1)));
        assert!(result.contains(&(T1RgnE2, T2RgnP0)));
    }

    #[test]
    fn test_box_faces() {
        // Geogram: has=1 n=1 (7,11)
        let (has, result) = run(
            [2.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
            [2.0, 2.0, 2.0],
            [1.0, 1.0, 1.0],
            [3.0, 1.0, 1.0],
            [3.0, 1.0, 3.0],
        );
        assert!(has);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], (T1RgnE1, T2RgnE2));
    }
}
