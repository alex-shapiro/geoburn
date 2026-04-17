use super::geogram::exact_pred::orient3d_f64 as orient3d;

/// Geometric tolerance for length/distance comparisons (not sign tests).
/// Sign tests use exact predicates and need no tolerance.
const GEO_TOL: f64 = 1e-12;

// ── Vector math helpers ─────────────────────────────────────────────

#[inline]
pub fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
pub fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
pub fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
pub fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
pub fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
pub fn lerp(a: [f64; 3], b: [f64; 3], t: f64) -> [f64; 3] {
    [
        a[0] + t * (b[0] - a[0]),
        a[1] + t * (b[1] - a[1]),
        a[2] + t * (b[2] - a[2]),
    ]
}

#[inline]
fn length_sq(v: [f64; 3]) -> f64 {
    dot(v, v)
}

// ── Exact orientation helpers ───────────────────────────────────────

/// Signed orientation of point `d` relative to the plane through `a`, `b`, `c`.
/// Positive → `d` is above the plane (left-hand rule), negative → below, zero → coplanar.
/// Uses Shewchuk's adaptive-precision arithmetic for exact results.
#[inline]
fn orientation(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> f64 {
    orient3d(a, b, c, d)
}

/// Sign of an exact predicate result: +1, -1, or 0.
#[inline]
fn sign(v: f64) -> i8 {
    if v > 0.0 {
        1
    } else if v < 0.0 {
        -1
    } else {
        0
    }
}

// ── Intersection types ──────────────────────────────────────────────

/// A cutting line that splits a triangle. Defined by a point on the line
/// and a direction.
#[derive(Debug, Clone)]
pub struct CutLine {
    pub point: [f64; 3],
    pub direction: [f64; 3],
}

/// A sub-triangle resulting from splitting.
#[derive(Debug, Clone)]
pub struct SubTriangle {
    pub verts: [[f64; 3]; 3],
    /// Index of the original triangle this came from.
    pub origin_tri: u32,
    /// Which mesh: 0 = A, 1 = B.
    pub mesh_id: u8,
}

/// An intersection segment between two triangles.
#[derive(Debug, Clone)]
pub struct IntersectionSegment {
    /// Start point of the intersection segment.
    pub p0: [f64; 3],
    /// End point of the intersection segment.
    pub p1: [f64; 3],
}

// ── Triangle-triangle intersection (exact predicates) ───────────────

/// Compute the intersection of two triangles using exact predicates.
///
/// Returns `None` if the triangles don't intersect, or `Some((cut_line, segment))`
/// giving the cut line for splitting and the actual intersection segment.
pub fn triangle_triangle_intersection(
    a: [[f64; 3]; 3],
    b: [[f64; 3]; 3],
) -> Option<(CutLine, IntersectionSegment)> {
    // Exact signed volumes: orientation of each vertex of A w.r.t. plane of B.
    let da = [
        orientation(b[0], b[1], b[2], a[0]),
        orientation(b[0], b[1], b[2], a[1]),
        orientation(b[0], b[1], b[2], a[2]),
    ];
    let sa = [sign(da[0]), sign(da[1]), sign(da[2])];

    // All same sign → no intersection.
    if (sa[0] > 0 && sa[1] > 0 && sa[2] > 0) || (sa[0] < 0 && sa[1] < 0 && sa[2] < 0) {
        return None;
    }

    // Exact signed volumes: orientation of each vertex of B w.r.t. plane of A.
    let db = [
        orientation(a[0], a[1], a[2], b[0]),
        orientation(a[0], a[1], a[2], b[1]),
        orientation(a[0], a[1], a[2], b[2]),
    ];
    let sb = [sign(db[0]), sign(db[1]), sign(db[2])];

    if (sb[0] > 0 && sb[1] > 0 && sb[2] > 0) || (sb[0] < 0 && sb[1] < 0 && sb[2] < 0) {
        return None;
    }

    // Coplanar triangles: all vertices of one triangle lie exactly on the
    // plane of the other. Coplanar faces share a plane but don't penetrate
    // — the inside/outside classification handles them correctly without splitting.
    if sa[0] == 0 && sa[1] == 0 && sa[2] == 0 {
        return None;
    }
    if sb[0] == 0 && sb[1] == 0 && sb[2] == 0 {
        return None;
    }

    // Compute intersection line direction (f64 is fine — only used for geometry, not decisions).
    let na = cross(sub(a[1], a[0]), sub(a[2], a[0]));
    let nb = cross(sub(b[1], b[0]), sub(b[2], b[0]));
    let dir = cross(na, nb);
    let dir_len_sq = length_sq(dir);
    if dir_len_sq < GEO_TOL * GEO_TOL {
        return None; // Parallel planes.
    }

    // Find edge crossings using exact signs for the sign test, f64 for positions.
    let cross_a = find_edge_crossings(&a, &da, sa)?;
    let cross_b = find_edge_crossings(&b, &db, sb)?;

    // Check interval overlap on the intersection line.
    let project = |p: [f64; 3]| -> f64 { dot(p, dir) };

    let a0 = project(cross_a.0);
    let a1 = project(cross_a.1);
    let (a_min, a_max) = if a0 <= a1 { (a0, a1) } else { (a1, a0) };

    let b0 = project(cross_b.0);
    let b1 = project(cross_b.1);
    let (b_min, b_max) = if b0 <= b1 { (b0, b1) } else { (b1, b0) };

    if a_min > b_max + GEO_TOL || b_min > a_max + GEO_TOL {
        return None;
    }

    let ov_min = a_min.max(b_min);
    let ov_max = a_max.min(b_max);

    if ov_max - ov_min < GEO_TOL {
        return None; // Point contact.
    }

    // Compute the intersection segment endpoints.
    let inv_dir_len_sq = 1.0 / dir_len_sq;
    let seg_p0 = {
        let t = ov_min - project(cross_a.0);
        add(cross_a.0, scale(dir, t * inv_dir_len_sq))
    };
    let seg_p1 = {
        let t = ov_max - project(cross_a.0);
        add(cross_a.0, scale(dir, t * inv_dir_len_sq))
    };

    // Use midpoint as the cut line reference point.
    let mid_t = f64::midpoint(ov_min, ov_max);
    let line_point = {
        let t = mid_t - project(cross_a.0);
        add(cross_a.0, scale(dir, t * inv_dir_len_sq))
    };

    Some((
        CutLine {
            point: line_point,
            direction: dir,
        },
        IntersectionSegment {
            p0: seg_p0,
            p1: seg_p1,
        },
    ))
}

/// Find the two edge crossing points of a triangle given exact orientation signs.
fn find_edge_crossings(
    verts: &[[f64; 3]; 3],
    dists: &[f64; 3],
    signs: [i8; 3],
) -> Option<([f64; 3], [f64; 3])> {
    let edges: [(usize, usize); 3] = [(0, 1), (1, 2), (2, 0)];
    let mut crossings = Vec::with_capacity(2);

    for &(i, j) in &edges {
        let si = signs[i];
        let sj = signs[j];

        if (si > 0 && sj < 0) || (si < 0 && sj > 0) {
            // Exact sign change → edge is crossed.
            let di = dists[i];
            let dj = dists[j];
            let t = di / (di - dj);
            crossings.push(lerp(verts[i], verts[j], t));
        } else if si == 0 && sj != 0 {
            crossings.push(verts[i]);
        } else if sj == 0 && si != 0 {
            crossings.push(verts[j]);
        }

        if crossings.len() == 2 {
            break;
        }
    }

    if crossings.len() > 2 {
        crossings.truncate(2);
    }

    if crossings.len() == 2 {
        Some((crossings[0], crossings[1]))
    } else {
        None
    }
}

// ── Triangle splitting by line (exact predicates) ───────────────────

/// Split a triangle along a cutting line using exact predicates for all
/// sign decisions.
pub fn split_triangle_by_line(
    verts: [[f64; 3]; 3],
    cut: &CutLine,
    origin_tri: u32,
    mesh_id: u8,
) -> Vec<SubTriangle> {
    let tri_normal = cross(sub(verts[1], verts[0]), sub(verts[2], verts[0]));
    if length_sq(tri_normal) < GEO_TOL * GEO_TOL {
        return vec![SubTriangle {
            verts,
            origin_tri,
            mesh_id,
        }];
    }

    // Build three points on the cutting plane: point, point+direction, point+tri_normal.
    // orient3d of each triangle vertex against these three points gives the exact sign.
    let p0 = cut.point;
    let p1 = add(cut.point, cut.direction);
    let p2 = add(cut.point, tri_normal);

    let signs = [
        sign(orientation(p0, p1, p2, verts[0])),
        sign(orientation(p0, p1, p2, verts[1])),
        sign(orientation(p0, p1, p2, verts[2])),
    ];

    // All same sign → line doesn't cross this triangle.
    if (signs[0] >= 0 && signs[1] >= 0 && signs[2] >= 0)
        || (signs[0] <= 0 && signs[1] <= 0 && signs[2] <= 0)
    {
        return vec![SubTriangle {
            verts,
            origin_tri,
            mesh_id,
        }];
    }

    // For edge crossing positions, compute signed distances using f64 dot product.
    // The sign decisions above are exact; the positions are f64-accurate.
    let cut_normal = cross(cut.direction, tri_normal);
    let dists = [
        dot(cut_normal, sub(verts[0], cut.point)),
        dot(cut_normal, sub(verts[1], cut.point)),
        dot(cut_normal, sub(verts[2], cut.point)),
    ];

    let lone_idx = find_lone_vertex(signs);

    if let Some(lone) = lone_idx {
        let i0 = lone;
        let i1 = (lone + 1) % 3;
        let i2 = (lone + 2) % 3;

        let d0 = dists[i0];
        let d1 = dists[i1];
        let d2 = dists[i2];

        let t01 = d0 / (d0 - d1);
        let p01 = lerp(verts[i0], verts[i1], t01);

        let t02 = d0 / (d0 - d2);
        let p02 = lerp(verts[i0], verts[i2], t02);

        vec![
            SubTriangle {
                verts: [verts[i0], p01, p02],
                origin_tri,
                mesh_id,
            },
            SubTriangle {
                verts: [p01, verts[i1], p02],
                origin_tri,
                mesh_id,
            },
            SubTriangle {
                verts: [verts[i1], verts[i2], p02],
                origin_tri,
                mesh_id,
            },
        ]
    } else {
        // One vertex exactly on the plane, the other two on opposite sides.
        if let Some(on_plane) = signs.iter().position(|&s| s == 0) {
            let i1 = (on_plane + 1) % 3;
            let i2 = (on_plane + 2) % 3;

            if (signs[i1] > 0 && signs[i2] < 0) || (signs[i1] < 0 && signs[i2] > 0) {
                let d1 = dists[i1];
                let d2 = dists[i2];
                let t12 = d1 / (d1 - d2);
                let p12 = lerp(verts[i1], verts[i2], t12);

                return vec![
                    SubTriangle {
                        verts: [verts[on_plane], verts[i1], p12],
                        origin_tri,
                        mesh_id,
                    },
                    SubTriangle {
                        verts: [verts[on_plane], p12, verts[i2]],
                        origin_tri,
                        mesh_id,
                    },
                ];
            }
        }

        vec![SubTriangle {
            verts,
            origin_tri,
            mesh_id,
        }]
    }
}

/// Find the vertex that is alone on one side of the cutting plane.
fn find_lone_vertex(signs: [i8; 3]) -> Option<usize> {
    for i in 0..3 {
        let j = (i + 1) % 3;
        let k = (i + 2) % 3;
        if (signs[i] > 0 && signs[j] <= 0 && signs[k] <= 0)
            || (signs[i] < 0 && signs[j] >= 0 && signs[k] >= 0)
        {
            return Some(i);
        }
    }
    None
}

/// Split a triangle by multiple cutting lines (iterative approach).
pub fn split_triangle(
    verts: [[f64; 3]; 3],
    cuts: &[CutLine],
    origin_tri: u32,
    mesh_id: u8,
) -> Vec<SubTriangle> {
    if cuts.is_empty() {
        return vec![SubTriangle {
            verts,
            origin_tri,
            mesh_id,
        }];
    }

    let mut current = vec![SubTriangle {
        verts,
        origin_tri,
        mesh_id,
    }];

    for cut in cuts {
        let mut next = Vec::new();
        for sub_tri in &current {
            next.extend(split_triangle_by_line(
                sub_tri.verts,
                cut,
                origin_tri,
                mesh_id,
            ));
        }
        current = next;
    }

    current
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intersecting_triangles() {
        let a = [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let b = [[0.0, -1.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, -0.5]];
        let result = triangle_triangle_intersection(a, b);
        assert!(result.is_some(), "triangles should intersect");

        let (_, seg) = result.unwrap();
        // Intersection segment should be non-degenerate.
        let seg_len_sq = length_sq(sub(seg.p1, seg.p0));
        assert!(seg_len_sq > 1e-10, "intersection segment too short");
    }

    #[test]
    fn non_intersecting_triangles() {
        let a = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let b = [[0.0, 0.0, 5.0], [1.0, 0.0, 5.0], [0.0, 1.0, 5.0]];
        assert!(triangle_triangle_intersection(a, b).is_none());
    }

    #[test]
    fn coplanar_triangles_do_not_intersect() {
        // Two triangles in the same plane — not an intersection.
        let a = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let b = [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 1.0, 0.0]];
        assert!(triangle_triangle_intersection(a, b).is_none());
    }

    #[test]
    fn axis_aligned_perpendicular_faces() {
        // Two triangles from axis-aligned box faces that genuinely intersect.
        // A's right face triangle at x=2.
        let a = [[2.0, 0.0, 0.0], [2.0, 2.0, 2.0], [2.0, 0.0, 2.0]];
        // B's front face triangle at y=1.
        let b = [[1.0, 1.0, 1.0], [3.0, 1.0, 1.0], [3.0, 1.0, 3.0]];
        let result = triangle_triangle_intersection(a, b);
        assert!(result.is_some(), "perpendicular faces should intersect");
    }

    #[test]
    fn split_by_line_basic() {
        let verts = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
        let cut = CutLine {
            point: [0.5, 0.5, 0.0],
            direction: [1.0, 1.0, 0.0],
        };
        let result = split_triangle_by_line(verts, &cut, 0, 0);
        assert!(
            result.len() >= 2,
            "should split: got {} sub-tris",
            result.len()
        );

        let original_area = triangle_area(verts);
        let total: f64 = result.iter().map(|s| triangle_area(s.verts)).sum();
        assert!(
            (total - original_area).abs() < 1e-6,
            "area mismatch: {total} vs {original_area}"
        );
    }

    #[test]
    fn split_preserves_area_with_multiple_cuts() {
        let verts = [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0]];
        let cuts = vec![
            CutLine {
                point: [1.0, 0.0, 0.0],
                direction: [0.0, 1.0, 0.0],
            },
            CutLine {
                point: [0.0, 1.0, 0.0],
                direction: [1.0, 0.0, 0.0],
            },
        ];
        let result = split_triangle(verts, &cuts, 0, 0);
        let original_area = triangle_area(verts);
        let total: f64 = result.iter().map(|s| triangle_area(s.verts)).sum();
        assert!(
            (total - original_area).abs() < 1e-6,
            "area mismatch: {total} vs {original_area} ({} sub-tris)",
            result.len()
        );
    }

    fn triangle_area(verts: [[f64; 3]; 3]) -> f64 {
        let e1 = sub(verts[1], verts[0]);
        let e2 = sub(verts[2], verts[0]);
        let c = cross(e1, e2);
        0.5 * length_sq(c).sqrt()
    }
}
