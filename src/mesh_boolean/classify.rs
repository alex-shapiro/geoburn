use super::geogram::exact_pred::orient3d_f64 as orient3d;

use super::IndexedMesh;

/// Test if a point is inside a closed triangle mesh using exact-predicate
/// ray casting.
///
/// Shoots a ray from `point` and counts crossings with the mesh using
/// orient3d for all sign decisions. Uses a single irrational direction to
/// avoid the measure-zero set of degenerate configurations (ray through
/// edge/vertex). If the chosen direction happens to be degenerate, falls
/// back to additional directions.
pub fn point_in_mesh(point: [f64; 3], mesh: &IndexedMesh) -> bool {
    // Irrational-component ray directions. The first direction avoids
    // axis-aligned and diagonal configurations. Additional directions are
    // tried only if the first hits an exact edge/vertex (extremely rare with
    // irrational components and exact predicates).
    #[allow(clippy::unreadable_literal)]
    const DIRECTIONS: [[f64; 3]; 3] = [
        [0.8572953, 0.3941726, 0.3310152],
        [0.2356781, 0.9422316, 0.2381749],
        [0.1423763, 0.3017459, 0.9426152],
    ];

    for dir in &DIRECTIONS {
        match count_crossings_exact(point, *dir, mesh) {
            CrossingResult::Count(n) => return n % 2 == 1,
            CrossingResult::Degenerate => {}
        }
    }

    // All directions degenerate (essentially impossible with irrational components).
    false
}

enum CrossingResult {
    Count(usize),
    Degenerate,
}

/// Count ray-mesh crossings using orient3d for all geometric decisions.
///
/// For each triangle, we test whether the ray crosses the triangle's
/// supporting plane and then whether the crossing point is inside the
/// triangle, all using exact predicates.
fn count_crossings_exact(origin: [f64; 3], dir: [f64; 3], mesh: &IndexedMesh) -> CrossingResult {
    // Construct a "far point" along the ray. The exact location doesn't
    // matter for sign tests — we just need a second point defining the ray.
    let far = [
        origin[0] + dir[0] * 1e6,
        origin[1] + dir[1] * 1e6,
        origin[2] + dir[2] * 1e6,
    ];

    let mut count = 0usize;

    for tri in &mesh.triangles {
        let v0 = mesh.vertices[tri[0] as usize];
        let v1 = mesh.vertices[tri[1] as usize];
        let v2 = mesh.vertices[tri[2] as usize];

        match ray_crosses_triangle_exact(origin, far, v0, v1, v2) {
            TriCrossing::Yes => count += 1,
            TriCrossing::No => {}
            TriCrossing::Degenerate => return CrossingResult::Degenerate,
        }
    }

    CrossingResult::Count(count)
}

enum TriCrossing {
    Yes,
    No,
    Degenerate,
}

/// Test if a ray (origin → far) crosses a triangle (v0, v1, v2) using
/// exact predicates.
///
/// The ray crosses the triangle if:
/// 1. Origin and far are on opposite sides of the triangle's plane.
/// 2. The crossing point is inside the triangle (tested by checking that
///    the four tetrahedra formed by the ray endpoints and triangle edges
///    all have consistent orientation).
fn ray_crosses_triangle_exact(
    origin: [f64; 3],
    far: [f64; 3],
    v0: [f64; 3],
    v1: [f64; 3],
    v2: [f64; 3],
) -> TriCrossing {
    // orient3d(v0, v1, v2, p) gives the signed volume of the tetrahedron.
    // Positive/negative tells us which side of the plane p is on.
    let s_origin = orient3d(v0, v1, v2, origin);
    let s_far = orient3d(v0, v1, v2, far);

    let sign_o = signum(s_origin);
    let sign_f = signum(s_far);

    // Both on same side → no crossing.
    if sign_o == sign_f {
        return TriCrossing::No;
    }

    // Origin exactly on the triangle plane → degenerate.
    if sign_o == 0 {
        return TriCrossing::Degenerate;
    }

    // Far point on the plane → crossing at infinity, doesn't count.
    // (This shouldn't happen with irrational directions, but handle it.)
    if sign_f == 0 {
        return TriCrossing::Degenerate;
    }

    // The ray crosses the plane. Now check if the crossing point is inside
    // the triangle. We test the three "side" tetrahedra:
    //   orient3d(origin, far, v0, v1) — crossing point relative to edge v0v1
    //   orient3d(origin, far, v1, v2) — crossing point relative to edge v1v2
    //   orient3d(origin, far, v2, v0) — crossing point relative to edge v2v0
    // If all three have the same sign, the crossing is inside the triangle.
    let s0 = orient3d(origin, far, v0, v1);
    let s1 = orient3d(origin, far, v1, v2);
    let s2 = orient3d(origin, far, v2, v0);

    let sg0 = signum(s0);
    let sg1 = signum(s1);
    let sg2 = signum(s2);

    // Any zero means the ray hits an edge or vertex — degenerate.
    if sg0 == 0 || sg1 == 0 || sg2 == 0 {
        return TriCrossing::Degenerate;
    }

    // All same sign → inside triangle.
    if sg0 == sg1 && sg1 == sg2 {
        TriCrossing::Yes
    } else {
        TriCrossing::No
    }
}

fn signum(v: f64) -> i8 {
    if v > 0.0 {
        1
    } else if v < 0.0 {
        -1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_inside_box() {
        let mesh = crate::mesh_boolean::tests::box_mesh([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
        assert!(point_in_mesh([1.0, 1.0, 1.0], &mesh));
    }

    #[test]
    fn point_outside_box() {
        let mesh = crate::mesh_boolean::tests::box_mesh([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
        assert!(!point_in_mesh([5.0, 5.0, 5.0], &mesh));
    }

    #[test]
    fn point_between_boxes() {
        let mesh = crate::mesh_boolean::tests::box_mesh([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert!(!point_in_mesh([2.0, 0.5, 0.5], &mesh));
    }

    #[test]
    fn point_near_corner() {
        let mesh = crate::mesh_boolean::tests::box_mesh([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert!(point_in_mesh([0.1, 0.1, 0.1], &mesh));
        assert!(point_in_mesh([0.9, 0.9, 0.9], &mesh));
    }

    #[test]
    fn point_at_exact_center_of_box() {
        // This case previously required SoS perturbation. With exact
        // predicates and irrational ray direction, it works directly.
        let mesh = crate::mesh_boolean::tests::box_mesh([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
        assert!(point_in_mesh([1.0, 1.0, 1.0], &mesh));
    }
}
