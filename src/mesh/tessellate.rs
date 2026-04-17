//! Error-bounded tessellation of B-rep solids.
//!
//! Two strategies:
//! - **Planar faces**: fan triangulation from boundary vertices
//! - **Curved faces**: adaptive UV grid with midpoint deviation check
//!
//! Watertightness is guaranteed by sampling shared edges once and reusing
//! the same vertex indices for adjacent faces.

use std::collections::HashMap;

use crate::el_surfaces;
use crate::geom::surface::Surface;
use crate::gp::{Pnt, precision};
use crate::shape::{Edge, Face, Ref, Shell, Solid};

use super::types::TriangleMesh;

/// Tessellate a solid into a triangle mesh.
///
/// `tolerance` controls the maximum distance from the mesh to the true
/// surface. Smaller tolerance = more triangles, better approximation.
pub fn tessellate_solid(solid: &Ref<Solid>, tolerance: f64) -> TriangleMesh {
    tessellate_shell(solid.shell(), tolerance)
}

/// Tessellate a shell.
pub fn tessellate_shell(shell: &Ref<Shell>, tolerance: f64) -> TriangleMesh {
    let mut mesh = TriangleMesh::new();

    // Stage 1: Sample shared edges → global vertex indices.
    // Key: Arc pointer address of the Edge data. Value: ordered vertex indices.
    let mut edge_samples: HashMap<usize, Vec<usize>> = HashMap::new();

    for face_ref in shell.faces() {
        let wire = face_ref.outer_wire();
        for edge_ref in wire.edges() {
            let edge_ptr = edge_ref.ptr_id();
            if edge_samples.contains_key(&edge_ptr) {
                continue;
            }
            let indices = sample_edge(edge_ref, tolerance, &mut mesh);
            edge_samples.insert(edge_ptr, indices);
        }
    }

    // Stage 2: Tessellate each face.
    for face_ref in shell.faces() {
        tessellate_face(face_ref, &edge_samples, tolerance, &mut mesh);
    }

    mesh
}

/// Sample an edge curve at adaptive intervals, returning global vertex indices.
fn sample_edge(edge: &Ref<Edge>, tolerance: f64, mesh: &mut TriangleMesh) -> Vec<usize> {
    let curve = edge.curve();
    let t0 = edge.first();
    let t1 = edge.last();

    // Adaptive sampling: subdivide until chord-to-curve deviation < tolerance.
    let mut params = vec![t0, t1];
    let mut i = 0;
    while i < params.len() - 1 {
        let ta = params[i];
        let tb = params[i + 1];
        let tm = (ta + tb) / 2.0;

        let pa = curve.value(ta);
        let pb = curve.value(tb);
        let pm = curve.value(tm);

        // Midpoint of the chord
        let chord_mid = Pnt::from((pa.coords + pb.coords) * 0.5);
        let dev = (pm - chord_mid).norm();

        if dev > tolerance && params.len() < 1000 {
            params.insert(i + 1, tm);
        } else {
            i += 1;
        }
    }

    // Create vertices, deduplicating by position against all existing mesh vertices.
    let mut indices = Vec::with_capacity(params.len());
    for &t in &params {
        let p = curve.value(t);
        // Check if this point coincides with any existing vertex
        let existing = mesh
            .vertices
            .iter()
            .enumerate()
            .find(|(_, v)| (v.coords - p.coords).norm() < precision::CONFUSION);
        if let Some((idx, _)) = existing {
            indices.push(idx);
        } else {
            let idx = mesh.vertices.len();
            mesh.vertices.push(p);
            indices.push(idx);
        }
    }

    indices
}

/// Tessellate a single face.
fn tessellate_face(
    face: &Ref<Face>,
    edge_samples: &HashMap<usize, Vec<usize>>,
    tolerance: f64,
    mesh: &mut TriangleMesh,
) {
    match face.surface() {
        Surface::Plane { .. } => tessellate_planar_face(face, edge_samples, mesh),
        _ => tessellate_parametric_face(face, edge_samples, tolerance, mesh),
    }
}

/// Fan-triangulate a planar face from its boundary vertices.
fn tessellate_planar_face(
    face: &Ref<Face>,
    edge_samples: &HashMap<usize, Vec<usize>>,
    mesh: &mut TriangleMesh,
) {
    let boundary = collect_boundary_indices(face, edge_samples);
    if boundary.len() < 3 {
        return;
    }

    // Determine winding: compute the polygon's winding direction and compare
    // to the face's outward normal.
    let face_normal = face.surface().normal(0.0, 0.0);

    // Compute the polygon's area-weighted normal using Newell's method
    let mut poly_normal = nalgebra::Vector3::zeros();
    let n = boundary.len();
    for i in 0..n {
        let vi = &mesh.vertices[boundary[i]];
        let vj = &mesh.vertices[boundary[(i + 1) % n]];
        poly_normal.x += (vi.y - vj.y) * (vi.z + vj.z);
        poly_normal.y += (vi.z - vj.z) * (vi.x + vj.x);
        poly_normal.z += (vi.x - vj.x) * (vi.y + vj.y);
    }
    let flip = poly_normal.dot(&face_normal) < 0.0;
    let center = boundary[0];

    for i in 1..boundary.len() - 1 {
        if flip {
            mesh.triangles.push([center, boundary[i + 1], boundary[i]]);
        } else {
            mesh.triangles.push([center, boundary[i], boundary[i + 1]]);
        }
    }
}

/// Tessellate a curved face using an adaptive UV grid.
///
/// Boundary grid rows/columns use shared edge sample vertices directly
/// (not snapped after the fact). Interior vertices are newly created.
#[allow(clippy::needless_range_loop)]
fn tessellate_parametric_face(
    face: &Ref<Face>,
    edge_samples: &HashMap<usize, Vec<usize>>,
    tolerance: f64,
    mesh: &mut TriangleMesh,
) {
    let surf = face.surface();
    let (u0, u1, v0, v1) = compute_face_uv_domain(face, edge_samples, surf, mesh);
    let (nu, nv) = adaptive_grid_resolution(surf, u0, u1, v0, v1, tolerance);

    // Collect edge sample vertices with their UV coordinates.
    let boundary_uv = collect_boundary_with_uv(face, edge_samples, surf, mesh);

    // Build grid. Boundary points (i=0, i=nu, j=0, j=nv) are matched
    // to the nearest shared edge vertex. Interior points are new.
    let mut grid = vec![vec![0usize; nv + 1]; nu + 1];

    let u_periodic = (u1 - u0 - std::f64::consts::TAU).abs() < 0.01;
    let v_periodic = (v1 - v0 - std::f64::consts::TAU).abs() < 0.01;

    // UV distance threshold for matching: half a grid cell diagonal
    let du = (u1 - u0) / nu as f64;
    let dv = (v1 - v0) / nv as f64;
    let match_radius_sq = (du * du + dv * dv) * 0.25; // half cell diagonal, squared

    for i in 0..=nu {
        for j in 0..=nv {
            let u = u0 + (u1 - u0) * i as f64 / nu as f64;
            let v = v0 + (v1 - v0) * j as f64 / nv as f64;
            let is_boundary = i == 0 || i == nu || j == 0 || j == nv;

            if is_boundary
                && let Some(idx) = find_nearest_boundary_vertex(&boundary_uv, u, v, match_radius_sq)
            {
                grid[i][j] = idx;
                continue;
            }

            // Interior or no matching boundary vertex: create new
            let p = surf.value(u, v);
            let idx = mesh.vertices.len();
            mesh.vertices.push(p);
            grid[i][j] = idx;
        }
    }

    // For periodic surfaces, merge first and last columns/rows
    if u_periodic {
        for j in 0..=nv {
            grid[nu][j] = grid[0][j];
        }
    }
    if v_periodic {
        for i in 0..=nu {
            grid[i][nv] = grid[i][0];
        }
    }

    // Merge coincident 3D vertices (poles, apex)
    merge_coincident_grid_vertices(&mut grid, nu, nv, mesh);

    // Generate triangles
    for i in 0..nu {
        for j in 0..nv {
            let v00 = grid[i][j];
            let v10 = grid[i + 1][j];
            let v01 = grid[i][j + 1];
            let v11 = grid[i + 1][j + 1];

            if v00 != v10 && v10 != v11 && v00 != v11 {
                mesh.triangles.push([v00, v10, v11]);
            }
            if v00 != v11 && v11 != v01 && v00 != v01 {
                mesh.triangles.push([v00, v11, v01]);
            }
        }
    }
}

/// Collect boundary edge vertices with their UV coordinates on the surface.
fn collect_boundary_with_uv(
    face: &Ref<Face>,
    edge_samples: &HashMap<usize, Vec<usize>>,
    surf: &Surface,
    mesh: &TriangleMesh,
) -> Vec<(usize, f64, f64)> {
    let mut result = Vec::new();
    let wire = face.outer_wire();

    for edge_ref in wire.edges() {
        let edge_ptr = edge_ref.ptr_id();
        if let Some(indices) = edge_samples.get(&edge_ptr) {
            for &idx in indices {
                let pt = &mesh.vertices[idx];
                let (u, v) = invert_point_on_surface(surf, pt);
                result.push((idx, u, v));
            }
        }
    }

    result
}

/// Find the nearest boundary vertex to a UV point, returning its mesh index.
///
/// `max_dist_sq` is the maximum squared UV distance for a match — should be
/// relative to the grid cell size so it works for any UV domain scale.
fn find_nearest_boundary_vertex(
    boundary_uv: &[(usize, f64, f64)],
    u: f64,
    v: f64,
    max_dist_sq: f64,
) -> Option<usize> {
    let mut best_dist = f64::MAX;
    let mut best_idx = None;

    for &(idx, bu, bv) in boundary_uv {
        let du = u - bu;
        let dv = v - bv;
        let dist = du * du + dv * dv;
        if dist < best_dist {
            best_dist = dist;
            best_idx = Some(idx);
        }
    }

    if best_dist < max_dist_sq {
        best_idx
    } else {
        None
    }
}

/// Merge grid vertices that coincide in 3D (poles, apex).
#[allow(clippy::needless_range_loop)]
fn merge_coincident_grid_vertices(
    grid: &mut [Vec<usize>],
    nu: usize,
    nv: usize,
    mesh: &TriangleMesh,
) {
    for i in 0..=nu {
        for j in 0..=nv {
            let idx_ij = grid[i][j];
            let p_ij = mesh.vertices[idx_ij];
            for ii in 0..=i {
                let jj_end = if ii == i { j } else { nv + 1 };
                for jj in 0..jj_end {
                    let idx_other = grid[ii][jj];
                    if idx_other == idx_ij {
                        continue;
                    }
                    if (p_ij.coords - mesh.vertices[idx_other].coords).norm() < precision::CONFUSION
                    {
                        grid[i][j] = idx_other;
                        break;
                    }
                }
                if grid[i][j] != idx_ij {
                    break;
                }
            }
        }
    }
}

/// Determine UV grid resolution based on midpoint deviation.
fn adaptive_grid_resolution(
    surf: &Surface,
    u0: f64,
    u1: f64,
    v0: f64,
    v1: f64,
    tolerance: f64,
) -> (usize, usize) {
    // Start with a coarse grid and refine until the midpoint deviation is within tolerance
    let mut nu = 4;
    let mut nv = 4;

    for _ in 0..8 {
        let max_dev = max_midpoint_deviation(surf, u0, u1, v0, v1, nu, nv);
        if max_dev <= tolerance {
            break;
        }
        // Refine the direction with larger spans
        let u_span = (u1 - u0) / nu as f64;
        let v_span = (v1 - v0) / nv as f64;
        if u_span > v_span {
            nu *= 2;
        } else {
            nv *= 2;
        }
    }

    // Minimum resolution
    nu = nu.max(2);
    nv = nv.max(2);

    (nu, nv)
}

/// Compute the maximum midpoint deviation over a grid.
fn max_midpoint_deviation(
    surf: &Surface,
    u0: f64,
    u1: f64,
    v0: f64,
    v1: f64,
    nu: usize,
    nv: usize,
) -> f64 {
    let mut max_dev = 0.0f64;

    for i in 0..nu {
        for j in 0..nv {
            let ua = u0 + (u1 - u0) * i as f64 / nu as f64;
            let ub = u0 + (u1 - u0) * (i + 1) as f64 / nu as f64;
            let va = v0 + (v1 - v0) * j as f64 / nv as f64;
            let vb = v0 + (v1 - v0) * (j + 1) as f64 / nv as f64;

            let um = (ua + ub) / 2.0;
            let vm = (va + vb) / 2.0;

            let p_mid = surf.value(um, vm);
            let p00 = surf.value(ua, va);
            let p11 = surf.value(ub, vb);
            let p_bilinear = Pnt::from((p00.coords + p11.coords) * 0.5);

            let dev = (p_mid - p_bilinear).norm();
            max_dev = max_dev.max(dev);
        }
    }

    max_dev
}

/// Collect boundary vertex indices for a face from its wire's edges.
///
/// Builds a clean vertex loop by taking each edge's samples in the
/// correct order and removing shared junction vertices.
fn collect_boundary_indices(
    face: &Ref<Face>,
    edge_samples: &HashMap<usize, Vec<usize>>,
) -> Vec<usize> {
    // First, collect all edge sample sequences in wire order
    let wire = face.outer_wire();
    let mut segments: Vec<Vec<usize>> = Vec::new();

    for edge_ref in wire.edges() {
        let edge_ptr = edge_ref.ptr_id();
        let Some(indices) = edge_samples.get(&edge_ptr) else {
            continue;
        };

        let is_reversed = edge_ref.orientation() == crate::shape::Orientation::Reversed;
        let ordered: Vec<usize> = if is_reversed {
            indices.iter().rev().copied().collect()
        } else {
            indices.clone()
        };
        segments.push(ordered);
    }

    if segments.is_empty() {
        return vec![];
    }

    // Build boundary: for each segment, add all vertices except the last
    // (which is the same as the next segment's first)
    let mut boundary = Vec::new();
    for seg in &segments {
        if seg.is_empty() {
            continue;
        }
        // Add all but the last vertex of this segment
        for &idx in &seg[..seg.len() - 1] {
            if boundary.last() != Some(&idx) {
                boundary.push(idx);
            }
        }
        // If this is a single-vertex segment (closed edge like a circle),
        // add it if not already present
        if seg.len() == 1 && boundary.last() != Some(&seg[0]) {
            boundary.push(seg[0]);
        }
    }

    boundary
}

/// Compute the UV domain of a face by inverting its boundary vertex
/// positions onto the surface.
///
/// For surfaces with finite parameter ranges (sphere, torus), uses those
/// directly. For surfaces with infinite ranges (cylinder, cone), inverts
/// boundary positions to determine the actual face extent.
fn compute_face_uv_domain(
    face: &Ref<Face>,
    edge_samples: &HashMap<usize, Vec<usize>>,
    surf: &Surface,
    mesh: &TriangleMesh,
) -> (f64, f64, f64, f64) {
    let ((u_lo, u_hi), (v_lo, v_hi)) = surf.parameter_range();

    // If both ranges are finite, use them directly
    let u_infinite = u_lo < -1e50 || u_hi > 1e50;
    let v_infinite = v_lo < -1e50 || v_hi > 1e50;

    if !u_infinite && !v_infinite {
        return (u_lo, u_hi, v_lo, v_hi);
    }

    // Invert boundary vertices onto the surface to find the actual UV extent
    let boundary_uv = collect_boundary_with_uv(face, edge_samples, surf, mesh);
    if boundary_uv.is_empty() {
        return (
            u_lo.max(-10.0),
            u_hi.min(10.0),
            v_lo.max(-10.0),
            v_hi.min(10.0),
        );
    }

    let mut u_min = f64::INFINITY;
    let mut u_max = f64::NEG_INFINITY;
    let mut v_min = f64::INFINITY;
    let mut v_max = f64::NEG_INFINITY;

    for &(_, u, v) in &boundary_uv {
        u_min = u_min.min(u);
        u_max = u_max.max(u);
        v_min = v_min.min(v);
        v_max = v_max.max(v);
    }

    // Use inverted values for infinite directions, surface range for finite
    let u0 = if u_infinite { u_min } else { u_lo };
    let u1 = if u_infinite { u_max } else { u_hi };
    let v0 = if v_infinite { v_min } else { v_lo };
    let v1 = if v_infinite { v_max } else { v_hi };

    (u0, u1, v0, v1)
}

/// Invert a 3D point onto a surface, returning (u, v) parameters.
pub(crate) fn invert_point_on_surface(surf: &Surface, pt: &Pnt) -> (f64, f64) {
    match surf {
        Surface::Plane { pos } => el_surfaces::plane_parameters(pos, pt),
        Surface::Cylinder { pos, radius } => el_surfaces::cylinder_parameters(pos, *radius, pt),
        Surface::Cone {
            pos,
            radius,
            semi_angle,
        } => el_surfaces::cone_parameters(pos, *radius, *semi_angle, pt),
        Surface::Sphere { pos, radius } => el_surfaces::sphere_parameters(pos, *radius, pt),
        Surface::Torus {
            pos,
            major_radius,
            minor_radius,
        } => el_surfaces::torus_parameters(pos, *major_radius, *minor_radius, pt),
        Surface::BSpline { .. } => {
            // TODO: Newton iteration for B-spline point inversion
            (0.0, 0.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solids;
    use nalgebra::Point3;
    use std::f64::consts::PI;

    #[test]
    fn tessellate_box() {
        let b = solids::make_box(Point3::origin(), 1.0, 1.0, 1.0);
        let mesh = tessellate_solid(&b, 0.1);
        assert!(
            mesh.num_triangles() >= 12,
            "box needs at least 12 triangles, got {}",
            mesh.num_triangles()
        );
        assert!(mesh.num_vertices() > 0);

        let vol = mesh.volume().abs();
        assert!(
            (vol - 1.0).abs() < 0.01,
            "box volume = {vol}, expected ~1.0"
        );
    }

    #[test]
    fn tessellate_cylinder() {
        let c = solids::make_cylinder(
            Point3::origin(),
            crate::gp::Dir::from(nalgebra::Unit::new_normalize(nalgebra::Vector3::new(
                0.0, 0.0, 1.0,
            ))),
            1.0,
            2.0,
        );
        let mesh = tessellate_solid(&c, 0.05);
        assert!(mesh.num_triangles() > 0);

        // Volume should be close to π * r² * h = π * 1 * 2 ≈ 6.28
        let expected_vol = PI * 1.0 * 1.0 * 2.0;
        let vol = mesh.volume().abs();
        assert!(
            (vol - expected_vol).abs() / expected_vol < 0.05,
            "cylinder volume = {vol}, expected ~{expected_vol}"
        );
    }

    #[test]
    fn tessellate_sphere() {
        let s = solids::make_sphere(
            Point3::origin(),
            crate::gp::Dir::from(nalgebra::Unit::new_normalize(nalgebra::Vector3::new(
                0.0, 0.0, 1.0,
            ))),
            1.0,
        );
        let mesh = tessellate_solid(&s, 0.05);
        assert!(mesh.num_triangles() > 0);

        // Volume should be close to (4/3)π ≈ 4.19
        let expected_vol = 4.0 / 3.0 * PI;
        let vol = mesh.volume().abs();
        assert!(
            (vol - expected_vol).abs() / expected_vol < 0.1,
            "sphere volume = {vol}, expected ~{expected_vol}"
        );
    }

    #[test]
    fn tessellate_box_manifold() {
        let b = solids::make_box(Point3::origin(), 1.0, 1.0, 1.0);
        let mesh = tessellate_solid(&b, 0.1);
        assert!(mesh.is_manifold(), "box mesh should be manifold");
    }

    #[test]
    fn tessellate_box_area() {
        let b = solids::make_box(Point3::origin(), 2.0, 3.0, 4.0);
        let mesh = tessellate_solid(&b, 0.1);
        let expected_area = 2.0 * (2.0 * 3.0 + 2.0 * 4.0 + 3.0 * 4.0);
        let area = mesh.area();
        assert!(
            (area - expected_area).abs() / expected_area < 0.01,
            "box area = {area}, expected {expected_area}"
        );
    }

    fn make_dir(x: f64, y: f64, z: f64) -> crate::gp::Dir {
        nalgebra::Unit::new_normalize(nalgebra::Vector3::new(x, y, z))
    }

    #[test]
    fn tessellate_cone() {
        let c = solids::make_cone(Point3::origin(), make_dir(0.0, 0.0, 1.0), 1.0, 2.0);
        let mesh = tessellate_solid(&c, 0.05);
        assert!(mesh.num_triangles() > 0);

        // Volume of a cone = (1/3)π r² h = (1/3)π * 1 * 2 ≈ 2.094
        let expected_vol = PI / 3.0 * 1.0 * 1.0 * 2.0;
        let vol = mesh.volume().abs();
        assert!(
            (vol - expected_vol).abs() / expected_vol < 0.1,
            "cone volume = {vol}, expected ~{expected_vol}"
        );
    }

    #[test]
    fn tessellate_torus() {
        let t = solids::make_torus(Point3::origin(), make_dir(0.0, 0.0, 1.0), 5.0, 1.0);
        let mesh = tessellate_solid(&t, 0.05);
        assert!(mesh.num_triangles() > 0);

        // Volume of a torus = 2π²Rr² = 2π² * 5 * 1 ≈ 98.7
        let expected_vol = 2.0 * PI * PI * 5.0 * 1.0 * 1.0;
        let vol = mesh.volume().abs();
        assert!(
            (vol - expected_vol).abs() / expected_vol < 0.1,
            "torus volume = {vol}, expected ~{expected_vol}"
        );
    }

    // -- Manifold tests --

    #[test]
    fn tessellate_cylinder_manifold() {
        let c = solids::make_cylinder(Point3::origin(), make_dir(0.0, 0.0, 1.0), 1.0, 2.0);
        let mesh = tessellate_solid(&c, 0.1);
        assert!(mesh.is_manifold(), "cylinder mesh should be manifold");
    }

    #[test]
    fn tessellate_sphere_manifold() {
        let s = solids::make_sphere(Point3::origin(), make_dir(0.0, 0.0, 1.0), 1.0);
        let mesh = tessellate_solid(&s, 0.1);
        let bad = mesh.non_manifold_edges();
        assert!(
            mesh.is_manifold(),
            "sphere mesh has {bad} non-manifold edges ({} verts, {} tris)",
            mesh.num_vertices(),
            mesh.num_triangles()
        );
    }

    #[test]
    fn tessellate_cone_manifold() {
        let c = solids::make_cone(Point3::origin(), make_dir(0.0, 0.0, 1.0), 1.0, 2.0);
        let mesh = tessellate_solid(&c, 0.1);
        let bad = mesh.non_manifold_edges();
        assert!(mesh.is_manifold(), "cone mesh has {bad} non-manifold edges");
    }

    #[test]
    fn tessellate_torus_manifold() {
        let t = solids::make_torus(Point3::origin(), make_dir(0.0, 0.0, 1.0), 5.0, 1.0);
        let mesh = tessellate_solid(&t, 0.1);
        assert!(mesh.is_manifold(), "torus mesh should be manifold");
    }
}
