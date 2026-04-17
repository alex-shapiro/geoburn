//! Exact mesh boolean operations.
//!
//! Pure Rust port of Geogram's exact mesh boolean (Lévy 2024).
//! Uses Shewchuk's adaptive-precision arithmetic for all sign decisions.

mod bvh;
mod classify;
pub mod geogram;
mod intersect;
mod types;

pub use types::*;

use std::collections::HashMap;

use bvh::{Bvh, find_overlapping_pairs};
use classify::point_in_mesh;
use intersect::{
    CutLine, IntersectionSegment, SubTriangle, split_triangle, split_triangle_by_line,
};

/// Perform an exact mesh boolean operation on two triangle meshes.
pub fn mesh_boolean(
    a: &IndexedMesh,
    b: &IndexedMesh,
    op: BooleanOp,
) -> Result<BooleanResult, BooleanError> {
    if a.triangles.is_empty() {
        return Err(BooleanError::EmptyMesh("mesh A is empty"));
    }
    if b.triangles.is_empty() {
        return Err(BooleanError::EmptyMesh("mesh B is empty"));
    }

    let bvh_a = Bvh::build(a);
    let bvh_b = Bvh::build(b);
    let pairs = find_overlapping_pairs(&bvh_a, &bvh_b);

    let mut cuts_a: HashMap<u32, Vec<CutLine>> = HashMap::new();
    let mut cuts_b: HashMap<u32, Vec<CutLine>> = HashMap::new();
    let mut intersection_segments: Vec<IntersectionSegment> = Vec::new();

    for &(ai, bi) in &pairs {
        let a_verts = a.triangle_verts(ai);
        let b_verts = b.triangle_verts(bi);
        if let Some((cut_line, segment)) =
            intersect::triangle_triangle_intersection(a_verts, b_verts)
        {
            cuts_a.entry(ai).or_default().push(cut_line.clone());
            cuts_b.entry(bi).or_default().push(cut_line);
            intersection_segments.push(segment);
        }
    }

    let mut all_sub_tris: Vec<SubTriangle> = Vec::new();

    for i in 0..a.num_triangles() {
        let verts = a.triangle_verts(i);
        if let Some(cuts) = cuts_a.get(&i) {
            if cuts.len() == 1 {
                all_sub_tris.extend(split_triangle_by_line(verts, &cuts[0], i, 0));
            } else {
                all_sub_tris.extend(split_triangle(verts, cuts, i, 0));
            }
        } else {
            all_sub_tris.push(SubTriangle {
                verts,
                origin_tri: i,
                mesh_id: 0,
            });
        }
    }

    for i in 0..b.num_triangles() {
        let verts = b.triangle_verts(i);
        if let Some(cuts) = cuts_b.get(&i) {
            if cuts.len() == 1 {
                all_sub_tris.extend(split_triangle_by_line(verts, &cuts[0], i, 1));
            } else {
                all_sub_tris.extend(split_triangle(verts, cuts, i, 1));
            }
        } else {
            all_sub_tris.push(SubTriangle {
                verts,
                origin_tri: i,
                mesh_id: 1,
            });
        }
    }

    let mut selected: Vec<(SubTriangle, bool)> = Vec::new();
    for sub_tri in &all_sub_tris {
        let centroid = triangle_centroid(&sub_tri.verts);
        let is_from_a = sub_tri.mesh_id == 0;
        let inside_other = if is_from_a {
            point_in_mesh(centroid, b)
        } else {
            point_in_mesh(centroid, a)
        };
        let (keep, flip) = match (op, is_from_a) {
            (BooleanOp::Union, _) | (BooleanOp::Difference, true) => (!inside_other, false),
            (BooleanOp::Intersection, _) => (inside_other, false),
            (BooleanOp::Difference, false) => (inside_other, true),
        };
        if keep {
            selected.push((sub_tri.clone(), flip));
        }
    }

    Ok(build_output_mesh(&selected, intersection_segments))
}

fn triangle_centroid(verts: &[[f64; 3]; 3]) -> [f64; 3] {
    [
        (verts[0][0] + verts[1][0] + verts[2][0]) / 3.0,
        (verts[0][1] + verts[1][1] + verts[2][1]) / 3.0,
        (verts[0][2] + verts[1][2] + verts[2][2]) / 3.0,
    ]
}

fn build_output_mesh(
    selected: &[(SubTriangle, bool)],
    intersection_segments: Vec<IntersectionSegment>,
) -> BooleanResult {
    let mut vertices: Vec<[f64; 3]> = Vec::new();
    let mut index_map: HashMap<[i64; 3], u32> = HashMap::new();
    let mut triangles: Vec<[u32; 3]> = Vec::new();
    let mut origins: Vec<FaceOrigin> = Vec::new();

    let quantize = |v: [f64; 3]| -> [i64; 3] {
        [
            (v[0] * 1e12).round() as i64,
            (v[1] * 1e12).round() as i64,
            (v[2] * 1e12).round() as i64,
        ]
    };

    let mut add_vertex = |v: [f64; 3]| -> u32 {
        let key = quantize(v);
        let len = vertices.len() as u32;
        *index_map.entry(key).or_insert_with(|| {
            vertices.push(v);
            len
        })
    };

    for (sub_tri, flip) in selected {
        let i0 = add_vertex(sub_tri.verts[0]);
        let i1 = add_vertex(sub_tri.verts[1]);
        let i2 = add_vertex(sub_tri.verts[2]);
        if *flip {
            triangles.push([i0, i2, i1]);
        } else {
            triangles.push([i0, i1, i2]);
        }
        origins.push(if sub_tri.mesh_id == 0 {
            FaceOrigin::FromA(sub_tri.origin_tri)
        } else {
            FaceOrigin::FromB(sub_tri.origin_tri)
        });
    }

    let intersection_points: Vec<[f64; 3]> = intersection_segments
        .iter()
        .flat_map(|seg| [seg.p0, seg.p1])
        .collect();

    BooleanResult {
        mesh: IndexedMesh::new(vertices, triangles),
        face_origins: FaceOriginMap { origins },
        intersection_points,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn box_mesh(min: [f64; 3], max: [f64; 3]) -> IndexedMesh {
        let [x0, y0, z0] = min;
        let [x1, y1, z1] = max;
        let vertices = vec![
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ];
        let triangles = vec![
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ];
        IndexedMesh::new(vertices, triangles)
    }

    fn mesh_volume(mesh: &IndexedMesh) -> f64 {
        let mut vol = 0.0;
        for tri in &mesh.triangles {
            let v0 = mesh.vertices[tri[0] as usize];
            let v1 = mesh.vertices[tri[1] as usize];
            let v2 = mesh.vertices[tri[2] as usize];
            vol += v0[0] * (v1[1] * v2[2] - v1[2] * v2[1])
                + v0[1] * (v1[2] * v2[0] - v1[0] * v2[2])
                + v0[2] * (v1[0] * v2[1] - v1[1] * v2[0]);
        }
        vol / 6.0
    }

    #[test]
    fn union_of_disjoint_boxes() {
        let r = mesh_boolean(
            &box_mesh([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            &box_mesh([5.0, 5.0, 5.0], [6.0, 6.0, 6.0]),
            BooleanOp::Union,
        )
        .unwrap();
        assert_eq!(r.mesh.triangles.len(), 24);
        assert!((mesh_volume(&r.mesh) - 2.0).abs() < 0.01);
    }

    #[test]
    fn union_of_overlapping_boxes() {
        let r = mesh_boolean(
            &box_mesh([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]),
            &box_mesh([1.0, 1.0, 1.0], [3.0, 3.0, 3.0]),
            BooleanOp::Union,
        )
        .unwrap();
        let vol = mesh_volume(&r.mesh);
        assert!((vol - 15.0).abs() < 1.0, "got {vol}");
    }

    #[test]
    fn intersection_of_overlapping_boxes() {
        let r = mesh_boolean(
            &box_mesh([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]),
            &box_mesh([1.0, 1.0, 1.0], [3.0, 3.0, 3.0]),
            BooleanOp::Intersection,
        )
        .unwrap();
        let vol = mesh_volume(&r.mesh);
        assert!((vol - 1.0).abs() < 0.5, "got {vol}");
    }

    #[test]
    fn difference_of_overlapping_boxes() {
        let r = mesh_boolean(
            &box_mesh([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]),
            &box_mesh([1.0, 1.0, 1.0], [3.0, 3.0, 3.0]),
            BooleanOp::Difference,
        )
        .unwrap();
        let vol = mesh_volume(&r.mesh);
        assert!((vol - 7.0).abs() < 1.0, "got {vol}");
    }

    #[test]
    fn intersection_of_disjoint_is_empty() {
        let r = mesh_boolean(
            &box_mesh([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            &box_mesh([5.0, 5.0, 5.0], [6.0, 6.0, 6.0]),
            BooleanOp::Intersection,
        )
        .unwrap();
        assert!(r.mesh.triangles.is_empty());
    }
}
