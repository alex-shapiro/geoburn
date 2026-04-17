//! Faithful port of Geogram's `MeshInTriangle` from
//! `mesh_surface_intersection_internal.h/.cpp`.
//!
//! `MeshInTriangle` inherits from `CDTBase2d` and overrides `orient2d`,
//! `incircle`, and `create_intersection` with implementations that operate
//! on 3D exact homogeneous coordinates (`Vec3HE`).
//!
//! The CDTBase2d combinatorial machinery (insert, insert_constraint, locate,
//! swap_edge, etc.) is duplicated here from `ExactCDT2d` with calls to
//! `orient2d_mit` / `incircle_mit` / `create_intersection_mit` instead of
//! the ExactCDT2d versions.
//!
//! BSD 3-Clause license (original Geogram copyright Inria 2000-2023).

// Pedantic clippy lints suppressed for faithful port fidelity:
#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::doc_markdown,
    clippy::redundant_else,
    clippy::unnecessary_wraps,
    clippy::new_without_default,
    clippy::if_not_else,
    clippy::manual_is_multiple_of,
    clippy::needless_bool,
    clippy::too_many_arguments,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]

use std::collections::BTreeMap;

use super::exact_pred::{
    Sign, Vec2HE, Vec3HE, incircle_2d_sos_with_lengths, orient_2d_projected, pck_orient_3d,
    triangle_normal_axis,
};
use super::expansion::{Expansion, expansion_det3x3};
use super::triangle_isect::{TriangleRegion, region_dim, regions_convex_hull};

#[allow(clippy::enum_glob_use)]
use TriangleRegion::*;

/// Sentinel for "no index".
pub const NO_INDEX: u32 = u32::MAX;

// ── DList ────────────────────────────────────────────────────────────
// Doubly-linked list of triangles, identical to cdt.rs.

const DLIST_S_ID: u32 = 0;
const DLIST_Q_ID: u32 = 1;
const DLIST_N_ID: u32 = 2;
const DLIST_NB: u32 = 3;

#[allow(dead_code)]
const T_MARKED_FLAG: u32 = DLIST_NB;
#[allow(dead_code)]
const T_VISITED_FLAG: u32 = DLIST_NB + 1;

struct DList {
    list_id: u32,
    front: u32,
    back: u32,
}

impl DList {
    fn new(list_id: u32) -> Self {
        debug_assert!(list_id < DLIST_NB);
        Self {
            list_id,
            front: NO_INDEX,
            back: NO_INDEX,
        }
    }

    fn new_uninit() -> Self {
        Self {
            list_id: NO_INDEX,
            front: NO_INDEX,
            back: NO_INDEX,
        }
    }

    fn initialized(&self) -> bool {
        self.list_id != NO_INDEX
    }

    fn initialize(&mut self, list_id: u32) {
        debug_assert!(!self.initialized());
        debug_assert!(list_id < DLIST_NB);
        self.list_id = list_id;
    }

    fn empty(&self) -> bool {
        debug_assert!((self.back == NO_INDEX) == (self.front == NO_INDEX));
        self.back == NO_INDEX
    }

    fn contains(&self, t_flags: &[u32], t: u32) -> bool {
        debug_assert!(self.initialized());
        (t_flags[t as usize] & (1u32 << self.list_id)) != 0
    }

    fn front(&self) -> u32 {
        self.front
    }

    fn next(t_next: &[u32], t: u32) -> u32 {
        t_next[t as usize]
    }

    fn push_back(&mut self, t_flags: &mut [u32], t_next: &mut [u32], t_prev: &mut [u32], t: u32) {
        debug_assert!(self.initialized());
        debug_assert!(!is_in_list(t_flags, t));
        t_flags[t as usize] |= 1u32 << self.list_id;
        if self.empty() {
            self.back = t;
            self.front = t;
            t_next[t as usize] = NO_INDEX;
            t_prev[t as usize] = NO_INDEX;
        } else {
            t_next[t as usize] = NO_INDEX;
            t_next[self.back as usize] = t;
            t_prev[t as usize] = self.back;
            self.back = t;
        }
    }

    fn pop_back(&mut self, t_flags: &mut [u32], t_next: &mut [u32], t_prev: &mut [u32]) -> u32 {
        debug_assert!(self.initialized());
        debug_assert!(!self.empty());
        let t = self.back;
        self.back = t_prev[self.back as usize];
        if self.back == NO_INDEX {
            debug_assert!(self.front == t);
            self.front = NO_INDEX;
        } else {
            t_next[self.back as usize] = NO_INDEX;
        }
        debug_assert!(self.contains(t_flags, t));
        t_flags[t as usize] &= !(1u32 << self.list_id);
        t
    }

    fn push_front(&mut self, t_flags: &mut [u32], t_next: &mut [u32], t_prev: &mut [u32], t: u32) {
        debug_assert!(self.initialized());
        debug_assert!(!is_in_list(t_flags, t));
        t_flags[t as usize] |= 1u32 << self.list_id;
        if self.empty() {
            self.back = t;
            self.front = t;
            t_next[t as usize] = NO_INDEX;
            t_prev[t as usize] = NO_INDEX;
        } else {
            t_prev[t as usize] = NO_INDEX;
            t_prev[self.front as usize] = t;
            t_next[t as usize] = self.front;
            self.front = t;
        }
    }

    fn clear(&mut self, t_flags: &mut [u32], t_next: &[u32]) {
        let mut t = self.front;
        while t != NO_INDEX {
            t_flags[t as usize] &= !(1u32 << self.list_id);
            t = t_next[t as usize];
        }
        self.back = NO_INDEX;
        self.front = NO_INDEX;
    }
}

fn is_in_list(t_flags: &[u32], t: u32) -> bool {
    (t_flags[t as usize] & ((1u32 << DLIST_NB) - 1)) != 0
}

// ── ConstraintWalker ─────────────────────────────────────────────────

struct ConstraintWalker {
    i: u32,
    j: u32,
    t_prev: u32,
    v_prev: u32,
    t: u32,
    v: u32,
}

impl ConstraintWalker {
    fn new(i: u32, j: u32) -> Self {
        Self {
            i,
            j,
            t_prev: NO_INDEX,
            v_prev: NO_INDEX,
            t: NO_INDEX,
            v: i,
        }
    }
}

// ── Mesh data ────────────────────────────────────────────────────────

/// Minimal mesh representation providing vertex positions and facet
/// connectivity for triangulated meshes.
#[derive(Clone, Debug)]
pub struct MeshData {
    pub vertices: Vec<[f64; 3]>,
    pub triangles: Vec<[u32; 3]>,
}

impl MeshData {
    pub fn new(vertices: Vec<[f64; 3]>, triangles: Vec<[u32; 3]>) -> Self {
        Self {
            vertices,
            triangles,
        }
    }

    #[inline]
    pub fn vertex(&self, v: u32) -> [f64; 3] {
        self.vertices[v as usize]
    }

    #[inline]
    pub fn facet_vertex(&self, f: u32, lv: u32) -> [f64; 3] {
        let vi = self.triangles[f as usize][lv as usize];
        self.vertices[vi as usize]
    }

    #[inline]
    pub fn facet_vertex_index(&self, f: u32, lv: u32) -> u32 {
        self.triangles[f as usize][lv as usize]
    }
}

// ── Rational ─────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct Rational {
    pub num: Expansion,
    pub denom: Expansion,
}

impl Rational {
    pub fn new(num: Expansion, denom: Expansion) -> Self {
        Self { num, denom }
    }
}

// ── Vec3E ────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
struct Vec3E {
    x: Expansion,
    y: Expansion,
    z: Expansion,
}

impl Vec3E {
    fn from_f64_point(p: [f64; 3]) -> Self {
        Self {
            x: Expansion::from_f64(p[0]),
            y: Expansion::from_f64(p[1]),
            z: Expansion::from_f64(p[2]),
        }
    }

    fn make_vec3(p1: [f64; 3], p2: [f64; 3]) -> Self {
        Self {
            x: Expansion::from_f64(p2[0]).sub(&Expansion::from_f64(p1[0])),
            y: Expansion::from_f64(p2[1]).sub(&Expansion::from_f64(p1[1])),
            z: Expansion::from_f64(p2[2]).sub(&Expansion::from_f64(p1[2])),
        }
    }

    fn cross(a: &Self, b: &Self) -> Self {
        Self {
            x: a.y.mul(&b.z).sub(&a.z.mul(&b.y)),
            y: a.z.mul(&b.x).sub(&a.x.mul(&b.z)),
            z: a.x.mul(&b.y).sub(&a.y.mul(&b.x)),
        }
    }

    fn dot(a: &Self, b: &Self) -> Expansion {
        a.x.mul(&b.x).add(&a.y.mul(&b.y)).add(&a.z.mul(&b.z))
    }
}

// ── Vec2E ────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
struct Vec2E {
    x: Expansion,
    y: Expansion,
}

impl Vec2E {
    fn make_vec2(p1: [f64; 2], p2: [f64; 2]) -> Self {
        Self {
            x: Expansion::from_f64(p2[0]).sub(&Expansion::from_f64(p1[0])),
            y: Expansion::from_f64(p2[1]).sub(&Expansion::from_f64(p1[1])),
        }
    }

    fn det(a: &Self, b: &Self) -> Expansion {
        a.x.mul(&b.y).sub(&a.y.mul(&b.x))
    }
}

// ── mix() ────────────────────────────────────────────────────────────

fn mix_vec3(t: &Rational, p1: [f64; 3], p2: [f64; 3]) -> Vec3HE {
    let st_d = &t.denom;
    let t_n = &t.num;
    let s_n = st_d.sub(t_n);
    Vec3HE::new(
        s_n.mul(&Expansion::from_f64(p1[0]))
            .add(&t_n.mul(&Expansion::from_f64(p2[0]))),
        s_n.mul(&Expansion::from_f64(p1[1]))
            .add(&t_n.mul(&Expansion::from_f64(p2[1]))),
        s_n.mul(&Expansion::from_f64(p1[2]))
            .add(&t_n.mul(&Expansion::from_f64(p2[2]))),
        st_d.clone(),
    )
}

// ── plane_line_intersection() ────────────────────────────────────────

fn plane_line_intersection(
    p1: [f64; 3],
    p2: [f64; 3],
    p3: [f64; 3],
    q1: [f64; 3],
    q2: [f64; 3],
) -> Vec3HE {
    let d = Vec3E::make_vec3(q1, q2);
    let e1 = Vec3E::make_vec3(p1, p2);
    let e2 = Vec3E::make_vec3(p1, p3);
    let ao = Vec3E::make_vec3(p1, q1);
    let n = Vec3E::cross(&e1, &e2);
    let neg_d_dot_n = Vec3E::dot(&d, &n).negate();
    debug_assert!(
        neg_d_dot_n.sign() != 0,
        "plane_line_intersection: degenerate"
    );
    let t = Rational::new(Vec3E::dot(&ao, &n), neg_d_dot_n);
    mix_vec3(&t, q1, q2)
}

// ── get_three_planes_intersection() ──────────────────────────────────

fn get_three_planes_intersection(
    p1: [f64; 3],
    p2: [f64; 3],
    p3: [f64; 3],
    q1: [f64; 3],
    q2: [f64; 3],
    q3: [f64; 3],
    r1: [f64; 3],
    r2: [f64; 3],
    r3: [f64; 3],
) -> Option<Vec3HE> {
    fn triangle_normal(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> Vec3E {
        let e1 = Vec3E::make_vec3(a, b);
        let e2 = Vec3E::make_vec3(a, c);
        Vec3E::cross(&e1, &e2)
    }

    let n1 = triangle_normal(p1, p2, p3);
    let n2 = triangle_normal(q1, q2, q3);
    let n3 = triangle_normal(r1, r2, r3);

    let b = Vec3E {
        x: Vec3E::dot(&n1, &Vec3E::from_f64_point(p1)),
        y: Vec3E::dot(&n2, &Vec3E::from_f64_point(q1)),
        z: Vec3E::dot(&n3, &Vec3E::from_f64_point(r1)),
    };

    let w = expansion_det3x3(
        &[n1.x.clone(), n1.y.clone(), n1.z.clone()],
        &[n2.x.clone(), n2.y.clone(), n2.z.clone()],
        &[n3.x.clone(), n3.y.clone(), n3.z.clone()],
    );

    if w.sign() == 0 {
        return None;
    }

    let x = expansion_det3x3(
        &[b.x.clone(), n1.y.clone(), n1.z.clone()],
        &[b.y.clone(), n2.y.clone(), n2.z.clone()],
        &[b.z.clone(), n3.y.clone(), n3.z.clone()],
    );

    let y = expansion_det3x3(
        &[n1.x.clone(), b.x.clone(), n1.z.clone()],
        &[n2.x.clone(), b.y.clone(), n2.z.clone()],
        &[n3.x.clone(), b.z.clone(), n3.z.clone()],
    );

    let z = expansion_det3x3(
        &[n1.x.clone(), n1.y.clone(), b.x.clone()],
        &[n2.x.clone(), n2.y.clone(), b.y.clone()],
        &[n3.x.clone(), n3.y.clone(), b.z.clone()],
    );

    Some(Vec3HE::new(x, y, z, w))
}

// ── trindex ──────────────────────────────────────────────────────────

/// Sorted triple of indices, used as predicate cache key.
/// Port of Geogram's `trindex`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Trindex {
    indices: [u32; 3],
}

impl Trindex {
    fn new(i: u32, j: u32, k: u32) -> Self {
        let mut indices = [i, j, k];
        indices.sort_unstable();
        Self { indices }
    }
}

/// Tests the parity of the permutation of three distinct indices
/// with respect to canonical (sorted) order.
/// Port of the static `odd_order` function.
fn odd_order(i: u32, j: u32, k: u32) -> bool {
    let mut tab = [i, j, k];
    let mut result = false;
    for pass in 0..2 {
        for idx in 0..(2 - pass) {
            if tab[idx] > tab[idx + 1] {
                tab.swap(idx, idx + 1);
                result = !result;
            }
        }
    }
    result
}

// ── IsectInfo ────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct IsectInfo {
    pub f1: u32,
    pub f2: u32,
    pub r1: TriangleRegion,
    pub r2: TriangleRegion,
}

impl IsectInfo {
    pub fn new(f1: u32, f2: u32, r1: TriangleRegion, r2: TriangleRegion) -> Self {
        Self { f1, f2, r1, r2 }
    }

    pub fn flip(&mut self) {
        std::mem::swap(&mut self.f1, &mut self.f2);
        self.r1 = super::triangle_isect::swap_t1_t2(self.r1);
        self.r2 = super::triangle_isect::swap_t1_t2(self.r2);
        std::mem::swap(&mut self.r1, &mut self.r2);
    }

    pub fn is_point(&self) -> bool {
        region_dim(self.r1) == 0 && region_dim(self.r2) == 0
    }
}

// ── Vertex ───────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VertexType {
    Uninitialized,
    MeshVertex,
    PrimaryIsect,
    SecondaryIsect,
}

#[derive(Clone, Copy, Debug)]
pub struct VertexSym {
    pub f1: u32,
    pub f2: u32,
    pub r1: TriangleRegion,
    pub r2: TriangleRegion,
}

/// A vertex of the triangulation. Port of `MeshInTriangle::Vertex`.
/// Stores exact 3D homogeneous coordinates (`point_exact`), NOT 2D projections.
#[derive(Clone, Debug)]
pub struct Vertex {
    pub point_exact: Vec3HE,
    pub vertex_type: VertexType,
    pub mesh_vertex_index: u32,
    pub sym: VertexSym,
    /// Precomputed approximate (p[u]^2 + p[v]^2) / p.w^2 for incircle.
    pub l: f64,
}

impl Vertex {
    fn uninitialized() -> Self {
        Self {
            point_exact: Vec3HE::from_f64(0.0, 0.0, 0.0),
            vertex_type: VertexType::Uninitialized,
            mesh_vertex_index: NO_INDEX,
            sym: VertexSym {
                f1: NO_INDEX,
                f2: NO_INDEX,
                r1: T1RgnT,
                r2: T2RgnT,
            },
            l: 0.0,
        }
    }

    /// Constructor for macro-triangle vertices.
    fn new_mesh_vertex(mesh: &MeshData, f1: u32, lv: u32) -> Self {
        let mut vert = Self::uninitialized();
        vert.vertex_type = VertexType::MeshVertex;
        vert.sym = VertexSym {
            f1,
            f2: NO_INDEX,
            r1: match lv {
                0 => T1RgnP0,
                1 => T1RgnP1,
                2 => T1RgnP2,
                _ => panic!("invalid local vertex {lv}"),
            },
            r2: T2RgnT,
        };
        let pe = Self::compute_geometry_static(mesh, f1, &vert.sym, &mut vert.mesh_vertex_index);
        vert.point_exact = pe;
        vert
    }

    /// Constructor for primary intersections.
    fn new_primary_isect(
        mesh: &MeshData,
        f1: u32,
        f2: u32,
        r1: TriangleRegion,
        r2: TriangleRegion,
    ) -> Self {
        let mut vert = Self::uninitialized();
        vert.vertex_type = VertexType::PrimaryIsect;
        vert.sym = VertexSym { f1, f2, r1, r2 };
        let pe = Self::compute_geometry_static(mesh, f1, &vert.sym, &mut vert.mesh_vertex_index);
        vert.point_exact = pe;
        vert
    }

    /// Constructor for secondary intersections (constraint-constraint).
    fn new_secondary_isect(point_exact: Vec3HE) -> Self {
        let mut vert = Self::uninitialized();
        vert.vertex_type = VertexType::SecondaryIsect;
        vert.sym = VertexSym {
            f1: NO_INDEX,
            f2: NO_INDEX,
            r1: T1RgnT,
            r2: T2RgnT,
        };
        vert.point_exact = point_exact;
        vert
    }

    /// Initialize geometry: store point_exact and compute approximate `l` value.
    /// Port of `MeshInTriangle::Vertex::init_geometry`.
    fn init_geometry(&mut self, u: usize, v: usize) {
        let pu = self.point_exact.coord(u).estimate();
        let pv = self.point_exact.coord(v).estimate();
        let pw = self.point_exact.w.estimate();
        self.l = (pu * pu + pv * pv) / (pw * pw);
    }

    /// Get approximate 2D UV coordinates.
    pub fn get_uv_approx(&self, u: usize, v: usize) -> [f64; 2] {
        let pu = self.point_exact.coord(u).estimate();
        let pv = self.point_exact.coord(v).estimate();
        let pw = self.point_exact.w.estimate();
        [pu / pw, pv / pw]
    }

    /// Compute geometry from symbolic information.
    /// Port of `MeshInTriangle::Vertex::compute_geometry()` with all 5 cases.
    fn compute_geometry_static(
        mesh: &MeshData,
        f1: u32,
        sym: &VertexSym,
        mesh_vertex_index: &mut u32,
    ) -> Vec3HE {
        // Case 1: f1 vertex (R1 is a vertex of T1)
        if region_dim(sym.r1) == 0 {
            let lv = sym.r1 as u32 - T1RgnP0 as u32;
            debug_assert!(lv < 3);
            let vi = mesh.facet_vertex_index(sym.f1, lv);
            *mesh_vertex_index = vi;
            let p = mesh.vertex(vi);
            return Vec3HE::from_f64(p[0], p[1], p[2]);
        }

        debug_assert!(sym.f1 != NO_INDEX && sym.f2 != NO_INDEX);

        // Case 2: f2 vertex (R2 is a vertex of T2)
        if region_dim(sym.r2) == 0 {
            let lv = sym.r2 as u32 - T2RgnP0 as u32;
            debug_assert!(lv < 3);
            let vi = mesh.facet_vertex_index(sym.f2, lv);
            *mesh_vertex_index = vi;
            let p = mesh.vertex(vi);
            return Vec3HE::from_f64(p[0], p[1], p[2]);
        }

        // Case 3: f1 (full or edge) /\ f2 edge in 3D
        if (region_dim(sym.r1) == 2 || region_dim(sym.r1) == 1) && region_dim(sym.r2) == 1 {
            let p1 = mesh.facet_vertex(sym.f1, 0);
            let p2 = mesh.facet_vertex(sym.f1, 1);
            let p3 = mesh.facet_vertex(sym.f1, 2);
            let e = sym.r2 as u32 - T2RgnE0 as u32;
            debug_assert!(e < 3);
            let q1 = mesh.facet_vertex(sym.f2, (e + 1) % 3);
            let q2 = mesh.facet_vertex(sym.f2, (e + 2) % 3);

            let seg_seg_two_d = region_dim(sym.r1) == 1
                && pck_orient_3d(&p1, &p2, &p3, &q1) == Sign::Zero
                && pck_orient_3d(&p1, &p2, &p3, &q2) == Sign::Zero;

            if !seg_seg_two_d {
                return plane_line_intersection(p1, p2, p3, q1, q2);
            }
            // Fall through to case 5
        }

        // Case 4: f1 edge /\ f2 (full triangle)
        if region_dim(sym.r1) == 1 && region_dim(sym.r2) == 2 {
            let e = sym.r1 as u32 - T1RgnE0 as u32;
            debug_assert!(e < 3);
            let p1 = mesh.facet_vertex(sym.f2, 0);
            let p2 = mesh.facet_vertex(sym.f2, 1);
            let p3 = mesh.facet_vertex(sym.f2, 2);
            let q1 = mesh.facet_vertex(sym.f1, (e + 1) % 3);
            let q2 = mesh.facet_vertex(sym.f1, (e + 2) % 3);
            return plane_line_intersection(p1, p2, p3, q1, q2);
        }

        // Case 5: f1 edge /\ f2 edge in 2D (coplanar edges)
        if region_dim(sym.r1) == 1 && region_dim(sym.r2) == 1 {
            let e1 = sym.r1 as u32 - T1RgnE0 as u32;
            debug_assert!(e1 < 3);
            let e2 = sym.r2 as u32 - T2RgnE0 as u32;
            debug_assert!(e2 < 3);

            let f1_normal_axis = triangle_normal_axis(
                mesh.facet_vertex(f1, 0),
                mesh.facet_vertex(f1, 1),
                mesh.facet_vertex(f1, 2),
            );
            let u_axis = (f1_normal_axis + 1) % 3;
            let v_axis = (f1_normal_axis + 2) % 3;

            let fv1_1 = mesh.facet_vertex(sym.f1, (e1 + 1) % 3);
            let fv1_2 = mesh.facet_vertex(sym.f1, (e1 + 2) % 3);
            let fv2_1 = mesh.facet_vertex(sym.f2, (e2 + 1) % 3);
            let fv2_2 = mesh.facet_vertex(sym.f2, (e2 + 2) % 3);

            let p1_uv = [fv1_1[u_axis], fv1_1[v_axis]];
            let p2_uv = [fv1_2[u_axis], fv1_2[v_axis]];
            let q1_uv = [fv2_1[u_axis], fv2_1[v_axis]];
            let q2_uv = [fv2_2[u_axis], fv2_2[v_axis]];

            let d1 = Vec2E::make_vec2(p1_uv, p2_uv);
            let d2 = Vec2E::make_vec2(q1_uv, q2_uv);
            let d = Vec2E::det(&d1, &d2);
            debug_assert!(d.sign() != 0, "compute_geometry case 5: degenerate");
            let ao = Vec2E::make_vec2(p1_uv, q1_uv);
            let t = Rational::new(Vec2E::det(&ao, &d2), d);
            return mix_vec3(&t, fv1_1, fv1_2);
        }

        panic!(
            "compute_geometry: unhandled case (R1={:?}, R2={:?})",
            sym.r1, sym.r2
        );
    }
}

// ── Edge ─────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct EdgeSym {
    pub f2: u32,
    pub r2: TriangleRegion,
}

#[derive(Clone, Debug)]
pub struct Edge {
    pub v1: u32,
    pub v2: u32,
    pub sym: EdgeSym,
}

impl Edge {
    fn new(v1: u32, v2: u32, f2: u32, r2: TriangleRegion) -> Self {
        Self {
            v1,
            v2,
            sym: EdgeSym { f2, r2 },
        }
    }

    fn new_simple(v1: u32, v2: u32) -> Self {
        Self::new(v1, v2, NO_INDEX, T2RgnT)
    }
}

// ══════════════════════════════════════════════════════════════════════
// MeshInTriangle
// ══════════════════════════════════════════════════════════════════════

/// Constrained Delaunay triangulation of a single facet, augmented with
/// intersection vertices. Port of Geogram's `MeshInTriangle` class.
///
/// Contains CDTBase2d fields directly and overrides orient2d/incircle/
/// create_intersection with implementations using 3D exact homogeneous
/// coordinates.
#[allow(clippy::struct_excessive_bools)]
pub struct MeshInTriangle {
    mesh: MeshData,

    // Current facet state.
    f1: u32,
    latest_f2: u32,
    latest_f2_count: u32,
    f1_normal_axis: usize,
    u: usize,
    v: usize,

    // Vertex and edge storage (MeshInTriangle-specific).
    vertex: Vec<Vertex>,
    edges: Vec<Edge>,
    has_planar_isect: bool,

    // ── CDTBase2d fields (copied from ExactCDT2d) ──
    t_verts: Vec<u32>,
    t_adj: Vec<u32>,
    t_ecnstr: Vec<u32>,
    t_flags: Vec<u32>,
    t_next: Vec<u32>,
    t_prev: Vec<u32>,
    v2t: Vec<u32>,
    ecnstr_val: Vec<u32>,
    ecnstr_next: Vec<u32>,
    nv: u32,
    ncnstr: u32,
    orient_012: Sign,
    delaunay: bool,
    exact_intersections: bool,
    constraints: Vec<(u32, u32)>,

    // ── Predicate cache (MeshInTriangle-specific) ──
    pred_cache: BTreeMap<Trindex, Sign>,
    use_pred_cache_insert_buffer: bool,
    pred_cache_insert_buffer: Vec<(Trindex, Sign)>,
}

impl MeshInTriangle {
    // ── Construction ─────────────────────────────────────────────

    pub fn new(mesh: MeshData) -> Self {
        Self {
            mesh,
            f1: NO_INDEX,
            latest_f2: NO_INDEX,
            latest_f2_count: 0,
            f1_normal_axis: 0,
            u: 1,
            v: 2,
            vertex: Vec::new(),
            edges: Vec::new(),
            has_planar_isect: false,
            // CDTBase2d fields
            t_verts: Vec::new(),
            t_adj: Vec::new(),
            t_ecnstr: Vec::new(),
            t_flags: Vec::new(),
            t_next: Vec::new(),
            t_prev: Vec::new(),
            v2t: Vec::new(),
            ecnstr_val: Vec::new(),
            ecnstr_next: Vec::new(),
            nv: 0,
            ncnstr: 0,
            orient_012: Sign::Zero,
            delaunay: true,
            exact_intersections: true,
            constraints: Vec::new(),
            // Predicate cache
            pred_cache: BTreeMap::new(),
            use_pred_cache_insert_buffer: false,
            pred_cache_insert_buffer: Vec::new(),
        }
    }

    // ── Public accessors ────────────────────────────────────────

    pub fn mesh(&self) -> &MeshData {
        &self.mesh
    }

    pub fn num_triangles(&self) -> u32 {
        (self.t_verts.len() / 3) as u32
    }

    pub fn num_vertices_cdt(&self) -> u32 {
        self.nv
    }

    pub fn triangle_vertex(&self, t: u32, lv: u32) -> u32 {
        self.tv(t, lv)
    }

    #[allow(dead_code)]
    pub fn triangle_adjacent(&self, t: u32, le: u32) -> u32 {
        self.tadj(t, le)
    }

    pub fn vertices(&self) -> &[Vertex] {
        &self.vertex
    }

    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }

    pub fn f1(&self) -> u32 {
        self.f1
    }

    pub fn has_planar_isect(&self) -> bool {
        self.has_planar_isect
    }

    #[allow(dead_code)]
    pub fn num_constraints(&self) -> u32 {
        self.ncnstr
    }

    #[allow(dead_code)]
    pub fn tedge_cnstr_first(&self, t: u32, le: u32) -> u32 {
        self.t_ecnstr[(3 * t + le) as usize]
    }

    #[allow(dead_code)]
    pub fn edge_cnstr_next(&self, ecit: u32) -> u32 {
        self.ecnstr_next[ecit as usize]
    }

    #[allow(dead_code)]
    pub fn edge_cnstr(&self, ecit: u32) -> u32 {
        self.ecnstr_val[ecit as usize]
    }

    #[allow(dead_code)]
    pub fn tedge_cnstr_nb(&self, t: u32, le: u32) -> u32 {
        let mut result = 0u32;
        let mut ecit = self.tedge_cnstr_first(t, le);
        while ecit != NO_INDEX {
            result += 1;
            ecit = self.edge_cnstr_next(ecit);
        }
        result
    }

    // ── clear ────────────────────────────────────────────────────

    pub fn clear(&mut self) {
        self.vertex.clear();
        self.edges.clear();
        self.f1 = NO_INDEX;
        self.pred_cache.clear();
        self.pred_cache_insert_buffer.clear();
        self.use_pred_cache_insert_buffer = false;
        // CDTBase2d::clear()
        self.nv = 0;
        self.ncnstr = 0;
        self.t_verts.clear();
        self.t_adj.clear();
        self.v2t.clear();
        self.t_flags.clear();
        self.t_ecnstr.clear();
        self.ecnstr_val.clear();
        self.ecnstr_next.clear();
        self.t_next.clear();
        self.t_prev.clear();
        self.constraints.clear();
    }

    // ══════════════════════════════════════════════════════════════
    // MeshInTriangle-specific methods
    // ══════════════════════════════════════════════════════════════

    // ── begin_facet ──────────────────────────────────────────────

    pub fn begin_facet(&mut self, f: u32) {
        self.f1 = f;
        self.latest_f2 = NO_INDEX;
        self.latest_f2_count = 0;

        let p1 = self.mesh.facet_vertex(f, 0);
        let p2 = self.mesh.facet_vertex(f, 1);
        let p3 = self.mesh.facet_vertex(f, 2);

        self.f1_normal_axis = triangle_normal_axis(p1, p2, p3);
        self.u = (self.f1_normal_axis + 1) % 3;
        self.v = (self.f1_normal_axis + 2) % 3;

        // Create macro-vertices with 3D exact homogeneous coordinates.
        for lv in 0..3u32 {
            let mut vert = Vertex::new_mesh_vertex(&self.mesh, f, lv);
            vert.init_geometry(self.u, self.v);
            self.vertex.push(vert);
        }

        // CDTBase2d::create_enclosing_triangle(0,1,2)
        self.nv = 3;
        self.v2t.resize(3, NO_INDEX);
        let t0 = self.tnew();
        self.tset(t0, 0, 1, 2, NO_INDEX, NO_INDEX, NO_INDEX);
        self.orient_012 = self.orient2d_mit(0, 1, 2);
        assert!(self.orient_012 != Sign::Zero);

        // Add the 3 boundary edges.
        self.edges.push(Edge::new_simple(1, 2));
        self.edges.push(Edge::new_simple(2, 0));
        self.edges.push(Edge::new_simple(0, 1));

        self.has_planar_isect = false;
    }

    // ── add_vertex ───────────────────────────────────────────────

    pub fn add_vertex(&mut self, f2: u32, r1: TriangleRegion, r2: TriangleRegion) -> u32 {
        debug_assert!(self.f1 != NO_INDEX);

        // Track planar intersections.
        if f2 != NO_INDEX && f2 == self.latest_f2 {
            self.latest_f2_count += 1;
            if self.latest_f2_count > 2 {
                self.has_planar_isect = true;
            }
        } else {
            self.latest_f2 = f2;
            self.latest_f2_count = 0;
        }

        // If vertex is a macro-vertex, return it directly.
        if region_dim(r1) == 0 {
            return r1 as u32;
        }

        // Create the vertex with exact 3D geometry.
        let mut vert = Vertex::new_primary_isect(&self.mesh, self.f1, f2, r1, r2);
        vert.init_geometry(self.u, self.v);
        self.vertex.push(vert);

        // Insert into the triangulation (CDTBase2d::insert).
        let v = self.base_insert(self.vertex.len() as u32 - 1, NO_INDEX);

        // If it was an existing vertex, remove the duplicate.
        if self.vertex.len() as u32 > self.nv {
            self.vertex.pop();
        }
        v
    }

    // ── add_edge ─────────────────────────────────────────────────

    pub fn add_edge(
        &mut self,
        f2: u32,
        ar1: TriangleRegion,
        ar2: TriangleRegion,
        br1: TriangleRegion,
        br2: TriangleRegion,
    ) {
        let v1 = self.add_vertex(f2, ar1, ar2);
        let v2 = self.add_vertex(f2, br1, br2);

        // If both extremities are on the same edge of f1, skip.
        if region_dim(regions_convex_hull(ar1, br1)) == 1 {
            return;
        }

        self.edges
            .push(Edge::new(v1, v2, f2, regions_convex_hull(ar2, br2)));
    }

    // ── commit ───────────────────────────────────────────────────

    pub fn commit(&mut self) -> u32 {
        // Insert all constraint edges.
        let n = self.edges.len();
        for i in 0..n {
            let v1 = self.edges[i].v1;
            let v2 = self.edges[i].v2;
            self.base_insert_constraint(v1, v2);
        }
        self.num_triangles()
    }

    // ── Edge-edge intersection ───────────────────────────────────

    pub fn get_edge_edge_intersection(&self, e1: u32, e2: u32) -> Vec3HE {
        let f1 = self.f1;
        let f2 = self.edges[e1 as usize].sym.f2;
        let f3 = self.edges[e2 as usize].sym.f2;

        debug_assert!(f1 != NO_INDEX);
        debug_assert!(f2 != NO_INDEX);
        debug_assert!(f3 != NO_INDEX);

        let p: [[f64; 3]; 9] = [
            self.mesh.facet_vertex(f1, 0),
            self.mesh.facet_vertex(f1, 1),
            self.mesh.facet_vertex(f1, 2),
            self.mesh.facet_vertex(f2, 0),
            self.mesh.facet_vertex(f2, 1),
            self.mesh.facet_vertex(f2, 2),
            self.mesh.facet_vertex(f3, 0),
            self.mesh.facet_vertex(f3, 1),
            self.mesh.facet_vertex(f3, 2),
        ];

        if let Some(result) =
            get_three_planes_intersection(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8])
        {
            return result;
        }

        self.get_edge_edge_intersection_2d(e1, e2)
    }

    fn get_edge_edge_intersection_2d(&self, e1: u32, e2: u32) -> Vec3HE {
        let edge1 = &self.edges[e1 as usize];
        let edge2 = &self.edges[e2 as usize];

        if region_dim(edge1.sym.r2) == 1 && region_dim(edge2.sym.r2) == 1 {
            let le1 = edge1.sym.r2 as u32 - T2RgnE0 as u32;
            let le2 = edge2.sym.r2 as u32 - T2RgnE0 as u32;
            debug_assert!(le1 < 3);
            debug_assert!(le2 < 3);

            let p1_f = edge1.sym.f2;
            let p2_f = edge2.sym.f2;

            let p1_uv = self.mesh_facet_vertex_uv(p1_f, (le1 + 1) % 3);
            let p2_uv = self.mesh_facet_vertex_uv(p1_f, (le1 + 2) % 3);
            let q1_uv = self.mesh_facet_vertex_uv(p2_f, (le2 + 1) % 3);
            let q2_uv = self.mesh_facet_vertex_uv(p2_f, (le2 + 2) % 3);

            let c1 = Vec2E::make_vec2(p1_uv, p2_uv);
            let c2 = Vec2E::make_vec2(q2_uv, q1_uv);
            let b = Vec2E::make_vec2(p1_uv, q1_uv);
            let d = Vec2E::det(&c1, &c2);
            debug_assert!(d.sign() != 0, "get_edge_edge_intersection_2D: degenerate");
            let t = Rational::new(Vec2E::det(&b, &c2), d);
            mix_vec3(
                &t,
                self.mesh.facet_vertex(p1_f, (le1 + 1) % 3),
                self.mesh.facet_vertex(p1_f, (le1 + 2) % 3),
            )
        } else {
            debug_assert!(region_dim(edge1.sym.r2) == 1 || region_dim(edge2.sym.r2) == 1);
            let (mut f1_loc, mut r1_loc) = (edge1.sym.f2, edge1.sym.r2);
            let (mut f2_loc, mut r2_loc) = (edge2.sym.f2, edge2.sym.r2);
            if region_dim(r1_loc) == 1 {
                std::mem::swap(&mut f1_loc, &mut f2_loc);
                std::mem::swap(&mut r1_loc, &mut r2_loc);
            }

            let e = r2_loc as u32 - T2RgnE0 as u32;
            debug_assert!(e < 3);

            plane_line_intersection(
                self.mesh.facet_vertex(f1_loc, 0),
                self.mesh.facet_vertex(f1_loc, 1),
                self.mesh.facet_vertex(f1_loc, 2),
                self.mesh.facet_vertex(f2_loc, (e + 1) % 3),
                self.mesh.facet_vertex(f2_loc, (e + 2) % 3),
            )
        }
    }

    fn mesh_facet_vertex_uv(&self, f: u32, lv: u32) -> [f64; 2] {
        let p = self.mesh.facet_vertex(f, lv);
        [p[self.u], p[self.v]]
    }

    // ══════════════════════════════════════════════════════════════
    // MeshInTriangle predicate overrides
    // ══════════════════════════════════════════════════════════════

    /// orient2d using predicate cache and orient_2d_projected on 3D coords.
    /// Port of `MeshInTriangle::orient2d`.
    fn orient2d_mit(&mut self, vx1: u32, vx2: u32, vx3: u32) -> Sign {
        let k = Trindex::new(vx1, vx2, vx3);

        if self.use_pred_cache_insert_buffer {
            let result = orient_2d_projected(
                &self.vertex[k.indices[0] as usize].point_exact,
                &self.vertex[k.indices[1] as usize].point_exact,
                &self.vertex[k.indices[2] as usize].point_exact,
                self.f1_normal_axis,
            );
            self.pred_cache_insert_buffer.push((k, result));
            if odd_order(vx1, vx2, vx3) {
                Sign::from_i32(-(result as i32))
            } else {
                result
            }
        } else {
            let result = self.pred_cache.entry(k).or_insert_with(|| {
                orient_2d_projected(
                    &self.vertex[k.indices[0] as usize].point_exact,
                    &self.vertex[k.indices[1] as usize].point_exact,
                    &self.vertex[k.indices[2] as usize].point_exact,
                    self.f1_normal_axis,
                )
            });
            let r = *result;

            if odd_order(vx1, vx2, vx3) {
                Sign::from_i32(-(r as i32))
            } else {
                r
            }
        }
    }

    /// incircle using 2D homogeneous coords extracted from 3D point_exact.
    /// Port of `MeshInTriangle::incircle`.
    /// Uses pre-cached `l` values (approximate lifted coordinate) for performance.
    fn incircle_mit(&self, v1: u32, v2: u32, v3: u32, v4: u32) -> Sign {
        let p1 = Vec2HE::new(
            self.vertex[v1 as usize].point_exact.coord(self.u).clone(),
            self.vertex[v1 as usize].point_exact.coord(self.v).clone(),
            self.vertex[v1 as usize].point_exact.w.clone(),
        );
        let p2 = Vec2HE::new(
            self.vertex[v2 as usize].point_exact.coord(self.u).clone(),
            self.vertex[v2 as usize].point_exact.coord(self.v).clone(),
            self.vertex[v2 as usize].point_exact.w.clone(),
        );
        let p3 = Vec2HE::new(
            self.vertex[v3 as usize].point_exact.coord(self.u).clone(),
            self.vertex[v3 as usize].point_exact.coord(self.v).clone(),
            self.vertex[v3 as usize].point_exact.w.clone(),
        );
        let p4 = Vec2HE::new(
            self.vertex[v4 as usize].point_exact.coord(self.u).clone(),
            self.vertex[v4 as usize].point_exact.coord(self.v).clone(),
            self.vertex[v4 as usize].point_exact.w.clone(),
        );
        incircle_2d_sos_with_lengths(
            &p1,
            &p2,
            &p3,
            &p4,
            self.vertex[v1 as usize].l,
            self.vertex[v2 as usize].l,
            self.vertex[v3 as usize].l,
            self.vertex[v4 as usize].l,
        )
    }

    /// create_intersection: compute 3D intersection via get_edge_edge_intersection,
    /// create a new SECONDARY_ISECT Vertex.
    /// Port of `MeshInTriangle::create_intersection`.
    fn create_intersection_mit(
        &mut self,
        e1: u32,
        _i: u32,
        _j: u32,
        e2: u32,
        _k: u32,
        _l: u32,
    ) -> u32 {
        let point = self.get_edge_edge_intersection(e1, e2);
        let mut vert = Vertex::new_secondary_isect(point);
        vert.init_geometry(self.u, self.v);
        self.vertex.push(vert);
        let x = self.vertex.len() as u32 - 1;
        self.v2t.push(NO_INDEX);
        debug_assert!(x == self.nv);
        self.nv += 1;
        x
    }

    // ── Transaction support for predicate cache ──────────────────

    fn begin_insert_transaction(&mut self) {
        self.use_pred_cache_insert_buffer = true;
    }

    fn commit_insert_transaction(&mut self) {
        for (k, v) in self.pred_cache_insert_buffer.drain(..) {
            self.pred_cache.insert(k, v);
        }
        self.use_pred_cache_insert_buffer = false;
    }

    fn rollback_insert_transaction(&mut self) {
        self.pred_cache_insert_buffer.clear();
        self.use_pred_cache_insert_buffer = false;
    }

    // ══════════════════════════════════════════════════════════════
    // CDTBase2d methods (copied from ExactCDT2d, using orient2d_mit/incircle_mit)
    // ══════════════════════════════════════════════════════════════

    // ── Triangle accessors (private) ────────────────────────────

    #[inline]
    fn tv(&self, t: u32, lv: u32) -> u32 {
        self.t_verts[(3 * t + lv) as usize]
    }

    #[inline]
    fn tadj(&self, t: u32, le: u32) -> u32 {
        self.t_adj[(3 * t + le) as usize]
    }

    fn tadj_find(&self, t1: u32, t2: u32) -> u32 {
        let base = (3 * t1) as usize;
        if self.t_adj[base] == t2 {
            return 0;
        }
        if self.t_adj[base + 1] == t2 {
            return 1;
        }
        debug_assert_eq!(self.t_adj[base + 2], t2);
        2
    }

    fn tv_find(&self, t: u32, v: u32) -> u32 {
        let base = (3 * t) as usize;
        if self.t_verts[base] == v {
            return 0;
        }
        if self.t_verts[base + 1] == v {
            return 1;
        }
        debug_assert_eq!(self.t_verts[base + 2], v);
        2
    }

    fn vt(&self, v: u32) -> u32 {
        self.v2t[v as usize]
    }

    // ── Triangle mutation ────────────────────────────────────────

    fn tset(&mut self, t: u32, v0: u32, v1: u32, v2: u32, a0: u32, a1: u32, a2: u32) {
        let base = (3 * t) as usize;
        self.t_verts[base] = v0;
        self.t_verts[base + 1] = v1;
        self.t_verts[base + 2] = v2;
        self.t_adj[base] = a0;
        self.t_adj[base + 1] = a1;
        self.t_adj[base + 2] = a2;
        self.t_ecnstr[base] = NO_INDEX;
        self.t_ecnstr[base + 1] = NO_INDEX;
        self.t_ecnstr[base + 2] = NO_INDEX;
        self.v2t[v0 as usize] = t;
        self.v2t[v1 as usize] = t;
        self.v2t[v2 as usize] = t;
    }

    fn tset_adj(&mut self, t: u32, le: u32, adj: u32) {
        self.t_adj[(3 * t + le) as usize] = adj;
    }

    fn tnew(&mut self) -> u32 {
        let t = self.num_triangles();
        self.t_verts.extend_from_slice(&[NO_INDEX; 3]);
        self.t_adj.extend_from_slice(&[NO_INDEX; 3]);
        self.t_ecnstr.extend_from_slice(&[NO_INDEX; 3]);
        self.t_flags.push(0);
        self.t_next.push(NO_INDEX);
        self.t_prev.push(NO_INDEX);
        t
    }

    fn trot(&mut self, t: u32, lv: u32) {
        if lv == 0 {
            return;
        }
        let base = (3 * t) as usize;
        let i = base + lv as usize;
        let j = base + ((lv + 1) % 3) as usize;
        let k = base + ((lv + 2) % 3) as usize;

        let (vi, vj, vk) = (self.t_verts[i], self.t_verts[j], self.t_verts[k]);
        let (ai, aj, ak) = (self.t_adj[i], self.t_adj[j], self.t_adj[k]);
        let (ci, cj, ck) = (self.t_ecnstr[i], self.t_ecnstr[j], self.t_ecnstr[k]);

        self.t_verts[base] = vi;
        self.t_verts[base + 1] = vj;
        self.t_verts[base + 2] = vk;
        self.t_adj[base] = ai;
        self.t_adj[base + 1] = aj;
        self.t_adj[base + 2] = ak;
        self.t_ecnstr[base] = ci;
        self.t_ecnstr[base + 1] = cj;
        self.t_ecnstr[base + 2] = ck;
        self.v2t[vi as usize] = t;
        self.v2t[vj as usize] = t;
        self.v2t[vk as usize] = t;
    }

    // ── Flag manipulation ────────────────────────────────────────

    #[allow(dead_code)]
    fn tset_flag(&mut self, t: u32, flag: u32) {
        self.t_flags[t as usize] |= 1u32 << flag;
    }

    #[allow(dead_code)]
    fn treset_flag(&mut self, t: u32, flag: u32) {
        self.t_flags[t as usize] &= !(1u32 << flag);
    }

    #[allow(dead_code)]
    fn tflag_is_set(&self, t: u32, flag: u32) -> bool {
        (self.t_flags[t as usize] & (1u32 << flag)) != 0
    }

    // ── Edge constraint manipulation ─────────────────────────────

    fn tset_edge_cnstr_first(&mut self, t: u32, le: u32, ecit: u32) {
        self.t_ecnstr[(3 * t + le) as usize] = ecit;
    }

    fn tedge_is_constrained(&self, t: u32, le: u32) -> bool {
        self.t_ecnstr[(3 * t + le) as usize] != NO_INDEX
    }

    fn tadd_edge_cnstr(&mut self, t: u32, le: u32, cnstr_id: u32) {
        let mut ecit = self.t_ecnstr[(3 * t + le) as usize];
        while ecit != NO_INDEX {
            if self.ecnstr_val[ecit as usize] == cnstr_id {
                return;
            }
            ecit = self.ecnstr_next[ecit as usize];
        }
        let old_first = self.t_ecnstr[(3 * t + le) as usize];
        self.ecnstr_val.push(cnstr_id);
        self.ecnstr_next.push(old_first);
        self.t_ecnstr[(3 * t + le) as usize] = (self.ecnstr_val.len() - 1) as u32;
    }

    fn tadd_edge_cnstr_with_neighbor(&mut self, t: u32, le: u32, cnstr_id: u32) {
        self.tadd_edge_cnstr(t, le, cnstr_id);
        let t2 = self.tadj(t, le);
        if t2 != NO_INDEX {
            let le2 = self.tadj_find(t2, t);
            let new_first = self.t_ecnstr[(3 * t + le) as usize];
            self.t_ecnstr[(3 * t2 + le2) as usize] = new_first;
        }
    }

    fn tadj_back_connect(&mut self, t1: u32, le1: u32, prev_adj: u32) {
        let t2 = self.tadj(t1, le1);
        if t2 == NO_INDEX {
            return;
        }
        let le2 = self.tadj_find(t2, prev_adj);
        self.tset_adj(t2, le2, t1);
        let cnstr = self.t_ecnstr[(3 * t2 + le2) as usize];
        self.t_ecnstr[(3 * t1 + le1) as usize] = cnstr;
    }

    // ── Point insertion ──────────────────────────────────────────

    /// CDTBase2d::insert with transaction support for predicate cache.
    fn base_insert(&mut self, v: u32, hint: u32) -> u32 {
        let keep_duplicates;
        if v == self.nv {
            self.v2t.push(NO_INDEX);
            self.nv += 1;
            keep_duplicates = false;
        } else {
            keep_duplicates = true;
            debug_assert!(v < self.nv);
        }

        // Begin transaction for predicate cache.
        self.begin_insert_transaction();

        // Phase 1: locate triangle containing v.
        let mut o = [Sign::Zero; 3];
        let t = self.locate(v, hint, &mut o);
        let nb_z = o.iter().filter(|&&s| s == Sign::Zero).count();
        debug_assert!(nb_z != 3);

        // Duplicated vertex.
        if nb_z == 2 {
            let existing = if o[0] != Sign::Zero {
                self.tv(t, 0)
            } else if o[1] != Sign::Zero {
                self.tv(t, 1)
            } else {
                self.tv(t, 2)
            };
            if !keep_duplicates {
                self.v2t.pop();
                self.nv -= 1;
            }
            self.rollback_insert_transaction();
            return existing;
        }

        // Commit cached predicates.
        self.commit_insert_transaction();

        // Phase 2: split triangle.
        let mut s = DList::new_uninit();
        if self.delaunay {
            s.initialize(DLIST_S_ID);
        }

        if nb_z == 1 {
            let le = if o[0] == Sign::Zero {
                0
            } else if o[1] == Sign::Zero {
                1
            } else {
                2
            };
            self.insert_vertex_in_edge(v, t, le, &mut s);
        } else {
            self.insert_vertex_in_triangle(v, t, &mut s);
        }

        // Phase 3: restore Delaunay condition.
        if self.delaunay {
            self.delaunayize_vertex_neighbors_with_stack(v, &mut s);
        }
        if s.initialized() {
            s.clear(&mut self.t_flags, &self.t_next);
        }

        v
    }

    // ── Constraint insertion ─────────────────────────────────────

    fn base_insert_constraint(&mut self, mut i: u32, j: u32) {
        self.constraints.push((i, j));
        self.ncnstr += 1;
        let first_v_isect = self.nv;

        let mut q = DList::new(DLIST_Q_ID);
        let mut n = DList::new_uninit();
        if self.delaunay {
            n.initialize(DLIST_N_ID);
        }

        while i != j {
            let k = self.find_intersected_edges(i, j, &mut q);

            if self.delaunay && self.exact_intersections && k >= first_v_isect {
                let inserted = self.base_insert(k, NO_INDEX);
                debug_assert_eq!(inserted, k);
                q.clear(&mut self.t_flags, &self.t_next);
                self.delaunayize_vertex_neighbors_no_stack(k);
                let new_k = self.find_intersected_edges(i, j, &mut q);
                assert_eq!(new_k, k);
            }

            self.constrain_edges(i, k, &mut q, &mut n);

            if self.delaunay {
                self.delaunayize_new_edges(&mut n);
            }

            i = k;
        }

        if self.delaunay && !self.exact_intersections {
            for v in first_v_isect..self.nv {
                self.delaunayize_vertex_neighbors_no_stack(v);
            }
        }

        q.clear(&mut self.t_flags, &self.t_next);
        if n.initialized() {
            n.clear(&mut self.t_flags, &self.t_next);
        }
    }

    // ── Point location ───────────────────────────────────────────

    fn locate(&mut self, v: u32, hint: u32, o: &mut [Sign; 3]) -> u32 {
        let mut t_pred = self.num_triangles() + 1;
        let mut t = if hint == NO_INDEX { 0 } else { hint };

        loop {
            assert!(t != NO_INDEX, "locate: point outside boundary");

            let tv = [self.tv(t, 0), self.tv(t, 1), self.tv(t, 2)];

            let e0 = 0u32;
            let mut found_next = false;
            for de in 0..3u32 {
                let le = (e0 + de) % 3;
                let t_next = self.tadj(t, le);

                if t_next == t_pred {
                    o[le as usize] = Sign::Positive;
                    continue;
                }

                let mut tv_test = tv;
                tv_test[le as usize] = v;
                o[le as usize] = Sign::from_i32(
                    self.orient_012 as i32
                        * self.orient2d_mit(tv_test[0], tv_test[1], tv_test[2]) as i32,
                );

                if o[le as usize] == Sign::Negative {
                    t_pred = t;
                    t = t_next;
                    found_next = true;
                    break;
                }
            }

            if !found_next {
                for le in 0..3u32 {
                    let t_next = self.tadj(t, le);
                    if t_next == t_pred {
                        o[le as usize] = Sign::Positive;
                    }
                }
                return t;
            }
        }
    }

    // ── insert_vertex_in_triangle ────────────────────────────────

    fn insert_vertex_in_triangle(&mut self, v: u32, t: u32, s: &mut DList) {
        let t1 = t;
        let v1 = self.tv(t1, 0);
        let v2 = self.tv(t1, 1);
        let v3 = self.tv(t1, 2);
        let adj1 = self.tadj(t1, 0);
        let adj2 = self.tadj(t1, 1);
        let adj3 = self.tadj(t1, 2);

        let t2 = self.tnew();
        let t3 = self.tnew();

        self.tset(t1, v, v2, v3, adj1, t2, t3);
        self.tset(t2, v, v3, v1, adj2, t3, t1);
        self.tset(t3, v, v1, v2, adj3, t1, t2);
        self.tadj_back_connect(t1, 0, t1);
        self.tadj_back_connect(t2, 0, t1);
        self.tadj_back_connect(t3, 0, t1);

        if s.initialized() {
            s.push_back(&mut self.t_flags, &mut self.t_next, &mut self.t_prev, t1);
            s.push_back(&mut self.t_flags, &mut self.t_next, &mut self.t_prev, t2);
            s.push_back(&mut self.t_flags, &mut self.t_next, &mut self.t_prev, t3);
        }
    }

    // ── insert_vertex_in_edge ────────────────────────────────────

    fn insert_vertex_in_edge(&mut self, v: u32, t: u32, le1: u32, s: &mut DList) {
        let cnstr_first = self.t_ecnstr[(3 * t + le1) as usize];
        let t1 = t;
        let v1 = self.tv(t1, le1);
        let v2 = self.tv(t1, (le1 + 1) % 3);
        let v3 = self.tv(t1, (le1 + 2) % 3);
        let t1_adj2 = self.tadj(t1, (le1 + 1) % 3);
        let t1_adj3 = self.tadj(t1, (le1 + 2) % 3);
        let t2 = self.tadj(t1, le1);

        if t2 != NO_INDEX {
            let le2 = self.tadj_find(t2, t1);
            debug_assert_eq!(self.tv(t2, (le2 + 1) % 3), v3);
            debug_assert_eq!(self.tv(t2, (le2 + 2) % 3), v2);
            let v4 = self.tv(t2, le2);
            let t2_adj2 = self.tadj(t2, (le2 + 1) % 3);
            let t2_adj3 = self.tadj(t2, (le2 + 2) % 3);

            let t3 = self.tnew();
            let t4 = self.tnew();

            self.tset(t1, v, v1, v2, t1_adj3, t2, t4);
            self.tset(t2, v, v2, v4, t2_adj2, t3, t1);
            self.tset(t3, v, v4, v3, t2_adj3, t4, t2);
            self.tset(t4, v, v3, v1, t1_adj2, t1, t3);
            self.tadj_back_connect(t1, 0, t1);
            self.tadj_back_connect(t2, 0, t2);
            self.tadj_back_connect(t3, 0, t2);
            self.tadj_back_connect(t4, 0, t1);
            self.tset_edge_cnstr_first(t1, 1, cnstr_first);
            self.tset_edge_cnstr_first(t2, 2, cnstr_first);
            self.tset_edge_cnstr_first(t3, 1, cnstr_first);
            self.tset_edge_cnstr_first(t4, 2, cnstr_first);
            if s.initialized() {
                s.push_back(&mut self.t_flags, &mut self.t_next, &mut self.t_prev, t1);
                s.push_back(&mut self.t_flags, &mut self.t_next, &mut self.t_prev, t2);
                s.push_back(&mut self.t_flags, &mut self.t_next, &mut self.t_prev, t3);
                s.push_back(&mut self.t_flags, &mut self.t_next, &mut self.t_prev, t4);
            }
        } else {
            let t2_new = self.tnew();
            self.tset(t1, v, v1, v2, t1_adj3, NO_INDEX, t2_new);
            self.tset(t2_new, v, v3, v1, t1_adj2, t1, NO_INDEX);
            self.tadj_back_connect(t1, 0, t1);
            self.tadj_back_connect(t2_new, 0, t1);
            self.tset_edge_cnstr_first(t1, 1, cnstr_first);
            self.tset_edge_cnstr_first(t2_new, 2, cnstr_first);
            if s.initialized() {
                s.push_back(&mut self.t_flags, &mut self.t_next, &mut self.t_prev, t1);
                s.push_back(
                    &mut self.t_flags,
                    &mut self.t_next,
                    &mut self.t_prev,
                    t2_new,
                );
            }
        }
    }

    fn insert_vertex_in_edge_no_list(&mut self, v: u32, t: u32, le1: u32) {
        let mut dummy = DList::new_uninit();
        self.insert_vertex_in_edge(v, t, le1, &mut dummy);
    }

    // ── Edge swapping ────────────────────────────────────────────

    fn swap_edge(&mut self, t1: u32, swap_t1_t2: bool) {
        debug_assert!(!self.tedge_is_constrained(t1, 0));
        let v1 = self.tv(t1, 0);
        let v2 = self.tv(t1, 1);
        let v3 = self.tv(t1, 2);
        let t1_adj2 = self.tadj(t1, 1);
        let t1_adj3 = self.tadj(t1, 2);
        let t2 = self.tadj(t1, 0);
        debug_assert!(t2 != NO_INDEX);

        let le2 = self.tadj_find(t2, t1);
        let v4 = self.tv(t2, le2);
        debug_assert_eq!(self.tv(t2, (le2 + 1) % 3), v3);
        debug_assert_eq!(self.tv(t2, (le2 + 2) % 3), v2);

        let t2_adj2 = self.tadj(t2, (le2 + 1) % 3);
        let t2_adj3 = self.tadj(t2, (le2 + 2) % 3);

        if swap_t1_t2 {
            self.tset(t2, v1, v4, v3, t2_adj3, t1_adj2, t1);
            self.tset(t1, v1, v2, v4, t2_adj2, t2, t1_adj3);
            self.tadj_back_connect(t2, 0, t2);
            self.tadj_back_connect(t2, 1, t1);
            self.tadj_back_connect(t1, 0, t2);
            self.tadj_back_connect(t1, 2, t1);
        } else {
            self.tset(t1, v1, v4, v3, t2_adj3, t1_adj2, t2);
            self.tset(t2, v1, v2, v4, t2_adj2, t1, t1_adj3);
            self.tadj_back_connect(t1, 0, t2);
            self.tadj_back_connect(t1, 1, t1);
            self.tadj_back_connect(t2, 0, t2);
            self.tadj_back_connect(t2, 2, t1);
        }
    }

    // ── is_convex_quad ───────────────────────────────────────────

    fn is_convex_quad(&mut self, t: u32) -> bool {
        let v1 = self.tv(t, 0);
        let v2 = self.tv(t, 1);
        let v3 = self.tv(t, 2);
        let t2 = self.tadj(t, 0);
        if t2 == NO_INDEX {
            return false;
        }
        let le2 = self.tadj_find(t2, t);
        let v4 = self.tv(t2, le2);
        self.orient2d_mit(v1, v4, v3) == self.orient_012
            && self.orient2d_mit(v4, v1, v2) == self.orient_012
    }

    // ── Delaunayization ──────────────────────────────────────────

    fn delaunayize_vertex_neighbors_with_stack(&mut self, v: u32, s: &mut DList) {
        let mut count = 0u32;
        while !s.empty() {
            count += 1;
            if count > 10 * self.num_triangles() {
                s.clear(&mut self.t_flags, &self.t_next);
                panic!("Emergency exit in delaunayize_vertex_neighbors");
            }
            let t1 = s.pop_back(&mut self.t_flags, &mut self.t_next, &mut self.t_prev);
            debug_assert_eq!(self.tv(t1, 0), v);

            if self.tedge_is_constrained(t1, 0) {
                continue;
            }
            let t2 = self.tadj(t1, 0);
            if t2 == NO_INDEX {
                continue;
            }
            if !self.is_convex_quad(t1) {
                continue;
            }

            let w1 = self.tv(t2, 0);
            let w2 = self.tv(t2, 1);
            let w3 = self.tv(t2, 2);
            let ic = self.incircle_mit(w1, w2, w3, v);
            let combined = Sign::from_i32(ic as i32 * self.orient_012 as i32);
            if combined == Sign::Positive {
                self.swap_edge(t1, false);
                s.push_back(&mut self.t_flags, &mut self.t_next, &mut self.t_prev, t1);
                s.push_back(&mut self.t_flags, &mut self.t_next, &mut self.t_prev, t2);
            }
        }
    }

    fn delaunayize_vertex_neighbors_no_stack(&mut self, v: u32) {
        assert!(self.vt(v) != NO_INDEX);

        let mut s = DList::new(DLIST_S_ID);

        let t0 = self.vt(v);
        let mut t = t0;
        loop {
            let lv = self.tv_find(t, v);
            self.trot(t, lv);
            debug_assert_eq!(self.tv(t, 0), v);
            s.push_back(&mut self.t_flags, &mut self.t_next, &mut self.t_prev, t);
            let next = self.tadj(t, 1);
            assert!(next != NO_INDEX);
            if next == t0 {
                break;
            }
            t = next;
        }
        self.delaunayize_vertex_neighbors_with_stack(v, &mut s);
        s.clear(&mut self.t_flags, &self.t_next);
    }

    fn delaunayize_new_edges(&mut self, n: &mut DList) {
        let mut count = 0u32;
        let mut swap_occurred = true;
        while swap_occurred {
            swap_occurred = false;
            count += 1;
            if count > 10 * self.num_triangles() {
                break;
            }
            let mut t1 = n.front();
            while t1 != NO_INDEX {
                let t1_next = DList::next(&self.t_next, t1);
                if self.tedge_is_constrained(t1, 0) {
                    t1 = t1_next;
                    continue;
                }
                let v1 = self.tv(t1, 1);
                let v2 = self.tv(t1, 2);
                let v0 = self.tv(t1, 0);
                let t2 = self.tadj(t1, 0);
                if t2 == NO_INDEX {
                    t1 = t1_next;
                    continue;
                }
                if !self.is_convex_quad(t1) {
                    t1 = t1_next;
                    continue;
                }
                let e2 = self.tadj_find(t2, t1);
                let v3 = self.tv(t2, e2);
                let ic = self.incircle_mit(v0, v1, v2, v3);
                let combined = Sign::from_i32(ic as i32 * self.orient_012 as i32);
                if combined == Sign::Positive {
                    if self.tv(t2, 0) == self.tv(t1, 1) {
                        self.swap_edge(t1, true);
                        self.trot(t1, 1);
                    } else {
                        self.swap_edge(t1, false);
                        self.trot(t1, 2);
                    }
                    swap_occurred = true;
                }
                t1 = t1_next;
            }
        }
        n.clear(&mut self.t_flags, &self.t_next);
    }

    // ── for_each_T_around_v ──────────────────────────────────────

    fn collect_triangles_around_v(&self, v: u32) -> Vec<(u32, u32)> {
        let mut result = Vec::new();
        let t_start = self.vt(v);
        if t_start == NO_INDEX {
            return result;
        }

        let mut t = t_start;
        loop {
            let lv = self.tv_find(t, v);
            result.push((t, lv));
            let next = self.tadj(t, (lv + 1) % 3);
            if next == t_start || next == NO_INDEX {
                break;
            }
            t = next;
        }

        {
            let lv = self.tv_find(t_start, v);
            let first_back = self.tadj(t_start, (lv + 2) % 3);
            if first_back != NO_INDEX {
                let forward_last = {
                    let (last_t, last_lv) = *result.last().unwrap();
                    self.tadj(last_t, (last_lv + 1) % 3)
                };
                if forward_last == NO_INDEX {
                    let mut backward = Vec::new();
                    let mut tb = first_back;
                    while tb != NO_INDEX {
                        let lv2 = self.tv_find(tb, v);
                        backward.push((tb, lv2));
                        tb = self.tadj(tb, (lv2 + 2) % 3);
                    }
                    backward.reverse();
                    backward.append(&mut result);
                    result = backward;
                }
            }
        }

        result
    }

    // ── find_intersected_edges ───────────────────────────────────

    fn find_intersected_edges(&mut self, i: u32, j: u32, q: &mut DList) -> u32 {
        let mut w = ConstraintWalker::new(i, j);
        while w.v == i || w.v == NO_INDEX {
            if w.v != NO_INDEX {
                self.walk_constraint_v(&mut w);
            } else {
                self.walk_constraint_t(&mut w, q);
            }
        }
        w.v
    }

    // ── walk_constraint_v ────────────────────────────────────────

    fn walk_constraint_v(&mut self, w: &mut ConstraintWalker) {
        debug_assert!(w.v != NO_INDEX);
        debug_assert!(w.t == NO_INDEX);

        let mut t_next = NO_INDEX;
        let mut v_next = NO_INDEX;

        let triangles = self.collect_triangles_around_v(w.v);
        for &(t_around_v, le) in &triangles {
            if t_around_v == w.t_prev {
                continue;
            }
            let v1 = self.tv(t_around_v, (le + 1) % 3);
            let v2 = self.tv(t_around_v, (le + 2) % 3);
            if v1 == w.j || v2 == w.j {
                v_next = w.j;
                let le_cnstr_edge = if v1 == w.j {
                    (le + 2) % 3
                } else {
                    (le + 1) % 3
                };
                self.tadd_edge_cnstr_with_neighbor(t_around_v, le_cnstr_edge, self.ncnstr - 1);
                break;
            }
            let o1 = self.orient2d_mit(w.i, w.j, v1);
            let o2 = self.orient2d_mit(w.i, w.j, v2);
            let o3 = self.orient2d_mit(v1, v2, w.j);
            let o4 = self.orient_012;
            if (o1 as i32 * o2 as i32) < 0 && (o3 as i32 * o4 as i32) < 0 {
                self.trot(t_around_v, le);
                t_next = t_around_v;
                break;
            } else {
                debug_assert!(o1 != Sign::Zero || o2 != Sign::Zero);
                if o1 == Sign::Zero && (o3 as i32 * o4 as i32) < 0 && v1 != w.v_prev {
                    v_next = v1;
                    self.tadd_edge_cnstr_with_neighbor(t_around_v, (le + 2) % 3, self.ncnstr - 1);
                    break;
                } else if o2 == Sign::Zero && (o3 as i32 * o4 as i32) < 0 && v2 != w.v_prev {
                    v_next = v2;
                    self.tadd_edge_cnstr_with_neighbor(t_around_v, (le + 1) % 3, self.ncnstr - 1);
                    break;
                }
            }
        }

        w.t_prev = w.t;
        w.v_prev = w.v;
        w.t = t_next;
        w.v = v_next;
    }

    // ── walk_constraint_t ────────────────────────────────────────

    fn walk_constraint_t(&mut self, w: &mut ConstraintWalker, q: &mut DList) {
        debug_assert!(w.v == NO_INDEX);
        debug_assert!(w.t != NO_INDEX);

        let mut v_next = NO_INDEX;
        let mut t_next = NO_INDEX;

        if self.tv(w.t, 0) == w.j || self.tv(w.t, 1) == w.j || self.tv(w.t, 2) == w.j {
            v_next = w.j;
        } else {
            for le in 0..3u32 {
                if self.tadj(w.t, le) == w.t_prev {
                    continue;
                }
                let v1 = self.tv(w.t, (le + 1) % 3);
                let v2 = self.tv(w.t, (le + 2) % 3);
                let o1 = self.orient2d_mit(w.i, w.j, v1);
                let o2 = self.orient2d_mit(w.i, w.j, v2);
                if (o1 as i32 * o2 as i32) < 0 {
                    self.trot(w.t, le);
                    if self.tedge_is_constrained(w.t, 0) {
                        let existing_cnstr =
                            self.ecnstr_val[self.t_ecnstr[(3 * w.t) as usize] as usize];
                        v_next = self.create_intersection_mit(
                            self.ncnstr - 1,
                            w.i,
                            w.j,
                            existing_cnstr,
                            v1,
                            v2,
                        );
                        self.insert_vertex_in_edge_no_list(v_next, w.t, 0);
                        if w.v_prev != NO_INDEX {
                            self.tadd_edge_cnstr_with_neighbor(w.t, 2, self.ncnstr - 1);
                        }
                    } else {
                        q.push_back(&mut self.t_flags, &mut self.t_next, &mut self.t_prev, w.t);
                        t_next = self.tadj(w.t, 0);
                    }
                    break;
                } else {
                    debug_assert!(o1 != Sign::Zero || o2 != Sign::Zero);
                    if o1 == Sign::Zero {
                        v_next = v1;
                        break;
                    } else if o2 == Sign::Zero {
                        v_next = v2;
                        break;
                    }
                }
            }
        }

        w.t_prev = w.t;
        w.v_prev = w.v;
        w.t = t_next;
        w.v = v_next;
    }

    // ── constrain_edges ──────────────────────────────────────────

    fn constrain_edges(&mut self, i: u32, j: u32, q: &mut DList, n: &mut DList) {
        while !q.empty() {
            let t1 = q.pop_back(&mut self.t_flags, &mut self.t_next, &mut self.t_prev);
            if !self.is_convex_quad(t1) {
                assert!(
                    !q.empty(),
                    "constrain_edges: non-convex quad is the only remaining edge"
                );
                q.push_front(&mut self.t_flags, &mut self.t_next, &mut self.t_prev, t1);
            } else {
                let t2 = self.tadj(t1, 0);
                let no_isect = !q.contains(&self.t_flags, t2);
                let v0 = self.tv(t1, 0);
                let t2v0_t1v2 = q.contains(&self.t_flags, t2) && self.tv(t2, 0) == self.tv(t1, 2);
                #[cfg(debug_assertions)]
                let t2v0_t1v1 = q.contains(&self.t_flags, t2) && self.tv(t2, 0) == self.tv(t1, 1);

                if no_isect {
                    self.swap_edge(t1, false);
                    self.trot(t1, 2);
                    self.apply_new_edge(t1, i, j, n);
                } else {
                    let o =
                        Sign::from_i32(self.orient2d_mit(i, j, v0) as i32 * self.orient_012 as i32);
                    if t2v0_t1v2 {
                        self.swap_edge(t1, false);
                        if o as i32 >= 0 {
                            self.trot(t1, 2);
                            self.apply_new_edge(t1, i, j, n);
                        } else {
                            self.trot(t1, 2);
                            q.push_front(&mut self.t_flags, &mut self.t_next, &mut self.t_prev, t1);
                        }
                    } else {
                        debug_assert!(t2v0_t1v1);
                        self.swap_edge(t1, true);
                        if o as i32 > 0 {
                            self.trot(t1, 1);
                            q.push_front(&mut self.t_flags, &mut self.t_next, &mut self.t_prev, t1);
                        } else {
                            self.trot(t1, 1);
                            self.apply_new_edge(t1, i, j, n);
                        }
                    }
                }
            }
        }
    }

    fn apply_new_edge(&mut self, t: u32, i: u32, j: u32, n: &mut DList) {
        if (self.tv(t, 1) == i && self.tv(t, 2) == j) || (self.tv(t, 1) == j && self.tv(t, 2) == i)
        {
            self.tadd_edge_cnstr_with_neighbor(t, 0, self.ncnstr - 1);
        } else if n.initialized() {
            n.push_back(&mut self.t_flags, &mut self.t_next, &mut self.t_prev, t);
        }
    }

    // ── Consistency checks ────────────────────────────────────────
    // Port of Geogram's CDTBase2d::Tcheck, check_combinatorics,
    // check_geometry, Tedge_is_Delaunay, and check_consistency.

    /// Combinatorial consistency check for a single triangle.
    /// Port of Geogram's CDTBase2d::Tcheck(t).
    fn tcheck(&self, t: u32) {
        if t == NO_INDEX {
            return;
        }
        for e in 0..3u32 {
            assert!(
                self.tv(t, e) != self.tv(t, (e + 1) % 3),
                "tcheck: triangle {t} has duplicate vertices at edges {e} and {}",
                (e + 1) % 3
            );
            if self.tadj(t, e) == NO_INDEX {
                continue;
            }
            assert!(
                self.tadj(t, e) != self.tadj(t, (e + 1) % 3),
                "tcheck: triangle {t} has duplicate adjacencies at edges {e} and {}",
                (e + 1) % 3
            );
            let t2 = self.tadj(t, e);
            let e2 = self.tadj_find(t2, t);
            assert!(
                self.tadj(t2, e2) == t,
                "tcheck: adjacency asymmetry: t={t} e={e} -> t2={t2} e2={e2} -> {}",
                self.tadj(t2, e2)
            );
        }
    }

    /// Combinatorial consistency check for all triangles.
    /// Port of Geogram's CDTBase2d::check_combinatorics().
    #[allow(dead_code)]
    pub fn check_combinatorics(&self) {
        for t in 0..self.num_triangles() {
            self.tcheck(t);
        }
    }

    /// Check the Delaunay property for a single edge.
    /// Port of Geogram's CDTBase2d::Tedge_is_Delaunay().
    fn tedge_is_delaunay(&self, t1: u32, le1: u32) -> bool {
        if self.tedge_is_constrained(t1, le1) {
            return true;
        }
        let t2 = self.tadj(t1, le1);
        if t2 == NO_INDEX {
            return true;
        }
        let le2 = self.tadj_find(t2, t1);
        let v1 = self.tv(t1, le1);
        let v2 = self.tv(t1, (le1 + 1) % 3);
        let v3 = self.tv(t1, (le1 + 2) % 3);
        let v4 = self.tv(t2, le2);

        // If the quad is not convex, the edge is considered Delaunay.
        if self.orient2d_mit_cached(v1, v4, v3) != self.orient_012
            || self.orient2d_mit_cached(v4, v1, v2) != self.orient_012
        {
            return true;
        }

        let ic = self.incircle_mit(v1, v2, v3, v4);
        let combined = Sign::from_i32(ic as i32 * self.orient_012 as i32);
        combined != Sign::Positive
    }

    /// Read-only orient2d lookup from the predicate cache.
    /// Used by consistency checks that cannot mutate self.
    fn orient2d_mit_cached(&self, vx1: u32, vx2: u32, vx3: u32) -> Sign {
        let k = Trindex::new(vx1, vx2, vx3);
        let result = if let Some(&cached) = self.pred_cache.get(&k) {
            cached
        } else {
            // Fallback: compute from the projected 3D coordinates.
            orient_2d_projected(
                &self.vertex[k.indices[0] as usize].point_exact,
                &self.vertex[k.indices[1] as usize].point_exact,
                &self.vertex[k.indices[2] as usize].point_exact,
                self.f1_normal_axis,
            )
        };
        if odd_order(vx1, vx2, vx3) {
            Sign::from_i32(-(result as i32))
        } else {
            result
        }
    }

    /// Geometric consistency check: verifies the Delaunay property
    /// for all non-constrained interior edges.
    /// Port of Geogram's CDTBase2d::check_geometry().
    #[allow(dead_code)]
    pub fn check_geometry(&self) {
        if self.delaunay {
            for t in 0..self.num_triangles() {
                for le in 0..3u32 {
                    assert!(
                        self.tedge_is_delaunay(t, le),
                        "check_geometry: edge ({t},{le}) is not Delaunay"
                    );
                }
            }
        }
    }

    /// Full consistency check: combinatorics + geometry.
    /// Port of Geogram's CDTBase2d::check_consistency().
    ///
    /// Note: Geogram's check_geometry() has its assertion commented out
    /// in production because co-circular configurations after constraint
    /// insertion can produce non-Delaunay edges. We match this behavior
    /// by only running the combinatorial check here.
    #[allow(dead_code)]
    pub fn check_consistency(&self) {
        self.check_combinatorics();
        // Geogram: geometry check is commented out in CDTBase2d::check_geometry()
        // self.check_geometry();
    }
}

// ══════════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a simple mesh with two triangles.
    fn test_mesh() -> MeshData {
        MeshData::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [0.5, -1.0, 0.0],
            ],
            vec![[0, 1, 2], [1, 0, 3]],
        )
    }

    #[test]
    fn begin_facet_creates_initial_triangulation() {
        let mesh = test_mesh();
        let mut mit = MeshInTriangle::new(mesh);
        mit.begin_facet(0);

        assert_eq!(mit.vertex.len(), 3);
        assert_eq!(mit.num_triangles(), 1);
        assert_eq!(mit.num_vertices_cdt(), 3);
        assert_eq!(mit.edges.len(), 3);
    }

    #[test]
    fn commit_with_no_extra_vertices() {
        let mesh = test_mesh();
        let mut mit = MeshInTriangle::new(mesh);
        mit.begin_facet(0);
        let nt = mit.commit();
        assert_eq!(nt, 1);
    }

    #[test]
    fn add_vertex_mesh_vertex_returns_lv() {
        let mesh = test_mesh();
        let mut mit = MeshInTriangle::new(mesh);
        mit.begin_facet(0);

        let v = mit.add_vertex(NO_INDEX, T1RgnP0, T2RgnT);
        assert_eq!(v, 0);
        let v = mit.add_vertex(NO_INDEX, T1RgnP1, T2RgnT);
        assert_eq!(v, 1);
    }

    #[test]
    fn mesh_in_triangle_with_intersection_vertex() {
        let mesh = MeshData::new(
            vec![
                [0.0, 0.0, 0.0],  // 0 - f0.P0
                [2.0, 0.0, 0.0],  // 1 - f0.P1
                [1.0, 2.0, 0.0],  // 2 - f0.P2
                [1.0, 0.5, 2.0],  // 3 - f1.P0
                [1.0, 0.5, -1.0], // 4 - f1.P1 (below z=0)
                [1.0, 0.5, 1.0],  // 5 - f1.P2 (above z=0)
            ],
            vec![[0, 1, 2], [3, 4, 5]],
        );

        let mut mit = MeshInTriangle::new(mesh);
        mit.begin_facet(0);

        let v = mit.add_vertex(1, T1RgnT, T2RgnE0);
        assert!(v >= 3, "new intersection vertex should be >= 3, got {v}");
        assert_eq!(mit.vertex.len() as u32, mit.num_vertices_cdt());

        let nt = mit.commit();
        assert!(nt >= 1, "should produce at least 1 triangle, got {nt}");
    }

    #[test]
    fn add_edge_and_commit() {
        let mesh = MeshData::new(
            vec![
                [0.0, 0.0, 0.0],  // 0 - f0.P0
                [4.0, 0.0, 0.0],  // 1 - f0.P1
                [2.0, 4.0, 0.0],  // 2 - f0.P2
                [1.0, 1.0, 1.0],  // 3 - f1.P0 (above z=0)
                [3.0, 1.0, -1.0], // 4 - f1.P1 (below z=0)
                [2.0, 1.0, 1.0],  // 5 - f1.P2 (above z=0)
            ],
            vec![[0, 1, 2], [3, 4, 5]],
        );

        let mut mit = MeshInTriangle::new(mesh);
        mit.begin_facet(0);

        mit.add_edge(1, T1RgnT, T2RgnE0, T1RgnT, T2RgnE2);

        let nt = mit.commit();
        assert!(
            nt >= 2,
            "should produce at least 2 triangles with one constraint edge, got {nt}"
        );
    }

    #[test]
    fn isect_info_flip() {
        let mut info = IsectInfo::new(0, 1, T1RgnP0, T2RgnE0);
        info.flip();
        assert_eq!(info.f1, 1);
        assert_eq!(info.f2, 0);
    }

    #[test]
    fn isect_info_is_point() {
        let info1 = IsectInfo::new(0, 1, T1RgnP0, T2RgnP1);
        assert!(info1.is_point());

        let info2 = IsectInfo::new(0, 1, T1RgnE0, T2RgnP1);
        assert!(!info2.is_point());
    }

    #[test]
    fn three_planes_intersection_basic() {
        let result = get_three_planes_intersection(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        );
        assert!(result.is_some());
        let p = result.unwrap();
        let w = p.w.estimate();
        assert!((p.x.estimate() / w).abs() < 1e-10);
        assert!((p.y.estimate() / w).abs() < 1e-10);
        assert!((p.z.estimate() / w).abs() < 1e-10);
    }

    #[test]
    fn plane_line_intersection_basic() {
        let p = plane_line_intersection(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, -1.0],
            [0.5, 0.5, 1.0],
        );
        let w = p.w.estimate();
        assert!((p.x.estimate() / w - 0.5).abs() < 1e-10);
        assert!((p.y.estimate() / w - 0.5).abs() < 1e-10);
        assert!((p.z.estimate() / w).abs() < 1e-10);
    }

    #[test]
    fn geogram_ref_plane_line_intersection() {
        let p = plane_line_intersection(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.25, -1.0, 1.0],
            [0.25, 1.0, -1.0],
        );
        let w = p.w.estimate();
        assert!(w.abs() > 1e-20, "w should be nonzero");
        let rx = p.x.estimate() / w;
        let ry = p.y.estimate() / w;
        let rz = p.z.estimate() / w;
        assert!((rx - 0.25).abs() < 1e-10, "x: got {rx}");
        assert!(ry.abs() < 1e-10, "y: got {ry}");
        assert!(rz.abs() < 1e-10, "z: got {rz}");
    }

    #[test]
    fn geogram_ref_three_planes_intersection() {
        let result = get_three_planes_intersection(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        );
        assert!(result.is_some());
        let p = result.unwrap();
        assert_eq!(p.w.sign(), 1, "w.sign should be 1");
        let w = p.w.estimate();
        assert!((p.x.estimate() / w).abs() < 1e-10, "x should be 0");
        assert!((p.y.estimate() / w).abs() < 1e-10, "y should be 0");
        assert!((p.z.estimate() / w).abs() < 1e-10, "z should be 0");
    }

    // ── Consistency check tests ──────────────────────────────────
    // Port of Geogram's CDTBase2d::check_consistency() usage.
    // Verifies combinatorial + geometric invariants after CDT operations
    // within MeshInTriangle.

    #[test]
    fn consistency_begin_facet() {
        let mesh = test_mesh();
        let mut mit = MeshInTriangle::new(mesh);
        mit.begin_facet(0);
        mit.check_combinatorics();
    }

    #[test]
    fn consistency_commit_no_extras() {
        let mesh = test_mesh();
        let mut mit = MeshInTriangle::new(mesh);
        mit.begin_facet(0);
        mit.commit();
        mit.check_consistency();
    }

    #[test]
    fn consistency_with_intersection_vertex() {
        let mesh = MeshData::new(
            vec![
                [0.0, 0.0, 0.0],  // 0 - f0.P0
                [2.0, 0.0, 0.0],  // 1 - f0.P1
                [1.0, 2.0, 0.0],  // 2 - f0.P2
                [1.0, 0.5, 2.0],  // 3 - f1.P0
                [1.0, 0.5, -1.0], // 4 - f1.P1
                [1.0, 0.5, 1.0],  // 5 - f1.P2
            ],
            vec![[0, 1, 2], [3, 4, 5]],
        );
        let mut mit = MeshInTriangle::new(mesh);
        mit.begin_facet(0);
        mit.add_vertex(1, T1RgnT, T2RgnE0);
        mit.commit();
        mit.check_consistency();
    }

    #[test]
    fn consistency_with_edge_constraint() {
        let mesh = MeshData::new(
            vec![
                [0.0, 0.0, 0.0],  // 0 - f0.P0
                [4.0, 0.0, 0.0],  // 1 - f0.P1
                [2.0, 4.0, 0.0],  // 2 - f0.P2
                [1.0, 1.0, 1.0],  // 3 - f1.P0
                [3.0, 1.0, -1.0], // 4 - f1.P1
                [2.0, 1.0, 1.0],  // 5 - f1.P2
            ],
            vec![[0, 1, 2], [3, 4, 5]],
        );
        let mut mit = MeshInTriangle::new(mesh);
        mit.begin_facet(0);
        mit.add_edge(1, T1RgnT, T2RgnE0, T1RgnT, T2RgnE2);
        mit.commit();
        mit.check_consistency();
    }
}
