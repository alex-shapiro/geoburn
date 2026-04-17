// Faithful port of Geogram's CDT_2d.h/.cpp (Sloan 1992 algorithm).
// BSD 3-Clause license (original Geogram copyright Inria).
//
// Clippy: the port preserves Geogram's variable naming conventions (single-char
// names from the algorithm, similar names for related quantities) and control
// flow patterns that trigger pedantic lints.
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
    clippy::needless_bool
)]

use super::exact_pred::{Sign, Vec2HE, incircle_2d_sos, orient_2d};
// Expansion is used indirectly through Vec2HE operations.
#[allow(unused_imports)]
use super::expansion::Expansion;

/// Sentinel for "no index".
pub const NO_INDEX: u32 = u32::MAX;

/// DList flag IDs.
const DLIST_S_ID: u32 = 0; // Stack for Delaunayization
const DLIST_Q_ID: u32 = 1; // Queue for constraint enforcement
const DLIST_N_ID: u32 = 2; // New edges for re-Delaunayization
const DLIST_NB: u32 = 3;

/// Triangle flags beyond the DList bits.
const T_MARKED_FLAG: u32 = DLIST_NB;
const T_VISITED_FLAG: u32 = DLIST_NB + 1;

// ── DList ────────────────────────────────────────────────────────────
//
// In Geogram, DList holds a mutable reference to the CDT and modifies
// its Tnext_, Tprev_, and Tflags_ arrays. In Rust we cannot have a
// mutable borrow of ExactCDT2d while also lending mutable references
// to its internals. Instead we store the DList state (front, back,
// list_id) in a separate struct and pass the raw arrays by reference.

/// State for a doubly-linked list of triangles.
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

    #[allow(dead_code)]
    fn pop_front(&mut self, t_flags: &mut [u32], t_next: &mut [u32], t_prev: &mut [u32]) -> u32 {
        debug_assert!(self.initialized());
        debug_assert!(!self.empty());
        let t = self.front;
        self.front = t_next[self.front as usize];
        if self.front == NO_INDEX {
            debug_assert!(self.back == t);
            self.back = NO_INDEX;
        } else {
            t_prev[self.front as usize] = NO_INDEX;
        }
        debug_assert!(self.contains(t_flags, t));
        t_flags[t as usize] &= !(1u32 << self.list_id);
        t
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

/// State for walking a constraint segment through the triangulation.
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

// ── ExactCDT2d ───────────────────────────────────────────────────────

/// Combinatorial CDT engine operating on Vec2HE points with exact predicates.
pub struct ExactCDT2d {
    // Triangle data (flat arrays, 3 entries per triangle).
    t_verts: Vec<u32>,  // T_[3*t + lv] = vertex index
    t_adj: Vec<u32>,    // Tadj_[3*t + le] = adjacent triangle
    t_ecnstr: Vec<u32>, // Tecnstr_first_[3*t + le] = constraint linked list head

    // Per-triangle DList support.
    t_flags: Vec<u32>, // Bitfield per triangle
    t_next: Vec<u32>,  // DList next pointer (per triangle)
    t_prev: Vec<u32>,  // DList prev pointer

    // Vertex data.
    v2t: Vec<u32>,       // v2T_[v] = one triangle incident to v
    points: Vec<Vec2HE>, // Exact point coordinates

    // Edge constraint linked list pools.
    ecnstr_val: Vec<u32>,
    ecnstr_next: Vec<u32>,

    // Counts.
    nv: u32,
    ncnstr: u32,

    // Orientation of the initial triangle (0,1,2).
    orient_012: Sign,

    // Configuration.
    delaunay: bool,
    exact_intersections: bool,

    // Constraint endpoints for exact intersection.
    constraints: Vec<(u32, u32)>,

    // Per-vertex external IDs (for simplify_coplanar_facets).
    vertex_ids: Vec<u32>,
}

impl ExactCDT2d {
    pub fn new() -> Self {
        Self {
            t_verts: Vec::new(),
            t_adj: Vec::new(),
            t_ecnstr: Vec::new(),
            t_flags: Vec::new(),
            t_next: Vec::new(),
            t_prev: Vec::new(),
            v2t: Vec::new(),
            points: Vec::new(),
            ecnstr_val: Vec::new(),
            ecnstr_next: Vec::new(),
            nv: 0,
            ncnstr: 0,
            orient_012: Sign::Zero,
            delaunay: true,
            exact_intersections: true,
            constraints: Vec::new(),
            vertex_ids: Vec::new(),
        }
    }

    // ── Public accessors ─────────────────────────────────────────

    pub fn num_triangles(&self) -> u32 {
        (self.t_verts.len() / 3) as u32
    }

    pub fn num_vertices(&self) -> u32 {
        self.nv
    }

    #[allow(dead_code)]
    pub fn num_constraints(&self) -> u32 {
        self.ncnstr
    }

    pub fn triangle_vertex(&self, t: u32, lv: u32) -> u32 {
        self.tv(t, lv)
    }

    #[allow(dead_code)]
    pub fn triangle_adjacent(&self, t: u32, le: u32) -> u32 {
        self.tadj(t, le)
    }

    #[allow(dead_code)]
    pub fn set_delaunay(&mut self, delaunay: bool) {
        self.delaunay = delaunay;
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

    // ── Triangle accessors (private) ─────────────────────────────

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

    #[allow(dead_code)]
    fn topp(&self, t: u32, e: u32) -> u32 {
        let t2 = self.tadj(t, e);
        if t2 == NO_INDEX {
            return NO_INDEX;
        }
        let e2 = self.tadj_find(t2, t);
        self.tv(t2, e2)
    }

    // ── Triangle mutation ────────────────────────────────────────

    #[allow(clippy::too_many_arguments)]
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

    /// Tset with explicit edge constraint first pointers.
    #[allow(clippy::too_many_arguments)]
    fn tset_with_cnstr(
        &mut self,
        t: u32,
        v0: u32,
        v1: u32,
        v2: u32,
        a0: u32,
        a1: u32,
        a2: u32,
        ec0: u32,
        ec1: u32,
        ec2: u32,
    ) {
        let base = (3 * t) as usize;
        self.t_verts[base] = v0;
        self.t_verts[base + 1] = v1;
        self.t_verts[base + 2] = v2;
        self.t_adj[base] = a0;
        self.t_adj[base + 1] = a1;
        self.t_adj[base + 2] = a2;
        self.t_ecnstr[base] = ec0;
        self.t_ecnstr[base + 1] = ec1;
        self.t_ecnstr[base + 2] = ec2;
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

    fn tset_flag(&mut self, t: u32, flag: u32) {
        self.t_flags[t as usize] |= 1u32 << flag;
    }

    fn treset_flag(&mut self, t: u32, flag: u32) {
        self.t_flags[t as usize] &= !(1u32 << flag);
    }

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
        // Check whether the edge is already constrained with the same constraint.
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

    /// Fix reverse adjacency: the neighbor at edge le1 should point back to t1.
    /// `prev_adj` is the triangle that the neighbor currently thinks it's adjacent to.
    fn tadj_back_connect(&mut self, t1: u32, le1: u32, prev_adj: u32) {
        let t2 = self.tadj(t1, le1);
        if t2 == NO_INDEX {
            return;
        }
        let le2 = self.tadj_find(t2, prev_adj);
        self.tset_adj(t2, le2, t1);
        // Copy edge constraint from neighbor to t1.
        let cnstr = self.t_ecnstr[(3 * t2 + le2) as usize];
        self.t_ecnstr[(3 * t1 + le1) as usize] = cnstr;
    }

    // ── Predicates ───────────────────────────────────────────────

    fn orient2d_pred(&self, i: u32, j: u32, k: u32) -> Sign {
        orient_2d(
            &self.points[i as usize],
            &self.points[j as usize],
            &self.points[k as usize],
        )
    }

    fn incircle_pred(&self, i: u32, j: u32, k: u32, l: u32) -> Sign {
        incircle_2d_sos(
            &self.points[i as usize],
            &self.points[j as usize],
            &self.points[k as usize],
            &self.points[l as usize],
        )
    }

    // ── clear ────────────────────────────────────────────────────

    #[allow(dead_code)]
    pub fn clear(&mut self) {
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
        self.points.clear();
        self.constraints.clear();
        self.vertex_ids.clear();
    }

    // ── Enclosing structures ─────────────────────────────────────

    fn base_create_enclosing_triangle(&mut self, v0: u32, v1: u32, v2: u32) {
        self.nv = 3;
        self.v2t.resize(3, NO_INDEX);
        let t0 = self.tnew();
        self.tset(t0, v0, v1, v2, NO_INDEX, NO_INDEX, NO_INDEX);
        self.orient_012 = self.orient2d_pred(0, 1, 2);
        assert!(self.orient_012 != Sign::Zero);
    }

    fn base_create_enclosing_quad(&mut self, v0: u32, v1: u32, v2: u32, v3: u32) {
        self.nv = 4;
        self.v2t.resize(4, NO_INDEX);
        let t0 = self.tnew();
        let t1 = self.tnew();
        self.tset(t0, v0, v1, v3, t1, NO_INDEX, NO_INDEX);
        self.tset(t1, v3, v1, v2, NO_INDEX, NO_INDEX, t0);
        self.orient_012 = self.orient2d_pred(0, 1, 2);
        debug_assert!(self.is_convex_quad(t0));
        let ic = self.incircle_pred(v0, v1, v2, v3);
        let combined = Sign::from_i32(ic as i32 * self.orient_012 as i32);
        if combined == Sign::Positive {
            self.swap_edge(t0, false);
        }
    }

    /// Create an initial triangulation from a bounding rectangle.
    pub fn create_enclosing_rectangle(&mut self, x1: f64, y1: f64, x2: f64, y2: f64) {
        self.create_enclosing_quad(
            Vec2HE::from_f64(x1, y1),
            Vec2HE::from_f64(x2, y1),
            Vec2HE::from_f64(x2, y2),
            Vec2HE::from_f64(x1, y2),
        );
    }

    /// Create an enclosing quad with external vertex IDs for the four corners.
    #[allow(clippy::too_many_arguments)]
    pub fn create_enclosing_quad_with_ids(
        &mut self,
        p0: Vec2HE,
        id0: u32,
        p1: Vec2HE,
        id1: u32,
        p2: Vec2HE,
        id2: u32,
        p3: Vec2HE,
        id3: u32,
    ) {
        assert!(self.nv == 0);
        assert!(self.num_triangles() == 0);
        self.points.push(p0);
        self.points.push(p1);
        self.points.push(p2);
        self.points.push(p3);
        self.vertex_ids.push(id0);
        self.vertex_ids.push(id1);
        self.vertex_ids.push(id2);
        self.vertex_ids.push(id3);
        self.base_create_enclosing_quad(0, 1, 2, 3);
    }

    pub fn create_enclosing_quad(&mut self, p0: Vec2HE, p1: Vec2HE, p2: Vec2HE, p3: Vec2HE) {
        assert!(self.nv == 0);
        assert!(self.num_triangles() == 0);
        self.points.push(p0);
        self.points.push(p1);
        self.points.push(p2);
        self.points.push(p3);
        self.vertex_ids.resize(4, NO_INDEX);
        self.base_create_enclosing_quad(0, 1, 2, 3);
    }

    #[allow(dead_code)]
    pub fn create_enclosing_triangle(&mut self, p0: Vec2HE, p1: Vec2HE, p2: Vec2HE) {
        assert!(self.nv == 0);
        assert!(self.num_triangles() == 0);
        self.points.push(p0);
        self.points.push(p1);
        self.points.push(p2);
        self.vertex_ids.resize(3, NO_INDEX);
        self.base_create_enclosing_triangle(0, 1, 2);
    }

    // ── Point insertion ──────────────────────────────────────────

    /// Insert a point. Returns its vertex index (may be an existing vertex if duplicate).
    pub fn insert(&mut self, p: Vec2HE) -> u32 {
        self.insert_with_id(p, NO_INDEX)
    }

    /// Insert a point with an external vertex ID.
    /// Returns its vertex index (may be an existing vertex if duplicate).
    pub fn insert_with_id(&mut self, p: Vec2HE, id: u32) -> u32 {
        self.points.push(p);
        self.vertex_ids.push(id);
        let v = self.nv; // v == nv() at this point, so base_insert will allocate
        let v = self.base_insert(v, NO_INDEX);
        // If inserted point already existed, nv() did not increase.
        if self.points.len() as u32 > self.nv {
            self.points.pop();
            self.vertex_ids.pop();
        }
        v
    }

    /// Get the external vertex ID for a CDT vertex.
    pub fn vertex_id(&self, v: u32) -> u32 {
        if (v as usize) < self.vertex_ids.len() {
            self.vertex_ids[v as usize]
        } else {
            NO_INDEX
        }
    }

    /// CDTBase2d::insert — core insert logic.
    /// Faithfully matches Geogram: v == nv() means new vertex.
    fn base_insert(&mut self, v: u32, hint: u32) -> u32 {
        let keep_duplicates;
        if v == self.nv {
            // Brand new vertex — allocate its slot.
            self.v2t.push(NO_INDEX);
            self.nv += 1;
            keep_duplicates = false;
        } else {
            // Inserting a vertex in the middle (batch-insertion or re-insertion).
            keep_duplicates = true;
            debug_assert!(v < self.nv);
        }

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
            return existing;
        }

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

    /// Insert a constraint between two existing vertices.
    pub fn insert_constraint(&mut self, i: u32, j: u32) {
        self.constraints.push((i, j));
        self.base_insert_constraint(i, j);
    }

    /// CDTBase2d::insert_constraint.
    fn base_insert_constraint(&mut self, mut i: u32, j: u32) {
        self.ncnstr += 1;
        let first_v_isect = self.nv;

        let mut q = DList::new(DLIST_Q_ID);
        let mut n = DList::new_uninit();
        if self.delaunay {
            n.initialize(DLIST_N_ID);
        }

        while i != j {
            // Step 1: find intersected edges.
            let k = self.find_intersected_edges(i, j, &mut q);

            // If we found a constraint intersection, we need to Delaunayize
            // the neighborhood of the newly created vertex.
            if self.delaunay && self.exact_intersections && k >= first_v_isect {
                let inserted = self.base_insert(k, NO_INDEX);
                debug_assert_eq!(inserted, k);
                q.clear(&mut self.t_flags, &self.t_next);
                self.delaunayize_vertex_neighbors_no_stack(k);
                let new_k = self.find_intersected_edges(i, j, &mut q);
                assert_eq!(new_k, k);
            }

            // Step 2: constrain edges.
            self.constrain_edges(i, k, &mut q, &mut n);

            // Step 3: restore Delaunay condition.
            if self.delaunay {
                self.delaunayize_new_edges(&mut n);
            }

            i = k;
        }

        // Delaunayize neighborhood of vertices yielded by constraint
        // intersections if not done before.
        if self.delaunay && !self.exact_intersections {
            for v in first_v_isect..self.nv {
                self.delaunayize_vertex_neighbors_no_stack(v);
            }
        }

        // Clean up any remaining list state.
        q.clear(&mut self.t_flags, &self.t_next);
        if n.initialized() {
            n.clear(&mut self.t_flags, &self.t_next);
        }
    }

    // ── Point location ("walking the triangulation") ─────────────

    fn locate(&self, v: u32, hint: u32, o: &mut [Sign; 3]) -> u32 {
        let mut t_pred = self.num_triangles() + 1; // Needs to be different from NO_INDEX
        let mut t = if hint == NO_INDEX {
            0 // simple deterministic start
        } else {
            hint
        };

        loop {
            assert!(t != NO_INDEX, "locate: point outside boundary");

            let tv = [self.tv(t, 0), self.tv(t, 1), self.tv(t, 2)];

            // Start from a random edge (using e0=0 for determinism).
            let e0 = 0u32;
            let mut found_next = false;
            for de in 0..3u32 {
                let le = (e0 + de) % 3;
                let t_next = self.tadj(t, le);

                if t_next == t_pred {
                    o[le as usize] = Sign::Positive;
                    continue;
                }

                // Replace vertex le with v to test orientation.
                let mut tv_test = tv;
                tv_test[le as usize] = v;
                o[le as usize] = Sign::from_i32(
                    self.orient_012 as i32
                        * self.orient2d_pred(tv_test[0], tv_test[1], tv_test[2]) as i32,
                );

                if o[le as usize] == Sign::Negative {
                    t_pred = t;
                    t = t_next;
                    found_next = true;
                    break;
                }
            }

            if !found_next {
                // All orientations non-negative: v is inside t.
                // Fill in any orientations we haven't computed.
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
            // Internal edge: 4 new triangles.
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
            // Border edge: 2 new triangles.
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

    /// Overload without DList for use from walk_constraint_t.
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

    fn is_convex_quad(&self, t: u32) -> bool {
        let v1 = self.tv(t, 0);
        let v2 = self.tv(t, 1);
        let v3 = self.tv(t, 2);
        let t2 = self.tadj(t, 0);
        if t2 == NO_INDEX {
            return false;
        }
        let le2 = self.tadj_find(t2, t);
        let v4 = self.tv(t2, le2);
        self.orient2d_pred(v1, v4, v3) == self.orient_012
            && self.orient2d_pred(v4, v1, v2) == self.orient_012
    }

    // ── Delaunayization ──────────────────────────────────────────

    /// Delaunayize_vertex_neighbors(v, S) — using the provided DList S.
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
            // Geogram: if(!exact_incircle_ && !is_convex_quad(t1)) { continue; }
            // ExactCDT2d always has exact_incircle=true, so this never skips.

            let w1 = self.tv(t2, 0);
            let w2 = self.tv(t2, 1);
            let w3 = self.tv(t2, 2);
            let ic = self.incircle_pred(w1, w2, w3, v);
            let combined = Sign::from_i32(ic as i32 * self.orient_012 as i32);
            if combined == Sign::Positive {
                self.swap_edge(t1, false);
                s.push_back(&mut self.t_flags, &mut self.t_next, &mut self.t_prev, t1);
                s.push_back(&mut self.t_flags, &mut self.t_next, &mut self.t_prev, t2);
            }
        }
    }

    /// Delaunayize_vertex_neighbors(v) — builds its own DList S internally.
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

    /// Delaunayize_new_edges: Delaunay-restore edges after constraint enforcement.
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
                // Geogram: if(!exact_incircle_ && !is_convex_quad(t1)) { continue; }
                // ExactCDT2d always has exact_incircle=true, so this never skips.
                let e2 = self.tadj_find(t2, t1);
                let v3 = self.tv(t2, e2);
                let ic = self.incircle_pred(v0, v1, v2, v3);
                let combined = Sign::from_i32(ic as i32 * self.orient_012 as i32);
                if combined == Sign::Positive {
                    // t2 may also encode a new edge, we need to preserve it.
                    if self.tv(t2, 0) == self.tv(t1, 1) {
                        self.swap_edge(t1, true); // t2 on top
                        self.trot(t1, 1);
                    } else {
                        self.swap_edge(t1, false); // t1 on top
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
    //
    // Geogram uses a lambda-based for_each_T_around_v. In Rust we
    // collect the (triangle, local_vertex) pairs into a Vec and
    // iterate over that to avoid borrowing issues.

    fn collect_triangles_around_v(&self, v: u32) -> Vec<(u32, u32)> {
        let mut result = Vec::new();
        let t_start = self.vt(v);
        if t_start == NO_INDEX {
            return result;
        }

        // Forward pass.
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

        // Check if we hit a border; if so, traverse backward.
        {
            let lv = self.tv_find(t_start, v);
            let first_back = self.tadj(t_start, (lv + 2) % 3);
            if first_back != NO_INDEX {
                // Only if the forward pass ended at a border (not a cycle).
                let forward_last = {
                    let (last_t, last_lv) = *result.last().unwrap();
                    self.tadj(last_t, (last_lv + 1) % 3)
                };
                if forward_last == NO_INDEX {
                    // Forward hit border, need backward pass.
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
                continue; // Don't go backwards!
            }
            let v1 = self.tv(t_around_v, (le + 1) % 3);
            let v2 = self.tv(t_around_v, (le + 2) % 3);
            if v1 == w.j || v2 == w.j {
                // Arrived at j.
                v_next = w.j;
                let le_cnstr_edge = if v1 == w.j {
                    (le + 2) % 3
                } else {
                    (le + 1) % 3
                };
                self.tadd_edge_cnstr_with_neighbor(t_around_v, le_cnstr_edge, self.ncnstr - 1);
                break;
            }
            let o1 = self.orient2d_pred(w.i, w.j, v1);
            let o2 = self.orient2d_pred(w.i, w.j, v2);
            let o3 = self.orient2d_pred(v1, v2, w.j);
            let o4 = self.orient_012; // equivalent to orient2d(v1,v2,i)
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
                let o1 = self.orient2d_pred(w.i, w.j, v1);
                let o2 = self.orient2d_pred(w.i, w.j, v2);
                if (o1 as i32 * o2 as i32) < 0 {
                    // Frank intersection.
                    self.trot(w.t, le);
                    if self.tedge_is_constrained(w.t, 0) {
                        // Constraint-constraint intersection.
                        let existing_cnstr =
                            self.ecnstr_val[self.t_ecnstr[(3 * w.t) as usize] as usize];
                        v_next = self.create_intersection(
                            self.ncnstr - 1,
                            w.i,
                            w.j,
                            existing_cnstr,
                            v1,
                            v2,
                        );
                        self.insert_vertex_in_edge_no_list(v_next, w.t, 0);
                        // Mark new edge as constraint if walker was previously on a vertex.
                        if w.v_prev != NO_INDEX {
                            self.tadd_edge_cnstr_with_neighbor(w.t, 2, self.ncnstr - 1);
                        }
                    } else {
                        q.push_back(&mut self.t_flags, &mut self.t_next, &mut self.t_prev, w.t);
                        t_next = self.tadj(w.t, 0);
                    }
                    break;
                } else {
                    // Special case: v1 or v2 is exactly on [i,j].
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
                    // new_edge
                    self.trot(t1, 2);
                    self.apply_new_edge(t1, i, j, n);
                } else {
                    let o = Sign::from_i32(
                        self.orient2d_pred(i, j, v0) as i32 * self.orient_012 as i32,
                    );
                    if t2v0_t1v2 {
                        self.swap_edge(t1, false); // "new t1 on top"
                        if o as i32 >= 0 {
                            // new_edge(t1, 2)
                            self.trot(t1, 2);
                            self.apply_new_edge(t1, i, j, n);
                        } else {
                            // isect_edge(t1, 2)
                            self.trot(t1, 2);
                            q.push_front(&mut self.t_flags, &mut self.t_next, &mut self.t_prev, t1);
                        }
                    } else {
                        debug_assert!(t2v0_t1v1);
                        self.swap_edge(t1, true); // "new t1 on bottom"
                        if o as i32 > 0 {
                            // isect_edge(t1, 1)
                            self.trot(t1, 1);
                            q.push_front(&mut self.t_flags, &mut self.t_next, &mut self.t_prev, t1);
                        } else {
                            // new_edge(t1, 1)
                            self.trot(t1, 1);
                            self.apply_new_edge(t1, i, j, n);
                        }
                    }
                }
            }
        }
    }

    /// Called when an edge has no intersection with the constraint.
    /// If the edge IS the constraint, mark it; otherwise add to N for re-Delaunayization.
    fn apply_new_edge(&mut self, t: u32, i: u32, j: u32, n: &mut DList) {
        if (self.tv(t, 1) == i && self.tv(t, 2) == j) || (self.tv(t, 1) == j && self.tv(t, 2) == i)
        {
            self.tadd_edge_cnstr_with_neighbor(t, 0, self.ncnstr - 1);
        } else if n.initialized() {
            n.push_back(&mut self.t_flags, &mut self.t_next, &mut self.t_prev, t);
        }
    }

    // ── create_intersection ──────────────────────────────────────
    //
    // Exact intersection of two constraint segments using expansion arithmetic.
    // Faithful port of ExactCDT2d::create_intersection.

    fn create_intersection(&mut self, e1: u32, _i: u32, _j: u32, e2: u32, _k: u32, _l: u32) -> u32 {
        // Use the original constraint endpoints (simpler coordinates).
        let (ci0, ci1) = self.constraints[e1 as usize];
        let (ck0, ck1) = self.constraints[e2 as usize];

        // U = point[j] - point[i] in homogeneous coords.
        // For Vec2HE: (x/w), so diff = (x_j * w_i - x_i * w_j) / (w_j * w_i).
        let pi = &self.points[ci0 as usize];
        let pj = &self.points[ci1 as usize];
        let pk = &self.points[ck0 as usize];
        let pl = &self.points[ck1 as usize];

        // U = pj - pi (homogeneous)
        let ux = pj.x.mul(&pi.w).sub(&pi.x.mul(&pj.w));
        let uy = pj.y.mul(&pi.w).sub(&pi.y.mul(&pj.w));
        let uw = pj.w.mul(&pi.w);

        // V = pl - pk (homogeneous)
        let vx = pl.x.mul(&pk.w).sub(&pk.x.mul(&pl.w));
        let vy = pl.y.mul(&pk.w).sub(&pk.y.mul(&pl.w));
        let _vw = pl.w.mul(&pk.w);

        // D = pk - pi (homogeneous)
        let dx = pk.x.mul(&pi.w).sub(&pi.x.mul(&pk.w));
        let dy = pk.y.mul(&pi.w).sub(&pi.y.mul(&pk.w));
        let dw = pk.w.mul(&pi.w);

        // t = det(D, V) / det(U, V) but in homogeneous coordinates we need:
        // det(D,V) = dx*vy - dy*vx but adjusted for weights:
        //   det(D,V) = (dx/dw)*(vy/vw) - (dy/dw)*(vx/vw)
        //            = (dx*vy - dy*vx) / (dw*vw)
        // Similarly det(U,V) = (ux*vy - uy*vx) / (uw*vw)
        //
        // t = det(D,V) * uw / (det(U,V) * dw)
        //   = (dx*vy - dy*vx) * uw / ((ux*vy - uy*vx) * dw)
        //
        // Then P = pi + t * U in homogeneous coords:
        //   Px = pi.x/pi.w + t * ux/uw
        //   Px = (pi.x * uw + t_num/t_den * ux) / pi.w
        //
        // Following Geogram's approach (rational t, then mix):
        // t_num = det(D,V) * uw = (dx*vy - dy*vx) * uw
        // t_den = det(U,V) * dw = (ux*vy - uy*vx) * dw

        let det_dv = dx.mul(&vy).sub(&dy.mul(&vx));
        let det_uv = ux.mul(&vy).sub(&uy.mul(&vx));

        let t_num = det_dv.mul(&uw);
        let t_den = det_uv.mul(&dw);

        // P = (1 - t) * pi + t * pj in homogeneous coordinates:
        // Px = pi.x * t_den * pj.w + t_num * (pj.x * pi.w - pi.x * pj.w) / (pi.w * pj.w)
        // But more simply:
        // P.x = t_den * pj.w * pi.x + t_num * ux  (note: ux = pj.x * pi.w - pi.x * pj.w)
        // P.y = t_den * pj.w * pi.y + t_num * uy
        // P.w = t_den * pj.w * pi.w + t_num * uw   (but uw = pi.w * pj.w, so this simplifies)
        //     = pi.w * pj.w * (t_den + t_num) ... no, let's be precise:
        //
        // Actually, mix(t, pi, pj) where t = t_num/t_den:
        //   result.x = pi.x + t * ux/uw * pi.w  => but this gets complicated.
        //
        // Let's use a simpler approach matching Geogram:
        // P.x = (pi.x/pi.w) + t * (ux/uw) = (pi.x * uw + t * ux * pi.w) / (pi.w * uw)
        //     = (pi.x * uw * t_den + t_num * ux * pi.w) / (pi.w * uw * t_den)
        //
        // In homogeneous coordinates:
        // P.x = pi.x * uw * t_den + t_num * ux * pi.w
        // P.y = pi.y * uw * t_den + t_num * uy * pi.w
        // P.w = pi.w * uw * t_den

        let px = pi.x.mul(&uw).mul(&t_den).add(&t_num.mul(&ux).mul(&pi.w));
        let py = pi.y.mul(&uw).mul(&t_den).add(&t_num.mul(&uy).mul(&pi.w));
        let pw = pi.w.mul(&uw).mul(&t_den);

        let new_point = Vec2HE::new(px, py, pw);

        self.points.push(new_point);
        self.v2t.push(NO_INDEX);
        self.vertex_ids.push(NO_INDEX);
        let x = self.nv;
        self.nv += 1;
        x
    }

    // ── remove_external_triangles ────────────────────────────────

    #[allow(dead_code)]
    pub fn remove_external_triangles(&mut self, remove_internal_holes: bool) {
        if remove_internal_holes {
            let mut s = DList::new(DLIST_S_ID);

            // Step 1: get triangles adjacent to the border.
            for t in 0..self.num_triangles() {
                for le in 0..3u32 {
                    if self.tadj(t, le) == NO_INDEX {
                        let outside = (self.tedge_cnstr_nb(t, le) % 2) == 0;
                        self.tset_flag(t, T_VISITED_FLAG);
                        if outside {
                            self.tset_flag(t, T_MARKED_FLAG);
                        }
                        s.push_back(&mut self.t_flags, &mut self.t_next, &mut self.t_prev, t);
                        break;
                    }
                }
            }

            // Step 2: recursive traversal.
            while !s.empty() {
                let t1 = s.pop_back(&mut self.t_flags, &mut self.t_next, &mut self.t_prev);
                let t1_outside = self.tflag_is_set(t1, T_MARKED_FLAG);
                for le in 0..3u32 {
                    let t2 = self.tadj(t1, le);
                    if t2 != NO_INDEX && !self.tflag_is_set(t2, T_VISITED_FLAG) {
                        let t2_outside = t1_outside ^ ((self.tedge_cnstr_nb(t1, le) % 2) != 0);
                        self.tset_flag(t2, T_VISITED_FLAG);
                        if t2_outside {
                            self.tset_flag(t2, T_MARKED_FLAG);
                        }
                        s.push_back(&mut self.t_flags, &mut self.t_next, &mut self.t_prev, t2);
                    }
                }
            }

            // Step 3: reset visited flag.
            for t in 0..self.num_triangles() {
                self.treset_flag(t, T_VISITED_FLAG);
            }
        } else {
            let mut s = DList::new(DLIST_S_ID);

            // Step 1: get triangles adjacent to the border.
            for t in 0..self.num_triangles() {
                if (!self.tedge_is_constrained(t, 0) && self.tadj(t, 0) == NO_INDEX)
                    || (!self.tedge_is_constrained(t, 1) && self.tadj(t, 1) == NO_INDEX)
                    || (!self.tedge_is_constrained(t, 2) && self.tadj(t, 2) == NO_INDEX)
                {
                    self.tset_flag(t, T_MARKED_FLAG);
                    s.push_back(&mut self.t_flags, &mut self.t_next, &mut self.t_prev, t);
                }
            }

            // Step 2: recursive traversal.
            while !s.empty() {
                let t1 = s.pop_back(&mut self.t_flags, &mut self.t_next, &mut self.t_prev);
                for le in 0..3u32 {
                    let t2 = self.tadj(t1, le);
                    if t2 != NO_INDEX
                        && !self.tedge_is_constrained(t1, le)
                        && !self.tflag_is_set(t2, T_MARKED_FLAG)
                    {
                        self.tset_flag(t2, T_MARKED_FLAG);
                        s.push_back(&mut self.t_flags, &mut self.t_next, &mut self.t_prev, t2);
                    }
                }
            }
        }

        // Step 3: remove marked triangles.
        s_cleanup_for_remove(&mut self.t_flags, &self.t_next, DLIST_S_ID);
        self.remove_marked_triangles();
    }

    // ── remove_marked_triangles ──────────────────────────────────

    fn remove_marked_triangles(&mut self) {
        // Step 1: compute old2new map (reusing t_next storage).
        let nt = self.num_triangles();
        let mut old2new = std::mem::take(&mut self.t_next);
        old2new.resize(nt as usize, NO_INDEX);
        let mut cur_t_new = 0u32;
        for t in 0..nt {
            if self.tflag_is_set(t, T_MARKED_FLAG) {
                old2new[t as usize] = NO_INDEX;
            } else {
                old2new[t as usize] = cur_t_new;
                cur_t_new += 1;
            }
        }
        let nt_new = cur_t_new;

        // Step 2: translate adjacency and move triangles.
        for t in 0..nt {
            let t_new = old2new[t as usize];
            if t_new == NO_INDEX {
                continue;
            }
            let mut adj0 = self.tadj(t, 0);
            if adj0 != NO_INDEX {
                adj0 = old2new[adj0 as usize];
            }
            let mut adj1 = self.tadj(t, 1);
            if adj1 != NO_INDEX {
                adj1 = old2new[adj1 as usize];
            }
            let mut adj2 = self.tadj(t, 2);
            if adj2 != NO_INDEX {
                adj2 = old2new[adj2 as usize];
            }
            let ec0 = self.t_ecnstr[(3 * t) as usize];
            let ec1 = self.t_ecnstr[(3 * t + 1) as usize];
            let ec2 = self.t_ecnstr[(3 * t + 2) as usize];
            self.tset_with_cnstr(
                t_new,
                self.tv(t, 0),
                self.tv(t, 1),
                self.tv(t, 2),
                adj0,
                adj1,
                adj2,
                ec0,
                ec1,
                ec2,
            );
            self.t_flags[t_new as usize] = 0;
        }

        // Step 3: resize arrays.
        let nc = (3 * nt_new) as usize;
        self.t_verts.truncate(nc);
        self.t_adj.truncate(nc);
        self.t_flags.truncate(nt_new as usize);
        self.t_ecnstr.truncate(nc);

        // Restore t_next / t_prev.
        self.t_next = vec![NO_INDEX; nt_new as usize];
        self.t_prev = vec![NO_INDEX; nt_new as usize];

        // Step 4: fix v2t.
        for v in 0..self.nv {
            self.v2t[v as usize] = NO_INDEX;
        }
        for t in 0..nt_new {
            let v0 = self.t_verts[(3 * t) as usize];
            let v1 = self.t_verts[(3 * t + 1) as usize];
            let v2 = self.t_verts[(3 * t + 2) as usize];
            self.v2t[v0 as usize] = t;
            self.v2t[v1 as usize] = t;
            self.v2t[v2 as usize] = t;
        }
    }

    // ── Consistency checks ────────────────────────────────────────
    // Port of Geogram's CDTBase2d::Tcheck, check_combinatorics,
    // check_geometry, and check_consistency.

    /// Combinatorial consistency check for a single triangle.
    /// Port of Geogram's CDTBase2d::Tcheck(t).
    /// Panics if inconsistency is detected.
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

    // ── Tedge_is_Delaunay ────────────────────────────────────────

    #[allow(dead_code)]
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

        if self.orient2d_pred(v1, v4, v3) != self.orient_012
            || self.orient2d_pred(v4, v1, v2) != self.orient_012
        {
            return true;
        }

        let ic = self.incircle_pred(v1, v2, v3, v4);
        let combined = Sign::from_i32(ic as i32 * self.orient_012 as i32);
        combined != Sign::Positive
    }
}

/// Helper: clear a DList's flags without needing a DList instance.
fn s_cleanup_for_remove(t_flags: &mut [u32], t_next: &[u32], list_id: u32) {
    // Walk is not possible without front, so just clear all flags for this list.
    let mask = !(1u32 << list_id);
    for f in t_flags.iter_mut() {
        *f &= mask;
    }
    let _ = t_next; // unused but matches the API shape
}

// ══════════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a unit-square CDT.
    fn make_square() -> ExactCDT2d {
        let mut cdt = ExactCDT2d::new();
        cdt.create_enclosing_rectangle(0.0, 0.0, 1.0, 1.0);
        cdt
    }

    #[test]
    fn square_produces_2_triangles() {
        // Geogram reference: square: nT=2 nv=4
        let cdt = make_square();
        assert_eq!(cdt.num_triangles(), 2, "nT");
        assert_eq!(cdt.num_vertices(), 4, "nv");
    }

    #[test]
    fn square_plus_center() {
        // Geogram reference: square+center: nT=4 nv=5
        let mut cdt = make_square();
        cdt.insert(Vec2HE::from_f64(0.5, 0.5));
        assert_eq!(cdt.num_triangles(), 4, "nT");
        assert_eq!(cdt.num_vertices(), 5, "nv");
    }

    #[test]
    fn square_plus_diag() {
        // Geogram reference: square+diag: nT=2 nv=4
        // A diagonal constraint between two existing vertices doesn't add vertices.
        let mut cdt = make_square();
        // Vertices: 0=(0,0), 1=(1,0), 2=(1,1), 3=(0,1).
        // Diagonal: 0-2 or 1-3.
        cdt.insert_constraint(0, 2);
        assert_eq!(cdt.num_triangles(), 2, "nT");
        assert_eq!(cdt.num_vertices(), 4, "nv");
    }

    #[test]
    fn square_plus_cross() {
        // Geogram reference: square+cross: nT=4 nv=5
        // Two crossing diagonal constraints create one intersection vertex.
        let mut cdt = make_square();
        cdt.insert_constraint(0, 2);
        cdt.insert_constraint(1, 3);
        assert_eq!(cdt.num_triangles(), 4, "nT");
        assert_eq!(cdt.num_vertices(), 5, "nv");
    }

    #[test]
    fn exact_cross() {
        // Geogram reference: exact_cross: nT=4 nv=5
        let mut cdt = ExactCDT2d::new();
        cdt.create_enclosing_rectangle(0.0, 0.0, 1.0, 1.0);
        cdt.insert_constraint(0, 2);
        cdt.insert_constraint(1, 3);
        assert_eq!(cdt.num_triangles(), 4, "nT");
        assert_eq!(cdt.num_vertices(), 5, "nv");
    }

    #[test]
    fn exact_vert_cnstr() {
        // Geogram reference: exact_vert_cnstr: nT=4 nv=6
        let mut cdt = ExactCDT2d::new();
        cdt.create_enclosing_rectangle(0.0, 0.0, 1.0, 1.0);
        let v4 = cdt.insert(Vec2HE::from_f64(0.5, 0.0));
        let v5 = cdt.insert(Vec2HE::from_f64(0.5, 1.0));
        cdt.insert_constraint(v4, v5);
        assert_eq!(cdt.num_triangles(), 4, "nT");
        assert_eq!(cdt.num_vertices(), 6, "nv");
    }

    // ── Geogram thorough reference tests ────────────────────────────

    #[test]
    fn star_8_constraints() {
        // Geogram: star_8: nT=20 nv=13
        let mut cdt = ExactCDT2d::new();
        cdt.create_enclosing_rectangle(-2.0, -2.0, 2.0, 2.0);
        let center = cdt.insert(Vec2HE::from_f64(0.0, 0.0));
        let mut tips = Vec::new();
        for i in 0..8 {
            let angle = 2.0 * std::f64::consts::PI * f64::from(i) / 8.0;
            tips.push(cdt.insert(Vec2HE::from_f64(angle.cos(), angle.sin())));
        }
        for &tip in &tips {
            cdt.insert_constraint(center, tip);
        }
        assert_eq!(cdt.num_triangles(), 20, "nT");
        assert_eq!(cdt.num_vertices(), 13, "nv");
    }

    #[test]
    fn star_16_constraints() {
        // Geogram: star_16: nT=36 nv=21
        let mut cdt = ExactCDT2d::new();
        cdt.create_enclosing_rectangle(-2.0, -2.0, 2.0, 2.0);
        let center = cdt.insert(Vec2HE::from_f64(0.0, 0.0));
        let mut tips = Vec::new();
        for i in 0..16 {
            let angle = 2.0 * std::f64::consts::PI * f64::from(i) / 16.0;
            tips.push(cdt.insert(Vec2HE::from_f64(angle.cos(), angle.sin())));
        }
        for &tip in &tips {
            cdt.insert_constraint(center, tip);
        }
        assert_eq!(cdt.num_triangles(), 36, "nT");
        assert_eq!(cdt.num_vertices(), 21, "nv");
    }

    #[test]
    fn crossing_grid_3x3() {
        // 3 horizontal + 3 vertical constraints crossing each other.
        // Creates 9 intersection vertices.
        // Geogram: crossing_grid: nT=32 nv=25
        let mut cdt = ExactCDT2d::new();
        cdt.create_enclosing_rectangle(0.0, 0.0, 4.0, 4.0);
        let mut h_left = Vec::new();
        let mut h_right = Vec::new();
        for j in 1..=3 {
            h_left.push(cdt.insert(Vec2HE::from_f64(0.0, f64::from(j))));
            h_right.push(cdt.insert(Vec2HE::from_f64(4.0, f64::from(j))));
        }
        let mut v_bot = Vec::new();
        let mut v_top = Vec::new();
        for i in 1..=3 {
            v_bot.push(cdt.insert(Vec2HE::from_f64(f64::from(i), 0.0)));
            v_top.push(cdt.insert(Vec2HE::from_f64(f64::from(i), 4.0)));
        }
        for j in 0..3 {
            cdt.insert_constraint(h_left[j], h_right[j]);
        }
        for i in 0..3 {
            cdt.insert_constraint(v_bot[i], v_top[i]);
        }
        assert_eq!(cdt.num_triangles(), 32, "nT");
        assert_eq!(cdt.num_vertices(), 25, "nv");
    }

    #[test]
    fn nearly_degenerate_points() {
        // Points very close together — tests exact predicate robustness.
        // Geogram: nearly_degen: nT=10 nv=8
        let mut cdt = ExactCDT2d::new();
        cdt.create_enclosing_rectangle(-1.0, -1.0, 2.0, 2.0);
        cdt.insert(Vec2HE::from_f64(0.0, 0.0));
        cdt.insert(Vec2HE::from_f64(1.0, 0.0));
        cdt.insert(Vec2HE::from_f64(0.5, 1e-10));
        cdt.insert(Vec2HE::from_f64(0.5, -1e-10));
        assert_eq!(cdt.num_triangles(), 10, "nT");
        assert_eq!(cdt.num_vertices(), 8, "nv");
    }

    #[test]
    fn overlapping_collinear_constraints() {
        // Two constraints that share a collinear portion.
        // Geogram: overlapping_cnstr: nT=10 nv=8
        let mut cdt = ExactCDT2d::new();
        cdt.create_enclosing_rectangle(-1.0, -1.0, 4.0, 4.0);
        let v0 = cdt.insert(Vec2HE::from_f64(0.0, 0.0));
        let v1 = cdt.insert(Vec2HE::from_f64(1.0, 1.0));
        let v2 = cdt.insert(Vec2HE::from_f64(2.0, 2.0));
        let v3 = cdt.insert(Vec2HE::from_f64(3.0, 3.0));
        cdt.insert_constraint(v0, v2);
        cdt.insert_constraint(v1, v3);
        assert_eq!(cdt.num_triangles(), 10, "nT");
        assert_eq!(cdt.num_vertices(), 8, "nv");
    }

    // ── Consistency check tests ──────────────────────────────────
    // Port of Geogram's CDTBase2d::check_consistency() usage.
    // Geogram calls debug_check_consistency() after every insert
    // and insert_constraint (under CDT_DEBUG). We verify it here
    // at the end of each scenario.

    #[test]
    fn consistency_square() {
        let cdt = make_square();
        cdt.check_consistency();
    }

    #[test]
    fn consistency_square_plus_center() {
        let mut cdt = make_square();
        cdt.insert(Vec2HE::from_f64(0.5, 0.5));
        cdt.check_consistency();
    }

    #[test]
    fn consistency_square_plus_cross() {
        let mut cdt = make_square();
        cdt.insert_constraint(0, 2);
        cdt.insert_constraint(1, 3);
        cdt.check_consistency();
    }

    #[test]
    fn consistency_star_8() {
        let mut cdt = ExactCDT2d::new();
        cdt.create_enclosing_rectangle(-2.0, -2.0, 2.0, 2.0);
        let center = cdt.insert(Vec2HE::from_f64(0.0, 0.0));
        let n = 8;
        let mut rim = Vec::new();
        for i in 0..n {
            let a = std::f64::consts::PI * 2.0 * f64::from(i) / f64::from(n);
            rim.push(cdt.insert(Vec2HE::from_f64(a.cos(), a.sin())));
        }
        for i in 0..n as usize {
            cdt.insert_constraint(center, rim[i]);
        }
        cdt.check_consistency();
    }

    #[test]
    fn consistency_crossing_grid() {
        let mut cdt = ExactCDT2d::new();
        cdt.create_enclosing_rectangle(-1.0, -1.0, 5.0, 5.0);
        let mut h_left = Vec::new();
        let mut h_right = Vec::new();
        for j in 1..=3 {
            h_left.push(cdt.insert(Vec2HE::from_f64(0.0, f64::from(j))));
            h_right.push(cdt.insert(Vec2HE::from_f64(4.0, f64::from(j))));
        }
        let mut v_bot = Vec::new();
        let mut v_top = Vec::new();
        for i in 1..=3 {
            v_bot.push(cdt.insert(Vec2HE::from_f64(f64::from(i), 0.0)));
            v_top.push(cdt.insert(Vec2HE::from_f64(f64::from(i), 4.0)));
        }
        for j in 0..3 {
            cdt.insert_constraint(h_left[j], h_right[j]);
        }
        for i in 0..3 {
            cdt.insert_constraint(v_bot[i], v_top[i]);
        }
        cdt.check_consistency();
    }

    #[test]
    fn consistency_nearly_degenerate() {
        let mut cdt = ExactCDT2d::new();
        cdt.create_enclosing_rectangle(-1.0, -1.0, 2.0, 2.0);
        cdt.insert(Vec2HE::from_f64(0.0, 0.0));
        cdt.insert(Vec2HE::from_f64(1.0, 0.0));
        cdt.insert(Vec2HE::from_f64(0.5, 1e-10));
        cdt.insert(Vec2HE::from_f64(0.5, -1e-10));
        cdt.check_consistency();
    }

    #[test]
    fn consistency_overlapping_collinear() {
        let mut cdt = ExactCDT2d::new();
        cdt.create_enclosing_rectangle(-1.0, -1.0, 4.0, 4.0);
        let v0 = cdt.insert(Vec2HE::from_f64(0.0, 0.0));
        let v1 = cdt.insert(Vec2HE::from_f64(1.0, 1.0));
        let v2 = cdt.insert(Vec2HE::from_f64(2.0, 2.0));
        let v3 = cdt.insert(Vec2HE::from_f64(3.0, 3.0));
        cdt.insert_constraint(v0, v2);
        cdt.insert_constraint(v1, v3);
        cdt.check_consistency();
    }

    // ── Bit-identical comparison with Geogram C++ reference ──────
    //
    // These tests compare the full triangle connectivity (vertex indices
    // and adjacency) against output dumped from Geogram's ExactCDT2d.
    // Any divergence means our CDT produces different combinatorics
    // than Geogram for the same input.

    /// Collect (verts, adj) for all triangles, for comparison with Geogram dump.
    fn dump(cdt: &ExactCDT2d) -> Vec<([u32; 3], [i32; 3])> {
        (0..cdt.num_triangles())
            .map(|t| {
                let v = [
                    cdt.triangle_vertex(t, 0),
                    cdt.triangle_vertex(t, 1),
                    cdt.triangle_vertex(t, 2),
                ];
                let a = [
                    cdt.triangle_adjacent(t, 0) as i32,
                    cdt.triangle_adjacent(t, 1) as i32,
                    cdt.triangle_adjacent(t, 2) as i32,
                ];
                (v, a)
            })
            .collect()
    }

    /// Compare against Geogram reference, printing diffs on failure.
    fn assert_matches_geogram(name: &str, cdt: &ExactCDT2d, expected: &[([u32; 3], [i32; 3])]) {
        let actual = dump(cdt);
        assert_eq!(
            actual.len(),
            expected.len(),
            "{name}: nT mismatch: got {} expected {}",
            actual.len(),
            expected.len()
        );
        let mut diffs = Vec::new();
        for (t, (act, exp)) in actual.iter().zip(expected.iter()).enumerate() {
            if act != exp {
                diffs.push(format!(
                    "  T[{t}]: got v={:?} adj={:?}  expect v={:?} adj={:?}",
                    act.0, act.1, exp.0, exp.1
                ));
            }
        }
        assert!(
            diffs.is_empty(),
            "{name}: {} differences:\n{}",
            diffs.len(),
            diffs.join("\n")
        );
    }

    // Geogram reference: NO_INDEX = -1 in i32
    const NI: i32 = -1;

    #[test]
    fn geogram_bitident_square() {
        let cdt = make_square();
        assert_matches_geogram(
            "square",
            &cdt,
            &[([0, 2, 3], [NI, NI, 1]), ([0, 1, 2], [NI, 0, NI])],
        );
    }

    #[test]
    fn geogram_bitident_square_center() {
        let mut cdt = make_square();
        cdt.insert(Vec2HE::from_f64(0.5, 0.5));
        assert_matches_geogram(
            "square_center",
            &cdt,
            &[
                ([4, 3, 0], [NI, 1, 3]),
                ([4, 0, 1], [NI, 2, 0]),
                ([4, 1, 2], [NI, 3, 1]),
                ([4, 2, 3], [NI, 0, 2]),
            ],
        );
    }

    #[test]
    fn geogram_bitident_square_diag() {
        let mut cdt = make_square();
        cdt.insert_constraint(0, 2);
        assert_matches_geogram(
            "square_diag",
            &cdt,
            &[([0, 2, 3], [NI, NI, 1]), ([0, 1, 2], [NI, 0, NI])],
        );
    }

    #[test]
    fn geogram_bitident_square_cross() {
        let mut cdt = make_square();
        cdt.insert_constraint(0, 2);
        cdt.insert_constraint(1, 3);
        assert_matches_geogram(
            "square_cross",
            &cdt,
            &[
                ([4, 2, 3], [NI, 2, 1]),
                ([4, 1, 2], [NI, 0, 3]),
                ([4, 3, 0], [NI, 3, 0]),
                ([4, 0, 1], [NI, 1, 2]),
            ],
        );
    }

    #[test]
    fn geogram_bitident_star8() {
        let mut cdt = ExactCDT2d::new();
        cdt.create_enclosing_rectangle(-2.0, -2.0, 2.0, 2.0);
        let center = cdt.insert(Vec2HE::from_f64(0.0, 0.0));
        let n = 8u32;
        let mut rim = Vec::new();
        for i in 0..n {
            let a = std::f64::consts::PI * 2.0 * f64::from(i) / f64::from(n);
            rim.push(cdt.insert(Vec2HE::from_f64(a.cos(), a.sin())));
        }
        for i in 0..n as usize {
            cdt.insert_constraint(center, rim[i]);
        }
        assert_matches_geogram(
            "star8",
            &cdt,
            &[
                ([9, 8, 3], [11, 13, 12]),
                ([11, 10, 0], [15, 17, 16]),
                ([5, 1, 2], [NI, 7, 5]),
                ([7, 6, 2], [7, 9, 8]),
                ([12, 5, 4], [6, 18, 5]),
                ([12, 1, 5], [2, 4, 19]),
                ([6, 4, 5], [4, 7, 8]),
                ([6, 5, 2], [2, 3, 6]),
                ([7, 4, 6], [6, 3, 10]),
                ([7, 2, 3], [NI, 11, 3]),
                ([8, 4, 7], [8, 11, 12]),
                ([8, 7, 3], [9, 0, 10]),
                ([9, 4, 8], [10, 0, 14]),
                ([9, 3, 0], [NI, 15, 0]),
                ([10, 4, 9], [12, 15, 16]),
                ([10, 9, 0], [13, 1, 14]),
                ([11, 4, 10], [14, 1, 18]),
                ([11, 0, 1], [NI, 19, 1]),
                ([12, 4, 11], [16, 19, 4]),
                ([12, 11, 1], [17, 5, 18]),
            ],
        );
    }

    #[test]
    fn geogram_bitident_nearly_degen() {
        let mut cdt = ExactCDT2d::new();
        cdt.create_enclosing_rectangle(-1.0, -1.0, 2.0, 2.0);
        cdt.insert(Vec2HE::from_f64(0.0, 0.0));
        cdt.insert(Vec2HE::from_f64(1.0, 0.0));
        cdt.insert(Vec2HE::from_f64(0.5, 1e-10));
        cdt.insert(Vec2HE::from_f64(0.5, -1e-10));
        assert_matches_geogram(
            "nearly_degen",
            &cdt,
            &[
                ([4, 3, 0], [NI, 9, 4]),
                ([7, 6, 4], [4, 9, 8]),
                ([5, 1, 2], [NI, 7, 6]),
                ([6, 2, 3], [NI, 4, 7]),
                ([6, 3, 4], [0, 1, 3]),
                ([7, 0, 1], [NI, 6, 9]),
                ([7, 1, 5], [2, 8, 5]),
                ([6, 5, 2], [2, 3, 8]),
                ([7, 5, 6], [7, 1, 6]),
                ([7, 4, 0], [0, 5, 1]),
            ],
        );
    }

    #[test]
    fn geogram_bitident_overlap_collinear() {
        // Geogram's locate() uses random_int32() for starting triangle and
        // edge, making output non-deterministic when points lie exactly on
        // edges (as collinear points do). We verify correctness properties
        // instead of bit-identity for this case.
        let mut cdt = ExactCDT2d::new();
        cdt.create_enclosing_rectangle(-1.0, -1.0, 4.0, 4.0);
        let v0 = cdt.insert(Vec2HE::from_f64(0.0, 0.0));
        let v1 = cdt.insert(Vec2HE::from_f64(1.0, 1.0));
        let v2 = cdt.insert(Vec2HE::from_f64(2.0, 2.0));
        let v3 = cdt.insert(Vec2HE::from_f64(3.0, 3.0));
        cdt.insert_constraint(v0, v2);
        cdt.insert_constraint(v1, v3);
        assert_eq!(cdt.num_triangles(), 10, "nT");
        assert_eq!(cdt.num_vertices(), 8, "nV");
        cdt.check_combinatorics();
    }

    #[test]
    fn geogram_bitident_grid3x3() {
        let mut cdt = ExactCDT2d::new();
        cdt.create_enclosing_rectangle(-1.0, -1.0, 5.0, 5.0);
        let mut h_left = Vec::new();
        let mut h_right = Vec::new();
        for j in 1..=3 {
            h_left.push(cdt.insert(Vec2HE::from_f64(0.0, f64::from(j))));
            h_right.push(cdt.insert(Vec2HE::from_f64(4.0, f64::from(j))));
        }
        let mut v_bot = Vec::new();
        let mut v_top = Vec::new();
        for i in 1..=3 {
            v_bot.push(cdt.insert(Vec2HE::from_f64(f64::from(i), 0.0)));
            v_top.push(cdt.insert(Vec2HE::from_f64(f64::from(i), 4.0)));
        }
        for j in 0..3 {
            cdt.insert_constraint(h_left[j], h_right[j]);
        }
        for i in 0..3 {
            cdt.insert_constraint(v_bot[i], v_top[i]);
        }
        assert_matches_geogram(
            "grid3x3",
            &cdt,
            &[
                ([13, 2, 3], [NI, 20, 24]),
                ([12, 10, 0], [15, 19, 27]),
                ([6, 3, 0], [NI, 6, 10]),
                ([7, 5, 1], [5, 7, 14]),
                ([23, 22, 5], [22, 14, 41]),
                ([14, 1, 5], [3, 22, 23]),
                ([6, 0, 4], [15, 26, 2]),
                ([7, 1, 2], [NI, 11, 3]),
                ([21, 13, 11], [20, 36, 42]),
                ([19, 12, 14], [23, 39, 33]),
                ([8, 3, 6], [2, 28, 16]),
                ([9, 7, 2], [7, 17, 12]),
                ([24, 7, 9], [11, 21, 25]),
                ([24, 15, 13], [24, 42, 21]),
                ([23, 5, 7], [3, 25, 4]),
                ([10, 4, 0], [6, 1, 18]),
                ([11, 3, 8], [10, 30, 20]),
                ([15, 9, 2], [11, 24, 21]),
                ([16, 4, 10], [15, 27, 26]),
                ([12, 0, 1], [NI, 23, 1]),
                ([13, 3, 11], [16, 8, 0]),
                ([24, 9, 15], [17, 13, 12]),
                ([22, 14, 5], [5, 4, 39]),
                ([14, 12, 1], [19, 5, 9]),
                ([15, 2, 13], [0, 13, 17]),
                ([24, 23, 7], [14, 12, 43]),
                ([16, 6, 4], [6, 18, 29]),
                ([16, 10, 12], [1, 33, 18]),
                ([17, 8, 6], [10, 29, 31]),
                ([17, 6, 16], [26, 32, 28]),
                ([18, 11, 8], [16, 31, 36]),
                ([18, 8, 17], [28, 34, 30]),
                ([19, 17, 16], [29, 33, 35]),
                ([19, 16, 12], [27, 9, 32]),
                ([20, 18, 17], [31, 35, 37]),
                ([20, 17, 19], [32, 38, 34]),
                ([21, 11, 18], [30, 37, 8]),
                ([21, 18, 20], [34, 40, 36]),
                ([22, 20, 19], [35, 39, 41]),
                ([22, 19, 14], [9, 22, 38]),
                ([23, 21, 20], [37, 41, 43]),
                ([23, 20, 22], [38, 4, 40]),
                ([24, 13, 21], [8, 43, 13]),
                ([24, 21, 23], [40, 25, 42]),
            ],
        );
    }
}
