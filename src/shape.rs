//! Strongly-typed topological data structures for boundary representation.
//!
//! Each topology level (Vertex, Edge, Wire, Face, Shell, Solid) is its own
//! type. References are wrapped in `Ref<T>`, which carries orientation and
//! location for sharing the same geometry in multiple contexts.
//!
//! Type safety is enforced at compile time — you can't pass a `Ref<Edge>`
//! where a `Ref<Face>` is expected. For dynamic dispatch, use `AnyRef`.

use crate::geom::curve2d::Curve2d;
use crate::geom::curve3d::Curve3d;
use crate::geom::surface::Surface;
use crate::gp::Pnt;
use crate::location::Location;
use std::ops::Deref;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Orientation (same as before)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Orientation {
    Forward,
    Reversed,
}

impl Orientation {
    pub fn reversed(self) -> Self {
        match self {
            Orientation::Forward => Orientation::Reversed,
            Orientation::Reversed => Orientation::Forward,
        }
    }
}

// ---------------------------------------------------------------------------
// Ref<T> — a typed, located, oriented reference to topology
// ---------------------------------------------------------------------------

/// A reference to a topological entity with orientation and location.
///
/// This is the typed equivalent of `Shape` — you can't accidentally
/// pass a `Ref<Edge>` where a `Ref<Face>` is expected.
#[derive(Debug)]
pub struct Ref<T> {
    data: Arc<T>,
    orientation: Orientation,
    location: Location,
}

impl<T> Clone for Ref<T> {
    fn clone(&self) -> Self {
        Self {
            data: Arc::clone(&self.data),
            orientation: self.orientation,
            location: self.location.clone(),
        }
    }
}

impl<T> Ref<T> {
    pub(crate) fn new(data: T) -> Self {
        Self {
            data: Arc::new(data),
            orientation: Orientation::Forward,
            location: Location::identity(),
        }
    }

    pub(crate) fn data(&self) -> &T {
        &self.data
    }

    pub fn orientation(&self) -> Orientation {
        self.orientation
    }

    pub fn location(&self) -> &Location {
        &self.location
    }

    pub fn reversed(&self) -> Self {
        Self {
            data: Arc::clone(&self.data),
            orientation: self.orientation.reversed(),
            location: self.location.clone(),
        }
    }

    pub fn located(&self, loc: &Location) -> Self {
        Self {
            data: Arc::clone(&self.data),
            orientation: self.orientation,
            location: loc.composed(&self.location),
        }
    }

    pub fn is_same(&self, other: &Ref<T>) -> bool {
        Arc::ptr_eq(&self.data, &other.data)
    }

    /// A unique identity key for this shape's underlying data.
    /// Two `Ref<T>` with the same `ptr_id` share the same `Arc` data.
    pub fn ptr_id(&self) -> usize {
        Arc::as_ptr(&self.data) as usize
    }
}

impl<T> Deref for Ref<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.data
    }
}

// ---------------------------------------------------------------------------
// Topology types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Vertex {
    pub(crate) point: Pnt,
    pub(crate) tolerance: f64,
}

#[derive(Debug, Clone)]
pub struct Edge {
    pub(crate) curve: Curve3d,
    pub(crate) first: f64,
    pub(crate) last: f64,
    pub(crate) tolerance: f64,
    pub(crate) front: Ref<Vertex>,
    pub(crate) back: Ref<Vertex>,
    pub(crate) pcurves: Vec<Curve2d>,
}

#[derive(Debug)]
pub struct Wire {
    pub(crate) edges: Vec<Ref<Edge>>,
    pub(crate) closed: bool,
}

#[derive(Debug)]
pub struct Face {
    pub(crate) surface: Surface,
    pub(crate) tolerance: f64,
    pub(crate) outer_wire: Ref<Wire>,
    pub(crate) holes: Vec<Ref<Wire>>,
}

#[derive(Debug)]
pub struct Shell {
    pub(crate) faces: Vec<Ref<Face>>,
}

#[derive(Debug)]
pub struct Solid {
    pub(crate) shell: Ref<Shell>,
}

#[derive(Debug)]
pub struct Compound {
    pub(crate) shapes: Vec<AnyRef>,
}

// ---------------------------------------------------------------------------
// AnyRef — for generic traversal
// ---------------------------------------------------------------------------

/// A dynamically-typed shape reference, for code that needs to handle
/// any topology level generically (serialization, bounding boxes, etc.).
#[derive(Debug, Clone)]
pub enum AnyRef {
    Vertex(Ref<Vertex>),
    Edge(Ref<Edge>),
    Wire(Ref<Wire>),
    Face(Ref<Face>),
    Shell(Ref<Shell>),
    Solid(Ref<Solid>),
    Compound(Ref<Compound>),
}

// ---------------------------------------------------------------------------
// Getters on data types
// ---------------------------------------------------------------------------

impl Vertex {
    pub fn point(&self) -> &Pnt {
        &self.point
    }
    pub fn tolerance(&self) -> f64 {
        self.tolerance
    }
}

impl Edge {
    pub fn curve(&self) -> &Curve3d {
        &self.curve
    }
    pub fn first(&self) -> f64 {
        self.first
    }
    pub fn last(&self) -> f64 {
        self.last
    }
    pub fn tolerance(&self) -> f64 {
        self.tolerance
    }
    pub fn front(&self) -> &Ref<Vertex> {
        &self.front
    }
    pub fn back(&self) -> &Ref<Vertex> {
        &self.back
    }
    pub fn pcurves(&self) -> &[Curve2d] {
        &self.pcurves
    }
}

impl Wire {
    pub fn edges(&self) -> &[Ref<Edge>] {
        &self.edges
    }
    pub fn is_closed(&self) -> bool {
        self.closed
    }
}

impl Face {
    pub fn surface(&self) -> &Surface {
        &self.surface
    }
    pub fn tolerance(&self) -> f64 {
        self.tolerance
    }
    pub fn outer_wire(&self) -> &Ref<Wire> {
        &self.outer_wire
    }
    pub fn holes(&self) -> &[Ref<Wire>] {
        &self.holes
    }
}

impl Shell {
    pub fn faces(&self) -> &[Ref<Face>] {
        &self.faces
    }
}

impl Solid {
    pub fn shell(&self) -> &Ref<Shell> {
        &self.shell
    }
}

impl Compound {
    pub fn shapes(&self) -> &[AnyRef] {
        &self.shapes
    }
}

// ---------------------------------------------------------------------------
// Constructors (free functions to avoid name collisions with accessors)
// ---------------------------------------------------------------------------

pub fn vertex(point: Pnt, tolerance: f64) -> Ref<Vertex> {
    Ref::new(Vertex { point, tolerance })
}

/// Create an edge. Validates that vertices lie on the curve.
pub fn edge(
    curve: Curve3d,
    front: Ref<Vertex>,
    back: Ref<Vertex>,
    first: f64,
    last: f64,
    tolerance: f64,
) -> Ref<Edge> {
    let p1 = curve.value(first);
    let p2 = curve.value(last);
    let d1 = (p1 - front.data().point).norm();
    let d2 = (p2 - back.data().point).norm();
    assert!(
        d1 <= tolerance,
        "front vertex is {d1} from curve at u={first} (tolerance={tolerance})"
    );
    assert!(
        d2 <= tolerance,
        "back vertex is {d2} from curve at u={last} (tolerance={tolerance})"
    );

    Ref::new(Edge {
        curve,
        first,
        last,
        tolerance,
        front,
        back,
        pcurves: vec![],
    })
}

/// Create a wire. Detects closedness by vertex identity.
pub fn wire(edges: Vec<Ref<Edge>>) -> Ref<Wire> {
    let closed = if edges.len() >= 2 {
        let first_start = edges[0].first_vertex();
        let last_end = edges[edges.len() - 1].last_vertex();
        first_start.is_same(&last_end)
    } else {
        false
    };
    Ref::new(Wire { edges, closed })
}

pub fn face(
    surface: Surface,
    outer_wire: Ref<Wire>,
    holes: Vec<Ref<Wire>>,
    tolerance: f64,
) -> Ref<Face> {
    Ref::new(Face {
        surface,
        tolerance,
        outer_wire,
        holes,
    })
}

pub fn shell(faces: Vec<Ref<Face>>) -> Ref<Shell> {
    Ref::new(Shell { faces })
}

pub fn solid(shell: Ref<Shell>) -> Ref<Solid> {
    Ref::new(Solid { shell })
}

pub fn compound(shapes: Vec<AnyRef>) -> Ref<Compound> {
    Ref::new(Compound { shapes })
}

// ---------------------------------------------------------------------------
// Typed accessors and exploration
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Methods that depend on orientation or aggregate children
// (simple getters go through Deref to the data type)
// ---------------------------------------------------------------------------

impl Ref<Edge> {
    /// Start vertex, respecting orientation.
    pub fn first_vertex(&self) -> Ref<Vertex> {
        match self.orientation {
            Orientation::Forward => self.data.front.clone(),
            Orientation::Reversed => self.data.back.clone(),
        }
    }

    /// End vertex, respecting orientation.
    pub fn last_vertex(&self) -> Ref<Vertex> {
        match self.orientation {
            Orientation::Forward => self.data.back.clone(),
            Orientation::Reversed => self.data.front.clone(),
        }
    }
}

impl Ref<Face> {
    /// All edges of this face (from outer wire + holes).
    pub fn all_edges(&self) -> Vec<Ref<Edge>> {
        let mut result: Vec<Ref<Edge>> = self.data.outer_wire.edges.clone();
        for hole in &self.data.holes {
            result.extend(hole.edges.clone());
        }
        result
    }

    /// All vertices of this face.
    pub fn all_vertices(&self) -> Vec<Ref<Vertex>> {
        let mut result = Vec::new();
        for edge in self.all_edges() {
            result.push(edge.first_vertex());
            result.push(edge.last_vertex());
        }
        result
    }
}

impl Ref<Shell> {
    pub fn all_edges(&self) -> Vec<Ref<Edge>> {
        self.data.faces.iter().flat_map(|f| f.all_edges()).collect()
    }

    pub fn all_vertices(&self) -> Vec<Ref<Vertex>> {
        self.data
            .faces
            .iter()
            .flat_map(|f| f.all_vertices())
            .collect()
    }
}

impl Ref<Solid> {
    pub fn all_faces(&self) -> &[Ref<Face>] {
        self.data.shell.faces()
    }

    pub fn all_edges(&self) -> Vec<Ref<Edge>> {
        self.data.shell.all_edges()
    }

    pub fn all_vertices(&self) -> Vec<Ref<Vertex>> {
        self.data.shell.all_vertices()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Point3, Unit, Vector3};

    fn dir(x: f64, y: f64, z: f64) -> crate::gp::Dir {
        Unit::new_normalize(Vector3::new(x, y, z))
    }

    fn pnt(x: f64, y: f64, z: f64) -> Pnt {
        Point3::new(x, y, z)
    }

    fn make_triangle() -> (
        Ref<Vertex>,
        Ref<Vertex>,
        Ref<Vertex>,
        Ref<Edge>,
        Ref<Edge>,
        Ref<Edge>,
        Ref<Wire>,
    ) {
        let v1 = vertex(pnt(0.0, 0.0, 0.0), 1e-7);
        let v2 = vertex(pnt(1.0, 0.0, 0.0), 1e-7);
        let v3 = vertex(pnt(0.0, 1.0, 0.0), 1e-7);

        let d23 = (2.0f64).sqrt();

        let e1 = edge(
            Curve3d::Line {
                origin: pnt(0.0, 0.0, 0.0),
                dir: dir(1.0, 0.0, 0.0),
            },
            v1.clone(),
            v2.clone(),
            0.0,
            1.0,
            1e-7,
        );
        let e2 = edge(
            Curve3d::Line {
                origin: pnt(1.0, 0.0, 0.0),
                dir: dir(-1.0, 1.0, 0.0),
            },
            v2.clone(),
            v3.clone(),
            0.0,
            d23,
            1e-7,
        );
        let e3 = edge(
            Curve3d::Line {
                origin: pnt(0.0, 1.0, 0.0),
                dir: dir(0.0, -1.0, 0.0),
            },
            v3.clone(),
            v1.clone(),
            0.0,
            1.0,
            1e-7,
        );

        let w = wire(vec![e1.clone(), e2.clone(), e3.clone()]);
        (v1, v2, v3, e1, e2, e3, w)
    }

    // -- Type safety: these wouldn't even compile if wrong --

    #[test]
    fn vertex_creation() {
        let v = vertex(pnt(1.0, 2.0, 3.0), 1e-7);
        assert_eq!(v.data().point, pnt(1.0, 2.0, 3.0));
    }

    #[test]
    fn edge_creation() {
        let (_, _, _, e1, _, _, _) = make_triangle();
        assert!((e1.data().first).abs() < 1e-15);
        assert!((e1.data().last - 1.0).abs() < 1e-15);
    }

    #[test]
    fn wire_closedness() {
        let (_, _, _, _, _, _, w) = make_triangle();
        assert!(w.data().closed);
    }

    #[test]
    fn open_wire() {
        let v1 = vertex(pnt(0.0, 0.0, 0.0), 1e-7);
        let v2 = vertex(pnt(1.0, 0.0, 0.0), 1e-7);
        let v3 = vertex(pnt(2.0, 0.0, 0.0), 1e-7);

        let e1 = edge(
            Curve3d::Line {
                origin: pnt(0.0, 0.0, 0.0),
                dir: dir(1.0, 0.0, 0.0),
            },
            v1,
            v2.clone(),
            0.0,
            1.0,
            1e-7,
        );
        let e2 = edge(
            Curve3d::Line {
                origin: pnt(1.0, 0.0, 0.0),
                dir: dir(1.0, 0.0, 0.0),
            },
            v2,
            v3,
            0.0,
            1.0,
            1e-7,
        );

        let w = wire(vec![e1, e2]);
        assert!(!w.data().closed);
    }

    #[test]
    fn sharing() {
        let v = vertex(pnt(0.0, 0.0, 0.0), 1e-7);
        let v2 = v.clone();
        assert!(v.is_same(&v2));
    }

    #[test]
    fn reversed_edge_swaps_vertices() {
        let (v1, v2, _, e1, _, _, _) = make_triangle();
        assert!(e1.first_vertex().is_same(&v1));
        assert!(e1.last_vertex().is_same(&v2));

        let e1_rev = e1.reversed();
        assert!(e1_rev.first_vertex().is_same(&v2));
        assert!(e1_rev.last_vertex().is_same(&v1));
    }

    #[test]
    fn face_typed_exploration() {
        use crate::gp::ax3::Ax3;
        let (_, _, _, _, _, _, w) = make_triangle();
        let surf = Surface::Plane {
            pos: Ax3::new(Point3::origin(), dir(0.0, 0.0, 1.0), dir(1.0, 0.0, 0.0)),
        };
        let f = face(surf, w, vec![], 1e-7);

        // Typed access — no runtime type checks needed
        let edges = f.all_edges();
        assert_eq!(edges.len(), 3);

        let verts = f.all_vertices();
        assert_eq!(verts.len(), 6);

        // Surface access is direct — no match/unwrap
        let p = f.surface().value(0.5, 0.5);
        assert!((p - pnt(0.5, 0.5, 0.0)).norm() < 1e-14);
    }

    #[test]
    fn solid_typed_exploration() {
        use crate::gp::ax3::Ax3;
        let (_, _, _, _, _, _, w) = make_triangle();
        let surf = Surface::Plane {
            pos: Ax3::new(Point3::origin(), dir(0.0, 0.0, 1.0), dir(1.0, 0.0, 0.0)),
        };
        let f = face(surf, w, vec![], 1e-7);
        let sh = shell(vec![f]);
        let s = solid(sh);

        // All typed, all compile-time checked
        assert_eq!(s.all_faces().len(), 1);
        assert_eq!(s.all_edges().len(), 3);
        assert_eq!(s.all_vertices().len(), 6);
    }

    // -- Edge validation still works --

    #[test]
    #[should_panic(expected = "front vertex")]
    fn edge_rejects_bad_vertex() {
        let v1 = vertex(pnt(0.0, 1.0, 0.0), 1e-7); // off the line
        let v2 = vertex(pnt(1.0, 0.0, 0.0), 1e-7);
        edge(
            Curve3d::Line {
                origin: pnt(0.0, 0.0, 0.0),
                dir: dir(1.0, 0.0, 0.0),
            },
            v1,
            v2,
            0.0,
            1.0,
            1e-7,
        );
    }
}
