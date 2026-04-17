use thiserror::Error;

/// A triangle mesh with indexed vertices.
#[derive(Debug, Clone)]
pub struct IndexedMesh {
    /// Vertex positions as `[x, y, z]`.
    pub vertices: Vec<[f64; 3]>,
    /// Triangles as `[v0, v1, v2]` indices into `vertices`.
    pub triangles: Vec<[u32; 3]>,
}

impl IndexedMesh {
    /// Create a new mesh from vertices and triangles.
    pub fn new(vertices: Vec<[f64; 3]>, triangles: Vec<[u32; 3]>) -> Self {
        Self {
            vertices,
            triangles,
        }
    }

    /// Get the three vertex positions of a triangle.
    pub fn triangle_verts(&self, idx: u32) -> [[f64; 3]; 3] {
        let tri = self.triangles[idx as usize];
        [
            self.vertices[tri[0] as usize],
            self.vertices[tri[1] as usize],
            self.vertices[tri[2] as usize],
        ]
    }

    /// Return the number of triangles.
    pub fn num_triangles(&self) -> u32 {
        self.triangles.len() as u32
    }
}

/// Boolean operation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BooleanOp {
    /// A ∪ B
    Union,
    /// A ∩ B
    Intersection,
    /// A \ B
    Difference,
}

/// Result of a mesh boolean operation.
#[derive(Debug, Clone)]
pub struct BooleanResult {
    /// The resulting mesh.
    pub mesh: IndexedMesh,
    /// For each output triangle, which input face it originated from.
    pub face_origins: FaceOriginMap,
    /// 3D points along the intersection boundary (pairs of segment endpoints).
    /// Used by the hybrid pipeline to map intersection curves back to
    /// parametric surfaces.
    pub intersection_points: Vec<[f64; 3]>,
}

/// Maps each output triangle to its source input face.
#[derive(Debug, Clone)]
pub struct FaceOriginMap {
    /// One entry per output triangle.
    pub origins: Vec<FaceOrigin>,
}

/// Origin of an output face.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaceOrigin {
    /// From mesh A, with the given original triangle index.
    FromA(u32),
    /// From mesh B, with the given original triangle index.
    FromB(u32),
}

/// Errors from mesh boolean operations.
#[derive(Debug, Error)]
pub enum BooleanError {
    #[error("empty mesh: {0}")]
    EmptyMesh(&'static str),

    #[error("degenerate geometry: {0}")]
    Degenerate(String),
}
