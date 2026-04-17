//! Triangle mesh types.

use crate::gp::Pnt;

/// A triangle mesh: vertices and triangle indices.
#[derive(Debug, Clone)]
pub struct TriangleMesh {
    /// Vertex positions.
    pub vertices: Vec<Pnt>,
    /// Triangle indices (each triple indexes into `vertices`).
    pub triangles: Vec<[usize; 3]>,
}

impl TriangleMesh {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            triangles: Vec::new(),
        }
    }

    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    pub fn num_triangles(&self) -> usize {
        self.triangles.len()
    }

    /// Check if the mesh is manifold: every edge is shared by exactly 2 triangles.
    pub fn is_manifold(&self) -> bool {
        self.non_manifold_edges() == 0
    }

    /// Count edges that are not shared by exactly 2 triangles.
    pub fn non_manifold_edges(&self) -> usize {
        use std::collections::HashMap;
        let mut edge_count: HashMap<(usize, usize), usize> = HashMap::new();
        for tri in &self.triangles {
            for i in 0..3 {
                let a = tri[i];
                let b = tri[(i + 1) % 3];
                let edge = if a < b { (a, b) } else { (b, a) };
                *edge_count.entry(edge).or_insert(0) += 1;
            }
        }
        edge_count.values().filter(|&&c| c != 2).count()
    }

    /// Compute the total surface area of the mesh.
    pub fn area(&self) -> f64 {
        let mut total = 0.0;
        for tri in &self.triangles {
            let a = &self.vertices[tri[0]];
            let b = &self.vertices[tri[1]];
            let c = &self.vertices[tri[2]];
            let ab = b - a;
            let ac = c - a;
            total += ab.cross(&ac).norm() * 0.5;
        }
        total
    }

    /// Compute the signed volume (assumes closed, consistently-oriented mesh).
    pub fn volume(&self) -> f64 {
        let mut total = 0.0;
        for tri in &self.triangles {
            let a = &self.vertices[tri[0]];
            let b = &self.vertices[tri[1]];
            let c = &self.vertices[tri[2]];
            // Signed volume of tetrahedron with origin
            total += a.coords.dot(&b.coords.cross(&c.coords));
        }
        total / 6.0
    }
}

impl Default for TriangleMesh {
    fn default() -> Self {
        Self::new()
    }
}
