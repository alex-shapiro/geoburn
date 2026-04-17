use super::IndexedMesh;

/// Axis-aligned bounding box in 3D.
#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    pub min: [f64; 3],
    pub max: [f64; 3],
}

impl Aabb {
    pub fn empty() -> Self {
        Self {
            min: [f64::INFINITY; 3],
            max: [f64::NEG_INFINITY; 3],
        }
    }

    pub fn from_triangle(v0: [f64; 3], v1: [f64; 3], v2: [f64; 3]) -> Self {
        Self {
            min: [
                v0[0].min(v1[0]).min(v2[0]),
                v0[1].min(v1[1]).min(v2[1]),
                v0[2].min(v1[2]).min(v2[2]),
            ],
            max: [
                v0[0].max(v1[0]).max(v2[0]),
                v0[1].max(v1[1]).max(v2[1]),
                v0[2].max(v1[2]).max(v2[2]),
            ],
        }
    }

    pub fn union(&self, other: &Self) -> Self {
        Self {
            min: [
                self.min[0].min(other.min[0]),
                self.min[1].min(other.min[1]),
                self.min[2].min(other.min[2]),
            ],
            max: [
                self.max[0].max(other.max[0]),
                self.max[1].max(other.max[1]),
                self.max[2].max(other.max[2]),
            ],
        }
    }

    pub fn intersects(&self, other: &Self) -> bool {
        self.min[0] <= other.max[0]
            && self.max[0] >= other.min[0]
            && self.min[1] <= other.max[1]
            && self.max[1] >= other.min[1]
            && self.min[2] <= other.max[2]
            && self.max[2] >= other.min[2]
    }

    fn center(&self, axis: usize) -> f64 {
        (self.min[axis] + self.max[axis]) * 0.5
    }

    fn longest_axis(&self) -> usize {
        let dx = self.max[0] - self.min[0];
        let dy = self.max[1] - self.min[1];
        let dz = self.max[2] - self.min[2];
        if dx >= dy && dx >= dz {
            0
        } else if dy >= dz {
            1
        } else {
            2
        }
    }
}

/// A node in a bounding volume hierarchy.
pub enum BvhNode {
    Leaf {
        aabb: Aabb,
        triangle_idx: u32,
    },
    Internal {
        aabb: Aabb,
        left: Box<BvhNode>,
        right: Box<BvhNode>,
    },
}

impl BvhNode {
    pub fn aabb(&self) -> &Aabb {
        match self {
            Self::Leaf { aabb, .. } | Self::Internal { aabb, .. } => aabb,
        }
    }
}

/// Bounding volume hierarchy for a triangle mesh.
pub struct Bvh {
    pub root: Option<Box<BvhNode>>,
}

impl Bvh {
    /// Build a BVH from an indexed mesh using median-split strategy.
    pub fn build(mesh: &IndexedMesh) -> Self {
        if mesh.triangles.is_empty() {
            return Self { root: None };
        }

        let aabbs: Vec<Aabb> = (0..mesh.num_triangles())
            .map(|i| {
                let [v0, v1, v2] = mesh.triangle_verts(i);
                Aabb::from_triangle(v0, v1, v2)
            })
            .collect();

        let mut indices: Vec<u32> = (0..mesh.num_triangles()).collect();
        let root = build_recursive(&aabbs, &mut indices);
        Self {
            root: Some(Box::new(root)),
        }
    }
}

fn build_recursive(aabbs: &[Aabb], indices: &mut [u32]) -> BvhNode {
    if indices.len() == 1 {
        return BvhNode::Leaf {
            aabb: aabbs[indices[0] as usize],
            triangle_idx: indices[0],
        };
    }

    let mut combined = Aabb::empty();
    for &idx in indices.iter() {
        combined = combined.union(&aabbs[idx as usize]);
    }

    if indices.len() == 2 {
        let left = BvhNode::Leaf {
            aabb: aabbs[indices[0] as usize],
            triangle_idx: indices[0],
        };
        let right = BvhNode::Leaf {
            aabb: aabbs[indices[1] as usize],
            triangle_idx: indices[1],
        };
        return BvhNode::Internal {
            aabb: combined,
            left: Box::new(left),
            right: Box::new(right),
        };
    }

    let axis = combined.longest_axis();
    indices.sort_unstable_by(|&a, &b| {
        let ca = aabbs[a as usize].center(axis);
        let cb = aabbs[b as usize].center(axis);
        ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mid = indices.len() / 2;
    let (left_indices, right_indices) = indices.split_at_mut(mid);

    let left = build_recursive(aabbs, left_indices);
    let right = build_recursive(aabbs, right_indices);

    BvhNode::Internal {
        aabb: combined,
        left: Box::new(left),
        right: Box::new(right),
    }
}

/// Find all pairs of potentially overlapping triangles between two BVHs.
pub fn find_overlapping_pairs(a: &Bvh, b: &Bvh) -> Vec<(u32, u32)> {
    let mut pairs = Vec::new();
    if let (Some(a_root), Some(b_root)) = (&a.root, &b.root) {
        traverse_pair(a_root, b_root, &mut pairs);
    }
    pairs
}

fn traverse_pair(a: &BvhNode, b: &BvhNode, pairs: &mut Vec<(u32, u32)>) {
    if !a.aabb().intersects(b.aabb()) {
        return;
    }

    match (a, b) {
        (
            BvhNode::Leaf {
                triangle_idx: ai, ..
            },
            BvhNode::Leaf {
                triangle_idx: bi, ..
            },
        ) => {
            pairs.push((*ai, *bi));
        }
        (
            BvhNode::Internal {
                left: al,
                right: ar,
                ..
            },
            BvhNode::Leaf { .. },
        ) => {
            traverse_pair(al, b, pairs);
            traverse_pair(ar, b, pairs);
        }
        (
            BvhNode::Leaf { .. },
            BvhNode::Internal {
                left: bl,
                right: br,
                ..
            },
        ) => {
            traverse_pair(a, bl, pairs);
            traverse_pair(a, br, pairs);
        }
        (
            BvhNode::Internal {
                left: al,
                right: ar,
                ..
            },
            BvhNode::Internal {
                left: bl,
                right: br,
                ..
            },
        ) => {
            traverse_pair(al, bl, pairs);
            traverse_pair(al, br, pairs);
            traverse_pair(ar, bl, pairs);
            traverse_pair(ar, br, pairs);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aabb_from_triangle() {
        let aabb = Aabb::from_triangle([0.0, 1.0, 2.0], [3.0, -1.0, 0.0], [1.0, 2.0, 5.0]);
        assert_eq!(aabb.min, [0.0, -1.0, 0.0]);
        assert_eq!(aabb.max, [3.0, 2.0, 5.0]);
    }

    #[test]
    fn aabb_intersects() {
        let a = Aabb {
            min: [0.0, 0.0, 0.0],
            max: [2.0, 2.0, 2.0],
        };
        let b = Aabb {
            min: [1.0, 1.0, 1.0],
            max: [3.0, 3.0, 3.0],
        };
        assert!(a.intersects(&b));

        let c = Aabb {
            min: [5.0, 5.0, 5.0],
            max: [6.0, 6.0, 6.0],
        };
        assert!(!a.intersects(&c));
    }

    #[test]
    fn bvh_overlapping_pair_count() {
        let a = crate::mesh_boolean::tests::box_mesh([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
        let b = crate::mesh_boolean::tests::box_mesh([1.0, 1.0, 1.0], [3.0, 3.0, 3.0]);

        let bvh_a = Bvh::build(&a);
        let bvh_b = Bvh::build(&b);
        let pairs = find_overlapping_pairs(&bvh_a, &bvh_b);

        // 12×12 = 144 possible, but only faces near the overlap region match.
        assert!(
            pairs.len() > 10 && pairs.len() < 100,
            "expected 10..100 overlapping pairs, got {}",
            pairs.len()
        );
    }

    #[test]
    fn bvh_disjoint_no_pairs() {
        let a = crate::mesh_boolean::tests::box_mesh([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let b = crate::mesh_boolean::tests::box_mesh([5.0, 5.0, 5.0], [6.0, 6.0, 6.0]);

        let bvh_a = Bvh::build(&a);
        let bvh_b = Bvh::build(&b);
        let pairs = find_overlapping_pairs(&bvh_a, &bvh_b);
        assert!(pairs.is_empty(), "disjoint boxes should have 0 pairs");
    }
}
