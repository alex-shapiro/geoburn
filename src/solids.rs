//! Shape builders for primitive solids.
//!
//! Each function constructs a complete, valid `Ref<Solid>` with all
//! topology levels: vertices, edges, wires, faces, shell, solid.

use crate::geom::curve3d::Curve3d;
use crate::geom::surface::Surface;
use crate::gp::ax3::Ax3;
use crate::gp::precision;
use crate::gp::{Dir, Pnt};
use crate::shape::{self, Edge, Face, Ref, Solid, Vertex};
use nalgebra::{Point3, Unit, Vector3};

/// Create a box solid from a corner point and three dimensions.
///
/// The box is axis-aligned with edges along X, Y, Z from `corner`.
/// Face normals point outward.
pub fn make_box(corner: Pnt, dx: f64, dy: f64, dz: f64) -> Ref<Solid> {
    assert!(
        dx > 0.0 && dy > 0.0 && dz > 0.0,
        "box dimensions must be positive"
    );

    let tol = precision::CONFUSION;

    // 8 vertices
    let v000 = shape::vertex(corner, tol);
    let v100 = shape::vertex(Point3::new(corner.x + dx, corner.y, corner.z), tol);
    let v010 = shape::vertex(Point3::new(corner.x, corner.y + dy, corner.z), tol);
    let v110 = shape::vertex(Point3::new(corner.x + dx, corner.y + dy, corner.z), tol);
    let v001 = shape::vertex(Point3::new(corner.x, corner.y, corner.z + dz), tol);
    let v101 = shape::vertex(Point3::new(corner.x + dx, corner.y, corner.z + dz), tol);
    let v011 = shape::vertex(Point3::new(corner.x, corner.y + dy, corner.z + dz), tol);
    let v111 = shape::vertex(
        Point3::new(corner.x + dx, corner.y + dy, corner.z + dz),
        tol,
    );

    // Helper: create a line edge between two vertices
    let line_edge = |va: &Ref<Vertex>, vb: &Ref<Vertex>| -> Ref<Edge> {
        let pa = *va.point();
        let pb = *vb.point();
        let diff = pb - pa;
        let len = diff.norm();
        let dir = Unit::new_normalize(diff);
        shape::edge(
            Curve3d::Line { origin: pa, dir },
            va.clone(),
            vb.clone(),
            0.0,
            len,
            tol,
        )
    };

    // 12 edges
    // Bottom face (z=0): e_b0..e_b3
    let e_b0 = line_edge(&v000, &v100); // along +X
    let e_b1 = line_edge(&v100, &v110); // along +Y
    let e_b2 = line_edge(&v110, &v010); // along -X
    let e_b3 = line_edge(&v010, &v000); // along -Y

    // Top face (z=dz): e_t0..e_t3
    let e_t0 = line_edge(&v001, &v101);
    let e_t1 = line_edge(&v101, &v111);
    let e_t2 = line_edge(&v111, &v011);
    let e_t3 = line_edge(&v011, &v001);

    // Vertical edges: e_v0..e_v3
    let e_v0 = line_edge(&v000, &v001);
    let e_v1 = line_edge(&v100, &v101);
    let e_v2 = line_edge(&v110, &v111);
    let e_v3 = line_edge(&v010, &v011);

    // Helper: make a planar face from 4 edges with outward normal
    let make_planar_face =
        |origin: Pnt, normal: Dir, x_dir: Dir, edges: Vec<Ref<Edge>>| -> Ref<Face> {
            let surf = Surface::Plane {
                pos: Ax3::new(origin, normal, x_dir),
            };
            let w = shape::wire(edges);
            shape::face(surf, w, vec![], tol)
        };

    fn d(x: f64, y: f64, z: f64) -> Dir {
        Unit::new_normalize(Vector3::new(x, y, z))
    }

    // 6 faces with outward normals
    // Bottom (z=0): normal -Z, edges reversed so wire is CCW when viewed from outside
    let f_bottom = make_planar_face(
        corner,
        d(0.0, 0.0, -1.0),
        d(1.0, 0.0, 0.0),
        vec![e_b0.clone(), e_b1.clone(), e_b2.clone(), e_b3.clone()],
    );

    // Top (z=dz): normal +Z
    let f_top = make_planar_face(
        Point3::new(corner.x, corner.y, corner.z + dz),
        d(0.0, 0.0, 1.0),
        d(1.0, 0.0, 0.0),
        vec![e_t0.clone(), e_t1.clone(), e_t2.clone(), e_t3.clone()],
    );

    // Front (y=0): normal -Y
    let f_front = make_planar_face(
        corner,
        d(0.0, -1.0, 0.0),
        d(1.0, 0.0, 0.0),
        vec![e_b0.clone(), e_v1.clone(), e_t0.reversed(), e_v0.reversed()],
    );

    // Back (y=dy): normal +Y, trace: v010→v110→v111→v011→v010
    let f_back = make_planar_face(
        Point3::new(corner.x, corner.y + dy, corner.z),
        d(0.0, 1.0, 0.0),
        d(1.0, 0.0, 0.0),
        vec![
            e_b2.reversed(),
            e_v2.clone(),
            e_t2.reversed(),
            e_v3.reversed(),
        ],
    );

    // Left (x=0): normal -X, trace: v000→v010→v011→v001→v000
    let f_left = make_planar_face(
        corner,
        d(-1.0, 0.0, 0.0),
        d(0.0, 1.0, 0.0),
        vec![
            e_b3.reversed(),
            e_v3.clone(),
            e_t3.reversed(),
            e_v0.reversed(),
        ],
    );

    // Right (x=dx): normal +X, trace: v100→v110→v111→v101→v100... no.
    // v100→v101→v111→v110... no. Let's think:
    // Right face vertices in CCW from outside (+X looking in -X): v100, v110, v111, v101
    // Wait — from outside the +X face, looking in -X direction:
    // Bottom-left=v100, bottom-right=v110, top-right=v111, top-left=v101
    // But CCW from that view: v100→v110→v111→v101
    let f_right = make_planar_face(
        Point3::new(corner.x + dx, corner.y, corner.z),
        d(1.0, 0.0, 0.0),
        d(0.0, 1.0, 0.0),
        vec![e_b1.clone(), e_v2.clone(), e_t1.reversed(), e_v1.reversed()],
    );

    let sh = shape::shell(vec![f_bottom, f_top, f_front, f_back, f_left, f_right]);
    shape::solid(sh)
}

/// Create a cylinder solid from a base center, axis direction, radius, and height.
///
/// The cylinder axis runs from `base_center` in the direction of `axis`
/// for `height` units. The lateral surface has a seam edge at the X-axis
/// of the local coordinate frame.
///
/// Topology: 2 circular edges (top/bottom), 1 seam edge (vertical line),
/// 2 cap faces (planar), 1 lateral face (cylindrical surface).
pub fn make_cylinder(base_center: Pnt, axis: Dir, radius: f64, height: f64) -> Ref<Solid> {
    assert!(radius > 0.0, "cylinder radius must be positive");
    assert!(height > 0.0, "cylinder height must be positive");

    let tol = precision::CONFUSION;
    let two_pi = std::f64::consts::TAU;

    // Build local coordinate frame: Z = axis, X and Y perpendicular
    let ax2 = crate::gp::ax2::Ax2::from_origin_z(base_center, axis);
    let x_dir = *ax2.x_direction();
    let z_dir = axis;

    let top_center = Pnt::from(base_center.coords + z_dir.as_ref() * height);

    // The seam point: where the circle parameter = 0 (on the X axis of the local frame)
    let seam_bottom = Pnt::from(base_center.coords + x_dir.as_ref() * radius);
    let seam_top = Pnt::from(top_center.coords + x_dir.as_ref() * radius);

    // 2 vertices at the seam
    let v_bot = shape::vertex(seam_bottom, tol);
    let v_top = shape::vertex(seam_top, tol);

    // 2 circular edges (bottom and top circles)
    let bottom_circle = Curve3d::Circle {
        pos: crate::gp::ax2::Ax2::new(base_center, z_dir, x_dir),
        radius,
    };
    let top_circle = Curve3d::Circle {
        pos: crate::gp::ax2::Ax2::new(top_center, z_dir, x_dir),
        radius,
    };

    // Circular edges: start and end at the same vertex (closed curve)
    // Parameter range [0, 2π]
    let e_bot_circle = shape::edge(
        bottom_circle,
        v_bot.clone(),
        v_bot.clone(),
        0.0,
        two_pi,
        tol,
    );
    let e_top_circle = shape::edge(top_circle, v_top.clone(), v_top.clone(), 0.0, two_pi, tol);

    // 1 seam edge: vertical line from seam_bottom to seam_top
    let seam_line = Curve3d::Line {
        origin: seam_bottom,
        dir: z_dir,
    };
    let e_seam = shape::edge(seam_line, v_bot.clone(), v_top.clone(), 0.0, height, tol);

    // Bottom cap face: planar, normal = -Z (outward = downward)
    let bottom_ax3 = Ax3::new(base_center, Unit::new_normalize(-*z_dir.as_ref()), x_dir);
    let f_bottom = shape::face(
        Surface::Plane { pos: bottom_ax3 },
        shape::wire(vec![e_bot_circle.reversed()]),
        vec![],
        tol,
    );

    // Top cap face: planar, normal = +Z (outward = upward)
    let top_ax3 = Ax3::new(top_center, z_dir, x_dir);
    let f_top = shape::face(
        Surface::Plane { pos: top_ax3 },
        shape::wire(vec![e_top_circle.clone()]),
        vec![],
        tol,
    );

    // Lateral face: cylindrical surface
    // Wire: seam_up → top_circle → seam_down → bottom_circle
    // The seam edge appears twice (forward going up, reversed going down)
    let cyl_ax3 = Ax3::new(base_center, z_dir, x_dir);
    let lateral_wire = shape::wire(vec![
        e_seam.clone(),
        e_top_circle.clone(),
        e_seam.reversed(),
        e_bot_circle.clone(),
    ]);
    let f_lateral = shape::face(
        Surface::Cylinder {
            pos: cyl_ax3,
            radius,
        },
        lateral_wire,
        vec![],
        tol,
    );

    let sh = shape::shell(vec![f_bottom, f_top, f_lateral]);
    shape::solid(sh)
}

/// Create a sphere solid from a center point, axis direction, and radius.
///
/// Topology: 2 vertices (poles), 1 seam edge (meridian, appears twice
/// in the wire with opposite orientations), 1 spherical face.
///
/// The seam meridian lies in the plane defined by the axis and the
/// auto-chosen X direction of the local frame.
pub fn make_sphere(center: Pnt, axis: Dir, radius: f64) -> Ref<Solid> {
    assert!(radius > 0.0, "sphere radius must be positive");

    let tol = precision::CONFUSION;
    let half_pi = std::f64::consts::FRAC_PI_2;

    // Local coordinate frame
    let ax2 = crate::gp::ax2::Ax2::from_origin_z(center, axis);
    let x_dir = *ax2.x_direction();
    let y_dir = *ax2.y_direction();
    let z_dir = axis;

    // Poles
    let south_pole = Pnt::from(center.coords - z_dir.as_ref() * radius);
    let north_pole = Pnt::from(center.coords + z_dir.as_ref() * radius);
    let v_south = shape::vertex(south_pole, tol);
    let v_north = shape::vertex(north_pole, tol);

    // Seam edge: semicircle from south pole to north pole along the u=0 meridian.
    // This circle lies in the plane spanned by x_dir and z_dir.
    // Circle parameterization: P(t) = center + R*cos(t)*x_dir + R*sin(t)*z_dir
    // At t=-π/2: south pole. At t=π/2: north pole.
    //
    // Circle Ax2: x_direction = sphere x_dir, y_direction = sphere z_dir,
    // so normal = x_dir × z_dir = -y_dir.
    let seam_circle_normal = Unit::new_normalize(-*y_dir.as_ref());
    let seam_circle = Curve3d::Circle {
        pos: crate::gp::ax2::Ax2::new(center, seam_circle_normal, x_dir),
        radius,
    };
    let e_seam = shape::edge(
        seam_circle,
        v_south.clone(),
        v_north.clone(),
        -half_pi,
        half_pi,
        tol,
    );

    // Sphere face: wire is [seam_forward, seam_reversed]
    // seam forward: south → north
    // seam reversed: north → south
    let sphere_wire = shape::wire(vec![e_seam.clone(), e_seam.reversed()]);
    let sphere_ax3 = Ax3::new(center, z_dir, x_dir);
    let f_sphere = shape::face(
        Surface::Sphere {
            pos: sphere_ax3,
            radius,
        },
        sphere_wire,
        vec![],
        tol,
    );

    let sh = shape::shell(vec![f_sphere]);
    shape::solid(sh)
}

/// Create a cone solid from a base center, axis direction, radius, and height.
///
/// The cone has a circular base of `radius` at `base_center` and narrows
/// to an apex at `base_center + height * axis`.
///
/// Topology: 3 vertices (base seam + apex), 3 edges (base circle + 2 seam lines),
/// 2 faces (base cap + lateral cone).
///
/// Note: the lateral face's seam is a line from the base seam point to the apex.
/// It appears twice in the wire (forward up, reversed down), similar to the
/// cylinder and sphere seam patterns.
pub fn make_cone(base_center: Pnt, axis: Dir, radius: f64, height: f64) -> Ref<Solid> {
    assert!(radius > 0.0, "cone radius must be positive");
    assert!(height > 0.0, "cone height must be positive");

    let tol = precision::CONFUSION;
    let two_pi = std::f64::consts::TAU;

    let ax2 = crate::gp::ax2::Ax2::from_origin_z(base_center, axis);
    let x_dir = *ax2.x_direction();
    let z_dir = axis;

    let apex = Pnt::from(base_center.coords + z_dir.as_ref() * height);
    let seam_base = Pnt::from(base_center.coords + x_dir.as_ref() * radius);

    // 3 vertices
    let v_base = shape::vertex(seam_base, tol);
    let v_apex = shape::vertex(apex, tol);

    // Base circle edge (closed, same vertex at both ends)
    let base_circle = Curve3d::Circle {
        pos: crate::gp::ax2::Ax2::new(base_center, z_dir, x_dir),
        radius,
    };
    let e_base_circle = shape::edge(
        base_circle,
        v_base.clone(),
        v_base.clone(),
        0.0,
        two_pi,
        tol,
    );

    // Seam edge: line from base seam point to apex
    let seam_line = Curve3d::Line {
        origin: seam_base,
        dir: Unit::new_normalize(apex - seam_base),
    };
    let seam_length = (apex - seam_base).norm();
    let e_seam = shape::edge(
        seam_line,
        v_base.clone(),
        v_apex.clone(),
        0.0,
        seam_length,
        tol,
    );

    // Base cap face: planar, normal = -Z (outward = downward)
    let base_ax3 = Ax3::new(base_center, Unit::new_normalize(-*z_dir.as_ref()), x_dir);
    let f_base = shape::face(
        Surface::Plane { pos: base_ax3 },
        shape::wire(vec![e_base_circle.reversed()]),
        vec![],
        tol,
    );

    // Lateral face: conical surface
    // The cone surface parameterization has origin at base_center,
    // radius R at v=0, and the apex at v = height / cos(semi_angle).
    let semi_angle = (radius / height).atan();
    let cone_ax3 = Ax3::new(base_center, z_dir, x_dir);
    let f_lateral = shape::face(
        Surface::Cone {
            pos: cone_ax3,
            radius,
            semi_angle,
        },
        shape::wire(vec![
            e_seam.clone(),
            e_seam.reversed(),
            e_base_circle.clone(),
        ]),
        vec![],
        tol,
    );

    let sh = shape::shell(vec![f_base, f_lateral]);
    shape::solid(sh)
}

/// Create a torus solid from a center, axis, major radius, and minor radius.
///
/// The torus is a ring with tube radius `minor_radius` centered at distance
/// `major_radius` from `center`, revolving around `axis`.
///
/// Topology: 1 vertex (seam intersection), 2 seam edges (major circle at v=0,
/// minor cross-section circle at u=0), 1 toroidal face.
/// Both seams meet at the single vertex and each appears twice in the wire.
pub fn make_torus(center: Pnt, axis: Dir, major_radius: f64, minor_radius: f64) -> Ref<Solid> {
    assert!(major_radius > 0.0, "torus major radius must be positive");
    assert!(minor_radius > 0.0, "torus minor radius must be positive");
    assert!(
        major_radius > minor_radius,
        "torus major radius must exceed minor radius (no self-intersecting tori)"
    );

    let tol = precision::CONFUSION;
    let two_pi = std::f64::consts::TAU;

    let ax2 = crate::gp::ax2::Ax2::from_origin_z(center, axis);
    let x_dir = *ax2.x_direction();
    let y_dir = *ax2.y_direction();
    let z_dir = axis;

    // The single seam vertex: at (center + (R+r)*x_dir), where the two seams cross
    let seam_point = Pnt::from(center.coords + x_dir.as_ref() * (major_radius + minor_radius));
    let v_seam = shape::vertex(seam_point, tol);

    // U-seam edge: the outer equator circle at v=0
    // Center = torus center, radius = R+r, in the plane perpendicular to axis
    let u_seam_circle = Curve3d::Circle {
        pos: crate::gp::ax2::Ax2::new(center, z_dir, x_dir),
        radius: major_radius + minor_radius,
    };
    let e_u_seam = shape::edge(
        u_seam_circle,
        v_seam.clone(),
        v_seam.clone(),
        0.0,
        two_pi,
        tol,
    );

    // V-seam edge: the cross-section circle at u=0
    // Center = torus center + R*x_dir (tube center), radius = r
    // Lies in the plane spanned by x_dir and z_dir
    // Circle x_dir = torus x_dir (outward), circle y_dir = z_dir (upward)
    // Circle normal = x_dir × z_dir = -y_dir
    let tube_center = Pnt::from(center.coords + x_dir.as_ref() * major_radius);
    let v_seam_normal = Unit::new_normalize(-*y_dir.as_ref());
    let v_seam_circle = Curve3d::Circle {
        pos: crate::gp::ax2::Ax2::new(tube_center, v_seam_normal, x_dir),
        radius: minor_radius,
    };
    let e_v_seam = shape::edge(
        v_seam_circle,
        v_seam.clone(),
        v_seam.clone(),
        0.0,
        two_pi,
        tol,
    );

    // Wire: traces the boundary of the UV rectangle
    // [u_seam, v_seam, u_seam.reversed, v_seam.reversed]
    let torus_wire = shape::wire(vec![
        e_u_seam.clone(),
        e_v_seam.clone(),
        e_u_seam.reversed(),
        e_v_seam.reversed(),
    ]);

    let torus_ax3 = Ax3::new(center, z_dir, x_dir);
    let f_torus = shape::face(
        Surface::Torus {
            pos: torus_ax3,
            major_radius,
            minor_radius,
        },
        torus_wire,
        vec![],
        tol,
    );

    let sh = shape::shell(vec![f_torus]);
    shape::solid(sh)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(x: f64, y: f64, z: f64) -> Dir {
        Unit::new_normalize(Vector3::new(x, y, z))
    }

    /// Helper: for a solid, check that all face normals point away from
    /// an interior point. Evaluates the normal at the midpoint of the
    /// surface parameter range.
    fn assert_normals_outward(solid: &Ref<Solid>, interior: Pnt) {
        for face_ref in solid.all_faces() {
            let surf = face_ref.surface();
            let ((u0, u1), (v0, v1)) = surf.parameter_range();
            let u_mid = (u0 + u1) / 2.0;
            let v_mid = (v0 + v1) / 2.0;
            let face_pt = surf.value(u_mid, v_mid);
            let n = surf.normal(u_mid, v_mid).normalize();
            let to_face = (face_pt - interior).normalize();
            let dot = n.dot(&to_face);
            assert!(
                dot > 0.0,
                "face normal points inward: dot={dot}, pt={face_pt:?}, n={n:?}"
            );
        }
    }

    #[test]
    fn box_topology_counts() {
        let b = make_box(Point3::origin(), 1.0, 2.0, 3.0);
        assert_eq!(b.all_faces().len(), 6);
        assert_eq!(b.all_edges().len(), 24); // 6 faces × 4 edges each (shared, but counted per-face)
        assert_eq!(b.all_vertices().len(), 48); // 24 edges × 2 vertices each
    }

    #[test]
    fn box_unique_edges() {
        let b = make_box(Point3::origin(), 1.0, 2.0, 3.0);
        let all_edges = b.all_edges();
        // Count unique edges by Arc identity
        let mut unique = vec![];
        for e in &all_edges {
            if !unique.iter().any(|u: &Ref<Edge>| u.is_same(e)) {
                unique.push(e.clone());
            }
        }
        assert_eq!(unique.len(), 12);
    }

    #[test]
    fn box_unique_vertices() {
        let b = make_box(Point3::origin(), 1.0, 2.0, 3.0);
        let all_verts = b.all_vertices();
        let mut unique = vec![];
        for v in &all_verts {
            if !unique.iter().any(|u: &Ref<Vertex>| u.is_same(v)) {
                unique.push(v.clone());
            }
        }
        assert_eq!(unique.len(), 8);
    }

    #[test]
    fn box_vertex_positions() {
        let b = make_box(Point3::new(1.0, 2.0, 3.0), 4.0, 5.0, 6.0);
        let all_verts = b.all_vertices();
        let mut unique_pts: Vec<Pnt> = vec![];
        for v in &all_verts {
            let p = *v.point();
            if !unique_pts.iter().any(|u| (u - p).norm() < 1e-10) {
                unique_pts.push(p);
            }
        }
        assert_eq!(unique_pts.len(), 8);

        // Check corners
        let expected = vec![
            Point3::new(1.0, 2.0, 3.0),
            Point3::new(5.0, 2.0, 3.0),
            Point3::new(1.0, 7.0, 3.0),
            Point3::new(5.0, 7.0, 3.0),
            Point3::new(1.0, 2.0, 9.0),
            Point3::new(5.0, 2.0, 9.0),
            Point3::new(1.0, 7.0, 9.0),
            Point3::new(5.0, 7.0, 9.0),
        ];
        for ep in &expected {
            assert!(
                unique_pts.iter().any(|p| (p - ep).norm() < 1e-10),
                "missing vertex at {ep:?}"
            );
        }
    }

    #[test]
    fn box_faces_are_planar() {
        let b = make_box(Point3::origin(), 1.0, 1.0, 1.0);
        for face in b.all_faces() {
            assert!(matches!(face.surface(), Surface::Plane { .. }));
        }
    }

    #[test]
    fn box_face_normals_point_outward() {
        let b = make_box(Point3::origin(), 2.0, 2.0, 2.0);
        assert_normals_outward(&b, Point3::new(1.0, 1.0, 1.0));
    }

    #[test]
    fn cylinder_face_normals_point_outward() {
        let c = make_cylinder(Point3::origin(), dir(0.0, 0.0, 1.0), 2.0, 4.0);
        assert_normals_outward(&c, Point3::new(0.0, 0.0, 2.0));
    }

    #[test]
    fn sphere_face_normals_point_outward() {
        let s = make_sphere(Point3::origin(), dir(0.0, 0.0, 1.0), 2.0);
        assert_normals_outward(&s, Point3::origin());
    }

    #[test]
    fn cone_face_normals_point_outward() {
        let c = make_cone(Point3::origin(), dir(0.0, 0.0, 1.0), 2.0, 4.0);
        assert_normals_outward(&c, Point3::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn torus_face_normals_point_outward() {
        let t = make_torus(Point3::origin(), dir(0.0, 0.0, 1.0), 5.0, 1.0);
        assert_normals_outward(&t, Point3::new(0.0, 5.0, 0.0));
    }

    #[test]
    fn box_all_wires_closed() {
        let b = make_box(Point3::origin(), 1.0, 1.0, 1.0);
        for face_ref in b.all_faces() {
            assert!(face_ref.outer_wire().is_closed());
        }
    }

    #[test]
    fn box_edge_sharing() {
        // Each edge should be shared by exactly 2 faces
        let b = make_box(Point3::origin(), 1.0, 1.0, 1.0);
        let all_edges = b.all_edges();
        let mut unique: Vec<Ref<Edge>> = vec![];
        let mut counts: Vec<usize> = vec![];

        for e in &all_edges {
            if let Some(idx) = unique.iter().position(|u| u.is_same(e)) {
                counts[idx] += 1;
            } else {
                unique.push(e.clone());
                counts.push(1);
            }
        }

        for (i, count) in counts.iter().enumerate() {
            assert_eq!(
                *count, 2,
                "edge {} shared by {} faces (expected 2)",
                i, count
            );
        }
    }

    // -- Cylinder tests --

    #[test]
    fn cylinder_topology_counts() {
        let c = make_cylinder(Point3::origin(), dir(0.0, 0.0, 1.0), 1.0, 2.0);
        assert_eq!(c.all_faces().len(), 3); // bottom, top, lateral
    }

    #[test]
    fn cylinder_unique_vertices() {
        let c = make_cylinder(Point3::origin(), dir(0.0, 0.0, 1.0), 1.0, 2.0);
        let all_verts = c.all_vertices();
        let mut unique = vec![];
        for v in &all_verts {
            if !unique.iter().any(|u: &Ref<Vertex>| u.is_same(v)) {
                unique.push(v.clone());
            }
        }
        // 2 unique vertices: seam bottom and seam top
        assert_eq!(unique.len(), 2);
    }

    #[test]
    fn cylinder_unique_edges() {
        let c = make_cylinder(Point3::origin(), dir(0.0, 0.0, 1.0), 1.0, 2.0);
        let all_edges = c.all_edges();
        let mut unique = vec![];
        for e in &all_edges {
            if !unique.iter().any(|u: &Ref<Edge>| u.is_same(e)) {
                unique.push(e.clone());
            }
        }
        // 3 unique edges: bottom circle, top circle, seam
        assert_eq!(unique.len(), 3);
    }

    #[test]
    fn cylinder_edge_sharing() {
        let c = make_cylinder(Point3::origin(), dir(0.0, 0.0, 1.0), 1.0, 2.0);
        let all_edges = c.all_edges();
        let mut unique: Vec<Ref<Edge>> = vec![];
        let mut counts: Vec<usize> = vec![];

        for e in &all_edges {
            if let Some(idx) = unique.iter().position(|u| u.is_same(e)) {
                counts[idx] += 1;
            } else {
                unique.push(e.clone());
                counts.push(1);
            }
        }

        // Each edge should be shared by exactly 2 faces
        for (i, count) in counts.iter().enumerate() {
            assert_eq!(
                *count, 2,
                "edge {} shared by {} faces (expected 2)",
                i, count
            );
        }
    }

    #[test]
    fn cylinder_all_wires_closed() {
        let c = make_cylinder(Point3::origin(), dir(0.0, 0.0, 1.0), 1.0, 2.0);
        for face_ref in c.all_faces() {
            // Cap wires have 1 edge (closed circle), lateral has 4
            let w = face_ref.outer_wire();
            let edges = w.edges();
            if edges.len() == 1 {
                // Single circular edge: starts and ends at same vertex
                assert!(edges[0].first_vertex().is_same(&edges[0].last_vertex()));
            } else {
                assert!(w.is_closed());
            }
        }
    }

    #[test]
    fn cylinder_vertex_positions() {
        let c = make_cylinder(Point3::new(1.0, 2.0, 3.0), dir(0.0, 0.0, 1.0), 5.0, 10.0);
        let all_verts = c.all_vertices();
        let mut unique_pts: Vec<Pnt> = vec![];
        for v in &all_verts {
            let p = *v.point();
            if !unique_pts.iter().any(|u| (u - p).norm() < 1e-10) {
                unique_pts.push(p);
            }
        }
        // Seam is at x_dir * radius from center. x_dir is auto-picked.
        // Bottom seam should be at distance `radius` from base_center in XY
        let base = Point3::new(1.0, 2.0, 3.0);
        let top = Point3::new(1.0, 2.0, 13.0);
        for p in &unique_pts {
            let dz = p.z;
            if (dz - base.z).abs() < 1e-10 {
                let dist = ((p.x - base.x).powi(2) + (p.y - base.y).powi(2)).sqrt();
                assert!((dist - 5.0).abs() < 1e-10, "bottom vertex not at radius");
            } else if (dz - top.z).abs() < 1e-10 {
                let dist = ((p.x - top.x).powi(2) + (p.y - top.y).powi(2)).sqrt();
                assert!((dist - 5.0).abs() < 1e-10, "top vertex not at radius");
            } else {
                panic!("vertex at unexpected z={dz}");
            }
        }
    }

    #[test]
    fn cylinder_surface_types() {
        let c = make_cylinder(Point3::origin(), dir(0.0, 0.0, 1.0), 1.0, 2.0);
        let faces = c.all_faces();
        let mut n_plane = 0;
        let mut n_cyl = 0;
        for f in faces {
            match f.surface() {
                Surface::Plane { .. } => n_plane += 1,
                Surface::Cylinder { .. } => n_cyl += 1,
                _ => panic!("unexpected surface type"),
            }
        }
        assert_eq!(n_plane, 2);
        assert_eq!(n_cyl, 1);
    }

    // -- Sphere tests --

    #[test]
    fn sphere_topology_counts() {
        let s = make_sphere(Point3::origin(), dir(0.0, 0.0, 1.0), 1.0);
        assert_eq!(s.all_faces().len(), 1);
    }

    #[test]
    fn sphere_unique_vertices() {
        let s = make_sphere(Point3::origin(), dir(0.0, 0.0, 1.0), 1.0);
        let all_verts = s.all_vertices();
        let mut unique = vec![];
        for v in &all_verts {
            if !unique.iter().any(|u: &Ref<Vertex>| u.is_same(v)) {
                unique.push(v.clone());
            }
        }
        assert_eq!(unique.len(), 2); // south and north poles
    }

    #[test]
    fn sphere_unique_edges() {
        let s = make_sphere(Point3::origin(), dir(0.0, 0.0, 1.0), 1.0);
        let all_edges = s.all_edges();
        let mut unique = vec![];
        for e in &all_edges {
            if !unique.iter().any(|u: &Ref<Edge>| u.is_same(e)) {
                unique.push(e.clone());
            }
        }
        assert_eq!(unique.len(), 1); // single seam edge
    }

    #[test]
    fn sphere_edge_sharing() {
        let s = make_sphere(Point3::origin(), dir(0.0, 0.0, 1.0), 1.0);
        let all_edges = s.all_edges();
        // The seam edge appears twice (forward + reversed) in one face
        assert_eq!(all_edges.len(), 2);
        assert!(all_edges[0].is_same(&all_edges[1]));
    }

    #[test]
    fn sphere_wire_closed() {
        let s = make_sphere(Point3::origin(), dir(0.0, 0.0, 1.0), 1.0);
        let face = &s.all_faces()[0];
        assert!(face.outer_wire().is_closed());
    }

    #[test]
    fn sphere_pole_positions() {
        let center = Point3::new(1.0, 2.0, 3.0);
        let s = make_sphere(center, dir(0.0, 0.0, 1.0), 5.0);
        let all_verts = s.all_vertices();
        let mut unique_pts: Vec<Pnt> = vec![];
        for v in &all_verts {
            let p = *v.point();
            if !unique_pts.iter().any(|u| (u - p).norm() < 1e-10) {
                unique_pts.push(p);
            }
        }
        assert_eq!(unique_pts.len(), 2);

        let south = Point3::new(1.0, 2.0, -2.0); // center.z - radius
        let north = Point3::new(1.0, 2.0, 8.0); // center.z + radius
        assert!(
            unique_pts.iter().any(|p| (p - south).norm() < 1e-10),
            "missing south pole"
        );
        assert!(
            unique_pts.iter().any(|p| (p - north).norm() < 1e-10),
            "missing north pole"
        );
    }

    #[test]
    fn sphere_surface_type() {
        let s = make_sphere(Point3::origin(), dir(0.0, 0.0, 1.0), 1.0);
        assert!(matches!(s.all_faces()[0].surface(), Surface::Sphere { .. }));
    }

    #[test]
    fn sphere_seam_is_on_surface() {
        // The seam edge (semicircle) should lie on the sphere surface
        let s = make_sphere(Point3::origin(), dir(0.0, 0.0, 1.0), 1.0);
        let face = &s.all_faces()[0];
        let edges = face.all_edges();
        let seam = &edges[0];
        let curve = seam.curve();

        // Sample points along the seam and check they're on the unit sphere
        for i in 0..=20 {
            let t = seam.first() + (seam.last() - seam.first()) * i as f64 / 20.0;
            let p = curve.value(t);
            let r = p.coords.norm();
            assert!(
                (r - 1.0).abs() < 1e-13,
                "seam point at t={t} has radius {r}"
            );
        }
    }

    // -- Cone tests --

    #[test]
    fn cone_topology_counts() {
        let c = make_cone(Point3::origin(), dir(0.0, 0.0, 1.0), 1.0, 2.0);
        assert_eq!(c.all_faces().len(), 2); // base + lateral
    }

    #[test]
    fn cone_unique_vertices() {
        let c = make_cone(Point3::origin(), dir(0.0, 0.0, 1.0), 1.0, 2.0);
        let all_verts = c.all_vertices();
        let mut unique = vec![];
        for v in &all_verts {
            if !unique.iter().any(|u: &Ref<Vertex>| u.is_same(v)) {
                unique.push(v.clone());
            }
        }
        // 2 unique vertices: base seam point + apex
        assert_eq!(unique.len(), 2);
    }

    #[test]
    fn cone_unique_edges() {
        let c = make_cone(Point3::origin(), dir(0.0, 0.0, 1.0), 1.0, 2.0);
        let all_edges = c.all_edges();
        let mut unique = vec![];
        for e in &all_edges {
            if !unique.iter().any(|u: &Ref<Edge>| u.is_same(e)) {
                unique.push(e.clone());
            }
        }
        // 2 unique edges: base circle + seam line
        assert_eq!(unique.len(), 2);
    }

    #[test]
    fn cone_edge_sharing() {
        let c = make_cone(Point3::origin(), dir(0.0, 0.0, 1.0), 1.0, 2.0);
        let all_edges = c.all_edges();
        let mut unique: Vec<Ref<Edge>> = vec![];
        let mut counts: Vec<usize> = vec![];

        for e in &all_edges {
            if let Some(idx) = unique.iter().position(|u| u.is_same(e)) {
                counts[idx] += 1;
            } else {
                unique.push(e.clone());
                counts.push(1);
            }
        }

        for (i, count) in counts.iter().enumerate() {
            assert_eq!(
                *count, 2,
                "edge {} shared by {} faces (expected 2)",
                i, count
            );
        }
    }

    #[test]
    fn cone_vertex_positions() {
        let c = make_cone(Point3::new(1.0, 2.0, 3.0), dir(0.0, 0.0, 1.0), 5.0, 10.0);
        let all_verts = c.all_vertices();
        let mut unique_pts: Vec<Pnt> = vec![];
        for v in &all_verts {
            let p = *v.point();
            if !unique_pts.iter().any(|u| (u - p).norm() < 1e-10) {
                unique_pts.push(p);
            }
        }
        assert_eq!(unique_pts.len(), 2);

        // Apex at center + height * axis
        let apex = Point3::new(1.0, 2.0, 13.0);
        assert!(
            unique_pts.iter().any(|p| (p - apex).norm() < 1e-10),
            "missing apex"
        );

        // Base seam at radius from base center in XY
        let base_center = Point3::new(1.0, 2.0, 3.0);
        let has_base = unique_pts.iter().any(|p| {
            (p.z - base_center.z).abs() < 1e-10
                && ((p.x - base_center.x).powi(2) + (p.y - base_center.y).powi(2)).sqrt() - 5.0
                    < 1e-10
        });
        assert!(has_base, "missing base seam vertex");
    }

    #[test]
    fn cone_surface_types() {
        let c = make_cone(Point3::origin(), dir(0.0, 0.0, 1.0), 1.0, 2.0);
        let faces = c.all_faces();
        let mut n_plane = 0;
        let mut n_cone = 0;
        for f in faces {
            match f.surface() {
                Surface::Plane { .. } => n_plane += 1,
                Surface::Cone { .. } => n_cone += 1,
                _ => panic!("unexpected surface type"),
            }
        }
        assert_eq!(n_plane, 1);
        assert_eq!(n_cone, 1);
    }

    #[test]
    fn cone_all_wires_closed() {
        let c = make_cone(Point3::origin(), dir(0.0, 0.0, 1.0), 1.0, 2.0);
        for face_ref in c.all_faces() {
            let w = face_ref.outer_wire();
            let edges = w.edges();
            if edges.len() == 1 {
                assert!(edges[0].first_vertex().is_same(&edges[0].last_vertex()));
            } else {
                assert!(w.is_closed());
            }
        }
    }

    #[test]
    fn cone_seam_connects_base_to_apex() {
        let c = make_cone(Point3::origin(), dir(0.0, 0.0, 1.0), 1.0, 2.0);
        // Find the seam edge (the line, not the circle)
        let all_edges = c.all_edges();
        let seam = all_edges
            .iter()
            .find(|e| matches!(e.curve(), Curve3d::Line { .. }))
            .expect("no line edge found");

        let fv = seam.first_vertex();
        let lv = seam.last_vertex();

        // One end at z=0 (base), other at z=2 (apex)
        let pts = [fv.point().z, lv.point().z];
        assert!(pts.contains(&0.0) || pts.iter().any(|z| z.abs() < 1e-10));
        assert!(pts.iter().any(|z| (z - 2.0).abs() < 1e-10));
    }

    // -- Torus tests --

    #[test]
    fn torus_topology_counts() {
        let t = make_torus(Point3::origin(), dir(0.0, 0.0, 1.0), 5.0, 1.0);
        assert_eq!(t.all_faces().len(), 1);
    }

    #[test]
    fn torus_unique_vertices() {
        let t = make_torus(Point3::origin(), dir(0.0, 0.0, 1.0), 5.0, 1.0);
        let all_verts = t.all_vertices();
        let mut unique = vec![];
        for v in &all_verts {
            if !unique.iter().any(|u: &Ref<Vertex>| u.is_same(v)) {
                unique.push(v.clone());
            }
        }
        assert_eq!(unique.len(), 1); // single seam vertex
    }

    #[test]
    fn torus_unique_edges() {
        let t = make_torus(Point3::origin(), dir(0.0, 0.0, 1.0), 5.0, 1.0);
        let all_edges = t.all_edges();
        let mut unique = vec![];
        for e in &all_edges {
            if !unique.iter().any(|u: &Ref<Edge>| u.is_same(e)) {
                unique.push(e.clone());
            }
        }
        assert_eq!(unique.len(), 2); // u-seam + v-seam
    }

    #[test]
    fn torus_edge_counts() {
        let t = make_torus(Point3::origin(), dir(0.0, 0.0, 1.0), 5.0, 1.0);
        // 4 edge references in the single wire (each seam appears twice)
        assert_eq!(t.all_edges().len(), 4);
    }

    #[test]
    fn torus_wire_closed() {
        let t = make_torus(Point3::origin(), dir(0.0, 0.0, 1.0), 5.0, 1.0);
        let face = &t.all_faces()[0];
        assert!(face.outer_wire().is_closed());
    }

    #[test]
    fn torus_surface_type() {
        let t = make_torus(Point3::origin(), dir(0.0, 0.0, 1.0), 5.0, 1.0);
        assert!(matches!(t.all_faces()[0].surface(), Surface::Torus { .. }));
    }

    #[test]
    fn torus_seam_vertex_position() {
        let center = Point3::new(1.0, 2.0, 3.0);
        let t = make_torus(center, dir(0.0, 0.0, 1.0), 5.0, 1.0);
        let all_verts = t.all_vertices();
        // The seam vertex is at center + (R+r)*x_dir
        // x_dir is auto-chosen, but distance from center should be R+r = 6
        let v = all_verts[0].point();
        let dist = ((v.x - center.x).powi(2) + (v.y - center.y).powi(2)).sqrt();
        assert!(
            (dist - 6.0).abs() < 1e-10,
            "seam vertex distance from axis: {dist}"
        );
        assert!(
            (v.z - center.z).abs() < 1e-10,
            "seam vertex should be in the equatorial plane"
        );
    }

    #[test]
    fn torus_seams_on_surface() {
        // Both seam edges should lie on the torus surface
        let t = make_torus(Point3::origin(), dir(0.0, 0.0, 1.0), 5.0, 1.0);
        let face = &t.all_faces()[0];
        let all_edges = face.all_edges();

        for edge_ref in &all_edges {
            let curve = edge_ref.curve();
            for i in 0..=20 {
                let param =
                    edge_ref.first() + (edge_ref.last() - edge_ref.first()) * i as f64 / 20.0;
                let p = curve.value(param);
                // Point should be on the torus: distance from axis = R ± r
                let rho = (p.x.powi(2) + p.y.powi(2)).sqrt(); // distance from Z axis
                let dz = p.z;
                // Distance from the tube center circle: sqrt((rho - R)^2 + dz^2) should = r
                let tube_dist = ((rho - 5.0).powi(2) + dz.powi(2)).sqrt();
                assert!(
                    (tube_dist - 1.0).abs() < 1e-12,
                    "seam point not on torus: tube_dist={tube_dist} at param={param}"
                );
            }
        }
    }
}
