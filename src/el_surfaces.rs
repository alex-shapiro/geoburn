//! Elementary surface evaluation: point-at-parameter and derivatives.

use crate::gp::ax3::Ax3;
use crate::gp::{Pnt, Vec3, precision};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Plane: P(u,v) = O + u*X + v*Y
// ---------------------------------------------------------------------------

pub fn plane_value(u: f64, v: f64, pos: &Ax3) -> Pnt {
    Pnt::from(pos.origin().coords + pos.x_direction().as_ref() * u + pos.y_direction().as_ref() * v)
}

pub fn plane_d1(u: f64, v: f64, pos: &Ax3) -> (Pnt, Vec3, Vec3) {
    let p = plane_value(u, v, pos);
    let vu: Vec3 = *pos.x_direction().as_ref();
    let vv: Vec3 = *pos.y_direction().as_ref();
    (p, vu, vv)
}

pub fn plane_parameters(pos: &Ax3, p: &Pnt) -> (f64, f64) {
    let d = p - pos.origin();
    let u = d.dot(pos.x_direction().as_ref());
    let v = d.dot(pos.y_direction().as_ref());
    (u, v)
}

// ---------------------------------------------------------------------------
// Cylinder: P(u,v) = O + R*cos(u)*X + R*sin(u)*Y + v*Z
// ---------------------------------------------------------------------------

pub fn cylinder_value(u: f64, v: f64, pos: &Ax3, radius: f64) -> Pnt {
    let (cu, su) = (u.cos(), u.sin());
    Pnt::from(
        pos.origin().coords
            + pos.x_direction().as_ref() * (radius * cu)
            + pos.y_direction().as_ref() * (radius * su)
            + pos.z_direction().as_ref() * v,
    )
}

pub fn cylinder_d1(u: f64, v: f64, pos: &Ax3, radius: f64) -> (Pnt, Vec3, Vec3) {
    let (cu, su) = (u.cos(), u.sin());
    let x: &Vec3 = pos.x_direction().as_ref();
    let y: &Vec3 = pos.y_direction().as_ref();
    let z: &Vec3 = pos.z_direction().as_ref();
    let p = Pnt::from(pos.origin().coords + x * (radius * cu) + y * (radius * su) + z * v);
    let vu = x * (-radius * su) + y * (radius * cu);
    let vv = *z;
    (p, vu, vv)
}

pub fn cylinder_parameters(pos: &Ax3, radius: f64, p: &Pnt) -> (f64, f64) {
    let d = p - pos.origin();
    let v = d.dot(pos.z_direction().as_ref());
    let dx = d.dot(pos.x_direction().as_ref());
    let dy = d.dot(pos.y_direction().as_ref());
    let mut u = dy.atan2(dx);
    normalize_angle(&mut u);
    let _ = radius; // radius not needed for parameter computation
    (u, v)
}

// ---------------------------------------------------------------------------
// Cone: P(u,v) = O + (R + v*sin(a))*cos(u)*X + (R + v*sin(a))*sin(u)*Y + v*cos(a)*Z
// ---------------------------------------------------------------------------

pub fn cone_value(u: f64, v: f64, pos: &Ax3, radius: f64, semi_angle: f64) -> Pnt {
    let (cu, su) = (u.cos(), u.sin());
    let r = radius + v * semi_angle.sin();
    let z_offset = v * semi_angle.cos();
    Pnt::from(
        pos.origin().coords
            + pos.x_direction().as_ref() * (r * cu)
            + pos.y_direction().as_ref() * (r * su)
            + pos.z_direction().as_ref() * z_offset,
    )
}

pub fn cone_d1(u: f64, v: f64, pos: &Ax3, radius: f64, semi_angle: f64) -> (Pnt, Vec3, Vec3) {
    let (cu, su) = (u.cos(), u.sin());
    let (sa, ca) = (semi_angle.sin(), semi_angle.cos());
    let r = radius + v * sa;
    let x: &Vec3 = pos.x_direction().as_ref();
    let y: &Vec3 = pos.y_direction().as_ref();
    let z: &Vec3 = pos.z_direction().as_ref();
    let p = Pnt::from(pos.origin().coords + x * (r * cu) + y * (r * su) + z * (v * ca));
    let vu = x * (-r * su) + y * (r * cu);
    let vv = x * (sa * cu) + y * (sa * su) + z * ca;
    (p, vu, vv)
}

pub fn cone_parameters(pos: &Ax3, radius: f64, semi_angle: f64, p: &Pnt) -> (f64, f64) {
    let d = p - pos.origin();
    let z_proj = d.dot(pos.z_direction().as_ref());
    let v = z_proj / semi_angle.cos();
    let dx = d.dot(pos.x_direction().as_ref());
    let dy = d.dot(pos.y_direction().as_ref());
    let mut u = dy.atan2(dx);
    normalize_angle(&mut u);
    let _ = radius; // used implicitly through the geometry
    (u, v)
}

// ---------------------------------------------------------------------------
// Sphere: P(u,v) = O + R*cos(v)*cos(u)*X + R*cos(v)*sin(u)*Y + R*sin(v)*Z
// ---------------------------------------------------------------------------

pub fn sphere_value(u: f64, v: f64, pos: &Ax3, radius: f64) -> Pnt {
    let (cu, su) = (u.cos(), u.sin());
    let (cv, sv) = (v.cos(), v.sin());
    let r = radius * cv;
    Pnt::from(
        pos.origin().coords
            + pos.x_direction().as_ref() * (r * cu)
            + pos.y_direction().as_ref() * (r * su)
            + pos.z_direction().as_ref() * (radius * sv),
    )
}

pub fn sphere_d1(u: f64, v: f64, pos: &Ax3, radius: f64) -> (Pnt, Vec3, Vec3) {
    let (cu, su) = (u.cos(), u.sin());
    let (cv, sv) = (v.cos(), v.sin());
    let r = radius * cv;
    let x: &Vec3 = pos.x_direction().as_ref();
    let y: &Vec3 = pos.y_direction().as_ref();
    let z: &Vec3 = pos.z_direction().as_ref();
    let p = Pnt::from(pos.origin().coords + x * (r * cu) + y * (r * su) + z * (radius * sv));
    let vu = x * (-r * su) + y * (r * cu);
    let vv = x * (-radius * sv * cu) + y * (-radius * sv * su) + z * (radius * cv);
    (p, vu, vv)
}

pub fn sphere_parameters(pos: &Ax3, _radius: f64, p: &Pnt) -> (f64, f64) {
    let d = p - pos.origin();
    let dx = d.dot(pos.x_direction().as_ref());
    let dy = d.dot(pos.y_direction().as_ref());
    let dz = d.dot(pos.z_direction().as_ref());
    let mut u = dy.atan2(dx);
    normalize_angle(&mut u);
    let v = dz.atan2((dx * dx + dy * dy).sqrt());
    (u, v)
}

// ---------------------------------------------------------------------------
// Torus: P(u,v) = O + (R + r*cos(v))*cos(u)*X + (R + r*cos(v))*sin(u)*Y + r*sin(v)*Z
// ---------------------------------------------------------------------------

pub fn torus_value(u: f64, v: f64, pos: &Ax3, major: f64, minor: f64) -> Pnt {
    let (cu, su) = (u.cos(), u.sin());
    let (cv, sv) = (v.cos(), v.sin());
    let r = major + minor * cv;
    let mut a1 = r * cu;
    let mut a2 = r * su;
    let mut a3 = minor * sv;
    let eps = 10.0 * (minor + major) * f64::EPSILON;
    if a1.abs() <= eps {
        a1 = 0.0;
    }
    if a2.abs() <= eps {
        a2 = 0.0;
    }
    if a3.abs() <= eps {
        a3 = 0.0;
    }
    Pnt::from(
        pos.origin().coords
            + pos.x_direction().as_ref() * a1
            + pos.y_direction().as_ref() * a2
            + pos.z_direction().as_ref() * a3,
    )
}

pub fn torus_d1(u: f64, v: f64, pos: &Ax3, major: f64, minor: f64) -> (Pnt, Vec3, Vec3) {
    let (cu, su) = (u.cos(), u.sin());
    let (cv, sv) = (v.cos(), v.sin());
    let r = major + minor * cv;
    let x: &Vec3 = pos.x_direction().as_ref();
    let y: &Vec3 = pos.y_direction().as_ref();
    let z: &Vec3 = pos.z_direction().as_ref();
    let p = Pnt::from(pos.origin().coords + x * (r * cu) + y * (r * su) + z * (minor * sv));
    let vu = x * (-r * su) + y * (r * cu);
    let vv = x * (-minor * sv * cu) + y * (-minor * sv * su) + z * (minor * cv);
    (p, vu, vv)
}

pub fn torus_parameters(pos: &Ax3, major: f64, minor: f64, p: &Pnt) -> (f64, f64) {
    let d = p - pos.origin();
    let dx = d.dot(pos.x_direction().as_ref());
    let dy = d.dot(pos.y_direction().as_ref());
    let dz = d.dot(pos.z_direction().as_ref());
    let mut u = dy.atan2(dx);
    normalize_angle(&mut u);
    // Distance from axis in XY plane, minus major radius → signed distance to tube center
    let rho = (dx * dx + dy * dy).sqrt() - major;
    let mut v = dz.atan2(rho);
    normalize_angle(&mut v);
    let _ = minor;
    (u, v)
}

// ---------------------------------------------------------------------------
// Angle normalization
// ---------------------------------------------------------------------------

fn normalize_angle(angle: &mut f64) {
    let two_pi = 2.0 * PI;
    while *angle < -precision::COMPUTATIONAL {
        *angle += two_pi;
    }
    while *angle > two_pi * (1.0 + crate::gp::RESOLUTION) {
        *angle -= two_pi;
    }
    if *angle < 0.0 {
        *angle = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gp::Dir;
    use nalgebra::{Point3, Unit, Vector3};
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};

    fn dir(x: f64, y: f64, z: f64) -> Dir {
        Unit::new_normalize(Vector3::new(x, y, z))
    }

    fn pnt(x: f64, y: f64, z: f64) -> Pnt {
        Point3::new(x, y, z)
    }

    fn assert_pnt_near(a: &Pnt, b: &Pnt, tol: f64) {
        assert!((a - b).norm() < tol, "{a:?} vs {b:?}");
    }

    fn standard_ax3() -> Ax3 {
        Ax3::new(Point3::origin(), dir(0.0, 0.0, 1.0), dir(1.0, 0.0, 0.0))
    }

    // Plane
    #[test]
    fn plane_at_origin() {
        assert_pnt_near(
            &plane_value(0.0, 0.0, &standard_ax3()),
            &Point3::origin(),
            1e-15,
        );
    }

    #[test]
    fn plane_parameter_roundtrip() {
        let pos = standard_ax3();
        let p = pnt(3.0, 4.0, 0.0);
        let (u, v) = plane_parameters(&pos, &p);
        assert_pnt_near(&plane_value(u, v, &pos), &p, 1e-14);
    }

    // Cylinder
    #[test]
    fn cylinder_at_u0_v0() {
        assert_pnt_near(
            &cylinder_value(0.0, 0.0, &standard_ax3(), 5.0),
            &pnt(5.0, 0.0, 0.0),
            1e-15,
        );
    }

    #[test]
    fn cylinder_at_v_offset() {
        assert_pnt_near(
            &cylinder_value(0.0, 3.0, &standard_ax3(), 5.0),
            &pnt(5.0, 0.0, 3.0),
            1e-15,
        );
    }

    // Cone
    #[test]
    fn cone_at_apex() {
        // v = -R/sin(a) should give the apex (radius goes to 0)
        let r = 5.0;
        let a = FRAC_PI_4;
        let v_apex = -r / a.sin();
        let p = cone_value(0.0, v_apex, &standard_ax3(), r, a);
        // All XY components should be ~0
        assert!(p.x.abs() < 1e-13);
        assert!(p.y.abs() < 1e-13);
    }

    // Sphere
    #[test]
    fn sphere_north_pole() {
        assert_pnt_near(
            &sphere_value(0.0, FRAC_PI_2, &standard_ax3(), 5.0),
            &pnt(0.0, 0.0, 5.0),
            1e-14,
        );
    }

    #[test]
    fn sphere_equator() {
        assert_pnt_near(
            &sphere_value(0.0, 0.0, &standard_ax3(), 5.0),
            &pnt(5.0, 0.0, 0.0),
            1e-15,
        );
    }

    #[test]
    fn sphere_parameter_roundtrip() {
        let pos = standard_ax3();
        let p = sphere_value(1.0, 0.5, &pos, 5.0);
        let (u, v) = sphere_parameters(&pos, 5.0, &p);
        assert_pnt_near(&sphere_value(u, v, &pos, 5.0), &p, 1e-13);
    }

    // Torus
    #[test]
    fn torus_outer_point() {
        // u=0, v=0: outermost point at (R+r, 0, 0)
        assert_pnt_near(
            &torus_value(0.0, 0.0, &standard_ax3(), 5.0, 1.0),
            &pnt(6.0, 0.0, 0.0),
            1e-14,
        );
    }

    #[test]
    fn torus_inner_point() {
        // u=0, v=PI: innermost at (R-r, 0, 0)
        assert_pnt_near(
            &torus_value(0.0, PI, &standard_ax3(), 5.0, 1.0),
            &pnt(4.0, 0.0, 0.0),
            1e-14,
        );
    }

    #[test]
    fn torus_top_point() {
        // u=0, v=PI/2: top at (R, 0, r)
        assert_pnt_near(
            &torus_value(0.0, FRAC_PI_2, &standard_ax3(), 5.0, 1.0),
            &pnt(5.0, 0.0, 1.0),
            1e-14,
        );
    }
}
