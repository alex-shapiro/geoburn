//! Elementary curve evaluation: point-at-parameter and derivatives.

use crate::gp::ax1::Ax1;
use crate::gp::ax2::Ax2;
use crate::gp::{Pnt, Vec3, precision};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Periodic parameter utilities
// ---------------------------------------------------------------------------

/// Wrap parameter `u` into the range `[u_first, u_last)`.
pub fn in_period(u: f64, u_first: f64, u_last: f64) -> f64 {
    if precision::is_infinite(u)
        || precision::is_infinite(u_first)
        || precision::is_infinite(u_last)
    {
        return u;
    }
    let period = u_last - u_first;
    if period < f64::EPSILON * u_last.abs() {
        return u;
    }
    (u + period * ((u_first - u) / period).ceil()).max(u_first)
}

/// Adjust two parameters into a periodic range, ensuring `u2 > u1`
/// with separation at least `preci`.
pub fn adjust_periodic(u_first: f64, u_last: f64, preci: f64, u1: &mut f64, u2: &mut f64) {
    if precision::is_infinite(u_first) || precision::is_infinite(u_last) {
        *u1 = u_first;
        *u2 = u_last;
        return;
    }
    let period = u_last - u_first;
    if period < f64::EPSILON * u_last.abs() {
        *u1 = u_first;
        *u2 = u_last;
        return;
    }
    *u1 -= ((*u1 - u_first) / period).floor() * period;
    if u_last - *u1 < preci {
        *u1 -= period;
    }
    *u2 -= ((*u2 - *u1) / period).floor() * period;
    if *u2 - *u1 < preci {
        *u2 += period;
    }
}

// ---------------------------------------------------------------------------
// Line
// ---------------------------------------------------------------------------

pub fn line_value(u: f64, pos: &Ax1) -> Pnt {
    Pnt::from(pos.origin.coords + pos.dir.as_ref() * u)
}

/// Returns (point, first_derivative).
pub fn line_d1(u: f64, pos: &Ax1) -> (Pnt, Vec3) {
    let d: &Vec3 = pos.dir.as_ref();
    (Pnt::from(pos.origin.coords + d * u), *d)
}

/// Nth derivative of a line. N=0 is not valid; N=1 is the direction; N≥2 is zero.
pub fn line_dn(u: f64, pos: &Ax1, n: u32) -> Vec3 {
    match n {
        0 => line_value(u, pos).coords,
        1 => *pos.dir.as_ref(),
        _ => Vec3::zeros(),
    }
}

/// Parameter of the point on the line closest to `p`.
pub fn line_parameter(pos: &Ax1, p: &Pnt) -> f64 {
    (p - pos.origin).dot(pos.dir.as_ref())
}

// ---------------------------------------------------------------------------
// Circle: P(u) = R*cos(u)*X + R*sin(u)*Y + O
// ---------------------------------------------------------------------------

pub fn circle_value(u: f64, pos: &Ax2, radius: f64) -> Pnt {
    let (cu, su) = (u.cos(), u.sin());
    Pnt::from(
        pos.origin().coords
            + pos.x_direction().as_ref() * (radius * cu)
            + pos.y_direction().as_ref() * (radius * su),
    )
}

pub fn circle_d1(u: f64, pos: &Ax2, radius: f64) -> (Pnt, Vec3) {
    let (cu, su) = (u.cos(), u.sin());
    let xc = radius * cu;
    let yc = radius * su;
    let x: &Vec3 = pos.x_direction().as_ref();
    let y: &Vec3 = pos.y_direction().as_ref();
    let p = Pnt::from(pos.origin().coords + x * xc + y * yc);
    let v1 = x * (-yc) + y * xc;
    (p, v1)
}

pub fn circle_d2(u: f64, pos: &Ax2, radius: f64) -> (Pnt, Vec3, Vec3) {
    let (cu, su) = (u.cos(), u.sin());
    let xc = radius * cu;
    let yc = radius * su;
    let x: &Vec3 = pos.x_direction().as_ref();
    let y: &Vec3 = pos.y_direction().as_ref();
    let p = Pnt::from(pos.origin().coords + x * xc + y * yc);
    let v1 = x * (-yc) + y * xc;
    let v2 = x * (-xc) + y * (-yc);
    (p, v1, v2)
}

pub fn circle_d3(u: f64, pos: &Ax2, radius: f64) -> (Pnt, Vec3, Vec3, Vec3) {
    let (cu, su) = (u.cos(), u.sin());
    let xc = radius * cu;
    let yc = radius * su;
    let x: &Vec3 = pos.x_direction().as_ref();
    let y: &Vec3 = pos.y_direction().as_ref();
    let p = Pnt::from(pos.origin().coords + x * xc + y * yc);
    let v1 = x * (-yc) + y * xc;
    let v2 = x * (-xc) + y * (-yc);
    let v3 = x * yc + y * (-xc);
    (p, v1, v2, v3)
}

pub fn circle_parameter(pos: &Ax2, p: &Pnt) -> f64 {
    let v = p - pos.origin();
    let mut u = v
        .dot(pos.y_direction().as_ref())
        .atan2(v.dot(pos.x_direction().as_ref()));
    if u < -precision::COMPUTATIONAL {
        u += 2.0 * PI;
    }
    if u < 0.0 {
        u = 0.0;
    }
    u
}

// ---------------------------------------------------------------------------
// Ellipse: P(u) = a*cos(u)*X + b*sin(u)*Y + O
// ---------------------------------------------------------------------------

pub fn ellipse_value(u: f64, pos: &Ax2, major: f64, minor: f64) -> Pnt {
    let (cu, su) = (u.cos(), u.sin());
    Pnt::from(
        pos.origin().coords
            + pos.x_direction().as_ref() * (major * cu)
            + pos.y_direction().as_ref() * (minor * su),
    )
}

pub fn ellipse_d1(u: f64, pos: &Ax2, major: f64, minor: f64) -> (Pnt, Vec3) {
    let (cu, su) = (u.cos(), u.sin());
    let x: &Vec3 = pos.x_direction().as_ref();
    let y: &Vec3 = pos.y_direction().as_ref();
    let p = Pnt::from(pos.origin().coords + x * (major * cu) + y * (minor * su));
    let v1 = x * (-major * su) + y * (minor * cu);
    (p, v1)
}

pub fn ellipse_d2(u: f64, pos: &Ax2, major: f64, minor: f64) -> (Pnt, Vec3, Vec3) {
    let (cu, su) = (u.cos(), u.sin());
    let x: &Vec3 = pos.x_direction().as_ref();
    let y: &Vec3 = pos.y_direction().as_ref();
    let p = Pnt::from(pos.origin().coords + x * (major * cu) + y * (minor * su));
    let v1 = x * (-major * su) + y * (minor * cu);
    let v2 = x * (-major * cu) + y * (-minor * su);
    (p, v1, v2)
}

pub fn ellipse_d3(u: f64, pos: &Ax2, major: f64, minor: f64) -> (Pnt, Vec3, Vec3, Vec3) {
    let (cu, su) = (u.cos(), u.sin());
    let x: &Vec3 = pos.x_direction().as_ref();
    let y: &Vec3 = pos.y_direction().as_ref();
    let p = Pnt::from(pos.origin().coords + x * (major * cu) + y * (minor * su));
    let v1 = x * (-major * su) + y * (minor * cu);
    let v2 = x * (-major * cu) + y * (-minor * su);
    let v3 = x * (major * su) + y * (-minor * cu);
    (p, v1, v2, v3)
}

pub fn ellipse_parameter(pos: &Ax2, major: f64, minor: f64, p: &Pnt) -> f64 {
    let v = p - pos.origin();
    let dx = v.dot(pos.x_direction().as_ref());
    let dy = v.dot(pos.y_direction().as_ref());
    // P = (a*cos(u), b*sin(u)) → atan2(dy/b, dx/a)
    let mut u = (dy / minor).atan2(dx / major);
    if u < -precision::COMPUTATIONAL {
        u += 2.0 * PI;
    }
    if u < 0.0 {
        u = 0.0;
    }
    u
}

// ---------------------------------------------------------------------------
// Hyperbola: P(u) = a*cosh(u)*X + b*sinh(u)*Y + O
// ---------------------------------------------------------------------------

pub fn hyperbola_value(u: f64, pos: &Ax2, major: f64, minor: f64) -> Pnt {
    Pnt::from(
        pos.origin().coords
            + pos.x_direction().as_ref() * (major * u.cosh())
            + pos.y_direction().as_ref() * (minor * u.sinh()),
    )
}

pub fn hyperbola_d1(u: f64, pos: &Ax2, major: f64, minor: f64) -> (Pnt, Vec3) {
    let (ch, sh) = (u.cosh(), u.sinh());
    let x: &Vec3 = pos.x_direction().as_ref();
    let y: &Vec3 = pos.y_direction().as_ref();
    let p = Pnt::from(pos.origin().coords + x * (major * ch) + y * (minor * sh));
    let v1 = x * (major * sh) + y * (minor * ch);
    (p, v1)
}

pub fn hyperbola_d2(u: f64, pos: &Ax2, major: f64, minor: f64) -> (Pnt, Vec3, Vec3) {
    let (ch, sh) = (u.cosh(), u.sinh());
    let x: &Vec3 = pos.x_direction().as_ref();
    let y: &Vec3 = pos.y_direction().as_ref();
    let v2 = x * (major * ch) + y * (minor * sh); // = P - O (same form as point offset)
    let p = Pnt::from(pos.origin().coords + v2);
    let v1 = x * (major * sh) + y * (minor * ch);
    (p, v1, v2)
}

pub fn hyperbola_d3(u: f64, pos: &Ax2, major: f64, minor: f64) -> (Pnt, Vec3, Vec3, Vec3) {
    let (ch, sh) = (u.cosh(), u.sinh());
    let x: &Vec3 = pos.x_direction().as_ref();
    let y: &Vec3 = pos.y_direction().as_ref();
    let v2 = x * (major * ch) + y * (minor * sh);
    let p = Pnt::from(pos.origin().coords + v2);
    let v1 = x * (major * sh) + y * (minor * ch);
    let v3 = v1; // d3 of cosh/sinh cycles back to sinh/cosh
    (p, v1, v2, v3)
}

pub fn hyperbola_parameter(pos: &Ax2, major: f64, minor: f64, p: &Pnt) -> f64 {
    let v = p - pos.origin();
    let dx = v.dot(pos.x_direction().as_ref());
    let _dy = v.dot(pos.y_direction().as_ref());
    // cosh(u) = dx/a, so u = acosh(dx/a), but we need sign from sinh
    // sinh(u) = dy/b, so u = asinh(dy/b)
    // Use asinh for the sign:
    let _ = dx / major; // cosh component (always positive)
    (v.dot(pos.y_direction().as_ref()) / minor).asinh()
}

// ---------------------------------------------------------------------------
// Parabola: P(u) = u²/(4f)*X + u*Y + O
// ---------------------------------------------------------------------------

pub fn parabola_value(u: f64, pos: &Ax2, focal: f64) -> Pnt {
    if focal.abs() <= crate::gp::RESOLUTION {
        return Pnt::from(pos.origin().coords + pos.x_direction().as_ref() * u);
    }
    Pnt::from(
        pos.origin().coords
            + pos.x_direction().as_ref() * (u * u / (4.0 * focal))
            + pos.y_direction().as_ref() * u,
    )
}

pub fn parabola_d1(u: f64, pos: &Ax2, focal: f64) -> (Pnt, Vec3) {
    let x: &Vec3 = pos.x_direction().as_ref();
    if focal.abs() <= crate::gp::RESOLUTION {
        return (Pnt::from(pos.origin().coords + x * u), *x);
    }
    let y: &Vec3 = pos.y_direction().as_ref();
    let p = Pnt::from(pos.origin().coords + x * (u * u / (4.0 * focal)) + y * u);
    let v1 = x * (u / (2.0 * focal)) + y;
    (p, v1)
}

pub fn parabola_d2(u: f64, pos: &Ax2, focal: f64) -> (Pnt, Vec3, Vec3) {
    let x: &Vec3 = pos.x_direction().as_ref();
    if focal.abs() <= crate::gp::RESOLUTION {
        return (Pnt::from(pos.origin().coords + x * u), *x, Vec3::zeros());
    }
    let y: &Vec3 = pos.y_direction().as_ref();
    let p = Pnt::from(pos.origin().coords + x * (u * u / (4.0 * focal)) + y * u);
    let v1 = x * (u / (2.0 * focal)) + y;
    let v2 = x * (1.0 / (2.0 * focal));
    (p, v1, v2)
}

pub fn parabola_parameter(pos: &Ax2, p: &Pnt) -> f64 {
    (p - pos.origin()).dot(pos.y_direction().as_ref())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gp::Dir;
    use nalgebra::{Point3, Unit, Vector3};
    use std::f64::consts::FRAC_PI_2;

    fn dir(x: f64, y: f64, z: f64) -> Dir {
        Unit::new_normalize(Vector3::new(x, y, z))
    }

    fn pnt(x: f64, y: f64, z: f64) -> Pnt {
        Point3::new(x, y, z)
    }

    fn assert_pnt_near(a: &Pnt, b: &Pnt, tol: f64) {
        assert!((a - b).norm() < tol, "{a:?} vs {b:?}");
    }

    fn xy_ax2() -> Ax2 {
        Ax2::new(Point3::origin(), dir(0.0, 0.0, 1.0), dir(1.0, 0.0, 0.0))
    }

    // Line tests
    #[test]
    fn line_at_origin() {
        let ax = Ax1::new(Point3::origin(), dir(1.0, 0.0, 0.0));
        assert_pnt_near(&line_value(5.0, &ax), &pnt(5.0, 0.0, 0.0), 1e-15);
    }

    #[test]
    fn line_parameter_roundtrip() {
        let ax = Ax1::new(pnt(1.0, 2.0, 3.0), dir(0.0, 0.0, 1.0));
        let u = 7.5;
        let p = line_value(u, &ax);
        assert!((line_parameter(&ax, &p) - u).abs() < 1e-14);
    }

    // Circle tests
    #[test]
    fn circle_at_0_is_x_axis() {
        assert_pnt_near(
            &circle_value(0.0, &xy_ax2(), 5.0),
            &pnt(5.0, 0.0, 0.0),
            1e-15,
        );
    }

    #[test]
    fn circle_at_pi_2_is_y_axis() {
        assert_pnt_near(
            &circle_value(FRAC_PI_2, &xy_ax2(), 5.0),
            &pnt(0.0, 5.0, 0.0),
            1e-14,
        );
    }

    #[test]
    fn circle_d1_tangent_is_perpendicular() {
        let (p, v1) = circle_d1(0.0, &xy_ax2(), 5.0);
        // At u=0, P=(5,0,0), tangent should be (0,5,0)
        assert_pnt_near(&p, &pnt(5.0, 0.0, 0.0), 1e-15);
        assert!((v1 - Vec3::new(0.0, 5.0, 0.0)).norm() < 1e-14);
    }

    #[test]
    fn circle_parameter_roundtrip() {
        let pos = xy_ax2();
        let u = 1.23;
        let p = circle_value(u, &pos, 5.0);
        assert!((circle_parameter(&pos, &p) - u).abs() < 1e-14);
    }

    // Ellipse tests
    #[test]
    fn ellipse_at_0() {
        assert_pnt_near(
            &ellipse_value(0.0, &xy_ax2(), 5.0, 3.0),
            &pnt(5.0, 0.0, 0.0),
            1e-15,
        );
    }

    #[test]
    fn ellipse_at_pi_2() {
        assert_pnt_near(
            &ellipse_value(FRAC_PI_2, &xy_ax2(), 5.0, 3.0),
            &pnt(0.0, 3.0, 0.0),
            1e-14,
        );
    }

    // Hyperbola tests
    #[test]
    fn hyperbola_at_0() {
        // cosh(0)=1, sinh(0)=0
        assert_pnt_near(
            &hyperbola_value(0.0, &xy_ax2(), 3.0, 4.0),
            &pnt(3.0, 0.0, 0.0),
            1e-15,
        );
    }

    // Parabola tests
    #[test]
    fn parabola_at_0_is_vertex() {
        assert_pnt_near(
            &parabola_value(0.0, &xy_ax2(), 5.0),
            &Point3::origin(),
            1e-15,
        );
    }

    #[test]
    fn parabola_shape() {
        // u=4, focal=2 → x = 16/8 = 2, y = 4
        assert_pnt_near(
            &parabola_value(4.0, &xy_ax2(), 2.0),
            &pnt(2.0, 4.0, 0.0),
            1e-14,
        );
    }

    // In-period
    #[test]
    fn in_period_wraps() {
        let u = in_period(7.0, 0.0, 2.0 * PI);
        assert!(u >= 0.0 && u < 2.0 * PI);
    }
}
