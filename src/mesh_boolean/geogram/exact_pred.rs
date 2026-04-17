// Pedantic clippy lints suppressed for faithful port fidelity:
#![allow(clippy::doc_markdown, clippy::too_many_arguments)]

//! Exact geometric predicates on homogeneous coordinates (Modules 2+3).
//!
//! Port of Geogram's `exact_geometry.h/.cpp` and `vechg.h`.
//! BSD 3-Clause license (original Geogram copyright Inria).

use super::expansion::{Expansion, expansion_det2x2, expansion_det3x3};
use std::cmp::Ordering;

/// Sign of a predicate result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sign {
    Negative = -1,
    Zero = 0,
    Positive = 1,
}

impl Sign {
    pub fn from_i32(v: i32) -> Self {
        match v.signum() {
            -1 => Self::Negative,
            1 => Self::Positive,
            _ => Self::Zero,
        }
    }

    /// Multiply two signs (sign product).
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, other: Self) -> Self {
        Self::from_i32(self as i32 * other as i32)
    }
}

/// 2D point in homogeneous coordinates with expansion components.
/// Represents the Cartesian point (x/w, y/w).
#[derive(Clone, Debug)]
pub struct Vec2HE {
    pub x: Expansion,
    pub y: Expansion,
    pub w: Expansion,
}

impl Vec2HE {
    pub fn new(x: Expansion, y: Expansion, w: Expansion) -> Self {
        Self { x, y, w }
    }

    pub fn from_f64(x: f64, y: f64) -> Self {
        Self {
            x: Expansion::from_f64(x),
            y: Expansion::from_f64(y),
            w: Expansion::from_f64(1.0),
        }
    }
}

/// 3D point in homogeneous coordinates with expansion components.
/// Represents the Cartesian point (x/w, y/w, z/w).
#[derive(Clone, Debug)]
pub struct Vec3HE {
    pub x: Expansion,
    pub y: Expansion,
    pub z: Expansion,
    pub w: Expansion,
}

impl Vec3HE {
    pub fn new(x: Expansion, y: Expansion, z: Expansion, w: Expansion) -> Self {
        Self { x, y, z, w }
    }

    pub fn from_f64(x: f64, y: f64, z: f64) -> Self {
        Self {
            x: Expansion::from_f64(x),
            y: Expansion::from_f64(y),
            z: Expansion::from_f64(z),
            w: Expansion::from_f64(1.0),
        }
    }

    /// Homogeneous subtraction: (a/wa) - (b/wb) = (a*wb - b*wa) / (wa*wb).
    pub fn sub(&self, other: &Self) -> Self {
        Self {
            x: self.x.mul(&other.w).sub(&other.x.mul(&self.w)),
            y: self.y.mul(&other.w).sub(&other.y.mul(&self.w)),
            z: self.z.mul(&other.w).sub(&other.z.mul(&self.w)),
            w: self.w.mul(&other.w),
        }
    }

    pub fn coord(&self, axis: usize) -> &Expansion {
        match axis {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("invalid axis {axis}"),
        }
    }
}

/// Exact 2D orientation predicate on homogeneous coordinates.
///
/// Computes sign of det|p0.x p0.y p0.w; p1.x p1.y p1.w; p2.x p2.y p2.w|
/// multiplied by sign(w0)*sign(w1)*sign(w2).
pub fn orient_2d(p0: &Vec2HE, p1: &Vec2HE, p2: &Vec2HE) -> Sign {
    let delta = expansion_det3x3(
        &[p0.x.clone(), p0.y.clone(), p0.w.clone()],
        &[p1.x.clone(), p1.y.clone(), p1.w.clone()],
        &[p2.x.clone(), p2.y.clone(), p2.w.clone()],
    );
    let w_sign = Sign::from_i32(p0.w.sign())
        .mul(Sign::from_i32(p1.w.sign()))
        .mul(Sign::from_i32(p2.w.sign()));
    Sign::from_i32(delta.sign()).mul(w_sign)
}

/// Exact 3D orientation predicate on homogeneous coordinates.
///
/// Computes sign of det(p1-p0, p2-p0, p3-p0), consistent with
/// Geogram's `PCK::orient_3d(p0, p1, p2, p3)`.
///
/// Note: this differs from `geometry_predicates::orient3d` by a sign
/// (Geogram subtracts p0; Shewchuk's convention subtracts p3).
pub fn orient_3d(p0: &Vec3HE, p1: &Vec3HE, p2: &Vec3HE, p3: &Vec3HE) -> Sign {
    let u = p1.sub(p0);
    let v = p2.sub(p0);
    let w = p3.sub(p0);

    let delta = expansion_det3x3(&[u.x, u.y, u.z], &[v.x, v.y, v.z], &[w.x, w.y, w.z]);

    let w_sign = Sign::from_i32(u.w.sign())
        .mul(Sign::from_i32(v.w.sign()))
        .mul(Sign::from_i32(w.w.sign()));

    Sign::from_i32(delta.sign()).mul(w_sign)
}

/// Exact 2D orientation projected along an axis.
///
/// Projects 3D homogeneous points onto the plane perpendicular to `axis`,
/// using coordinates `(axis+1)%3` and `(axis+2)%3`.
pub fn orient_2d_projected(p0: &Vec3HE, p1: &Vec3HE, p2: &Vec3HE, axis: usize) -> Sign {
    let u = (axis + 1) % 3;
    let v = (axis + 2) % 3;

    let delta = expansion_det3x3(
        &[p0.coord(u).clone(), p0.coord(v).clone(), p0.w.clone()],
        &[p1.coord(u).clone(), p1.coord(v).clone(), p1.w.clone()],
        &[p2.coord(u).clone(), p2.coord(v).clone(), p2.w.clone()],
    );

    let w_sign = Sign::from_i32(p0.w.sign())
        .mul(Sign::from_i32(p1.w.sign()))
        .mul(Sign::from_i32(p2.w.sign()));

    Sign::from_i32(delta.sign()).mul(w_sign)
}

/// Exact comparison of two ratios: returns sign of (a_num/a_den - b_num/b_den).
/// Port of Geogram's Numeric::ratio_compare.
pub fn ratio_compare(
    a_num: &Expansion,
    a_den: &Expansion,
    b_num: &Expansion,
    b_den: &Expansion,
) -> Sign {
    // sign(a_num/a_den - b_num/b_den)
    // = sign(a_num*b_den - b_num*a_den) * sign(a_den) * sign(b_den)
    let diff = a_num.mul(b_den).sub(&b_num.mul(a_den));
    let s = diff.sign();
    if s == 0 {
        return Sign::Zero;
    }
    Sign::from_i32(s * a_den.sign() * b_den.sign())
}

/// Lexicographic comparison of two Vec2HE points.
/// Port of Geogram's vec2HgLexicoCompare<expansion_nt>.
/// Returns true if v1 < v2 in lexicographic order (x/w first, then y/w).
fn vec2he_lexico_less(v1: &Vec2HE, v2: &Vec2HE) -> bool {
    let s = ratio_compare(&v2.x, &v2.w, &v1.x, &v1.w);
    if s == Sign::Positive {
        return true;
    }
    if s == Sign::Negative {
        return false;
    }
    let s = ratio_compare(&v2.y, &v2.w, &v1.y, &v1.w);
    s == Sign::Positive
}

/// Sign of the determinant:
/// | x1/w1  y1/w1  1 |
/// | x2/w2  y2/w2  1 |
/// | x3/w3  y3/w3  1 |
///
/// Port of Geogram's det3_111_sign (exact_geometry.cpp).
fn det3_111_sign(p1: &Vec2HE, p2: &Vec2HE, p3: &Vec2HE) -> Sign {
    let m1 = expansion_det2x2(&p2.x, &p2.y, &p3.x, &p3.y);
    let m2 = expansion_det2x2(&p1.x, &p1.y, &p3.x, &p3.y);
    let m3 = expansion_det2x2(&p1.x, &p1.y, &p2.x, &p2.y);
    let d = p1.w.mul(&m1).sub(&p2.w.mul(&m2)).add(&p3.w.mul(&m3));
    Sign::from_i32(p1.w.sign() * p2.w.sign() * p3.w.sign() * d.sign())
}

/// Simulation of Simplicity for 4 points.
/// Port of Geogram's SOS() template (PCK.h).
///
/// Sorts the 4 points by lexicographic order, then evaluates the SOS
/// expression for each point (in sorted order) until one returns non-zero.
fn sos_incircle(p0: &Vec2HE, p1: &Vec2HE, p2: &Vec2HE, p3: &Vec2HE) -> Sign {
    // Build an array of (index, &point) and sort by lexico order.
    let mut pts: [usize; 4] = [0, 1, 2, 3];
    let all = [p0, p1, p2, p3];
    pts.sort_by(|&a, &b| {
        if vec2he_lexico_less(all[a], all[b]) {
            Ordering::Less
        } else if vec2he_lexico_less(all[b], all[a]) {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    });

    // Evaluate SOS expressions in sorted order.
    // SOS expression for p0: +det3_111_sign(p1,p2,p3)
    // SOS expression for p1: -det3_111_sign(p0,p2,p3)
    // SOS expression for p2: +det3_111_sign(p0,p1,p3)
    // SOS expression for p3: -det3_111_sign(p0,p1,p2)
    for &i in &pts {
        let result = match i {
            0 => det3_111_sign(p1, p2, p3),
            1 => det3_111_sign(p0, p2, p3).mul(Sign::Negative),
            2 => det3_111_sign(p0, p1, p3),
            3 => det3_111_sign(p0, p1, p2).mul(Sign::Negative),
            _ => unreachable!(),
        };
        if result != Sign::Zero {
            return result;
        }
    }
    unreachable!("SOS should always break ties");
}

/// Exact in-circle test with Simulation of Simplicity and pre-computed lengths.
///
/// Port of Geogram's `incircle_2d_SOS_with_lengths` (exact_geometry.cpp).
///
/// The `l` values are approximate: `li = (xi^2 + yi^2) / wi^2`.
/// Using f64 lengths avoids expansion overflow for the lifted coordinate.
/// This makes it a "regular (weighted) triangulation" perturbation rather
/// than exact Delaunay, but CDT2d tests convexity separately.
pub fn incircle_2d_sos_with_lengths(
    p0: &Vec2HE,
    p1: &Vec2HE,
    p2: &Vec2HE,
    p3: &Vec2HE,
    l0: f64,
    l1: f64,
    l2: f64,
    l3: f64,
) -> Sign {
    // Exact computation.
    //
    // Subtract p3 from p0,p1,p2 (in homogeneous coords):
    //   Pi = pi - p3  (i=0,1,2, producing P1,P2,P3 in Geogram notation)
    //   Li = li - l3
    //
    // Then compute:
    //   D = L1*W1*M1 - L2*W2*M2 + L3*W3*M3
    // where Mi = det2x2 of the appropriate Pi's
    // Result = sign(D) * sign(W1) * sign(W2) * sign(W3)

    let big_l1 = Expansion::from_f64(l0).sub(&Expansion::from_f64(l3));
    let big_l2 = Expansion::from_f64(l1).sub(&Expansion::from_f64(l3));
    let big_l3 = Expansion::from_f64(l2).sub(&Expansion::from_f64(l3));

    // P1 = p0 - p3, P2 = p1 - p3, P3 = p2 - p3 (homogeneous subtraction)
    let p1x = p0.x.mul(&p3.w).sub(&p3.x.mul(&p0.w));
    let p1y = p0.y.mul(&p3.w).sub(&p3.y.mul(&p0.w));
    let p1w = p0.w.mul(&p3.w);

    let p2x = p1.x.mul(&p3.w).sub(&p3.x.mul(&p1.w));
    let p2y = p1.y.mul(&p3.w).sub(&p3.y.mul(&p1.w));
    let p2w = p1.w.mul(&p3.w);

    let p3x = p2.x.mul(&p3.w).sub(&p3.x.mul(&p2.w));
    let p3y = p2.y.mul(&p3.w).sub(&p3.y.mul(&p2.w));
    let p3w = p2.w.mul(&p3.w);

    let m1 = expansion_det2x2(&p2x, &p2y, &p3x, &p3y);
    let m2 = expansion_det2x2(&p1x, &p1y, &p3x, &p3y);
    let m3 = expansion_det2x2(&p1x, &p1y, &p2x, &p2y);

    let d = big_l1
        .mul(&p1w)
        .mul(&m1)
        .sub(&big_l2.mul(&p2w).mul(&m2))
        .add(&big_l3.mul(&p3w).mul(&m3));

    let result = Sign::from_i32(d.sign() * p1w.sign() * p2w.sign() * p3w.sign());

    if result != Sign::Zero {
        return result;
    }

    // Symbolic perturbation (SOS).
    sos_incircle(p0, p1, p2, p3)
}

/// Exact in-circle test with Simulation of Simplicity.
///
/// Port of Geogram's `incircle_2d_SOS` (exact_geometry.h inline).
/// Computes l values on the fly from the homogeneous coordinates.
pub fn incircle_2d_sos(p0: &Vec2HE, p1: &Vec2HE, p2: &Vec2HE, p3: &Vec2HE) -> Sign {
    let compute_l = |p: &Vec2HE| -> f64 {
        let x_est = p.x.estimate();
        let y_est = p.y.estimate();
        let w_est = p.w.estimate();
        (x_est * x_est + y_est * y_est) / (w_est * w_est)
    };
    incircle_2d_sos_with_lengths(
        p0,
        p1,
        p2,
        p3,
        compute_l(p0),
        compute_l(p1),
        compute_l(p2),
        compute_l(p3),
    )
}

/// Get the axis most normal to a triangle.
pub fn triangle_normal_axis(p1: [f64; 3], p2: [f64; 3], p3: [f64; 3]) -> usize {
    let ux = p2[0] - p1[0];
    let uy = p2[1] - p1[1];
    let uz = p2[2] - p1[2];
    let vx = p3[0] - p1[0];
    let vy = p3[1] - p1[1];
    let vz = p3[2] - p1[2];

    let nx = (uy * vz - uz * vy).abs();
    let ny = (uz * vx - ux * vz).abs();
    let nz = (ux * vy - uy * vx).abs();

    if nx >= ny && nx >= nz {
        0
    } else if ny >= nz {
        1
    } else {
        2
    }
}

/// Test whether three 3D homogeneous points are collinear.
pub fn aligned_3d(p0: &Vec3HE, p1: &Vec3HE, p2: &Vec3HE) -> bool {
    // Three points are collinear iff orient_2d_projected is zero
    // for all three projection axes.
    orient_2d_projected(p0, p1, p2, 0) == Sign::Zero
        && orient_2d_projected(p0, p1, p2, 1) == Sign::Zero
        && orient_2d_projected(p0, p1, p2, 2) == Sign::Zero
}

// ── PCK predicates on f64 coordinates ───────────────────────────────
//
// Port of Geogram's PCK::orient_2d, orient_3d, dot_3d, aligned_3d
// on const double* coordinates. Each uses a fast f64 filter and falls
// back to exact expansion arithmetic when the filter is inconclusive.

/// Exact 2D orientation on f64 coordinates.
/// Computes sign of det(p1-p0, p2-p0).
pub fn pck_orient_2d(p0: &[f64; 2], p1: &[f64; 2], p2: &[f64; 2]) -> Sign {
    // Fast filter.
    let a11 = p1[0] - p0[0];
    let a12 = p1[1] - p0[1];
    let a21 = p2[0] - p0[0];
    let a22 = p2[1] - p0[1];
    let det = a11 * a22 - a12 * a21;

    // Error bound for 2x2 determinant (Shewchuk).
    let abs_det = det.abs();
    let max_val = a11.abs().max(a12.abs()).max(a21.abs()).max(a22.abs());
    let eps = max_val * max_val * f64::EPSILON * 4.0;

    if abs_det > eps {
        return Sign::from_i32(det.signum() as i32);
    }

    // Exact fallback.
    let ea11 = Expansion::from_f64(p1[0]).sub(&Expansion::from_f64(p0[0]));
    let ea12 = Expansion::from_f64(p1[1]).sub(&Expansion::from_f64(p0[1]));
    let ea21 = Expansion::from_f64(p2[0]).sub(&Expansion::from_f64(p0[0]));
    let ea22 = Expansion::from_f64(p2[1]).sub(&Expansion::from_f64(p0[1]));
    let delta = expansion_det2x2(&ea11, &ea12, &ea21, &ea22);
    Sign::from_i32(delta.sign())
}

/// Exact 3D orientation on f64 coordinates.
/// Computes sign of det(p1-p0, p2-p0, p3-p0).
pub fn pck_orient_3d(p0: &[f64; 3], p1: &[f64; 3], p2: &[f64; 3], p3: &[f64; 3]) -> Sign {
    // Fast filter.
    let a11 = p1[0] - p0[0];
    let a12 = p1[1] - p0[1];
    let a13 = p1[2] - p0[2];
    let a21 = p2[0] - p0[0];
    let a22 = p2[1] - p0[1];
    let a23 = p2[2] - p0[2];
    let a31 = p3[0] - p0[0];
    let a32 = p3[1] - p0[1];
    let a33 = p3[2] - p0[2];

    let det = a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31)
        + a13 * (a21 * a32 - a22 * a31);

    let abs_det = det.abs();
    let max_val = a11
        .abs()
        .max(a12.abs())
        .max(a13.abs())
        .max(a21.abs())
        .max(a22.abs())
        .max(a23.abs())
        .max(a31.abs())
        .max(a32.abs())
        .max(a33.abs());
    let eps = max_val * max_val * max_val * f64::EPSILON * 24.0;

    if abs_det > eps {
        return Sign::from_i32(det.signum() as i32);
    }

    // Exact fallback.
    let row0 = [
        Expansion::from_f64(p1[0]).sub(&Expansion::from_f64(p0[0])),
        Expansion::from_f64(p1[1]).sub(&Expansion::from_f64(p0[1])),
        Expansion::from_f64(p1[2]).sub(&Expansion::from_f64(p0[2])),
    ];
    let row1 = [
        Expansion::from_f64(p2[0]).sub(&Expansion::from_f64(p0[0])),
        Expansion::from_f64(p2[1]).sub(&Expansion::from_f64(p0[1])),
        Expansion::from_f64(p2[2]).sub(&Expansion::from_f64(p0[2])),
    ];
    let row2 = [
        Expansion::from_f64(p3[0]).sub(&Expansion::from_f64(p0[0])),
        Expansion::from_f64(p3[1]).sub(&Expansion::from_f64(p0[1])),
        Expansion::from_f64(p3[2]).sub(&Expansion::from_f64(p0[2])),
    ];
    let delta = expansion_det3x3(&row0, &row1, &row2);
    Sign::from_i32(delta.sign())
}

/// Exact dot product sign on f64 coordinates.
/// Returns sign of (p1-p0)·(p2-p0).
pub fn pck_dot_3d(p0: &[f64; 3], p1: &[f64; 3], p2: &[f64; 3]) -> Sign {
    let u0 = p1[0] - p0[0];
    let u1 = p1[1] - p0[1];
    let u2 = p1[2] - p0[2];
    let v0 = p2[0] - p0[0];
    let v1 = p2[1] - p0[1];
    let v2 = p2[2] - p0[2];
    let dot = u0 * v0 + u1 * v1 + u2 * v2;

    let abs_dot = dot.abs();
    let max_u = u0.abs().max(u1.abs()).max(u2.abs());
    let max_v = v0.abs().max(v1.abs()).max(v2.abs());
    let eps = max_u * max_v * f64::EPSILON * 8.0;

    if abs_dot > eps {
        return Sign::from_i32(dot.signum() as i32);
    }

    // Exact fallback.
    let eu0 = Expansion::from_f64(p1[0]).sub(&Expansion::from_f64(p0[0]));
    let eu1 = Expansion::from_f64(p1[1]).sub(&Expansion::from_f64(p0[1]));
    let eu2 = Expansion::from_f64(p1[2]).sub(&Expansion::from_f64(p0[2]));
    let ev0 = Expansion::from_f64(p2[0]).sub(&Expansion::from_f64(p0[0]));
    let ev1 = Expansion::from_f64(p2[1]).sub(&Expansion::from_f64(p0[1]));
    let ev2 = Expansion::from_f64(p2[2]).sub(&Expansion::from_f64(p0[2]));
    let delta = eu0.mul(&ev0).add(&eu1.mul(&ev1)).add(&eu2.mul(&ev2));
    Sign::from_i32(delta.sign())
}

/// Test whether three f64 3D points are collinear.
/// Returns true if the cross product (p1-p0)×(p2-p0) is zero in all 3 components.
pub fn pck_aligned_3d(p0: &[f64; 3], p1: &[f64; 3], p2: &[f64; 3]) -> bool {
    let u0 = Expansion::from_f64(p1[0]).sub(&Expansion::from_f64(p0[0]));
    let u1 = Expansion::from_f64(p1[1]).sub(&Expansion::from_f64(p0[1]));
    let u2 = Expansion::from_f64(p1[2]).sub(&Expansion::from_f64(p0[2]));
    let v0 = Expansion::from_f64(p2[0]).sub(&Expansion::from_f64(p0[0]));
    let v1 = Expansion::from_f64(p2[1]).sub(&Expansion::from_f64(p0[1]));
    let v2 = Expansion::from_f64(p2[2]).sub(&Expansion::from_f64(p0[2]));

    let n0 = expansion_det2x2(&u1, &v1, &u2, &v2);
    let n1 = expansion_det2x2(&u2, &v2, &u0, &v0);
    let n2 = expansion_det2x2(&u0, &v0, &u1, &v1);

    n0.sign() == 0 && n1.sign() == 0 && n2.sign() == 0
}

/// Test whether point P lies on segment [Q1, Q2] in exact 3D homogeneous coords.
/// Port of Geogram's `PCK::on_segment_3d`.
///
/// P is on segment [Q1,Q2] iff:
///   1. cross(P-Q1, P-Q2) == 0 (collinear)
///   2. dot(P-Q1, P-Q2) <= 0   (between Q1 and Q2)
pub fn on_segment_3d(p: &Vec3HE, q1: &Vec3HE, q2: &Vec3HE) -> bool {
    // Check collinearity: cross(P-Q1, P-Q2) must be zero in all components.
    if !aligned_3d(p, q1, q2) {
        return false;
    }
    // Check betweenness: dot(P-Q1, P-Q2) <= 0.
    // d = P - Q1, e = P - Q2 (homogeneous subtraction).
    let d = p.sub(q1);
    let e = p.sub(q2);
    // dot(d, e) = d.x*e.x + d.y*e.y + d.z*e.z (but in homogeneous coords,
    // the actual Cartesian vectors are d/d.w and e/e.w, so the sign of the
    // dot product is sign(d.x*e.x + d.y*e.y + d.z*e.z) * sign(d.w) * sign(e.w)).
    let dot = d.x.mul(&e.x).add(&d.y.mul(&e.y)).add(&d.z.mul(&e.z));
    let dot_sign = dot.sign() * d.w.sign() * e.w.sign();
    dot_sign <= 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec3he_sub_basic() {
        let a = Vec3HE::from_f64(3.0, 4.0, 5.0);
        let b = Vec3HE::from_f64(1.0, 1.0, 1.0);
        let c = a.sub(&b);
        assert_eq!(c.x.sign(), 1); // 3*1 - 1*1 = 2
        assert_eq!(c.y.sign(), 1); // 4*1 - 1*1 = 3
        assert_eq!(c.z.sign(), 1); // 5*1 - 1*1 = 4
        assert_eq!(c.w.sign(), 1); // 1*1 = 1
    }

    #[test]
    fn orient_2d_ccw() {
        let p0 = Vec2HE::from_f64(0.0, 0.0);
        let p1 = Vec2HE::from_f64(1.0, 0.0);
        let p2 = Vec2HE::from_f64(0.0, 1.0);
        assert_eq!(orient_2d(&p0, &p1, &p2), Sign::Positive);
    }

    #[test]
    fn orient_2d_cw() {
        let p0 = Vec2HE::from_f64(0.0, 0.0);
        let p1 = Vec2HE::from_f64(0.0, 1.0);
        let p2 = Vec2HE::from_f64(1.0, 0.0);
        assert_eq!(orient_2d(&p0, &p1, &p2), Sign::Negative);
    }

    #[test]
    fn orient_2d_collinear() {
        let p0 = Vec2HE::from_f64(0.0, 0.0);
        let p1 = Vec2HE::from_f64(1.0, 1.0);
        let p2 = Vec2HE::from_f64(2.0, 2.0);
        assert_eq!(orient_2d(&p0, &p1, &p2), Sign::Zero);
    }

    #[test]
    fn orient_3d_positive_tet() {
        // Geogram reference: orient_3d_positive_tet=1
        let p0 = Vec3HE::from_f64(0.0, 0.0, 0.0);
        let p1 = Vec3HE::from_f64(1.0, 0.0, 0.0);
        let p2 = Vec3HE::from_f64(0.0, 1.0, 0.0);
        let p3 = Vec3HE::from_f64(0.0, 0.0, 1.0);
        assert_eq!(orient_3d(&p0, &p1, &p2, &p3), Sign::Positive);
    }

    #[test]
    fn orient_3d_coplanar() {
        let p0 = Vec3HE::from_f64(0.0, 0.0, 0.0);
        let p1 = Vec3HE::from_f64(1.0, 0.0, 0.0);
        let p2 = Vec3HE::from_f64(0.0, 1.0, 0.0);
        let p3 = Vec3HE::from_f64(1.0, 1.0, 0.0);
        assert_eq!(orient_3d(&p0, &p1, &p2, &p3), Sign::Zero);
    }

    #[test]
    fn orient_3d_matches_geogram_reference() {
        // Geogram reference: orient_3d_near_degen=1 (Positive)
        // for points (0,0,0),(1,0,0),(0,1,0),(0,0,1e-15).
        let p0 = Vec3HE::from_f64(0.0, 0.0, 0.0);
        let p1 = Vec3HE::from_f64(1.0, 0.0, 0.0);
        let p2 = Vec3HE::from_f64(0.0, 1.0, 0.0);
        let p3 = Vec3HE::from_f64(0.0, 0.0, 1e-15);
        assert_eq!(orient_3d(&p0, &p1, &p2, &p3), Sign::Positive);
    }

    #[test]
    fn orient_2d_projected_along_z() {
        let p0 = Vec3HE::from_f64(0.0, 0.0, 0.0);
        let p1 = Vec3HE::from_f64(1.0, 0.0, 0.0);
        let p2 = Vec3HE::from_f64(0.0, 1.0, 0.0);
        assert_eq!(orient_2d_projected(&p0, &p1, &p2, 2), Sign::Positive);
    }

    #[test]
    fn normal_axis_of_xy_triangle() {
        assert_eq!(
            triangle_normal_axis([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            2
        );
    }

    #[test]
    fn normal_axis_of_xz_triangle() {
        assert_eq!(
            triangle_normal_axis([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            1
        );
    }

    #[test]
    fn aligned_3d_collinear_points() {
        let p0 = Vec3HE::from_f64(0.0, 0.0, 0.0);
        let p1 = Vec3HE::from_f64(1.0, 1.0, 1.0);
        let p2 = Vec3HE::from_f64(2.0, 2.0, 2.0);
        assert!(aligned_3d(&p0, &p1, &p2));
    }

    #[test]
    fn aligned_3d_non_collinear_points() {
        let p0 = Vec3HE::from_f64(0.0, 0.0, 0.0);
        let p1 = Vec3HE::from_f64(1.0, 0.0, 0.0);
        let p2 = Vec3HE::from_f64(0.0, 1.0, 0.0);
        assert!(!aligned_3d(&p0, &p1, &p2));
    }

    #[test]
    fn incircle_2d_sos_inside() {
        // Point clearly inside the circumcircle.
        let p0 = Vec2HE::from_f64(0.0, 0.0);
        let p1 = Vec2HE::from_f64(1.0, 0.0);
        let p2 = Vec2HE::from_f64(0.0, 1.0);
        let p3 = Vec2HE::from_f64(0.25, 0.25); // inside the circumcircle
        let s = incircle_2d_sos(&p0, &p1, &p2, &p3);
        assert_eq!(
            s,
            Sign::Positive,
            "point inside circumcircle should be Positive"
        );
    }

    #[test]
    fn incircle_2d_sos_outside() {
        let p0 = Vec2HE::from_f64(0.0, 0.0);
        let p1 = Vec2HE::from_f64(1.0, 0.0);
        let p2 = Vec2HE::from_f64(0.0, 1.0);
        let p3 = Vec2HE::from_f64(10.0, 10.0); // far outside
        let s = incircle_2d_sos(&p0, &p1, &p2, &p3);
        assert_eq!(
            s,
            Sign::Negative,
            "point outside circumcircle should be Negative"
        );
    }

    // ── SOS co-circular tests ────────────────────────────────────
    // These exercise the SOS_LEXICO perturbation code path (the exact
    // determinant is zero, so the symbolic perturbation must break the tie).

    #[test]
    fn incircle_2d_sos_cocircular_on_circumcircle() {
        // Four points on the unit circle: the exact incircle determinant is zero.
        // SOS must return a definite sign (never Zero).
        let p0 = Vec2HE::from_f64(1.0, 0.0);
        let p1 = Vec2HE::from_f64(0.0, 1.0);
        let p2 = Vec2HE::from_f64(-1.0, 0.0);
        let p3 = Vec2HE::from_f64(0.0, -1.0);
        let s = incircle_2d_sos(&p0, &p1, &p2, &p3);
        assert_ne!(
            s,
            Sign::Zero,
            "SOS should never return Zero for co-circular points"
        );
    }

    #[test]
    fn incircle_2d_sos_cocircular_consistent() {
        // SOS must be consistent: swapping the query point (p3) with
        // a triangle vertex changes the sign in a predictable way.
        // Specifically, incircle(p0,p1,p2,p3) and incircle(p0,p1,p3,p2)
        // should have opposite signs (swapping p2,p3 negates the determinant,
        // and SOS preserves this antisymmetry).
        let p0 = Vec2HE::from_f64(1.0, 0.0);
        let p1 = Vec2HE::from_f64(0.0, 1.0);
        let p2 = Vec2HE::from_f64(-1.0, 0.0);
        let p3 = Vec2HE::from_f64(0.0, -1.0);
        let s1 = incircle_2d_sos(&p0, &p1, &p2, &p3);
        let s2 = incircle_2d_sos(&p0, &p1, &p3, &p2);
        assert_eq!(
            s1 as i32,
            -(s2 as i32),
            "swapping p2<->p3 should negate the SOS result: got {s1:?} and {s2:?}"
        );
    }

    #[test]
    fn incircle_2d_sos_cocircular_square_corners() {
        // Four corners of a unit square: all lie on a circle of radius sqrt(2)/2.
        // The exact determinant is zero. SOS must break the tie.
        let p0 = Vec2HE::from_f64(0.0, 0.0);
        let p1 = Vec2HE::from_f64(1.0, 0.0);
        let p2 = Vec2HE::from_f64(1.0, 1.0);
        let p3 = Vec2HE::from_f64(0.0, 1.0);
        let s = incircle_2d_sos(&p0, &p1, &p2, &p3);
        assert_ne!(s, Sign::Zero, "SOS should never return Zero");
    }

    #[test]
    fn incircle_2d_sos_with_lengths_cocircular() {
        // Same co-circular test using the with_lengths variant directly.
        let p0 = Vec2HE::from_f64(1.0, 0.0);
        let p1 = Vec2HE::from_f64(0.0, 1.0);
        let p2 = Vec2HE::from_f64(-1.0, 0.0);
        let p3 = Vec2HE::from_f64(0.0, -1.0);
        // All points on unit circle: l = x^2 + y^2 = 1.0
        let s = incircle_2d_sos_with_lengths(&p0, &p1, &p2, &p3, 1.0, 1.0, 1.0, 1.0);
        assert_ne!(s, Sign::Zero, "SOS should never return Zero");
        // Should match the non-with-lengths version.
        let s2 = incircle_2d_sos(&p0, &p1, &p2, &p3);
        assert_eq!(s, s2, "with_lengths and without should agree");
    }

    /// Geogram reference: incircle results for the collinear CDT configuration.
    /// Points: 0=(-1,-1) 1=(4,-1) 2=(4,4) 3=(-1,4) 4=(0,0) 5=(1,1) 6=(2,2)
    #[test]
    fn incircle_collinear_reference() {
        let p0 = Vec2HE::from_f64(-1.0, -1.0);
        let p1 = Vec2HE::from_f64(4.0, -1.0);
        let p2 = Vec2HE::from_f64(4.0, 4.0);
        let p3 = Vec2HE::from_f64(-1.0, 4.0);
        let p4 = Vec2HE::from_f64(0.0, 0.0);
        let p5 = Vec2HE::from_f64(1.0, 1.0);
        let p6 = Vec2HE::from_f64(2.0, 2.0);

        // Geogram C++ reference values:
        assert_eq!(
            incircle_2d_sos(&p5, &p4, &p1, &p6),
            Sign::Negative,
            "incircle(5,4,1,6)"
        );
        assert_eq!(
            incircle_2d_sos(&p5, &p1, &p2, &p6),
            Sign::Positive,
            "incircle(5,1,2,6)"
        );
        assert_eq!(
            incircle_2d_sos(&p5, &p2, &p3, &p6),
            Sign::Positive,
            "incircle(5,2,3,6)"
        );
        assert_eq!(
            incircle_2d_sos(&p5, &p3, &p4, &p6),
            Sign::Negative,
            "incircle(5,3,4,6)"
        );
        assert_eq!(
            incircle_2d_sos(&p4, &p3, &p0, &p6),
            Sign::Negative,
            "incircle(4,3,0,6)"
        );
        assert_eq!(
            incircle_2d_sos(&p4, &p0, &p1, &p6),
            Sign::Negative,
            "incircle(4,0,1,6)"
        );
    }
}

/// Exact orient3d on plain f64 coordinates.
///
/// Returns a positive value if d is above the plane (a, b, c),
/// negative if below, zero if coplanar. Uses Shewchuk's convention:
/// det3x3(a-d, b-d, c-d).
///
/// This replaces the `geometry_predicates::orient3d` dependency.
pub fn orient3d_f64(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> f64 {
    use super::expansion::{Expansion, expansion_det4x4};

    let ea = [
        Expansion::from_f64(a[0]),
        Expansion::from_f64(a[1]),
        Expansion::from_f64(a[2]),
    ];
    let eb = [
        Expansion::from_f64(b[0]),
        Expansion::from_f64(b[1]),
        Expansion::from_f64(b[2]),
    ];
    let ec = [
        Expansion::from_f64(c[0]),
        Expansion::from_f64(c[1]),
        Expansion::from_f64(c[2]),
    ];
    let ed = [
        Expansion::from_f64(d[0]),
        Expansion::from_f64(d[1]),
        Expansion::from_f64(d[2]),
    ];

    let det = expansion_det4x4(&ea, &eb, &ec, &ed);
    let s = det.sign();
    // Negate to match Shewchuk/geometry_predicates convention:
    // positive when d is above the plane oriented by (a, b, c).
    -(s as f64)
}
