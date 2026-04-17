//! Precision constants for geometric comparisons.
//! All values assume the working unit is millimeters.

/// Angular tolerance for comparing directions (radians). 1e-12.
pub const ANGULAR: f64 = 1e-12;

/// Tolerance for point coincidence in real space. 1e-7.
///
/// Two points closer than this are considered the same.
/// Also the threshold below which a vector is considered null.
pub const CONFUSION: f64 = 1e-7;

/// Square of `CONFUSION`, for avoiding sqrt in distance checks.
pub const SQUARE_CONFUSION: f64 = CONFUSION * CONFUSION;

/// Machine epsilon (`f64::EPSILON` ≈ 2.22e-16).
///
/// For low-level numerical checks (convergence, division-by-zero guards).
/// **Not** for geometric comparisons — use `CONFUSION` or `ANGULAR` instead.
pub const COMPUTATIONAL: f64 = f64::EPSILON;

/// Square of `COMPUTATIONAL`.
pub const SQUARE_COMPUTATIONAL: f64 = COMPUTATIONAL * COMPUTATIONAL;

/// Tolerance for intersection algorithms. `CONFUSION / 100` = 1e-9.
///
/// Tighter than confusion to force iterative solvers to converge
/// closer to true intersections, especially for tangent cases.
pub const INTERSECTION: f64 = CONFUSION * 0.01;

/// Tolerance for approximation algorithms. `CONFUSION * 10` = 1e-6.
///
/// Looser than confusion to keep approximation cost reasonable.
pub const APPROXIMATION: f64 = CONFUSION * 10.0;

/// Default parametric confusion: `CONFUSION / 100` = 1e-9.
///
/// Assumes a parametric tangent length of ~100 (i.e., parameter
/// change of 1 produces ~100 units of arc length).
pub const P_CONFUSION: f64 = CONFUSION * 0.01;

/// Square of `P_CONFUSION`.
pub const SQUARE_P_CONFUSION: f64 = P_CONFUSION * P_CONFUSION;

/// Default parametric intersection tolerance: `INTERSECTION / 100` = 1e-11.
pub const P_INTERSECTION: f64 = INTERSECTION * 0.01;

/// Default parametric approximation tolerance: `APPROXIMATION / 100` = 1e-8.
pub const P_APPROXIMATION: f64 = APPROXIMATION * 0.01;

/// A value large enough to be treated as infinite. 2e100.
pub const INFINITE: f64 = 2e100;

/// Convert a real-space precision to parametric space given a tangent magnitude.
pub fn parametric(precision: f64, tangent_magnitude: f64) -> f64 {
    precision / tangent_magnitude
}

/// Parametric confusion for a given tangent magnitude.
pub fn p_confusion(tangent_magnitude: f64) -> f64 {
    parametric(CONFUSION, tangent_magnitude)
}

/// Parametric intersection tolerance for a given tangent magnitude.
pub fn p_intersection(tangent_magnitude: f64) -> f64 {
    parametric(INTERSECTION, tangent_magnitude)
}

/// Parametric approximation tolerance for a given tangent magnitude.
pub fn p_approximation(tangent_magnitude: f64) -> f64 {
    parametric(APPROXIMATION, tangent_magnitude)
}

/// Returns true if the value is large enough to be considered infinite.
pub fn is_infinite(r: f64) -> bool {
    r.abs() >= 0.5 * INFINITE
}

/// Returns true if the value is a large positive number.
pub fn is_positive_infinite(r: f64) -> bool {
    r >= 0.5 * INFINITE
}

/// Returns true if the value is a large negative number.
pub fn is_negative_infinite(r: f64) -> bool {
    r <= -0.5 * INFINITE
}
