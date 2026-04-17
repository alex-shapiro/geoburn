//! Knot vector management for B-spline curves and surfaces.

use super::MAX_DEGREE;

/// Validate a flat knot vector and degree.
///
/// Checks:
/// - degree > 0 and degree <= MAX_DEGREE
/// - knot vector is long enough (>= 2 * (degree + 1))
/// - knot values are non-decreasing
///
/// Returns the number of control points (n + 1) or panics with a message.
pub fn validate_knots(degree: usize, knots: &[f64]) -> usize {
    assert!(degree > 0, "B-spline degree must be positive");
    assert!(
        degree <= MAX_DEGREE,
        "B-spline degree {degree} exceeds maximum {MAX_DEGREE}"
    );
    let min_knots = 2 * (degree + 1);
    assert!(
        knots.len() >= min_knots,
        "knot vector length {} too short for degree {degree} (need >= {min_knots})",
        knots.len()
    );
    for i in 1..knots.len() {
        assert!(
            knots[i] >= knots[i - 1],
            "knot vector not non-decreasing at index {i}: {} > {}",
            knots[i - 1],
            knots[i]
        );
    }
    knots.len() - degree - 1
}

/// Validate that poles array is consistent with knots and degree.
pub fn validate_poles(degree: usize, knots: &[f64], n_poles: usize) {
    let expected = knots.len() - degree - 1;
    assert!(
        n_poles == expected,
        "expected {expected} poles for degree {degree} with {} knots, got {n_poles}",
        knots.len()
    );
}

/// Find the knot span index for parameter `u`.
///
/// Returns the index `i` such that `knots[i] <= u < knots[i+1]`,
/// clamped to the valid range `[degree, n]`.
///
/// P&T Algorithm A2.1 (FindSpan).
///
/// `n` is the index of the last control point (knots.len() - degree - 2).
pub fn find_span(n: usize, degree: usize, u: f64, knots: &[f64]) -> usize {
    if u >= knots[n + 1] {
        return n;
    }
    if u <= knots[degree] {
        return degree;
    }

    let mut low = degree;
    let mut high = n + 1;
    let mut mid = (low + high) / 2;

    while u < knots[mid] || u >= knots[mid + 1] {
        if u < knots[mid] {
            high = mid;
        } else {
            low = mid;
        }
        mid = (low + high) / 2;
    }
    mid
}

/// Wrap a parameter value into the periodic range [knots[degree], knots[n+1]).
pub fn periodic_param(u: f64, degree: usize, knots: &[f64]) -> f64 {
    let n = knots.len() - degree - 2;
    let u_first = knots[degree];
    let u_last = knots[n + 1];
    let period = u_last - u_first;
    if period <= 0.0 {
        return u;
    }
    let mut result = u;
    while result < u_first {
        result += period;
    }
    while result >= u_last {
        result -= period;
    }
    result
}

/// Compute the multiplicity of a knot value in the flat knot vector.
pub fn knot_multiplicity(knot: f64, knots: &[f64], tolerance: f64) -> usize {
    knots
        .iter()
        .filter(|&&k| (k - knot).abs() <= tolerance)
        .count()
}

/// Generate a flat knot vector from distinct knots and multiplicities.
pub fn flat_knots(knots: &[f64], mults: &[usize]) -> Vec<f64> {
    assert_eq!(knots.len(), mults.len());
    let total: usize = mults.iter().sum();
    let mut flat = Vec::with_capacity(total);
    for (k, &m) in knots.iter().zip(mults.iter()) {
        for _ in 0..m {
            flat.push(*k);
        }
    }
    flat
}

/// Extract distinct knots and multiplicities from a flat knot vector.
pub fn distinct_knots(flat: &[f64], tolerance: f64) -> (Vec<f64>, Vec<usize>) {
    if flat.is_empty() {
        return (vec![], vec![]);
    }
    let mut knots = vec![flat[0]];
    let mut mults = vec![1usize];
    for &k in &flat[1..] {
        if (k - *knots.last().unwrap()).abs() <= tolerance {
            *mults.last_mut().unwrap() += 1;
        } else {
            knots.push(k);
            mults.push(1);
        }
    }
    (knots, mults)
}

/// Number of control points for given multiplicities and degree.
pub fn num_poles(degree: usize, mults: &[usize]) -> usize {
    let total: usize = mults.iter().sum();
    total - degree - 1
}

/// Reparametrize a knot vector to the range [u1, u2].
pub fn reparametrize(knots: &mut [f64], u1: f64, u2: f64) {
    assert!(!knots.is_empty());
    assert!(
        (u2 - u1).abs() > f64::EPSILON,
        "reparametrize: u1 and u2 must differ"
    );
    let k_first = knots[0];
    let k_last = *knots.last().unwrap();
    let scale = (u2 - u1) / (k_last - k_first);
    for k in knots.iter_mut() {
        *k = u1 + (*k - k_first) * scale;
    }
}

/// Reverse a knot vector in place.
pub fn reverse_knots(knots: &mut [f64]) {
    let last = *knots.last().unwrap();
    let first = knots[0];
    knots.reverse();
    for k in knots.iter_mut() {
        *k = first + last - *k;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cubic_knots() -> Vec<f64> {
        vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0]
    }

    #[test]
    fn validate_good_knots() {
        assert_eq!(validate_knots(3, &cubic_knots()), 7);
    }

    #[test]
    #[should_panic(expected = "non-decreasing")]
    fn validate_bad_order() {
        validate_knots(2, &[0.0, 0.0, 0.0, 1.0, 0.5, 2.0, 2.0, 2.0]);
    }

    #[test]
    #[should_panic(expected = "too short")]
    fn validate_too_short() {
        validate_knots(3, &[0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn find_span_interior() {
        let knots = cubic_knots();
        let n = knots.len() - 3 - 2;
        assert_eq!(find_span(n, 3, 0.5, &knots), 3);
        assert_eq!(find_span(n, 3, 1.5, &knots), 4);
        assert_eq!(find_span(n, 3, 2.5, &knots), 5);
        assert_eq!(find_span(n, 3, 3.5, &knots), 6);
    }

    #[test]
    fn find_span_at_knots() {
        let knots = cubic_knots();
        let n = knots.len() - 3 - 2;
        assert_eq!(find_span(n, 3, 0.0, &knots), 3);
        assert_eq!(find_span(n, 3, 1.0, &knots), 4);
        assert_eq!(find_span(n, 3, 4.0, &knots), n);
    }

    #[test]
    fn find_span_at_all_internal_knots() {
        let knots = cubic_knots();
        let n = knots.len() - 3 - 2;
        assert_eq!(find_span(n, 3, 1.0, &knots), 4);
        assert_eq!(find_span(n, 3, 2.0, &knots), 5);
        assert_eq!(find_span(n, 3, 3.0, &knots), 6);
    }

    #[test]
    fn flat_and_distinct_roundtrip() {
        let knots = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let mults = vec![4, 1, 1, 1, 4];
        let flat = flat_knots(&knots, &mults);
        assert_eq!(flat, cubic_knots());
        let (dk, dm) = distinct_knots(&flat, 1e-10);
        assert_eq!(dk, knots);
        assert_eq!(dm, mults);
    }

    #[test]
    fn num_poles_cubic() {
        assert_eq!(num_poles(3, &[4, 1, 1, 1, 4]), 7);
    }

    #[test]
    fn reparametrize_to_unit() {
        let mut knots = vec![0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0];
        reparametrize(&mut knots, 0.0, 1.0);
        assert!(knots[0].abs() < 1e-15);
        assert!((*knots.last().unwrap() - 1.0).abs() < 1e-15);
        assert!((knots[3] - 0.5).abs() < 1e-15);
    }

    #[test]
    fn reverse() {
        let mut knots = vec![0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0];
        reverse_knots(&mut knots);
        assert!(knots[0].abs() < 1e-15);
        assert!((knots[3] - 1.0).abs() < 1e-15);
        assert!((*knots.last().unwrap() - 2.0).abs() < 1e-15);
    }

    #[test]
    fn periodic_param_wraps() {
        let knots = vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0];
        assert!((periodic_param(5.5, 2, &knots) - 1.5).abs() < 1e-14);
        assert!((periodic_param(-0.5, 2, &knots) - 3.5).abs() < 1e-14);
    }
}
