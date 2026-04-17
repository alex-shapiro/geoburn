//! Property-based tests for the bspline module.
//!
//! These generate random B-spline curves and verify mathematical invariants
//! that must hold for any valid input, not just hand-picked test cases.

use geoburn::bspline::{basis, curves, degree, insert, knots};
use geoburn::gp::Pnt;
use nalgebra::Point3;
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Strategies for generating valid B-spline inputs
// ---------------------------------------------------------------------------

/// Generate a valid clamped B-spline: (degree, flat_knots, poles).
///
/// - degree in [1, 5]
/// - 1..4 internal knot values (each with multiplicity 1)
/// - poles with coordinates in [-10, 10]
fn arb_bspline() -> impl Strategy<Value = (usize, Vec<f64>, Vec<Pnt>)> {
    // degree 1..=5
    (1usize..=5).prop_flat_map(|degree| {
        // number of internal distinct knots: 0..=4
        (Just(degree), 0usize..=4).prop_flat_map(move |(degree, n_internal)| {
            // Generate sorted internal knot values in (0, 1)
            let internals =
                proptest::collection::vec(0.01f64..0.99, n_internal).prop_map(|mut v| {
                    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    // Ensure no duplicates (snap to grid)
                    v.dedup_by(|a, b| (*a - *b).abs() < 0.01);
                    v
                });

            (Just(degree), internals).prop_flat_map(move |(degree, internal_vals)| {
                // Build flat knot vector: [0]*order + internals + [1]*order
                let order = degree + 1;
                let n_poles = order + internal_vals.len();
                let mut flat = vec![0.0; order];
                flat.extend_from_slice(&internal_vals);
                flat.extend(vec![1.0; order]);

                // Generate poles
                let poles_strategy = proptest::collection::vec(
                    (-10.0f64..10.0, -10.0f64..10.0, -10.0f64..10.0),
                    n_poles,
                )
                .prop_map(|coords| {
                    coords
                        .into_iter()
                        .map(|(x, y, z)| Point3::new(x, y, z))
                        .collect::<Vec<Pnt>>()
                });

                (Just(degree), Just(flat), poles_strategy)
            })
        })
    })
}

/// Generate a valid clamped B-spline with internal knots that may have
/// multiplicity > 1 (creating reduced-continuity points).
///
/// - degree in [2, 5] (need degree >= 2 for mult > 1 to be meaningful)
/// - 1..3 internal knot values with multiplicity in [1, degree-1]
/// - poles with coordinates in [-10, 10]
fn arb_bspline_high_mult() -> impl Strategy<Value = (usize, Vec<f64>, Vec<Pnt>)> {
    (2usize..=5).prop_flat_map(|degree| {
        (Just(degree), 1usize..=3).prop_flat_map(move |(degree, n_internal)| {
            let internals =
                proptest::collection::vec(0.01f64..0.99, n_internal).prop_map(|mut v| {
                    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    v.dedup_by(|a, b| (*a - *b).abs() < 0.05);
                    v
                });

            (Just(degree), internals).prop_flat_map(move |(degree, internal_vals)| {
                let n_int = internal_vals.len();
                // Generate a multiplicity for each internal knot: 1..=degree-1
                let mults_strategy =
                    proptest::collection::vec(1usize..=degree.saturating_sub(1).max(1), n_int);

                (Just(degree), Just(internal_vals), mults_strategy).prop_flat_map(
                    move |(degree, internal_vals, mults)| {
                        let order = degree + 1;
                        let mut flat = vec![0.0; order];
                        let mut total_internal = 0usize;
                        for (val, &mult) in internal_vals.iter().zip(mults.iter()) {
                            for _ in 0..mult {
                                flat.push(*val);
                            }
                            total_internal += mult;
                        }
                        flat.extend(vec![1.0; order]);

                        let n_poles = flat.len() - degree - 1;

                        let poles_strategy = proptest::collection::vec(
                            (-10.0f64..10.0, -10.0f64..10.0, -10.0f64..10.0),
                            n_poles,
                        )
                        .prop_map(|coords| {
                            coords
                                .into_iter()
                                .map(|(x, y, z)| Point3::new(x, y, z))
                                .collect::<Vec<Pnt>>()
                        });

                        let _ = total_internal;
                        (Just(degree), Just(flat), poles_strategy)
                    },
                )
            })
        })
    })
}

/// Generate a valid B-spline with positive weights for rational testing.
fn arb_rational_bspline() -> impl Strategy<Value = (usize, Vec<f64>, Vec<Pnt>, Vec<f64>)> {
    arb_bspline().prop_flat_map(|(degree, kts, poles)| {
        let n = poles.len();
        let weights = proptest::collection::vec(0.1f64..10.0, n);
        (Just(degree), Just(kts), Just(poles), weights)
    })
}

// ---------------------------------------------------------------------------
// Basis function invariants
// ---------------------------------------------------------------------------

proptest! {
    #[test]
    fn basis_partition_of_unity((degree, kts, poles) in arb_bspline()) {
        let n = poles.len() - 1;
        // Test at 20 evenly-spaced parameters
        let u_min = kts[degree];
        let u_max = kts[kts.len() - degree - 1];
        for i in 0..=20 {
            let u = u_min + (u_max - u_min) * i as f64 / 20.0;
            let span = knots::find_span(n, degree, u, &kts);
            let basis_vals = basis::basis_funs(span, u, degree, &kts);
            let sum: f64 = basis_vals.iter().sum();
            prop_assert!(
                (sum - 1.0).abs() < 1e-12,
                "partition of unity failed: degree={degree}, u={u}, sum={sum}"
            );
        }
    }

    #[test]
    fn basis_non_negative((degree, kts, poles) in arb_bspline()) {
        let n = poles.len() - 1;
        let u_min = kts[degree];
        let u_max = kts[kts.len() - degree - 1];
        for i in 0..=20 {
            let u = u_min + (u_max - u_min) * i as f64 / 20.0;
            let span = knots::find_span(n, degree, u, &kts);
            let basis_vals = basis::basis_funs(span, u, degree, &kts);
            for (j, &val) in basis_vals.iter().enumerate() {
                prop_assert!(
                    val >= -1e-14,
                    "negative basis function: degree={degree}, u={u}, N[{j}]={val}"
                );
            }
        }
    }

    #[test]
    fn basis_derivative_sums_to_zero((degree, kts, poles) in arb_bspline()) {
        let n = poles.len() - 1;
        let u_min = kts[degree];
        let u_max = kts[kts.len() - degree - 1];
        let max_deriv = degree.min(3);
        for i in 0..=20 {
            let u = u_min + (u_max - u_min) * i as f64 / 20.0;
            let span = knots::find_span(n, degree, u, &kts);
            let ders = basis::ders_basis_funs(span, u, degree, max_deriv, &kts);
            for k in 1..=max_deriv {
                let dk_sum: f64 = ders[k].iter().sum();
                // Tolerance scales with derivative order: each order loses ~2 digits
                let tol = 1e-12_f64 * 100.0_f64.powi(k as i32);
                prop_assert!(
                    dk_sum.abs() < tol,
                    "d{k} sum != 0: degree={degree}, u={u}, sum={dk_sum}, tol={tol}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Knot insertion invariants
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn insert_knot_preserves_shape((degree, kts, poles) in arb_bspline()) {
        let u_min = kts[degree];
        let u_max = kts[kts.len() - degree - 1];
        // Insert at midpoint of the parameter range
        let u_insert = (u_min + u_max) / 2.0;

        // Check current multiplicity
        let s = kts.iter().filter(|&&v| v == u_insert).count();
        if s >= degree {
            return Ok(()); // already at full multiplicity, skip
        }

        let (new_kts, new_poles) = insert::insert_knot(degree, &kts, &poles, u_insert, 1);

        // Verify shape preservation at 20 sample points
        for i in 0..=20 {
            let u = u_min + (u_max - u_min) * i as f64 / 20.0;
            let p1: Pnt = curves::curve_point(degree, &kts, &poles, u);
            let p2: Pnt = curves::curve_point(degree, &new_kts, &new_poles, u);
            let dist = (p1 - p2).norm();
            prop_assert!(
                dist < 1e-10,
                "shape changed after knot insertion: u={u}, dist={dist}"
            );
        }
    }

    #[test]
    fn insert_knot_correct_sizes((degree, kts, poles) in arb_bspline()) {
        let u_min = kts[degree];
        let u_max = kts[kts.len() - degree - 1];
        let u_insert = (u_min + u_max) / 2.0;

        let s = kts.iter().filter(|&&v| v == u_insert).count();
        if s >= degree {
            return Ok(());
        }

        let (new_kts, new_poles) = insert::insert_knot(degree, &kts, &poles, u_insert, 1);

        prop_assert_eq!(new_kts.len(), kts.len() + 1);
        prop_assert_eq!(new_poles.len(), poles.len() + 1);
        // Fundamental B-spline identity: n_poles = n_knots - degree - 1
        prop_assert_eq!(new_poles.len(), new_kts.len() - degree - 1);
    }
}

// ---------------------------------------------------------------------------
// Bezier extraction invariants
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn bezier_extraction_preserves_shape((degree, kts, poles) in arb_bspline()) {
        let segments = insert::extract_bezier(degree, &kts, &poles);

        let u_min = kts[degree];
        let u_max = kts[kts.len() - degree - 1];

        // Each segment, evaluated on its span, should match the original
        for (seg_kts, seg_poles) in &segments {
            let seg_start = seg_kts[0];
            let seg_end = *seg_kts.last().unwrap();
            for j in 0..=10 {
                let u = seg_start + (seg_end - seg_start) * j as f64 / 10.0;
                // Clamp u to original range
                let u = u.max(u_min).min(u_max);
                let p_orig: Pnt = curves::curve_point(degree, &kts, &poles, u);
                let p_bez: Pnt = curves::curve_point(degree, seg_kts, seg_poles, u);
                let dist = (p_orig - p_bez).norm();
                prop_assert!(
                    dist < 1e-10,
                    "bezier extraction mismatch: u={u}, dist={dist}"
                );
            }
        }
    }

    #[test]
    fn bezier_segments_have_correct_size((degree, kts, poles) in arb_bspline()) {
        let segments = insert::extract_bezier(degree, &kts, &poles);
        let order = degree + 1;
        for (seg_kts, seg_poles) in &segments {
            prop_assert_eq!(seg_poles.len(), order);
            prop_assert_eq!(seg_kts.len(), 2 * order);
        }
    }
}

// ---------------------------------------------------------------------------
// Degree elevation invariants
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    #[test]
    fn elevate_preserves_shape((degree, kts, poles) in arb_bspline()) {
        let new_degree = degree + 1;
        let (new_kts, new_poles) = degree::elevate_degree(degree, new_degree, &kts, &poles);

        // Fundamental identity
        prop_assert_eq!(
            new_poles.len(), new_kts.len() - new_degree - 1,
            "pole/knot mismatch after elevation"
        );

        // Shape preservation
        let u_min = kts[degree];
        let u_max = kts[kts.len() - degree - 1];
        for i in 0..=20 {
            let u = u_min + (u_max - u_min) * i as f64 / 20.0;
            let p1: Pnt = curves::curve_point(degree, &kts, &poles, u);
            let p2: Pnt = curves::curve_point(new_degree, &new_kts, &new_poles, u);
            let dist = (p1 - p2).norm();
            prop_assert!(
                dist < 1e-9,
                "shape changed after degree elevation: degree {degree}→{new_degree}, u={u}, dist={dist}"
            );
        }
    }

    #[test]
    fn elevate_by_two_preserves_shape((degree, kts, poles) in arb_bspline()) {
        if degree > 4 { return Ok(()); } // new_degree would be 7+, keep it reasonable
        let new_degree = degree + 2;
        let (new_kts, new_poles) = degree::elevate_degree(degree, new_degree, &kts, &poles);

        prop_assert_eq!(new_poles.len(), new_kts.len() - new_degree - 1);

        let u_min = kts[degree];
        let u_max = kts[kts.len() - degree - 1];
        for i in 0..=20 {
            let u = u_min + (u_max - u_min) * i as f64 / 20.0;
            let p1: Pnt = curves::curve_point(degree, &kts, &poles, u);
            let p2: Pnt = curves::curve_point(new_degree, &new_kts, &new_poles, u);
            let dist = (p1 - p2).norm();
            prop_assert!(
                dist < 1e-9,
                "shape changed: degree {degree}→{new_degree}, u={u}, dist={dist}"
            );
        }
    }

    #[test]
    fn elevate_idempotent_shape((degree, kts, poles) in arb_bspline()) {
        // elevate(p→p+1) then elevate(p+1→p+2) should agree pointwise
        // with elevate(p→p+2), since both preserve the original shape.
        if degree > 4 { return Ok(()); }
        let (kts1, poles1) = degree::elevate_degree(degree, degree + 1, &kts, &poles);
        let (kts_12, poles_12) = degree::elevate_degree(degree + 1, degree + 2, &kts1, &poles1);
        let (kts_02, poles_02) = degree::elevate_degree(degree, degree + 2, &kts, &poles);

        let new_deg = degree + 2;
        let u_min = kts[degree];
        let u_max = kts[kts.len() - degree - 1];
        for i in 0..=20 {
            let u = u_min + (u_max - u_min) * i as f64 / 20.0;
            let p_12: Pnt = curves::curve_point(new_deg, &kts_12, &poles_12, u);
            let p_02: Pnt = curves::curve_point(new_deg, &kts_02, &poles_02, u);
            let dist = (p_12 - p_02).norm();
            prop_assert!(
                dist < 1e-8,
                "elevation not idempotent: degree {degree}→{new_deg}, u={u}, dist={dist}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Rational curve invariants
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    #[test]
    fn rational_unit_weights_matches_nonrational(
        (degree, kts, poles) in arb_bspline()
    ) {
        let weights = vec![1.0; poles.len()];
        let u_min = kts[degree];
        let u_max = kts[kts.len() - degree - 1];
        for i in 0..=20 {
            let u = u_min + (u_max - u_min) * i as f64 / 20.0;
            let p1: Pnt = curves::curve_point(degree, &kts, &poles, u);
            let p2: Pnt = curves::rational_curve_point(degree, &kts, &poles, &weights, u);
            let dist = (p1 - p2).norm();
            prop_assert!(
                dist < 1e-12,
                "rational with w=1 differs: u={u}, dist={dist}"
            );
        }
    }

    #[test]
    fn rational_insert_preserves_shape(
        (degree, kts, poles, weights) in arb_rational_bspline()
    ) {
        let u_min = kts[degree];
        let u_max = kts[kts.len() - degree - 1];
        let u_insert = (u_min + u_max) / 2.0;

        let s = kts.iter().filter(|&&v| v == u_insert).count();
        if s >= degree { return Ok(()); }

        let (new_kts, new_poles, new_weights) =
            insert::insert_knot_rational(degree, &kts, &poles, &weights, u_insert, 1);

        for i in 0..=20 {
            let u = u_min + (u_max - u_min) * i as f64 / 20.0;
            let p1: Pnt = curves::rational_curve_point(degree, &kts, &poles, &weights, u);
            let p2: Pnt = curves::rational_curve_point(degree, &new_kts, &new_poles, &new_weights, u);
            let dist = (p1 - p2).norm();
            prop_assert!(
                dist < 1e-9,
                "rational shape changed after insert: u={u}, dist={dist}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// High-multiplicity knot tests (internal knots with mult > 1)
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn high_mult_basis_partition_of_unity((degree, kts, poles) in arb_bspline_high_mult()) {
        let n = poles.len() - 1;
        let u_min = kts[degree];
        let u_max = kts[kts.len() - degree - 1];
        for i in 0..=20 {
            let u = u_min + (u_max - u_min) * i as f64 / 20.0;
            let span = knots::find_span(n, degree, u, &kts);
            let basis_vals = basis::basis_funs(span, u, degree, &kts);
            let sum: f64 = basis_vals.iter().sum();
            prop_assert!(
                (sum - 1.0).abs() < 1e-12,
                "partition of unity failed: degree={degree}, u={u}, sum={sum}, knots={kts:?}"
            );
        }
    }

    #[test]
    fn high_mult_bezier_extraction_preserves_shape((degree, kts, poles) in arb_bspline_high_mult()) {
        let segments = insert::extract_bezier(degree, &kts, &poles);

        let u_min = kts[degree];
        let u_max = kts[kts.len() - degree - 1];

        for (seg_kts, seg_poles) in &segments {
            let seg_start = seg_kts[0];
            let seg_end = *seg_kts.last().unwrap();
            for j in 0..=10 {
                let u = seg_start + (seg_end - seg_start) * j as f64 / 10.0;
                let u = u.max(u_min).min(u_max);
                let p_orig: Pnt = curves::curve_point(degree, &kts, &poles, u);
                let p_bez: Pnt = curves::curve_point(degree, seg_kts, seg_poles, u);
                let dist = (p_orig - p_bez).norm();
                prop_assert!(
                    dist < 1e-10,
                    "bezier mismatch with high mult knots: u={u}, dist={dist}"
                );
            }
        }
    }

    #[test]
    fn high_mult_elevate_preserves_shape((degree, kts, poles) in arb_bspline_high_mult()) {
        let new_degree = degree + 1;
        let (new_kts, new_poles) = degree::elevate_degree(degree, new_degree, &kts, &poles);

        prop_assert_eq!(
            new_poles.len(),
            new_kts.len() - new_degree - 1,
            "pole/knot mismatch after elevation with high mult knots"
        );

        let u_min = kts[degree];
        let u_max = kts[kts.len() - degree - 1];
        for i in 0..=20 {
            let u = u_min + (u_max - u_min) * i as f64 / 20.0;
            let p1: Pnt = curves::curve_point(degree, &kts, &poles, u);
            let p2: Pnt = curves::curve_point(new_degree, &new_kts, &new_poles, u);
            let dist = (p1 - p2).norm();
            prop_assert!(
                dist < 1e-9,
                "shape changed after elevation with high mult: degree {degree}→{new_degree}, u={u}, dist={dist}"
            );
        }
    }

    #[test]
    fn high_mult_insert_preserves_shape((degree, kts, poles) in arb_bspline_high_mult()) {
        let u_min = kts[degree];
        let u_max = kts[kts.len() - degree - 1];
        let u_insert = (u_min + u_max) / 2.0;

        let s = kts.iter().filter(|&&v| v == u_insert).count();
        if s >= degree { return Ok(()); }

        let (new_kts, new_poles) = insert::insert_knot(degree, &kts, &poles, u_insert, 1);

        for i in 0..=20 {
            let u = u_min + (u_max - u_min) * i as f64 / 20.0;
            let p1: Pnt = curves::curve_point(degree, &kts, &poles, u);
            let p2: Pnt = curves::curve_point(degree, &new_kts, &new_poles, u);
            let dist = (p1 - p2).norm();
            prop_assert!(
                dist < 1e-10,
                "shape changed after insert with high mult knots: u={u}, dist={dist}"
            );
        }
    }
}
