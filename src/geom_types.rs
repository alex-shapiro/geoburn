//! Classification enums for curves, surfaces, and geometric continuity.

/// Classification of curve types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CurveType {
    Line,
    Circle,
    Ellipse,
    Hyperbola,
    Parabola,
    BezierCurve,
    BSplineCurve,
    OffsetCurve,
    OtherCurve,
}

/// Classification of surface types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SurfaceType {
    Plane,
    Cylinder,
    Cone,
    Sphere,
    Torus,
    BezierSurface,
    BSplineSurface,
    SurfaceOfRevolution,
    SurfaceOfExtrusion,
    OffsetSurface,
    OtherSurface,
}

/// Geometric/parametric continuity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Continuity {
    /// Positional continuity only (C0).
    C0,
    /// Tangent continuity (G1): tangent vectors are collinear.
    G1,
    /// First derivative continuity.
    C1,
    /// Normal continuity (G2): curvature vectors agree in direction.
    G2,
    /// Second derivative continuity.
    C2,
    /// Third derivative continuity.
    C3,
    /// Infinite-order continuity.
    CN,
}

/// B-spline knot distribution classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KnotDistribution {
    /// Non-uniform knot spacing.
    NonUniform,
    /// Uniform knot spacing.
    Uniform,
    /// Quasi-uniform: multiplicity only at endpoints.
    QuasiUniform,
    /// Piecewise Bezier: internal knots have full multiplicity.
    PiecewiseBezier,
}

/// Join type for parallel curves / offset operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JoinType {
    /// Arc join between consecutive arcs.
    Arc,
    /// Tangent join.
    Tangent,
    /// Intersection join.
    Intersection,
}

/// Classification of isoparametric curves on a surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IsoType {
    /// U-isoparametric curve (constant U).
    IsoU,
    /// V-isoparametric curve (constant V).
    IsoV,
    /// Not an isoparametric curve.
    None,
}
