//! Pure Rust port of Geogram's exact mesh boolean (Lévy 2024).
//!
//! Ported bottom-up: expansion arithmetic → exact predicates →
//! homogeneous vectors → CDT → triangle intersection → Weiler model → boolean.

pub mod boolean;
pub mod cdt;
pub mod exact_pred;
pub mod expansion;
pub mod mesh_in_triangle;
pub mod triangle_isect;
