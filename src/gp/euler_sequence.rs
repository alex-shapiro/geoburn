/// All 24 variants of generalized Euler angles for 3D rotation.
///
/// Rotations are composed around three axes. The naming convention is:
/// - **Intrinsic**: rotations around axes of the rotating (body) frame
/// - **Extrinsic**: rotations around axes of the fixed (world) frame
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EulerSequence {
    /// Classic Euler angles (intrinsic ZXZ).
    EulerAngles,
    /// Yaw-pitch-roll / nautical angles (intrinsic ZYX).
    YawPitchRoll,

    // Tait-Bryan angles (three different axes)
    ExtrinsicXYZ,
    ExtrinsicXZY,
    ExtrinsicYZX,
    ExtrinsicYXZ,
    ExtrinsicZXY,
    ExtrinsicZYX,

    IntrinsicXYZ,
    IntrinsicXZY,
    IntrinsicYZX,
    IntrinsicYXZ,
    IntrinsicZXY,
    IntrinsicZYX,

    // Proper Euler angles (first and third axis the same)
    ExtrinsicXYX,
    ExtrinsicXZX,
    ExtrinsicYZY,
    ExtrinsicYXY,
    ExtrinsicZYZ,
    ExtrinsicZXZ,

    IntrinsicXYX,
    IntrinsicXZX,
    IntrinsicYZY,
    IntrinsicYXY,
    IntrinsicZXZ,
    IntrinsicZYZ,
}
