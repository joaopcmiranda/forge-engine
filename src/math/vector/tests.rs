#[cfg(test)]
use super::*;
use std::f32::consts::PI;
use crate::s;

// Helper function for floating point comparisons
fn approx_eq(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-6
}

// ============ Cross-type Tests ============

#[test]
fn test_vector2_to_vector3_conversion() {
    let v2 = Vec2::new(3.0, 4.0);
    let v3 = v2.vec3();
    assert_eq!(v3, Vec::new(3.0, 4.0, 0.0));
}

#[test]
fn test_vector3_to_vector2_conversion() {
    let v3 = Vec::new(3.0, 4.0, 5.0);
    let v2 = s!(v3, x y);
    assert_eq!(v2, Vec2::new(3.0, 4.0));
}

#[test]
fn test_vector3_to_vector4_conversion() {
    let v3 = Vec::new(1.0, 2.0, 3.0);
    let v4: Vec4 = v3.into();
    assert_eq!(v4, Vec4::new(1.0, 2.0, 3.0, 1.0));
}

#[test]
fn test_vector4_to_vector3_conversion() {
    let v4 = Vec4::new(4.0, 6.0, 8.0, 2.0);
    let v3 = s!(v4, x y z);
    assert_eq!(v3, Vec::new(4.0, 6.0, 8.0));
}

// ============ Edge Case Tests ============
#[test]
fn test_normalize_zero_vector() {
    let v2 = Vec2::ZERO;
    let v3 = Vec::ZERO;
    let v4 = Vec4::ZERO;

    // Should handle gracefully
    assert!(v2.normalized().is_zero());
    assert!(v3.normalized().is_zero());
    assert_eq!(v4.normalized(), Vec4::ZERO);
}

#[test]
fn test_division_by_zero() {
    let v2 = Vec2::new(1.0, 2.0);
    let v3 = Vec::new(1.0, 2.0, 3.0);
    let v4 = Vec4::new(1.0, 2.0, 3.0, 4.0);

    let result2 = v2 / 0.0;
    let result3 = v3 / 0.0;
    let result4 = v4 / 0.0;

    assert!(result2.x.is_infinite());
    assert!(result3.x.is_infinite());
    assert!(result4.x.is_infinite());
}

#[test]
fn test_very_small_vectors() {
    let tiny = Vec::new(1e-10, 1e-10, 1e-10);
    assert!(tiny.is_zero());
    assert!(tiny.safe_normal().is_none());
}

#[test]
fn test_very_large_vectors() {
    let huge = Vec::new(1e10, 1e10, 1e10);
    let normalized = huge.normalized();
    assert!(approx_eq(normalized.magnitude(), 1.0));
}

#[test]
fn test_perpendicular_to_zero() {
    let v = Vec2::ZERO;
    let perp = v.perpendicular();
    assert_eq!(perp, Vec2::ZERO);
}

#[test]
fn test_angle_between_parallel_vectors() {
    let a = Vec::new(1.0, 0.0, 0.0);
    let b = Vec::new(2.0, 0.0, 0.0);
    assert!(approx_eq(a.angle_between(b), 0.0));

    let c = Vec::new(-1.0, 0.0, 0.0);
    assert!(approx_eq(a.angle_between(c), PI));
}

#[test]
fn test_project_onto_zero_vector() {
    let v = Vec::new(3.0, 4.0, 5.0);
    let onto = Vec::ZERO;
    assert_eq!(v.project_onto(onto), Vec::ZERO);
}

#[test]
fn test_reflect_with_non_unit_normal() {
    // Reflection formula expects normalized normal
    let v = Vec::new(1.0, -1.0, 0.0);
    let normal = Vec::new(0.0, 2.0, 0.0);
    let normalized_normal = normal.normalized();
    let reflected = v.reflect(normalized_normal);
    assert_eq!(reflected, Vec::new(1.0, 1.0, 0.0));

    // Or test that non-normalized normal gives expected result
    let reflected_non_unit = v.reflect(normal);
    // With normal of length 2, the reflection is amplified
    assert_eq!(reflected_non_unit, Vec::new(1.0, 7.0, 0.0));
}

#[test]
fn test_nan_propagation() {
    let v = Vec::new(1.0, f32::NAN, 3.0);
    assert!(v.magnitude().is_nan());
    assert!(v.normalized().x.is_nan());
}

#[test]
fn test_infinity_handling() {
    let v = Vec::new(f32::INFINITY, 1.0, 1.0);
    assert!(v.magnitude().is_infinite());

    let normalized = v.normalized();
    assert!(normalized.x.is_nan() || normalized.x == 1.0);
}

// ============ Performance-Related Tests ============

#[test]
fn test_magnitude_squared_vs_magnitude() {
    let v = Vec::new(3.0, 4.0, 5.0);

    // magnitude_squared should give square of magnitude
    let mag = v.magnitude();
    let mag_sq = v.magnitude_squared();
    assert!(approx_eq(mag * mag, mag_sq));

    // Good for distance comparisons without sqrt
    let a = Vec::new(0.0, 0.0, 0.0);
    let b = Vec::new(3.0, 4.0, 0.0);
    assert_eq!(a.distance_squared(b), 25.0);
    assert_eq!(a.distance(b), 5.0);
}

#[test]
fn test_operators_produce_same_results_as_functions() {
    let a = Vec::new(1.0, 2.0, 3.0);
    let b = Vec::new(4.0, 5.0, 6.0);

    // Addition
    assert_eq!(a + b, Vec::new(a.x + b.x, a.y + b.y, a.z + b.z));

    // Scalar multiplication
    assert_eq!(a * 2.0, Vec::new(a.x * 2.0, a.y * 2.0, a.z * 2.0));
}
