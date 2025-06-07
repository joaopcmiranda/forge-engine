#[cfg(test)]
use super::*;
use std::f32::consts::PI;

// Helper function for floating point comparisons
fn approx_eq(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-6
}

// ============ Vector Tests ============

#[test]
fn test_vector3_construction() {
    let v = Vec3::new(1.0, 2.0, 3.0);
    assert_eq!(v.x, 1.0);
    assert_eq!(v.y, 2.0);
    assert_eq!(v.z, 3.0);
}

#[test]
fn test_vector3_constants() {
    assert_eq!(Vec3::ZERO, Vec3::new(0.0, 0.0, 0.0));
    assert_eq!(Vec3::ONE, Vec3::new(1.0, 1.0, 1.0));
    assert_eq!(Vec3::UP, Vec3::new(0.0, 1.0, 0.0));
    assert_eq!(Vec3::DOWN, Vec3::new(0.0, -1.0, 0.0));
    assert_eq!(Vec3::RIGHT, Vec3::new(1.0, 0.0, 0.0));
    assert_eq!(Vec3::LEFT, Vec3::new(-1.0, 0.0, 0.0));
    assert_eq!(Vec3::FORWARD, Vec3::new(0.0, 0.0, 1.0));
    assert_eq!(Vec3::BACKWARD, Vec3::new(0.0, 0.0, -1.0));
}

#[test]
fn test_vector3_from_tuple() {
    let v: Vec3 = (1.0, 2.0, 3.0).into();
    assert_eq!(v, Vec3::new(1.0, 2.0, 3.0));
}

#[test]
fn test_vector3_add() {
    let a = Vec3::new(1.0, 2.0, 3.0);
    let b = Vec3::new(4.0, 5.0, 6.0);
    assert_eq!(a + b, Vec3::new(5.0, 7.0, 9.0));
    assert_eq!(a + 10.0, Vec3::new(11.0, 12.0, 13.0));
    assert_eq!(10.0 + a, Vec3::new(11.0, 12.0, 13.0));
}

#[test]
fn test_vector3_add_assign() {
    let mut v = Vec3::new(1.0, 2.0, 3.0);
    v += Vec3::new(4.0, 5.0, 6.0);
    assert_eq!(v, Vec3::new(5.0, 7.0, 9.0));
}

#[test]
fn test_vector3_sub() {
    let a = Vec3::new(5.0, 7.0, 9.0);
    let b = Vec3::new(1.0, 2.0, 3.0);
    assert_eq!(a - b, Vec3::new(4.0, 5.0, 6.0));
    assert_eq!(a - 2.0, Vec3::new(3.0, 5.0, 7.0));
    assert_eq!(10.0 - a, Vec3::new(5.0, 3.0, 1.0));
}

#[test]
fn test_vector3_sub_assign() {
    let mut v = Vec3::new(5.0, 7.0, 9.0);
    v -= Vec3::new(1.0, 2.0, 3.0);
    assert_eq!(v, Vec3::new(4.0, 5.0, 6.0));
}

#[test]
fn test_vector3_mul() {
    let a = Vec3::new(2.0, 3.0, 4.0);
    let b = Vec3::new(5.0, 6.0, 7.0);
    assert_eq!(a * b, Vec3::new(10.0, 18.0, 28.0));
    assert_eq!(a * 2.0, Vec3::new(4.0, 6.0, 8.0));
    assert_eq!(2.0 * a, Vec3::new(4.0, 6.0, 8.0));
}

#[test]
fn test_vector3_mul_assign() {
    let mut v = Vec3::new(2.0, 3.0, 4.0);
    v *= Vec3::new(5.0, 6.0, 7.0);
    assert_eq!(v, Vec3::new(10.0, 18.0, 28.0));
}

#[test]
fn test_vector3_div() {
    let a = Vec3::new(10.0, 18.0, 28.0);
    let b = Vec3::new(5.0, 6.0, 7.0);
    assert_eq!(a / b, Vec3::new(2.0, 3.0, 4.0));
    assert_eq!(a / 2.0, Vec3::new(5.0, 9.0, 14.0));
    assert_eq!(60.0 / a, Vec3::new(6.0, 60.0 / 18.0, 60.0 / 28.0));
}

#[test]
fn test_vector3_div_assign() {
    let mut v = Vec3::new(10.0, 18.0, 28.0);
    v /= Vec3::new(5.0, 6.0, 7.0);
    assert_eq!(v, Vec3::new(2.0, 3.0, 4.0));
}

#[test]
fn test_vector3_neg() {
    let v = Vec3::new(1.0, -2.0, 3.0);
    assert_eq!(-v, Vec3::new(-1.0, 2.0, -3.0));
}

#[test]
fn test_vector3_dot() {
    let a = Vec3::new(1.0, 2.0, 3.0);
    let b = Vec3::new(4.0, 5.0, 6.0);
    assert_eq!(a.dot(b), 32.0);
}

#[test]
fn test_vector3_equals_default_precision() {
    let a = Vec3::new(1.0000001, 2.0, 3.0);
    let b = Vec3::new(1.0000002, 2.0, 3.0);
    assert!(a.equals(b)); // Should be equal at default precision (6)
}

#[test]
fn test_vector3_equals_with_precision_high() {
    let a = Vec3::new(1.0000001, 2.0, 3.0);
    let b = Vec3::new(1.0000009, 2.0, 3.0);
    assert!(a.equals_with_precision(b, 5)); // Equal at 5 decimals
    assert!(!a.equals_with_precision(b, 6)); // Not equal at 6 decimals
    assert!(!a.equals_with_precision(b, 7)); // Not equal at 7 decimals
}

#[test]
fn test_vector3_equals_with_precision_low() {
    let a = Vec3::new(1.001, 2.0, 3.0);
    let b = Vec3::new(1.002, 2.0, 3.0);
    assert!(a.equals_with_precision(b, 2)); // Equal at 2 decimals
    assert!(!a.equals_with_precision(b, 3)); // Not equal at 3 decimals
}

#[test]
fn test_vector3_not_equals() {
    let a = Vec3::new(1.0, 2.0, 3.0);
    let b = Vec3::new(1.1, 2.0, 3.0);
    assert!(!a.equals(b));
    assert!(!a.equals_with_precision(b, 6));
}

#[test]
fn test_vector3_cross() {
    let a = Vec3::new(1.0, 0.0, 0.0);
    let b = Vec3::new(0.0, 1.0, 0.0);
    assert_eq!(a.cross(b), Vec3::new(0.0, 0.0, 1.0));

    let a = Vec3::new(2.0, 3.0, 4.0);
    let b = Vec3::new(5.0, 6.0, 7.0);
    assert_eq!(a.cross(b), Vec3::new(-3.0, 6.0, -3.0));
}

#[test]
fn test_vector3_magnitude() {
    let v = Vec3::new(2.0, 3.0, 6.0);
    assert_eq!(v.magnitude(), 7.0);
}

#[test]
fn test_vector3_normalized() {
    let v = Vec3::new(0.0, 3.0, 4.0);
    let n = v.normalized();
    assert!(approx_eq(n.magnitude(), 1.0));
    assert_eq!(n, Vec3::new(0.0, 0.6, 0.8));
}

#[test]
fn test_vector3_is_zero() {
    assert!(Vec3::ZERO.is_zero());
    assert!(!Vec3::ONE.is_zero());
}

#[test]
fn test_vector3_is_normalised() {
    assert!(Vec3::RIGHT.is_normalised());
    assert!(Vec3::new(0.0, 0.6, 0.8).is_normalised());
    assert!(!Vec3::new(2.0, 3.0, 6.0).is_normalised());
}

#[test]
fn test_vector3_safe_normal() {
    let v = Vec3::new(3.0, 4.0, 0.0);
    assert_eq!(v.safe_normal(), Some(v.normalized()));
    assert_eq!(Vec3::ZERO.safe_normal(), None);
}

#[test]
fn test_vector3_lerp() {
    let a = Vec3::new(0.0, 0.0, 0.0);
    let b = Vec3::new(10.0, 20.0, 30.0);
    assert_eq!(a.lerp(b, 0.0), a);
    assert_eq!(a.lerp(b, 1.0), b);
    assert_eq!(a.lerp(b, 0.5), Vec3::new(5.0, 10.0, 15.0));
}

#[test]
fn test_vector3_distance() {
    let a = Vec3::new(1.0, 2.0, 3.0);
    let b = Vec3::new(4.0, 6.0, 8.0);
    assert!(approx_eq(a.distance(b), 50.0_f32.sqrt()));
    assert_eq!(a.distance_squared(b), 50.0);
}

#[test]
fn test_vector3_abs() {
    let v = Vec3::new(-1.0, 2.0, -3.0);
    assert_eq!(v.abs(), Vec3::new(1.0, 2.0, 3.0));
}

#[test]
fn test_vector3_min_max() {
    let a = Vec3::new(1.0, 5.0, 3.0);
    let b = Vec3::new(4.0, 2.0, 6.0);
    assert_eq!(a.min(b), Vec3::new(1.0, 2.0, 3.0));
    assert_eq!(a.max(b), Vec3::new(4.0, 5.0, 6.0));
}

#[test]
fn test_vector3_clamp() {
    let v = Vec3::new(-1.0, 5.0, 2.0);
    let min = Vec3::new(0.0, 0.0, 0.0);
    let max = Vec3::new(3.0, 3.0, 3.0);
    assert_eq!(v.clamp(min, max), Vec3::new(0.0, 3.0, 2.0));
}

#[test]
fn test_vector3_angle_between() {
    let a = Vec3::new(1.0, 0.0, 0.0);
    let b = Vec3::new(0.0, 1.0, 0.0);
    assert!(approx_eq(a.angle_between(b), PI / 2.0));
}

#[test]
fn test_vector3_project_reject() {
    let v = Vec3::new(3.0, 4.0, 5.0);
    let onto = Vec3::new(1.0, 0.0, 0.0);
    let proj = v.project_onto(onto);
    let rej = v.reject_from(onto);
    assert_eq!(proj, Vec3::new(3.0, 0.0, 0.0));
    assert_eq!(rej, Vec3::new(0.0, 4.0, 5.0));
    assert!(approx_eq((proj + rej).x, v.x));
    assert!(approx_eq((proj + rej).y, v.y));
    assert!(approx_eq((proj + rej).z, v.z));
}

#[test]
fn test_vector3_reflect() {
    let v = Vec3::new(1.0, -1.0, 0.0);
    let normal = Vec3::new(0.0, 1.0, 0.0);
    let reflected = v.reflect(normal);
    assert_eq!(reflected, Vec3::new(1.0, 1.0, 0.0));
}

#[test]
fn test_vector3_pow_sqrt() {
    let v = Vec3::new(4.0, 9.0, 16.0);
    assert_eq!(v.pow(2.0), Vec3::new(16.0, 81.0, 256.0));
    assert_eq!(v.sqrt(), Vec3::new(2.0, 3.0, 4.0));
}

#[test]
fn test_vector3_swizzle() {
    let v = Vec3::new(1.0, 2.0, 3.0);
    assert_eq!(v.xy(), Vec2::new(1.0, 2.0));
    assert_eq!(v.yx(), Vec2::new(2.0, 1.0));
    assert_eq!(v.xz(), Vec2::new(1.0, 3.0));
    assert_eq!(v.zx(), Vec2::new(3.0, 1.0));
    assert_eq!(v.yz(), Vec2::new(2.0, 3.0));
    assert_eq!(v.zy(), Vec2::new(3.0, 2.0));
    assert_eq!(v.xx(), Vec2::new(1.0, 1.0));
    assert_eq!(v.yy(), Vec2::new(2.0, 2.0));
    assert_eq!(v.zz(), Vec2::new(3.0, 3.0));
}

#[test]
fn test_vector3_conversions() {
    let v3 = Vec3::new(3.0, 4.0, 5.0);
    let v2 = v3.vec2();
    assert_eq!(v2, Vec2::new(3.0, 4.0));
}

#[test]
fn test_vector3_default() {
    let v: Vec3 = Default::default();
    assert_eq!(v, Vec3::ZERO);
}

#[test]
fn test_vector3_indexing() {
    let mut v = Vec3::new(1.0, 2.0, 3.0);

    // Read access
    assert_eq!(v[0], 1.0);
    assert_eq!(v[1], 2.0);
    assert_eq!(v[2], 3.0);

    // Write access
    v[0] = 5.0;
    v[1] = 6.0;
    v[2] = 7.0;
    assert_eq!(v, Vec3::new(5.0, 6.0, 7.0));
}

#[test]
#[should_panic(expected = "Vector index 3 out of bounds")]
fn test_vector3_index_out_of_bounds() {
    let v = Vec3::new(1.0, 2.0, 3.0);
    let _ = v[3];
}

#[test]
fn test_indexing_in_loops() {
    let mut v3 = Vec3::new(1.0, 2.0, 3.0);

    // Modify all components
    for i in 0..3 {
        v3[i] *= 2.0;
    }

    assert_eq!(v3, Vec3::new(2.0, 4.0, 6.0));
}
