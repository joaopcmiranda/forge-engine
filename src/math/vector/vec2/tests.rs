use super::*;

#[cfg(test)]
use std::f32::consts::PI;

// Helper function for floating point comparisons
fn approx_eq(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-6
}

// ============ Vector2 Tests ============

#[test]
fn test_vector2_construction() {
    let v = Vec2::new(3.0, 4.0);
    assert_eq!(v.x, 3.0);
    assert_eq!(v.y, 4.0);
}

#[test]
fn test_vector2_from_angle() {
    let v = Vec2::from_angle(0.0);
    assert!(approx_eq(v.x, 1.0));
    assert!(approx_eq(v.y, 0.0));

    let v = Vec2::from_angle(PI / 2.0);
    assert!(approx_eq(v.x, 0.0));
    assert!(approx_eq(v.y, 1.0));
}

#[test]
fn test_vector2_constants() {
    assert_eq!(Vec2::ZERO, Vec2::new(0.0, 0.0));
    assert_eq!(Vec2::ONE, Vec2::new(1.0, 1.0));
    assert_eq!(Vec2::UP, Vec2::new(0.0, 1.0));
    assert_eq!(Vec2::DOWN, Vec2::new(0.0, -1.0));
    assert_eq!(Vec2::RIGHT, Vec2::new(1.0, 0.0));
    assert_eq!(Vec2::LEFT, Vec2::new(-1.0, 0.0));
}

#[test]
fn test_vector2_from_tuple() {
    let v: Vec2 = (3.0, 4.0).into();
    assert_eq!(v, Vec2::new(3.0, 4.0));
}

#[test]
fn test_vector2_add() {
    let a = Vec2::new(1.0, 2.0);
    let b = Vec2::new(3.0, 4.0);
    assert_eq!(a + b, Vec2::new(4.0, 6.0));
    assert_eq!(a + 5.0, Vec2::new(6.0, 7.0));
    assert_eq!(5.0 + a, Vec2::new(6.0, 7.0));
}

#[test]
fn test_vector2_add_assign() {
    let mut v = Vec2::new(1.0, 2.0);
    v += Vec2::new(3.0, 4.0);
    assert_eq!(v, Vec2::new(4.0, 6.0));
}

#[test]
fn test_vector2_sub() {
    let a = Vec2::new(5.0, 7.0);
    let b = Vec2::new(2.0, 3.0);
    assert_eq!(a - b, Vec2::new(3.0, 4.0));
    assert_eq!(a - 2.0, Vec2::new(3.0, 5.0));
    assert_eq!(10.0 - a, Vec2::new(5.0, 3.0));
}

#[test]
fn test_vector2_sub_assign() {
    let mut v = Vec2::new(5.0, 7.0);
    v -= Vec2::new(2.0, 3.0);
    assert_eq!(v, Vec2::new(3.0, 4.0));
}

#[test]
fn test_vector2_mul() {
    let a = Vec2::new(2.0, 3.0);
    let b = Vec2::new(4.0, 5.0);
    assert_eq!(a * b, Vec2::new(8.0, 15.0));
    assert_eq!(a * 3.0, Vec2::new(6.0, 9.0));
    assert_eq!(3.0 * a, Vec2::new(6.0, 9.0));
}

#[test]
fn test_vector2_mul_assign() {
    let mut v = Vec2::new(2.0, 3.0);
    v *= Vec2::new(4.0, 5.0);
    assert_eq!(v, Vec2::new(8.0, 15.0));
}

#[test]
fn test_vector2_div() {
    let a = Vec2::new(8.0, 15.0);
    let b = Vec2::new(4.0, 5.0);
    assert_eq!(a / b, Vec2::new(2.0, 3.0));
    assert_eq!(a / 2.0, Vec2::new(4.0, 7.5));
    assert_eq!(24.0 / a, Vec2::new(3.0, 1.6));
}

#[test]
fn test_vector2_div_assign() {
    let mut v = Vec2::new(8.0, 15.0);
    v /= Vec2::new(4.0, 5.0);
    assert_eq!(v, Vec2::new(2.0, 3.0));
}

#[test]
fn test_vector2_neg() {
    let v = Vec2::new(3.0, -4.0);
    assert_eq!(-v, Vec2::new(-3.0, 4.0));
}

#[test]
fn test_vector2_dot() {
    let a = Vec2::new(3.0, 4.0);
    let b = Vec2::new(2.0, 1.0);
    assert_eq!(a.dot(b), 10.0);
}

#[test]
fn test_vector2_cross() {
    let a = Vec2::new(3.0, 4.0);
    let b = Vec2::new(2.0, 1.0);
    assert_eq!(a.cross(b), -5.0);
}

#[test]
fn test_vector2_magnitude() {
    let v = Vec2::new(3.0, 4.0);
    assert_eq!(v.magnitude(), 5.0);
    assert_eq!(v.magnitude_squared(), 25.0);
}

#[test]
fn test_vector2_normalized() {
    let v = Vec2::new(3.0, 4.0);
    let n = v.normalized();
    assert!(approx_eq(n.magnitude(), 1.0));
    assert!(approx_eq(n.x, 0.6));
    assert!(approx_eq(n.y, 0.8));
}

#[test]
fn test_vector2_is_zero() {
    assert!(Vec2::ZERO.is_zero());
    assert!(!Vec2::ONE.is_zero());
}

#[test]
fn test_vector2_is_normalised() {
    assert!(Vec2::new(1.0, 0.0).is_normalised());
    assert!(Vec2::new(0.6, 0.8).is_normalised());
    assert!(!Vec2::new(3.0, 4.0).is_normalised());
}

#[test]
fn test_vector2_safe_normal() {
    let v = Vec2::new(3.0, 4.0);
    assert_eq!(v.safe_normal(), Some(v.normalized()));
    assert_eq!(Vec2::ZERO.safe_normal(), None);
}

#[test]
fn test_vector2_lerp() {
    let a = Vec2::new(0.0, 0.0);
    let b = Vec2::new(10.0, 20.0);
    assert_eq!(a.lerp(b, 0.0), a);
    assert_eq!(a.lerp(b, 1.0), b);
    assert_eq!(a.lerp(b, 0.5), Vec2::new(5.0, 10.0));
}

#[test]
fn test_vector2_distance() {
    let a = Vec2::new(1.0, 2.0);
    let b = Vec2::new(4.0, 6.0);
    assert_eq!(a.distance(b), 5.0);
    assert_eq!(a.distance_squared(b), 25.0);
}

#[test]
fn test_vector2_angle() {
    let v = Vec2::new(1.0, 0.0);
    assert!(approx_eq(v.angle(), 0.0));

    let v = Vec2::new(0.0, 1.0);
    assert!(approx_eq(v.angle(), PI / 2.0));
}

#[test]
fn test_vector2_angle_between() {
    let a = Vec2::new(1.0, 0.0);
    let b = Vec2::new(0.0, 1.0);
    assert!(approx_eq(a.angle_between(b), PI / 2.0));
}

#[test]
fn test_vector2_rotate() {
    let v = Vec2::new(1.0, 0.0);
    let rotated = v.rotate(PI / 2.0);
    assert!(approx_eq(rotated.x, 0.0));
    assert!(approx_eq(rotated.y, 1.0));
}

#[test]
fn test_vector2_perpendicular() {
    let v = Vec2::new(3.0, 4.0);
    let perp = v.perpendicular();
    assert!(approx_eq(v.dot(perp), 0.0));
    assert_eq!(perp, Vec2::new(-4.0, 3.0));
}

#[test]
fn test_vector2_abs() {
    let v = Vec2::new(-3.0, 4.0);
    assert_eq!(v.abs(), Vec2::new(3.0, 4.0));
}

#[test]
fn test_vector2_min_max() {
    let a = Vec2::new(1.0, 4.0);
    let b = Vec2::new(3.0, 2.0);
    assert_eq!(a.min(b), Vec2::new(1.0, 2.0));
    assert_eq!(a.max(b), Vec2::new(3.0, 4.0));
}

#[test]
fn test_vector2_clamp() {
    let v = Vec2::new(5.0, -2.0);
    assert_eq!(
        v.clamp(Vec2::new(0.0, 0.0), Vec2::new(3.0, 3.0)),
        Vec2::new(3.0, 0.0)
    );
}

#[test]
fn test_vector2_project_reject() {
    let v = Vec2::new(3.0, 4.0);
    let onto = Vec2::new(1.0, 0.0);
    let proj = v.project_onto(onto);
    let rej = v.reject_from(onto);
    assert_eq!(proj, Vec2::new(3.0, 0.0));
    assert_eq!(rej, Vec2::new(0.0, 4.0));
    assert!(approx_eq((proj + rej).x, v.x));
    assert!(approx_eq((proj + rej).y, v.y));
}

#[test]
fn test_vector2_reflect() {
    let v = Vec2::new(1.0, -1.0);
    let normal = Vec2::new(0.0, 1.0);
    let reflected = v.reflect(normal);
    assert_eq!(reflected, Vec2::new(1.0, 1.0));
}

#[test]
fn test_vector2_pow_sqrt() {
    let v = Vec2::new(4.0, 9.0);
    assert_eq!(v.pow(2.0), Vec2::new(16.0, 81.0));
    assert_eq!(v.sqrt(), Vec2::new(2.0, 3.0));
}

#[test]
fn test_vector2_conversions() {
    let v2 = Vec2::new(3.0, 4.0);
    let v3 = v2.vec3();
    assert_eq!(v3, Vec::new(3.0, 4.0, 0.0));
}

#[test]
fn test_vector2_default() {
    let v: Vec2 = Default::default();
    assert_eq!(v, Vec2::ZERO);
}

#[test]
fn test_vector2_indexing() {
    let mut v = Vec2::new(3.0, 4.0);

    // Read access
    assert_eq!(v[0], 3.0);
    assert_eq!(v[1], 4.0);

    // Write access
    v[0] = 5.0;
    v[1] = 6.0;
    assert_eq!(v, Vec2::new(5.0, 6.0));
}

#[test]
#[should_panic(expected = "Vector2 index 2 out of bounds")]
fn test_vector2_index_out_of_bounds() {
    let v = Vec2::new(1.0, 2.0);
    let _ = v[2];
}
