#[cfg(test)]
use super::*;

// Helper function for floating point comparisons
fn approx_eq(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-6
}

// ============ Vector4 Tests ============

#[test]
fn test_vector4_construction() {
    let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
    assert_eq!(v.x, 1.0);
    assert_eq!(v.y, 2.0);
    assert_eq!(v.z, 3.0);
    assert_eq!(v.w, 4.0);
}

#[test]
fn test_vector4_from_vec3() {
    let v3 = Vec::new(1.0, 2.0, 3.0);
    let v4 = Vec4::from_vec3(v3, 4.0);
    assert_eq!(v4, Vec4::new(1.0, 2.0, 3.0, 4.0));
}

#[test]
fn test_vector4_constants() {
    assert_eq!(Vec4::ZERO, Vec4::new(0.0, 0.0, 0.0, 0.0));
    assert_eq!(Vec4::ONE, Vec4::new(1.0, 1.0, 1.0, 1.0));
    assert_eq!(Vec4::X, Vec4::new(1.0, 0.0, 0.0, 0.0));
    assert_eq!(Vec4::Y, Vec4::new(0.0, 1.0, 0.0, 0.0));
    assert_eq!(Vec4::Z, Vec4::new(0.0, 0.0, 1.0, 0.0));
    assert_eq!(Vec4::W, Vec4::new(0.0, 0.0, 0.0, 1.0));
}

#[test]
fn test_vector4_from_tuple() {
    let v: Vec4 = (1.0, 2.0, 3.0, 4.0).into();
    assert_eq!(v, Vec4::new(1.0, 2.0, 3.0, 4.0));
}

#[test]
fn test_vector4_from_array() {
    let v: Vec4 = [1.0, 2.0, 3.0, 4.0].into();
    assert_eq!(v, Vec4::new(1.0, 2.0, 3.0, 4.0));
}

#[test]
fn test_vector4_from_vector3() {
    let v3 = Vec::new(1.0, 2.0, 3.0);
    let v4: Vec4 = v3.into();
    assert_eq!(v4, Vec4::new(1.0, 2.0, 3.0, 1.0));
}

#[test]
fn test_vector4_from_vector2() {
    let v2 = Vec2::new(1.0, 2.0);
    let v4: Vec4 = v2.into();
    assert_eq!(v4, Vec4::new(1.0, 2.0, 0.0, 1.0));
}

#[test]
fn test_vector4_to_tuple() {
    let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
    let t: (f32, f32, f32, f32) = v.into();
    assert_eq!(t, (1.0, 2.0, 3.0, 4.0));
}

#[test]
fn test_vector4_to_array() {
    let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
    let arr: [f32; 4] = v.into();
    assert_eq!(arr, [1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_vector4_to_vector3() {
    let v = Vec4::new(2.0, 4.0, 6.0, 2.0);
    let v3: Vec = v.into();
    assert_eq!(v3, Vec::new(1.0, 2.0, 3.0));

    let v = Vec4::new(2.0, 4.0, 6.0, 0.0);
    let v3: Vec = v.into();
    assert_eq!(v3, Vec::new(2.0, 4.0, 6.0));
}

#[test]
fn test_vector4_add() {
    let a = Vec4::new(1.0, 2.0, 3.0, 4.0);
    let b = Vec4::new(5.0, 6.0, 7.0, 8.0);
    assert_eq!(a + b, Vec4::new(6.0, 8.0, 10.0, 12.0));
    assert_eq!(a + 10.0, Vec4::new(11.0, 12.0, 13.0, 14.0));
    assert_eq!(10.0 + a, Vec4::new(11.0, 12.0, 13.0, 14.0));
}

#[test]
fn test_vector4_add_assign() {
    let mut v = Vec4::new(1.0, 2.0, 3.0, 4.0);
    v += Vec4::new(5.0, 6.0, 7.0, 8.0);
    assert_eq!(v, Vec4::new(6.0, 8.0, 10.0, 12.0));
}

#[test]
fn test_vector4_sub() {
    let a = Vec4::new(5.0, 6.0, 7.0, 8.0);
    let b = Vec4::new(1.0, 2.0, 3.0, 4.0);
    assert_eq!(a - b, Vec4::new(4.0, 4.0, 4.0, 4.0));
    assert_eq!(a - 2.0, Vec4::new(3.0, 4.0, 5.0, 6.0));
    assert_eq!(10.0 - a, Vec4::new(5.0, 4.0, 3.0, 2.0));
}

#[test]
fn test_vector4_sub_assign() {
    let mut v = Vec4::new(5.0, 6.0, 7.0, 8.0);
    v -= Vec4::new(1.0, 2.0, 3.0, 4.0);
    assert_eq!(v, Vec4::new(4.0, 4.0, 4.0, 4.0));
}

#[test]
fn test_vector4_mul() {
    let a = Vec4::new(2.0, 3.0, 4.0, 5.0);
    let b = Vec4::new(6.0, 7.0, 8.0, 9.0);
    assert_eq!(a * b, Vec4::new(12.0, 21.0, 32.0, 45.0));
    assert_eq!(a * 2.0, Vec4::new(4.0, 6.0, 8.0, 10.0));
    assert_eq!(2.0 * a, Vec4::new(4.0, 6.0, 8.0, 10.0));
}

#[test]
fn test_vector4_mul_assign() {
    let mut v = Vec4::new(2.0, 3.0, 4.0, 5.0);
    v *= Vec4::new(6.0, 7.0, 8.0, 9.0);
    assert_eq!(v, Vec4::new(12.0, 21.0, 32.0, 45.0));
}

#[test]
fn test_vector4_div() {
    let a = Vec4::new(12.0, 21.0, 32.0, 45.0);
    let b = Vec4::new(6.0, 7.0, 8.0, 9.0);
    assert_eq!(a / b, Vec4::new(2.0, 3.0, 4.0, 5.0));
    assert_eq!(a / 2.0, Vec4::new(6.0, 10.5, 16.0, 22.5));
    assert_eq!(
        60.0 / a,
        Vec4::new(5.0, 60.0 / 21.0, 60.0 / 32.0, 60.0 / 45.0)
    );
}

#[test]
fn test_vector4_div_assign() {
    let mut v = Vec4::new(12.0, 21.0, 32.0, 45.0);
    v /= Vec4::new(6.0, 7.0, 8.0, 9.0);
    assert_eq!(v, Vec4::new(2.0, 3.0, 4.0, 5.0));
}

#[test]
fn test_vector4_neg() {
    let v = Vec4::new(1.0, -2.0, 3.0, -4.0);
    assert_eq!(-v, Vec4::new(-1.0, 2.0, -3.0, 4.0));
}

#[test]
fn test_vector4_dot() {
    let a = Vec4::new(1.0, 2.0, 3.0, 4.0);
    let b = Vec4::new(5.0, 6.0, 7.0, 8.0);
    assert_eq!(a.dot(b), 70.0);
}

#[test]
fn test_vector4_magnitude() {
    let v = Vec4::new(1.0, 2.0, 2.0, 4.0);
    assert_eq!(v.magnitude(), 5.0);
    assert_eq!(v.magnitude_squared(), 25.0);
}

#[test]
fn test_vector4_normalized() {
    let v = Vec4::new(0.0, 3.0, 0.0, 4.0);
    let n = v.normalized();
    assert!(approx_eq(n.magnitude(), 1.0));
    assert_eq!(n, Vec4::new(0.0, 0.6, 0.0, 0.8));
}

#[test]
fn test_vector4_normalized_zero() {
    let v = Vec4::ZERO;
    let n = v.normalized();
    assert_eq!(n, Vec4::ZERO);
}

#[test]
fn test_vector4_is_zero() {
    assert!(Vec4::ZERO.is_zero());
    assert!(!Vec4::ONE.is_zero());
}

#[test]
fn test_vector4_is_normalised() {
    assert!(Vec4::X.is_normalised());
    assert!(Vec4::new(0.0, 0.6, 0.0, 0.8).is_normalised());
    assert!(!Vec4::new(1.0, 2.0, 2.0, 4.0).is_normalised());
}

#[test]
fn test_vector4_safe_normal() {
    let v = Vec4::new(3.0, 4.0, 0.0, 0.0);
    assert_eq!(v.safe_normal(), Some(v.normalized()));
    assert_eq!(Vec4::ZERO.safe_normal(), None);
}

#[test]
fn test_vector4_lerp() {
    let a = Vec4::new(0.0, 0.0, 0.0, 0.0);
    let b = Vec4::new(10.0, 20.0, 30.0, 40.0);
    assert_eq!(a.lerp(b, 0.0), a);
    assert_eq!(a.lerp(b, 1.0), b);
    assert_eq!(a.lerp(b, 0.5), Vec4::new(5.0, 10.0, 15.0, 20.0));
}

#[test]
fn test_vector4_distance() {
    let a = Vec4::new(1.0, 2.0, 3.0, 4.0);
    let b = Vec4::new(5.0, 6.0, 7.0, 8.0);
    assert_eq!(a.distance(b), 8.0);
    assert_eq!(a.distance_squared(b), 64.0);
}

#[test]
fn test_vector4_abs() {
    let v = Vec4::new(-1.0, 2.0, -3.0, 4.0);
    assert_eq!(v.abs(), Vec4::new(1.0, 2.0, 3.0, 4.0));
}

#[test]
fn test_vector4_min_max() {
    let a = Vec4::new(1.0, 5.0, 3.0, 7.0);
    let b = Vec4::new(4.0, 2.0, 6.0, 1.0);
    assert_eq!(a.min(b), Vec4::new(1.0, 2.0, 3.0, 1.0));
    assert_eq!(a.max(b), Vec4::new(4.0, 5.0, 6.0, 7.0));
}

#[test]
fn test_vector4_clamp() {
    let v = Vec4::new(-1.0, 5.0, 2.0, 8.0);
    let min = Vec4::new(0.0, 0.0, 0.0, 0.0);
    let max = Vec4::new(3.0, 3.0, 3.0, 3.0);
    assert_eq!(v.clamp(min, max), Vec4::new(0.0, 3.0, 2.0, 3.0));
}

#[test]
fn test_vector4_project_reject() {
    let v = Vec4::new(3.0, 4.0, 5.0, 6.0);
    let onto = Vec4::new(1.0, 0.0, 0.0, 0.0);
    let proj = v.project_onto(onto);
    let rej = v.reject_from(onto);
    assert_eq!(proj, Vec4::new(3.0, 0.0, 0.0, 0.0));
    assert_eq!(rej, Vec4::new(0.0, 4.0, 5.0, 6.0));
    assert!(approx_eq((proj + rej).x, v.x));
    assert!(approx_eq((proj + rej).y, v.y));
    assert!(approx_eq((proj + rej).z, v.z));
    assert!(approx_eq((proj + rej).w, v.w));
}

#[test]
fn test_vector4_reflect() {
    let v = Vec4::new(1.0, -1.0, 0.0, 0.0);
    let normal = Vec4::new(0.0, 1.0, 0.0, 0.0);
    let reflected = v.reflect(normal);
    assert_eq!(reflected, Vec4::new(1.0, 1.0, 0.0, 0.0));
}

#[test]
fn test_vector4_pow_sqrt() {
    let v = Vec4::new(4.0, 9.0, 16.0, 25.0);
    assert_eq!(v.pow(2.0), Vec4::new(16.0, 81.0, 256.0, 625.0));
    assert_eq!(v.sqrt(), Vec4::new(2.0, 3.0, 4.0, 5.0));
}

#[test]
fn test_vector4_default() {
    let v: Vec4 = Default::default();
    assert_eq!(v, Vec4::ZERO);
}

#[test]
fn test_vector4_indexing() {
    let mut v = Vec4::new(1.0, 2.0, 3.0, 4.0);

    // Read access
    assert_eq!(v[0], 1.0);
    assert_eq!(v[1], 2.0);
    assert_eq!(v[2], 3.0);
    assert_eq!(v[3], 4.0);

    // Write access
    v[0] = 5.0;
    v[1] = 6.0;
    v[2] = 7.0;
    v[3] = 8.0;
    assert_eq!(v, Vec4::new(5.0, 6.0, 7.0, 8.0));
}

#[test]
#[should_panic(expected = "Vector4 index 4 out of bounds")]
fn test_vector4_index_out_of_bounds() {
    let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
    let _ = v[4];
}
