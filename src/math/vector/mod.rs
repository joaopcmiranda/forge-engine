mod vector3;
mod vector2;
mod vector4;

// Re-export the main types for convenience
pub use vector3::Vector3 as Vector;
pub use vector2::Vector2 as Vector2;
pub use vector4::Vector4 as Vector4;


#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    // Helper function for floating point comparisons
    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-6
    }


    // ============ Cross-type Tests ============
    #[cfg(test)]
    mod cross_type_tests {
        use super::*;

        #[test]
        fn test_vector2_to_vector3_conversion() {
            let v2 = Vector2::new(3.0, 4.0);
            let v3 = v2.vec3();
            assert_eq!(v3, Vector::new(3.0, 4.0, 0.0));
        }

        #[test]
        fn test_vector3_to_vector2_conversion() {
            let v3 = Vector::new(3.0, 4.0, 5.0);
            let v2 = v3.vec2();
            assert_eq!(v2, Vector2::new(3.0, 4.0));
        }

        #[test]
        fn test_vector3_to_vector4_conversion() {
            let v3 = Vector::new(1.0, 2.0, 3.0);
            let v4: Vector4 = v3.into();
            assert_eq!(v4, Vector4::new(1.0, 2.0, 3.0, 1.0));
        }

        #[test]
        fn test_vector4_to_vector3_conversion() {
            let v4 = Vector4::new(4.0, 6.0, 8.0, 2.0);
            let v3 = v4.vec3();
            assert_eq!(v3, Vector::new(2.0, 3.0, 4.0));
        }
    }

    // ============ Edge Case Tests ============
    #[cfg(test)]
    mod edge_case_tests {
        use super::*;

        #[test]
        fn test_normalize_zero_vector() {
            let v2 = Vector2::ZERO;
            let v3 = Vector::ZERO;
            let v4 = Vector4::ZERO;

            // Should handle gracefully
            assert!(v2.normalized().is_zero());
            assert!(v3.normalized().is_zero());
            assert_eq!(v4.normalized(), Vector4::ZERO);
        }

        #[test]
        fn test_division_by_zero() {
            let v2 = Vector2::new(1.0, 2.0);
            let v3 = Vector::new(1.0, 2.0, 3.0);
            let v4 = Vector4::new(1.0, 2.0, 3.0, 4.0);

            let result2 = v2 / 0.0;
            let result3 = v3 / 0.0;
            let result4 = v4 / 0.0;

            assert!(result2.x.is_infinite());
            assert!(result3.x.is_infinite());
            assert!(result4.x.is_infinite());
        }

        #[test]
        fn test_very_small_vectors() {
            let tiny = Vector::new(1e-10, 1e-10, 1e-10);
            assert!(tiny.is_zero());
            assert!(tiny.safe_normal().is_none());
        }

        #[test]
        fn test_very_large_vectors() {
            let huge = Vector::new(1e10, 1e10, 1e10);
            let normalized = huge.normalized();
            assert!(approx_eq(normalized.magnitude(), 1.0));
        }

        #[test]
        fn test_perpendicular_to_zero() {
            let v = Vector2::ZERO;
            let perp = v.perpendicular();
            assert_eq!(perp, Vector2::ZERO);
        }

        #[test]
        fn test_angle_between_parallel_vectors() {
            let a = Vector::new(1.0, 0.0, 0.0);
            let b = Vector::new(2.0, 0.0, 0.0);
            assert!(approx_eq(a.angle_between(b), 0.0));

            let c = Vector::new(-1.0, 0.0, 0.0);
            assert!(approx_eq(a.angle_between(c), PI));
        }

        #[test]
        fn test_project_onto_zero_vector() {
            let v = Vector::new(3.0, 4.0, 5.0);
            let onto = Vector::ZERO;
            assert_eq!(v.project_onto(onto), Vector::ZERO);
        }

        #[test]
        fn test_reflect_with_non_unit_normal() {
            // Reflection formula expects normalized normal
            let v = Vector::new(1.0, -1.0, 0.0);
            let normal = Vector::new(0.0, 2.0, 0.0);
            let normalized_normal = normal.normalized();
            let reflected = v.reflect(normalized_normal);
            assert_eq!(reflected, Vector::new(1.0, 1.0, 0.0));

            // Or test that non-normalized normal gives expected result
            let reflected_non_unit = v.reflect(normal);
            // With normal of length 2, the reflection is amplified
            assert_eq!(reflected_non_unit, Vector::new(1.0, 7.0, 0.0));
        }

        #[test]
        fn test_nan_propagation() {
            let v = Vector::new(1.0, f32::NAN, 3.0);
            assert!(v.magnitude().is_nan());
            assert!(v.normalized().x.is_nan());
        }

        #[test]
        fn test_infinity_handling() {
            let v = Vector::new(f32::INFINITY, 1.0, 1.0);
            assert!(v.magnitude().is_infinite());

            let normalized = v.normalized();
            assert!(normalized.x.is_nan() || normalized.x == 1.0);
        }
    }

    // ============ Performance-Related Tests ============
    #[cfg(test)]
    mod performance_tests {
        use super::*;

        #[test]
        fn test_magnitude_squared_vs_magnitude() {
            let v = Vector::new(3.0, 4.0, 5.0);

            // magnitude_squared should give square of magnitude
            let mag = v.magnitude();
            let mag_sq = v.magnitude_squared();
            assert!(approx_eq(mag * mag, mag_sq));

            // Good for distance comparisons without sqrt
            let a = Vector::new(0.0, 0.0, 0.0);
            let b = Vector::new(3.0, 4.0, 0.0);
            assert_eq!(a.distance_squared(b), 25.0);
            assert_eq!(a.distance(b), 5.0);
        }

        #[test]
        fn test_operators_produce_same_results_as_functions() {
            let a = Vector::new(1.0, 2.0, 3.0);
            let b = Vector::new(4.0, 5.0, 6.0);

            // Addition
            assert_eq!(a + b, Vector::new(a.x + b.x, a.y + b.y, a.z + b.z));

            // Scalar multiplication
            assert_eq!(a * 2.0, Vector::new(a.x * 2.0, a.y * 2.0, a.z * 2.0));
        }
    }
}