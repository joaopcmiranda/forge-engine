use super::*;
use crate::math::{Vec, Vec4};
#[cfg(test)]
use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};

#[test]
fn test_new() {
    let mat = Mat4::new();
    assert_eq!(mat, Mat4::IDENTITY);
}

#[test]
fn test_from_rows() {
    let row1 = Vec4::new(1.0, 2.0, 3.0, 4.0);
    let row2 = Vec4::new(5.0, 6.0, 7.0, 8.0);
    let row3 = Vec4::new(9.0, 10.0, 11.0, 12.0);
    let row4 = Vec4::new(13.0, 14.0, 15.0, 16.0);

    let mat = Mat4::from_rows(row1, row2, row3, row4);

    // Verify by getting rows back
    let retrieved_rows = mat.rows();
    assert_eq!(retrieved_rows[0], row1);
    assert_eq!(retrieved_rows[1], row2);
    assert_eq!(retrieved_rows[2], row3);
    assert_eq!(retrieved_rows[3], row4);

    // Verify internal column-major storage
    assert_eq!(mat.e[0], Vec4::new(1.0, 5.0, 9.0, 13.0)); // col 0
    assert_eq!(mat.e[1], Vec4::new(2.0, 6.0, 10.0, 14.0)); // col 1
    assert_eq!(mat.e[2], Vec4::new(3.0, 7.0, 11.0, 15.0)); // col 2
    assert_eq!(mat.e[3], Vec4::new(4.0, 8.0, 12.0, 16.0)); // col 3
}

#[test]
fn test_from_cols() {
    let col1 = Vec4::new(1.0, 5.0, 9.0, 13.0);
    let col2 = Vec4::new(2.0, 6.0, 10.0, 14.0);
    let col3 = Vec4::new(3.0, 7.0, 11.0, 15.0);
    let col4 = Vec4::new(4.0, 8.0, 12.0, 16.0);

    let mat = Mat4::from_cols(col1, col2, col3, col4);

    // Verify by getting columns back
    let retrieved_cols = mat.cols();
    assert_eq!(retrieved_cols[0], col1);
    assert_eq!(retrieved_cols[1], col2);
    assert_eq!(retrieved_cols[2], col3);
    assert_eq!(retrieved_cols[3], col4);

    // Verify internal storage
    assert_eq!(mat.e[0], col1);
    assert_eq!(mat.e[1], col2);
    assert_eq!(mat.e[2], col3);
    assert_eq!(mat.e[3], col4);
}

#[test]
fn test_from_rows_from_cols_consistency() {
    // Create matrix using rows
    let row1 = Vec4::new(1.0, 2.0, 3.0, 4.0);
    let row2 = Vec4::new(5.0, 6.0, 7.0, 8.0);
    let row3 = Vec4::new(9.0, 10.0, 11.0, 15.0);
    let row4 = Vec4::new(13.0, 14.0, 15.0, 16.0);
    let mat_from_rows = Mat4::from_rows(row1, row2, row3, row4);

    // Create same matrix using columns (transposed)
    let col1 = Vec4::new(1.0, 5.0, 9.0, 13.0);
    let col2 = Vec4::new(2.0, 6.0, 10.0, 14.0);
    let col3 = Vec4::new(3.0, 7.0, 11.0, 15.0);
    let col4 = Vec4::new(4.0, 8.0, 15.0, 16.0);
    let mat_from_cols = Mat4::from_cols(col1, col2, col3, col4);

    assert_eq!(mat_from_rows, mat_from_cols);
}

#[test]
fn test_from_mul_identity() {
    let identity = Mat4::IDENTITY;
    let test_mat = Mat4::from_rows(
        Vec4::new(1.0, 2.0, 3.0, 4.0),
        Vec4::new(5.0, 6.0, 7.0, 8.0),
        Vec4::new(9.0, 10.0, 11.0, 12.0),
        Vec4::new(13.0, 14.0, 15.0, 16.0),
    );

    // I * M = M
    let result1 = Mat4::from_mul(identity, test_mat);
    assert_eq!(result1, test_mat);

    // M * I = M
    let result2 = Mat4::from_mul(test_mat, identity);
    assert_eq!(result2, test_mat);
}

#[test]
fn test_from_mul_known_result() {
    // Simple 2x2 case embedded in 4x4 for easy verification
    let mat_a = Mat4::from_rows(
        Vec4::new(1.0, 2.0, 0.0, 0.0),
        Vec4::new(3.0, 4.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 1.0, 0.0),
        Vec4::new(0.0, 0.0, 0.0, 1.0),
    );

    let mat_b = Mat4::from_rows(
        Vec4::new(5.0, 6.0, 0.0, 0.0),
        Vec4::new(7.0, 8.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 1.0, 0.0),
        Vec4::new(0.0, 0.0, 0.0, 1.0),
    );

    let result = Mat4::from_mul(mat_a, mat_b);

    // Expected result for 2x2 multiplication:
    // [1 2] * [5 6] = [19 22]
    // [3 4]   [7 8]   [43 50]
    let expected = Mat4::from_rows(
        Vec4::new(19.0, 22.0, 0.0, 0.0),
        Vec4::new(43.0, 50.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 1.0, 0.0),
        Vec4::new(0.0, 0.0, 0.0, 1.0),
    );

    assert_eq!(result, expected);
}

#[test]
fn test_from_mul_translation_composition() {
    let translate1 = Mat4::translation(Vec::new(1.0, 2.0, 3.0));
    let translate2 = Mat4::translation(Vec::new(4.0, 5.0, 6.0));

    // Composing translations should add them
    let result = Mat4::from_mul(translate2, translate1);
    let expected = Mat4::translation(Vec::new(5.0, 7.0, 9.0));

    // Check the translation column (column 3)
    let result_translation = result.col(3);
    let expected_translation = expected.col(3);

    assert!((result_translation.x - expected_translation.x).abs() < 1e-6);
    assert!((result_translation.y - expected_translation.y).abs() < 1e-6);
    assert!((result_translation.z - expected_translation.z).abs() < 1e-6);
    assert!((result_translation.w - expected_translation.w).abs() < 1e-6);
}

#[test]
fn test_from_mul_non_commutative() {
    let mat_a = Mat4::from_rows(
        Vec4::new(1.0, 2.0, 0.0, 0.0),
        Vec4::new(0.0, 1.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 1.0, 0.0),
        Vec4::new(0.0, 0.0, 0.0, 1.0),
    );

    let mat_b = Mat4::from_rows(
        Vec4::new(1.0, 0.0, 0.0, 0.0),
        Vec4::new(3.0, 1.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 1.0, 0.0),
        Vec4::new(0.0, 0.0, 0.0, 1.0),
    );

    let result_ab = Mat4::from_mul(mat_a, mat_b);
    let result_ba = Mat4::from_mul(mat_b, mat_a);

    // Matrix multiplication is not commutative
    assert_ne!(result_ab, result_ba);
}

#[test]
fn test_from_mul_associativity() {
    let mat_a = Mat4::translation(Vec::new(1.0, 0.0, 0.0));
    let mat_b = Mat4::scaling(Vec::new(2.0, 2.0, 2.0));
    let mat_c = Mat4::rotation_z(std::f32::consts::PI / 4.0);

    // (A * B) * C
    let ab = Mat4::from_mul(mat_a, mat_b);
    let ab_c = Mat4::from_mul(ab, mat_c);

    // A * (B * C)
    let bc = Mat4::from_mul(mat_b, mat_c);
    let a_bc = Mat4::from_mul(mat_a, bc);

    // Should be equal (within floating point precision)
    let tolerance = 1e-6;
    for i in 0..4 {
        for j in 0..4 {
            let diff = (ab_c.e[i][j] - a_bc.e[i][j]).abs();
            assert!(
                diff < tolerance,
                "Matrices differ at [{i}][{j}]: {} vs {}",
                ab_c.e[i][j],
                a_bc.e[i][j]
            );
        }
    }
}

const TOLERANCE: f32 = 1e-6;

fn assert_vec_approx_eq(a: Vec, b: Vec, tolerance: f32) {
    assert!((a.x - b.x).abs() < tolerance, "x: {} vs {}", a.x, b.x);
    assert!((a.y - b.y).abs() < tolerance, "y: {} vs {}", a.y, b.y);
    assert!((a.z - b.z).abs() < tolerance, "z: {} vs {}", a.z, b.z);
}

#[test]
fn test_translation_identity() {
    let mat = Mat4::translation(Vec::new(0.0, 0.0, 0.0));
    assert_eq!(mat, Mat4::IDENTITY);
}

#[test]
fn test_translation_basic() {
    let translation = Vec::new(1.0, 2.0, 3.0);
    let mat = Mat4::translation(translation);

    // Translation matrix should have translation in the last column
    assert_eq!(mat.col(3), Vec4::new(1.0, 2.0, 3.0, 1.0));

    // Other columns should be identity
    assert_eq!(mat.col(0), Vec4::new(1.0, 0.0, 0.0, 0.0));
    assert_eq!(mat.col(1), Vec4::new(0.0, 1.0, 0.0, 0.0));
    assert_eq!(mat.col(2), Vec4::new(0.0, 0.0, 1.0, 0.0));
}

#[test]
fn test_translation_transform_point() {
    let translation = Vec::new(5.0, -3.0, 2.0);
    let mat = Mat4::translation(translation);

    let point = Vec::new(1.0, 1.0, 1.0);
    let transformed = mat.transform_point(point);
    let expected = Vec::new(6.0, -2.0, 3.0);

    assert_vec_approx_eq(transformed, expected, TOLERANCE);
}

#[test]
fn test_translation_transform_direction() {
    let translation = Vec::new(5.0, -3.0, 2.0);
    let mat = Mat4::translation(translation);

    let direction = Vec::new(1.0, 0.0, 0.0);
    let transformed = mat.transform_direction(direction);

    // Directions should not be affected by translation
    assert_vec_approx_eq(transformed, direction, TOLERANCE);
}

#[test]
fn test_scaling_identity() {
    let mat = Mat4::scaling(Vec::new(1.0, 1.0, 1.0));
    assert_eq!(mat, Mat4::IDENTITY);
}

#[test]
fn test_scaling_basic() {
    let scale = Vec::new(2.0, 3.0, 4.0);
    let mat = Mat4::scaling(scale);

    // Scaling matrix should have scales on the diagonal
    assert_eq!(mat.col(0), Vec4::new(2.0, 0.0, 0.0, 0.0));
    assert_eq!(mat.col(1), Vec4::new(0.0, 3.0, 0.0, 0.0));
    assert_eq!(mat.col(2), Vec4::new(0.0, 0.0, 4.0, 0.0));
    assert_eq!(mat.col(3), Vec4::new(0.0, 0.0, 0.0, 1.0));
}

#[test]
fn test_scaling_transform_point() {
    let scale = Vec::new(2.0, 0.5, -1.0);
    let mat = Mat4::scaling(scale);

    let point = Vec::new(3.0, 4.0, 5.0);
    let transformed = mat.transform_point(point);
    let expected = Vec::new(6.0, 2.0, -5.0);

    assert_vec_approx_eq(transformed, expected, TOLERANCE);
}

#[test]
fn test_rotation_x_identity() {
    let mat = Mat4::rotation_x(0.0);
    assert_eq!(mat, Mat4::IDENTITY);
}

#[test]
fn test_rotation_x_90_degrees() {
    let mat = Mat4::rotation_x(FRAC_PI_2);

    // Test rotating (0, 1, 0) -> (0, 0, 1)
    let point = Vec::new(0.0, 1.0, 0.0);
    let transformed = mat.transform_point(point);
    let expected = Vec::new(0.0, 0.0, 1.0);

    assert_vec_approx_eq(transformed, expected, TOLERANCE);
}

#[test]
fn test_rotation_x_180_degrees() {
    let mat = Mat4::rotation_x(PI);

    // Test rotating (0, 1, 0) -> (0, -1, 0)
    let point = Vec::new(0.0, 1.0, 0.0);
    let transformed = mat.transform_point(point);
    let expected = Vec::new(0.0, -1.0, 0.0);

    assert_vec_approx_eq(transformed, expected, TOLERANCE);
}

#[test]
fn test_rotation_y_identity() {
    let mat = Mat4::rotation_y(0.0);
    assert_eq!(mat, Mat4::IDENTITY);
}

#[test]
fn test_rotation_y_90_degrees() {
    let mat = Mat4::rotation_y(FRAC_PI_2);

    // Test rotating (1, 0, 0) -> (0, 0, -1)
    let point = Vec::new(1.0, 0.0, 0.0);
    let transformed = mat.transform_point(point);
    let expected = Vec::new(0.0, 0.0, -1.0);

    assert_vec_approx_eq(transformed, expected, TOLERANCE);
}

#[test]
fn test_rotation_z_identity() {
    let mat = Mat4::rotation_z(0.0);
    assert_eq!(mat, Mat4::IDENTITY);
}

#[test]
fn test_rotation_z_90_degrees() {
    let mat = Mat4::rotation_z(FRAC_PI_2);

    // Test rotating (1, 0, 0) -> (0, 1, 0)
    let point = Vec::new(1.0, 0.0, 0.0);
    let transformed = mat.transform_point(point);
    let expected = Vec::new(0.0, 1.0, 0.0);

    assert_vec_approx_eq(transformed, expected, TOLERANCE);
}

#[test]
fn test_rotation_arbitrary_axis_identity() {
    let axis = Vec::new(1.0, 0.0, 0.0).normalized();
    let mat = Mat4::rotation(axis, 0.0);

    // Should be approximately identity
    for i in 0..4 {
        for j in 0..4 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!((mat.e[i][j] - expected).abs() < TOLERANCE);
        }
    }
}

#[test]
fn test_rotation_arbitrary_axis_90_degrees() {
    let axis = Vec::new(1.0, 0.0, 0.0).normalized();
    let mat = Mat4::rotation(axis, FRAC_PI_2);

    // Should behave like rotation_x
    let expected_mat = Mat4::rotation_x(FRAC_PI_2);

    for i in 0..4 {
        for j in 0..4 {
            assert!((mat.e[i][j] - expected_mat.e[i][j]).abs() < TOLERANCE);
        }
    }
}

#[test]
fn test_rotation_preserves_axis() {
    let axis = Vec::new(1.0, 2.0, 3.0).normalized();
    let mat = Mat4::rotation(axis, FRAC_PI_4);

    // Rotating around an axis should preserve the axis direction
    let transformed = mat.transform_direction(axis);
    assert_vec_approx_eq(transformed, axis, TOLERANCE);
}

#[test]
fn test_rotation_preserves_length() {
    let mat = Mat4::rotation_z(FRAC_PI_4);
    let vector = Vec::new(3.0, 4.0, 0.0); // length = 5
    let transformed = mat.transform_direction(vector);

    let original_length = vector.length();
    let transformed_length = transformed.length();

    assert!((original_length - transformed_length).abs() < TOLERANCE);
}

#[test]
fn test_from_look_at_identity_case() {
    let eye = Vec::new(0.0, 0.0, 0.0);
    let target = Vec::new(0.0, 0.0, -1.0);
    let up = Vec::new(0.0, 1.0, 0.0);

    let mat = Mat4::from_look_at(eye, target, up);

    // Should create a view matrix that looks down negative Z
    // The matrix should transform world coordinates to view coordinates
    let forward_point = Vec::new(0.0, 0.0, -1.0);
    let transformed = mat.transform_point(forward_point);

    // Point in front should have negative Z in view space
    assert!(transformed.z < 0.0);
}

#[test]
fn test_from_look_at_orthogonal_basis() {
    let eye = Vec::new(1.0, 2.0, 3.0);
    let target = Vec::new(4.0, 5.0, 6.0);
    let up = Vec::new(0.0, 1.0, 0.0);

    let mat = Mat4::from_look_at(eye, target, up);

    // The first three columns should form an orthonormal basis
    let x_axis = s!(mat.col(0), x y z);
    let y_axis = s!(mat.col(1), x y z);
    let z_axis = s!(mat.col(2), x y z);

    // Check orthogonality
    assert!((x_axis.dot(y_axis)).abs() < TOLERANCE);
    assert!((x_axis.dot(z_axis)).abs() < TOLERANCE);
    assert!((y_axis.dot(z_axis)).abs() < TOLERANCE);

    // Check unit length
    assert!((x_axis.length() - 1.0).abs() < TOLERANCE);
    assert!((y_axis.length() - 1.0).abs() < TOLERANCE);
    assert!((z_axis.length() - 1.0).abs() < TOLERANCE);
}

#[test]
fn test_perspective_basic_properties() {
    let fov_y = FRAC_PI_2; // 90 degrees
    let aspect = 16.0 / 9.0;
    let near = 0.1;
    let far = 100.0;

    let mat = Mat4::perspective(fov_y, aspect, near, far);

    // Check that we have the expected structure
    // Column 2 (z column) should have non-zero third component
    assert!(mat.e[2].z != 0.0);
    // Column 3 should have -1 in the z component for perspective divide
    assert!((mat.e[3].z - (-1.0)).abs() < TOLERANCE);
    // w component of column 3 should be 0 (for perspective)
    assert!((mat.e[3].w - 0.0).abs() < TOLERANCE);
}

#[test]
fn test_perspective_aspect_ratio() {
    let fov_y = FRAC_PI_2;
    let aspect = 2.0;
    let near = 0.1;
    let far = 100.0;

    let mat = Mat4::perspective(fov_y, aspect, near, far);

    // The x scaling should be affected by aspect ratio
    // f/aspect where f = 1/tan(fov_y/2)
    let f = 1.0 / (fov_y / 2.0).tan();
    let expected_x_scale = f / aspect;

    assert!((mat.e[0].x - expected_x_scale).abs() < TOLERANCE);
}

#[test]
#[should_panic(expected = "Field of view must be positive")]
fn test_perspective_invalid_fov() {
    Mat4::perspective(-1.0, 1.0, 0.1, 100.0);
}

#[test]
#[should_panic(expected = "Aspect ratio must be positive")]
fn test_perspective_invalid_aspect() {
    Mat4::perspective(FRAC_PI_2, -1.0, 0.1, 100.0);
}

#[test]
#[should_panic(expected = "Near plane must be positive")]
fn test_perspective_invalid_near() {
    Mat4::perspective(FRAC_PI_2, 1.0, -0.1, 100.0);
}

#[test]
#[should_panic(expected = "Far plane must be greater than near plane")]
fn test_perspective_invalid_far() {
    Mat4::perspective(FRAC_PI_2, 1.0, 100.0, 0.1);
}

#[test]
fn test_transformation_composition() {
    // Test TRS (Translate, Rotate, Scale) composition
    let translation = Vec::new(1.0, 2.0, 3.0);
    let rotation_angle = FRAC_PI_4;
    let scale = Vec::new(2.0, 2.0, 2.0);

    let t_mat = Mat4::translation(translation);
    let r_mat = Mat4::rotation_z(rotation_angle);
    let s_mat = Mat4::scaling(scale);

    // TRS order: first scale, then rotate, then translate
    let trs = Mat4::from_mul(t_mat, Mat4::from_mul(r_mat, s_mat));

    // Test with a point
    let point = Vec::new(1.0, 0.0, 0.0);
    let transformed = trs.transform_point(point);

    // Manual computation: scale -> rotate -> translate
    let scaled = Vec::new(2.0, 0.0, 0.0);
    let cos45 = (FRAC_PI_4).cos();
    let sin45 = (FRAC_PI_4).sin();
    let rotated = Vec::new(scaled.x * cos45, scaled.x * sin45, 0.0);
    let translated = rotated + translation;

    assert_vec_approx_eq(transformed, translated, TOLERANCE);
}

#[test]
fn test_inverse_transformations() {
    let translation = Vec::new(5.0, -3.0, 2.0);
    let scale = Vec::new(2.0, 3.0, 0.5);

    let t_mat = Mat4::translation(translation);
    let s_mat = Mat4::scaling(scale);

    let inv_t = Mat4::translation(-translation);
    let inv_s = Mat4::scaling(Vec::new(1.0 / scale.x, 1.0 / scale.y, 1.0 / scale.z));

    // T * T^-1 should be identity
    let identity_test1 = Mat4::from_mul(t_mat, inv_t);
    let identity_test2 = Mat4::from_mul(s_mat, inv_s);

    for i in 0..4 {
        for j in 0..4 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!((identity_test1.e[i][j] - expected).abs() < TOLERANCE);
            assert!((identity_test2.e[i][j] - expected).abs() < TOLERANCE);
        }
    }
}
