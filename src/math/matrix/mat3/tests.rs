#[cfg(test)]
use super::*;
use std::f32::consts::PI;

#[test]
fn test_constructors() {
    let identity = Mat3::new();
    assert_eq!(identity, Mat3::IDENTITY);

    let from_rows = Mat3::from_rows(
        Vec::new(1.0, 2.0, 3.0),
        Vec::new(4.0, 5.0, 6.0),
        Vec::new(7.0, 8.0, 9.0),
    );

    let from_cols = Mat3::from_cols(
        Vec::new(1.0, 4.0, 7.0),
        Vec::new(2.0, 5.0, 8.0),
        Vec::new(3.0, 6.0, 9.0),
    );

    assert_eq!(from_rows, from_cols);
}

#[test]
fn test_constants() {
    let identity = Mat3::IDENTITY;
    assert_eq!(identity.e[0], Vec::new(1.0, 0.0, 0.0));
    assert_eq!(identity.e[1], Vec::new(0.0, 1.0, 0.0));
    assert_eq!(identity.e[2], Vec::new(0.0, 0.0, 1.0));

    let zero = Mat3::ZERO;
    assert_eq!(zero.e[0], Vec::ZERO);
    assert_eq!(zero.e[1], Vec::ZERO);
    assert_eq!(zero.e[2], Vec::ZERO);

    let flip_x = Mat3::FLIP_X;
    assert_eq!(flip_x.e[0], Vec::new(-1.0, 0.0, 0.0));
    assert_eq!(flip_x.e[1], Vec::new(0.0, 1.0, 0.0));
    assert_eq!(flip_x.e[2], Vec::new(0.0, 0.0, 1.0));
}

#[test]
fn test_transformation_constructors() {
    let scale = Mat3::scaling(Vec::new(2.0, 3.0, 4.0));
    let expected = Mat3::from_cols(
        Vec::new(2.0, 0.0, 0.0),
        Vec::new(0.0, 3.0, 0.0),
        Vec::new(0.0, 0.0, 4.0),
    );
    assert_eq!(scale, expected);

    // Test 90-degree rotations
    let rot_x = Mat3::rotation_x(PI / 2.0);
    let test_vec = Vec::new(1.0, 1.0, 0.0);
    let result = rot_x * test_vec;
    assert!((result.x - 1.0).abs() < 1e-6);
    assert!(result.y.abs() < 1e-6);
    assert!((result.z - 1.0).abs() < 1e-6);

    let rot_y = Mat3::rotation_y(PI / 2.0);
    let test_vec = Vec::new(1.0, 0.0, 1.0);
    let result = rot_y * test_vec;
    assert!((result.x - 1.0).abs() < 1e-6);
    assert!(result.y.abs() < 1e-6);
    assert!((result.z + 1.0).abs() < 1e-6);

    let rot_z = Mat3::rotation_z(PI / 2.0);
    let test_vec = Vec::new(1.0, 0.0, 1.0);
    let result = rot_z * test_vec;
    assert!(result.x.abs() < 1e-6);
    assert!((result.y - 1.0).abs() < 1e-6);
    assert!((result.z - 1.0).abs() < 1e-6);
}

#[test]
fn test_arbitrary_axis_rotation() {
    // Rotation around z-axis should match rotation_z
    let axis = Vec::new(0.0, 0.0, 1.0);
    let angle = PI / 4.0;

    let rot_axis = Mat3::rotation(axis, angle);
    let rot_z = Mat3::rotation_z(angle);

    let test_vec = Vec::new(1.0, 0.0, 0.0);
    let result_axis = rot_axis * test_vec;
    let result_z = rot_z * test_vec;

    assert!((result_axis.x - result_z.x).abs() < 1e-6);
    assert!((result_axis.y - result_z.y).abs() < 1e-6);
    assert!((result_axis.z - result_z.z).abs() < 1e-6);
}

#[test]
fn test_conversions() {
    let mat = Mat3::from_cols(
        Vec::new(1.0, 2.0, 3.0),
        Vec::new(4.0, 5.0, 6.0),
        Vec::new(7.0, 8.0, 9.0),
    );

    // Test array conversion
    let arr: [f32; 9] = mat.into();
    let expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    assert_eq!(arr, expected);

    let from_arr = Mat3::from(expected);
    assert_eq!(from_arr, mat);

    // Test 2D array conversion
    let arr_2d: [[f32; 3]; 3] = mat.into();
    let expected_2d = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    assert_eq!(arr_2d, expected_2d);

    let from_arr_2d = Mat3::from(expected_2d);
    assert_eq!(from_arr_2d, mat);

    // Test Vec array conversion
    let vec_arr: [Vec; 3] = mat.into();
    let from_vec_arr = Mat3::from(vec_arr);
    assert_eq!(from_vec_arr, mat);

    // Test scalar conversion
    let scalar_mat = Mat3::from(5.0);
    let expected_scalar = Mat3::from_cols(
        Vec::new(5.0, 5.0, 5.0),
        Vec::new(5.0, 5.0, 5.0),
        Vec::new(5.0, 5.0, 5.0),
    );
    assert_eq!(scalar_mat, expected_scalar);
}

#[test]
fn test_accessors() {
    let mat = Mat3::from_cols(
        Vec::new(1.0, 2.0, 3.0),
        Vec::new(4.0, 5.0, 6.0),
        Vec::new(7.0, 8.0, 9.0),
    );

    // Test column access
    assert_eq!(mat.col(0), Vec::new(1.0, 2.0, 3.0));
    assert_eq!(mat.col(1), Vec::new(4.0, 5.0, 6.0));
    assert_eq!(mat.col(2), Vec::new(7.0, 8.0, 9.0));

    let cols = mat.cols();
    assert_eq!(cols[0], Vec::new(1.0, 2.0, 3.0));
    assert_eq!(cols[1], Vec::new(4.0, 5.0, 6.0));
    assert_eq!(cols[2], Vec::new(7.0, 8.0, 9.0));

    // Test row access
    assert_eq!(mat.row(0), Vec::new(1.0, 4.0, 7.0));
    assert_eq!(mat.row(1), Vec::new(2.0, 5.0, 8.0));
    assert_eq!(mat.row(2), Vec::new(3.0, 6.0, 9.0));

    let rows = mat.rows();
    assert_eq!(rows[0], Vec::new(1.0, 4.0, 7.0));
    assert_eq!(rows[1], Vec::new(2.0, 5.0, 8.0));
    assert_eq!(rows[2], Vec::new(3.0, 6.0, 9.0));
}

#[test]
fn test_setters() {
    let mut mat = Mat3::ZERO;

    mat.set_col(0, Vec::new(1.0, 2.0, 3.0));
    mat.set_col(1, Vec::new(4.0, 5.0, 6.0));
    mat.set_col(2, Vec::new(7.0, 8.0, 9.0));

    assert_eq!(mat.col(0), Vec::new(1.0, 2.0, 3.0));
    assert_eq!(mat.col(1), Vec::new(4.0, 5.0, 6.0));
    assert_eq!(mat.col(2), Vec::new(7.0, 8.0, 9.0));

    let mut mat2 = Mat3::ZERO;
    mat2.set_row(0, Vec::new(1.0, 4.0, 7.0));
    mat2.set_row(1, Vec::new(2.0, 5.0, 8.0));
    mat2.set_row(2, Vec::new(3.0, 6.0, 9.0));

    assert_eq!(mat, mat2);

    // Test individual element setting
    let mut mat3 = Mat3::ZERO;
    mat3.set(0, 0, 1.0);
    mat3.set(1, 1, 2.0);
    mat3.set(2, 2, 3.0);

    assert_eq!(mat3.e[0].x, 1.0);
    assert_eq!(mat3.e[1].y, 2.0);
    assert_eq!(mat3.e[2].z, 3.0);
}

#[test]
fn test_indexing() {
    let mat = Mat3::from_cols(
        Vec::new(1.0, 2.0, 3.0),
        Vec::new(4.0, 5.0, 6.0),
        Vec::new(7.0, 8.0, 9.0),
    );

    assert_eq!(mat[0], Vec::new(1.0, 2.0, 3.0));
    assert_eq!(mat[1], Vec::new(4.0, 5.0, 6.0));
    assert_eq!(mat[2], Vec::new(7.0, 8.0, 9.0));

    let mut mat_mut = mat;
    mat_mut[0] = Vec::new(10.0, 11.0, 12.0);
    assert_eq!(mat_mut[0], Vec::new(10.0, 11.0, 12.0));
}

#[test]
fn test_matrix_addition() {
    let a = Mat3::from_cols(
        Vec::new(1.0, 2.0, 3.0),
        Vec::new(4.0, 5.0, 6.0),
        Vec::new(7.0, 8.0, 9.0),
    );
    let b = Mat3::from_cols(
        Vec::new(9.0, 8.0, 7.0),
        Vec::new(6.0, 5.0, 4.0),
        Vec::new(3.0, 2.0, 1.0),
    );

    let result = a + b;
    let expected = Mat3::from_cols(
        Vec::new(10.0, 10.0, 10.0),
        Vec::new(10.0, 10.0, 10.0),
        Vec::new(10.0, 10.0, 10.0),
    );
    assert_eq!(result, expected);

    let mut a_mut = a;
    a_mut += b;
    assert_eq!(a_mut, expected);
}

#[test]
fn test_matrix_subtraction() {
    let a = Mat3::from_cols(
        Vec::new(10.0, 10.0, 10.0),
        Vec::new(10.0, 10.0, 10.0),
        Vec::new(10.0, 10.0, 10.0),
    );
    let b = Mat3::from_cols(
        Vec::new(1.0, 2.0, 3.0),
        Vec::new(4.0, 5.0, 6.0),
        Vec::new(7.0, 8.0, 9.0),
    );

    let result = a - b;
    let expected = Mat3::from_cols(
        Vec::new(9.0, 8.0, 7.0),
        Vec::new(6.0, 5.0, 4.0),
        Vec::new(3.0, 2.0, 1.0),
    );
    assert_eq!(result, expected);

    let mut a_mut = a;
    a_mut -= b;
    assert_eq!(a_mut, expected);
}

#[test]
fn test_matrix_multiplication() {
    let a = Mat3::from_rows(
        Vec::new(1.0, 2.0, 3.0),
        Vec::new(4.0, 5.0, 6.0),
        Vec::new(7.0, 8.0, 9.0),
    );
    let b = Mat3::from_rows(
        Vec::new(9.0, 8.0, 7.0),
        Vec::new(6.0, 5.0, 4.0),
        Vec::new(3.0, 2.0, 1.0),
    );

    let result = a * b;
    let expected = Mat3::from_rows(
        Vec::new(30.0, 24.0, 18.0),
        Vec::new(84.0, 69.0, 54.0),
        Vec::new(138.0, 114.0, 90.0),
    );
    assert_eq!(result, expected);

    // Test identity multiplication
    let identity_result = a * Mat3::IDENTITY;
    assert_eq!(identity_result, a);

    let mut a_mut = a;
    a_mut *= Mat3::IDENTITY;
    assert_eq!(a_mut, a);
}

#[test]
fn test_matrix_vector_multiplication() {
    let mat = Mat3::from_rows(
        Vec::new(1.0, 2.0, 3.0),
        Vec::new(4.0, 5.0, 6.0),
        Vec::new(7.0, 8.0, 9.0),
    );
    let vec = Vec::new(1.0, 2.0, 3.0);

    let result = mat * vec;
    let expected = Vec::new(14.0, 32.0, 50.0); // [1*1+2*2+3*3, 4*1+5*2+6*3, 7*1+8*2+9*3]
    assert_eq!(result, expected);

    // Test identity transformation
    let identity_result = Mat3::IDENTITY * vec;
    assert_eq!(identity_result, vec);

    let mut vec_mut = vec;
    vec_mut *= Mat3::IDENTITY;
    assert_eq!(vec_mut, vec);
}

#[test]
fn test_scalar_multiplication() {
    let mat = Mat3::from_cols(
        Vec::new(1.0, 2.0, 3.0),
        Vec::new(4.0, 5.0, 6.0),
        Vec::new(7.0, 8.0, 9.0),
    );

    let result = mat * 2.0;
    let expected = Mat3::from_cols(
        Vec::new(2.0, 4.0, 6.0),
        Vec::new(8.0, 10.0, 12.0),
        Vec::new(14.0, 16.0, 18.0),
    );
    assert_eq!(result, expected);

    let result2 = 2.0 * mat;
    assert_eq!(result2, expected);

    let mut mat_mut = mat;
    mat_mut *= 2.0;
    assert_eq!(mat_mut, expected);
}

#[test]
fn test_scalar_division() {
    let mat = Mat3::from_cols(
        Vec::new(2.0, 4.0, 6.0),
        Vec::new(8.0, 10.0, 12.0),
        Vec::new(14.0, 16.0, 18.0),
    );

    let result = mat / 2.0;
    let expected = Mat3::from_cols(
        Vec::new(1.0, 2.0, 3.0),
        Vec::new(4.0, 5.0, 6.0),
        Vec::new(7.0, 8.0, 9.0),
    );
    assert_eq!(result, expected);

    let mut mat_mut = mat;
    mat_mut /= 2.0;
    assert_eq!(mat_mut, expected);
}

#[test]
#[should_panic(expected = "Division by zero in Mat3 division")]
fn test_scalar_division_by_zero() {
    let mat = Mat3::IDENTITY;
    let _ = mat / 0.0;
}

#[test]
fn test_matrix_division() {
    let a = Mat3::IDENTITY;
    let b = Mat3::scaling(Vec::new(2.0, 3.0, 4.0));

    if let Some(b_inv) = b.inverse() {
        let result = a / b;
        let expected = a * b_inv;

        // Use approximate equality for floating point
        for i in 0..3 {
            for j in 0..3 {
                assert!((result.e[i][j] - expected.e[i][j]).abs() < 1e-6);
            }
        }
    }
}

#[test]
fn test_negation() {
    let mat = Mat3::from_cols(
        Vec::new(1.0, -2.0, 3.0),
        Vec::new(-4.0, 5.0, -6.0),
        Vec::new(7.0, -8.0, 9.0),
    );

    let result = -mat;
    let expected = Mat3::from_cols(
        Vec::new(-1.0, 2.0, -3.0),
        Vec::new(4.0, -5.0, 6.0),
        Vec::new(-7.0, 8.0, -9.0),
    );
    assert_eq!(result, expected);
}

#[test]
fn test_determinant() {
    // Test identity
    assert_eq!(Mat3::IDENTITY.determinant(), 1.0);

    // Test simple case
    let mat = Mat3::from_rows(
        Vec::new(1.0, 2.0, 3.0),
        Vec::new(4.0, 5.0, 6.0),
        Vec::new(7.0, 8.0, 9.0),
    );
    assert_eq!(mat.determinant(), 0.0);

    // Test non-singular matrix
    let mat2 = Mat3::from_rows(
        Vec::new(1.0, 0.0, 2.0),
        Vec::new(0.0, 1.0, 3.0),
        Vec::new(1.0, 2.0, 1.0),
    );
    assert_eq!(mat2.determinant(), -7.0);
}

#[test]
fn test_minor() {
    let mat = Mat3::from_rows(
        Vec::new(1.0, 2.0, 3.0),
        Vec::new(4.0, 5.0, 6.0),
        Vec::new(7.0, 8.0, 9.0),
    );

    // Minor of (0,0) should be determinant of [[5,6],[8,9]] = 5*9 - 6*8 = 45 - 48 = -3
    assert_eq!(mat.minor(0, 0), -3.0);

    // Minor of (1,1) should be determinant of [[1,3],[7,9]] = 1*9 - 3*7 = 9 - 21 = -12
    assert_eq!(mat.minor(1, 1), -12.0);
}

#[test]
fn test_transpose() {
    let mat = Mat3::from_rows(
        Vec::new(1.0, 2.0, 3.0),
        Vec::new(4.0, 5.0, 6.0),
        Vec::new(7.0, 8.0, 9.0),
    );

    let transposed = mat.transpose();
    let expected = Mat3::from_cols(
        Vec::new(1.0, 2.0, 3.0),
        Vec::new(4.0, 5.0, 6.0),
        Vec::new(7.0, 8.0, 9.0),
    );
    assert_eq!(transposed, expected);

    // Test transpose_mut
    let mut mat_mut = mat;
    mat_mut.transpose_mut();
    assert_eq!(mat_mut, expected);

    // Test double transpose returns original
    assert_eq!(transposed.transpose(), mat);
}

#[test]
fn test_inverse() {
    // Test identity inverse
    let identity_inv = Mat3::IDENTITY.inverse().unwrap();
    assert_eq!(identity_inv, Mat3::IDENTITY);

    // Test scaling matrix inverse
    let scale = Mat3::scaling(Vec::new(2.0, 3.0, 4.0));
    let scale_inv = scale.inverse().unwrap();
    let expected_inv = Mat3::scaling(Vec::new(0.5, 1.0 / 3.0, 0.25));

    for i in 0..3 {
        for j in 0..3 {
            assert!((scale_inv.e[i][j] - expected_inv.e[i][j]).abs() < 1e-6);
        }
    }

    // Test that A * A^-1 = I
    let result = scale * scale_inv;
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!((result.e[i][j] - expected).abs() < 1e-6);
        }
    }

    // Test singular matrix returns None
    let singular = Mat3::from_rows(
        Vec::new(1.0, 2.0, 3.0),
        Vec::new(4.0, 5.0, 6.0),
        Vec::new(7.0, 8.0, 9.0),
    );
    assert!(singular.inverse().is_none());
}

#[test]
fn test_trace() {
    let mat = Mat3::from_rows(
        Vec::new(1.0, 2.0, 3.0),
        Vec::new(4.0, 5.0, 6.0),
        Vec::new(7.0, 8.0, 9.0),
    );
    assert_eq!(mat.trace(), 15.0); // 1 + 5 + 9

    assert_eq!(Mat3::IDENTITY.trace(), 3.0);
    assert_eq!(Mat3::ZERO.trace(), 0.0);
}

#[test]
fn test_utility_operations() {
    assert!(Mat3::IDENTITY.is_identity());
    assert!(!Mat3::ZERO.is_identity());

    assert!(Mat3::ZERO.is_zero());
    assert!(!Mat3::IDENTITY.is_zero());

    assert!(Mat3::IDENTITY.is_invertible());

    let singular = Mat3::from_rows(
        Vec::new(1.0, 2.0, 3.0),
        Vec::new(4.0, 5.0, 6.0),
        Vec::new(7.0, 8.0, 9.0),
    );
    assert!(!singular.is_invertible());
}

#[test]
fn test_chainable_operations() {
    let mat = Mat3::IDENTITY;

    let result = mat
        .scale(Vec::new(2.0, 2.0, 2.0))
        .rotate_z(PI / 2.0);

    // Should scale then rotate
    let test_vec = Vec::new(1.0, 0.0, 0.0);
    let transformed = result * test_vec;

    // After scaling: (2, 0, 0), after 90° z rotation: (0, 2, 0)
    assert!(transformed.x.abs() < 1e-6);
    assert!((transformed.y - 2.0).abs() < 1e-6);
    assert!(transformed.z.abs() < 1e-6);
}

#[test]
fn test_default() {
    let default_mat = Mat3::default();
    assert_eq!(default_mat, Mat3::IDENTITY);
}

#[test]
#[should_panic]
fn test_out_of_bounds_access() {
    let mat = Mat3::IDENTITY;
    let _ = mat[3]; // Should panic
}

#[test]
#[should_panic]
fn test_out_of_bounds_set() {
    let mut mat = Mat3::IDENTITY;
    mat.set(3, 0, 1.0); // Should panic
}

#[test]
#[should_panic]
fn test_out_of_bounds_row() {
    let mat = Mat3::IDENTITY;
    let _ = mat.row(3); // Should panic
}

#[test]
#[should_panic]
fn test_out_of_bounds_col_set() {
    let mut mat = Mat3::IDENTITY;
    mat.set_col(3, Vec::ZERO); // Should panic
}