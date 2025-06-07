#[cfg(test)]
use super::*;

#[test]
fn test_constructors() {
    let identity = Mat2::new();
    assert_eq!(identity, Mat2::IDENTITY);

    let from_rows = Mat2::from_rows(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));

    let from_cols = Mat2::from_cols(Vec2::new(1.0, 3.0), Vec2::new(2.0, 4.0));

    assert_eq!(from_rows, from_cols);
}

#[test]
fn test_constants() {
    let identity = Mat2::IDENTITY;
    assert_eq!(identity.e[0], Vec2::new(1.0, 0.0));
    assert_eq!(identity.e[1], Vec2::new(0.0, 1.0));

    let zero = Mat2::ZERO;
    assert_eq!(zero.e[0], Vec2::ZERO);
    assert_eq!(zero.e[1], Vec2::ZERO);

    let flip_x = Mat2::FLIP_X;
    assert_eq!(flip_x.e[0], Vec2::new(-1.0, 0.0));
    assert_eq!(flip_x.e[1], Vec2::new(0.0, 1.0));

    let flip_y = Mat2::FLIP_Y;
    assert_eq!(flip_y.e[0], Vec2::new(1.0, 0.0));
    assert_eq!(flip_y.e[1], Vec2::new(0.0, -1.0));
}

#[test]
fn test_conversions() {
    let mat = Mat2::from_cols(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));

    // Test array conversion
    let arr: [f32; 4] = mat.into();
    let expected = [1.0, 2.0, 3.0, 4.0];
    assert_eq!(arr, expected);

    let from_arr = Mat2::from(expected);
    assert_eq!(from_arr, mat);

    // Test 2D array conversion
    let arr_2d: [[f32; 2]; 2] = mat.into();
    let expected_2d = [[1.0, 2.0], [3.0, 4.0]];
    assert_eq!(arr_2d, expected_2d);

    let from_arr_2d = Mat2::from(expected_2d);
    assert_eq!(from_arr_2d, mat);

    // Test Vec2 array conversion
    let vec_arr: [Vec2; 2] = mat.into();
    let from_vec_arr = Mat2::from(vec_arr);
    assert_eq!(from_vec_arr, mat);

    // Test scalar conversion
    let scalar_mat = Mat2::from(5.0);
    let expected_scalar = Mat2::from_cols(Vec2::new(5.0, 5.0), Vec2::new(5.0, 5.0));
    assert_eq!(scalar_mat, expected_scalar);
}

#[test]
fn test_accessors() {
    let mat = Mat2::from_cols(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));

    // Test column access
    assert_eq!(mat.col(0), Vec2::new(1.0, 2.0));
    assert_eq!(mat.col(1), Vec2::new(3.0, 4.0));

    let cols = mat.cols();
    assert_eq!(cols[0], Vec2::new(1.0, 2.0));
    assert_eq!(cols[1], Vec2::new(3.0, 4.0));

    // Test row access
    assert_eq!(mat.row(0), Vec2::new(1.0, 3.0));
    assert_eq!(mat.row(1), Vec2::new(2.0, 4.0));

    let rows = mat.rows();
    assert_eq!(rows[0], Vec2::new(1.0, 3.0));
    assert_eq!(rows[1], Vec2::new(2.0, 4.0));
}

#[test]
fn test_setters() {
    let mut mat = Mat2::ZERO;

    mat.set_col(0, Vec2::new(1.0, 2.0));
    mat.set_col(1, Vec2::new(3.0, 4.0));

    assert_eq!(mat.col(0), Vec2::new(1.0, 2.0));
    assert_eq!(mat.col(1), Vec2::new(3.0, 4.0));

    let mut mat2 = Mat2::ZERO;
    mat2.set_row(0, Vec2::new(1.0, 3.0));
    mat2.set_row(1, Vec2::new(2.0, 4.0));

    assert_eq!(mat, mat2);

    // Test individual element setting
    let mut mat3 = Mat2::ZERO;
    mat3.set(0, 0, 1.0);
    mat3.set(1, 1, 2.0);

    assert_eq!(mat3.e[0].x, 1.0);
    assert_eq!(mat3.e[1].y, 2.0);
}

#[test]
fn test_indexing() {
    let mat = Mat2::from_cols(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));

    assert_eq!(mat[0], Vec2::new(1.0, 2.0));
    assert_eq!(mat[1], Vec2::new(3.0, 4.0));

    let mut mat_mut = mat;
    mat_mut[0] = Vec2::new(10.0, 11.0);
    assert_eq!(mat_mut[0], Vec2::new(10.0, 11.0));
}

#[test]
fn test_matrix_addition() {
    let a = Mat2::from_cols(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
    let b = Mat2::from_cols(Vec2::new(5.0, 6.0), Vec2::new(7.0, 8.0));

    let result = a + b;
    let expected = Mat2::from_cols(Vec2::new(6.0, 8.0), Vec2::new(10.0, 12.0));
    assert_eq!(result, expected);

    let mut a_mut = a;
    a_mut += b;
    assert_eq!(a_mut, expected);
}

#[test]
fn test_matrix_subtraction() {
    let a = Mat2::from_cols(Vec2::new(5.0, 6.0), Vec2::new(7.0, 8.0));
    let b = Mat2::from_cols(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));

    let result = a - b;
    let expected = Mat2::from_cols(Vec2::new(4.0, 4.0), Vec2::new(4.0, 4.0));
    assert_eq!(result, expected);

    let mut a_mut = a;
    a_mut -= b;
    assert_eq!(a_mut, expected);
}

#[test]
fn test_matrix_multiplication() {
    let a = Mat2::from_rows(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
    let b = Mat2::from_rows(Vec2::new(5.0, 6.0), Vec2::new(7.0, 8.0));

    let result = a * b;
    let expected = Mat2::from_rows(
        Vec2::new(19.0, 22.0), // [1*5+2*7, 1*6+2*8]
        Vec2::new(43.0, 50.0), // [3*5+4*7, 3*6+4*8]
    );
    assert_eq!(result, expected);

    // Test identity multiplication
    let identity_result = a * Mat2::IDENTITY;
    assert_eq!(identity_result, a);

    let mut a_mut = a;
    a_mut *= Mat2::IDENTITY;
    assert_eq!(a_mut, a);
}

#[test]
fn test_matrix_vector_multiplication() {
    let mat = Mat2::from_rows(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
    let vec = Vec2::new(5.0, 6.0);

    let result = mat * vec;
    let expected = Vec2::new(17.0, 39.0); // [1*5+2*6, 3*5+4*6]
    assert_eq!(result, expected);

    // Test identity transformation
    let identity_result = Mat2::IDENTITY * vec;
    assert_eq!(identity_result, vec);

    let mut vec_mut = vec;
    vec_mut *= Mat2::IDENTITY;
    assert_eq!(vec_mut, vec);
}

#[test]
fn test_scalar_multiplication() {
    let mat = Mat2::from_cols(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));

    let result = mat * 2.0;
    let expected = Mat2::from_cols(Vec2::new(2.0, 4.0), Vec2::new(6.0, 8.0));
    assert_eq!(result, expected);

    let result2 = 2.0 * mat;
    assert_eq!(result2, expected);

    let mut mat_mut = mat;
    mat_mut *= 2.0;
    assert_eq!(mat_mut, expected);
}

#[test]
fn test_scalar_division() {
    let mat = Mat2::from_cols(Vec2::new(2.0, 4.0), Vec2::new(6.0, 8.0));

    let result = mat / 2.0;
    let expected = Mat2::from_cols(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
    assert_eq!(result, expected);

    let mut mat_mut = mat;
    mat_mut /= 2.0;
    assert_eq!(mat_mut, expected);
}

#[test]
#[should_panic(expected = "Division by zero in Mat2 division")]
fn test_scalar_division_by_zero() {
    let mat = Mat2::IDENTITY;
    let _ = mat / 0.0;
}

#[test]
fn test_matrix_division() {
    let a = Mat2::IDENTITY;
    let b = Mat2::from_cols(Vec2::new(2.0, 0.0), Vec2::new(0.0, 3.0));

    if let Some(b_inv) = b.inverse() {
        let result = a / b;
        let expected = a * b_inv;

        // Use approximate equality for floating point
        for i in 0..2 {
            for j in 0..2 {
                assert!((result.e[i][j] - expected.e[i][j]).abs() < 1e-6);
            }
        }
    }
}

#[test]
fn test_negation() {
    let mat = Mat2::from_cols(Vec2::new(1.0, -2.0), Vec2::new(-3.0, 4.0));

    let result = -mat;
    let expected = Mat2::from_cols(Vec2::new(-1.0, 2.0), Vec2::new(3.0, -4.0));
    assert_eq!(result, expected);
}

#[test]
fn test_determinant() {
    // Test identity
    assert_eq!(Mat2::IDENTITY.determinant(), 1.0);

    // Test simple case
    let mat = Mat2::from_rows(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
    // det = 1*4 - 2*3 = 4 - 6 = -2
    assert_eq!(mat.determinant(), -2.0);

    // Test singular matrix (determinant = 0)
    let singular = Mat2::from_rows(Vec2::new(1.0, 2.0), Vec2::new(2.0, 4.0));
    assert_eq!(singular.determinant(), 0.0);

    // Test scaling matrix
    let scale = Mat2::from_cols(Vec2::new(2.0, 0.0), Vec2::new(0.0, 3.0));
    assert_eq!(scale.determinant(), 6.0);
}

#[test]
fn test_transpose() {
    let mat = Mat2::from_rows(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));

    let transposed = mat.transpose();
    let expected = Mat2::from_cols(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
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
    let identity_inv = Mat2::IDENTITY.inverse().unwrap();
    assert_eq!(identity_inv, Mat2::IDENTITY);

    // Test simple matrix inverse
    let mat = Mat2::from_rows(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
    // det = -2, so inverse should be (1/det) * [[4, -2], [-3, 1]]
    let mat_inv = mat.inverse().unwrap();
    let expected = Mat2::from_rows(Vec2::new(-2.0, 1.0), Vec2::new(1.5, -0.5));

    for i in 0..2 {
        for j in 0..2 {
            assert!((mat_inv.e[i][j] - expected.e[i][j]).abs() < 1e-6);
        }
    }

    // Test that A * A^-1 = I
    let result = mat * mat_inv;
    for i in 0..2 {
        for j in 0..2 {
            let expected_val = if i == j { 1.0 } else { 0.0 };
            assert!((result.e[i][j] - expected_val).abs() < 1e-6);
        }
    }

    // Test singular matrix returns None
    let singular = Mat2::from_rows(Vec2::new(1.0, 2.0), Vec2::new(2.0, 4.0));
    assert!(singular.inverse().is_none());
}

#[test]
fn test_trace() {
    let mat = Mat2::from_rows(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
    assert_eq!(mat.trace(), 5.0); // 1 + 4

    assert_eq!(Mat2::IDENTITY.trace(), 2.0);
    assert_eq!(Mat2::ZERO.trace(), 0.0);
}

#[test]
fn test_utility_operations() {
    assert!(Mat2::IDENTITY.is_identity());
    assert!(!Mat2::ZERO.is_identity());

    assert!(Mat2::ZERO.is_zero());
    assert!(!Mat2::IDENTITY.is_zero());

    assert!(Mat2::IDENTITY.is_invertible());

    let singular = Mat2::from_rows(Vec2::new(1.0, 2.0), Vec2::new(2.0, 4.0));
    assert!(!singular.is_invertible());

    // Test scaling matrix
    let scale = Mat2::from_cols(Vec2::new(2.0, 0.0), Vec2::new(0.0, 3.0));
    assert!(scale.is_invertible());
}

#[test]
fn test_default() {
    let default_mat = Mat2::default();
    assert_eq!(default_mat, Mat2::IDENTITY);
}

#[test]
fn test_multiplication_associativity() {
    let a = Mat2::from_rows(Vec2::new(1.0, 2.0), Vec2::new(3.0, 4.0));
    let b = Mat2::from_rows(Vec2::new(5.0, 6.0), Vec2::new(7.0, 8.0));
    let c = Mat2::from_rows(Vec2::new(1.0, 0.0), Vec2::new(0.0, 2.0));

    let result1 = (a * b) * c;
    let result2 = a * (b * c);

    for i in 0..2 {
        for j in 0..2 {
            assert!((result1.e[i][j] - result2.e[i][j]).abs() < 1e-6);
        }
    }
}

#[test]
fn test_flip_matrices() {
    let vec = Vec2::new(1.0, 1.0);

    // Test FLIP_X
    let flipped_x = Mat2::FLIP_X * vec;
    assert_eq!(flipped_x, Vec2::new(-1.0, 1.0));

    // Test FLIP_Y
    let flipped_y = Mat2::FLIP_Y * vec;
    assert_eq!(flipped_y, Vec2::new(1.0, -1.0));
}

#[test]
#[should_panic]
fn test_out_of_bounds_access() {
    let mat = Mat2::IDENTITY;
    let _ = mat[2]; // Should panic
}

#[test]
#[should_panic]
fn test_out_of_bounds_set() {
    let mut mat = Mat2::IDENTITY;
    mat.set(2, 0, 1.0); // Should panic
}

#[test]
#[should_panic]
fn test_out_of_bounds_row() {
    let mat = Mat2::IDENTITY;
    let _ = mat.row(2); // Should panic
}

#[test]
#[should_panic]
fn test_out_of_bounds_col_set() {
    let mut mat = Mat2::IDENTITY;
    mat.set_col(2, Vec2::ZERO); // Should panic
}
