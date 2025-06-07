/*
Constructors

from_quaternion(Quaternion) - When you add quaternions later
perspective(fov, aspect, near, far) - Projection matrix
orthographic(left, right, bottom, top, near, far) - Ortho projection

Decomposition

extract_translation() - Get translation component
extract_scale() - Get scale component (approximate)
extract_rotation() - Get rotation component (future quaternion)

Utility Operations

lerp(other, t) - Linear interpolation

 */

use crate::math::{Vec4,Mat3};
use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};
use crate::Vec;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat4 {
    e: [Vec4; 4],
}

// Constructors

impl Mat4 {
    pub fn new() -> Self {
        Mat4::IDENTITY
    }

    pub fn from_rows(row1: Vec4, row2: Vec4, row3: Vec4, row4: Vec4) -> Self {
        Mat4 {
            e: [
                Vec4::new(row1.x, row2.x, row3.x, row4.x), // col 0
                Vec4::new(row1.y, row2.y, row3.y, row4.y), // col 1
                Vec4::new(row1.z, row2.z, row3.z, row4.z), // col 2
                Vec4::new(row1.w, row2.w, row3.w, row4.w), // col 3
            ],
        }
    }

    pub fn from_cols(col1: Vec4, col2: Vec4, col3: Vec4, col4: Vec4) -> Self {
        Mat4 {
            e: [col1, col2, col3, col4],
        }
    }

    pub fn from_mul(one: Mat4, other: Mat4) -> Self {
        let rows = one.rows();
        let cols = other.cols();
        Mat4 {
            e: [
                // Column 0: each row of 'one' dotted with column 0 of 'other'
                Vec4::new(
                    rows[0].dot(cols[0]),
                    rows[1].dot(cols[0]),
                    rows[2].dot(cols[0]),
                    rows[3].dot(cols[0]),
                ),
                // Column 1: each row of 'one' dotted with column 1 of 'other'
                Vec4::new(
                    rows[0].dot(cols[1]),
                    rows[1].dot(cols[1]),
                    rows[2].dot(cols[1]),
                    rows[3].dot(cols[1]),
                ),
                // Column 2: each row of 'one' dotted with column 2 of 'other'
                Vec4::new(
                    rows[0].dot(cols[2]),
                    rows[1].dot(cols[2]),
                    rows[2].dot(cols[2]),
                    rows[3].dot(cols[2]),
                ),
                // Column 3: each row of 'one' dotted with column 3 of 'other'
                Vec4::new(
                    rows[0].dot(cols[3]),
                    rows[1].dot(cols[3]),
                    rows[2].dot(cols[3]),
                    rows[3].dot(cols[3]),
                ),
            ],
        }
    }
}

// Transformation Constructors

impl Mat4 {


    pub fn translation(translation: Vec) -> Self {
        Mat4 {
            e: [
                Vec4::new(1.0, 0.0, 0.0, 0.0),
                Vec4::new(0.0, 1.0, 0.0, 0.0),
                Vec4::new(0.0, 0.0, 1.0, 0.0),
                Vec4::new(translation.x, translation.y, translation.z, 1.0),
            ],
        }
    }

    pub fn scaling(scaling: Vec) -> Self {
        Mat4 {
            e: [
                Vec4::new(scaling.x, 0.0, 0.0, 0.0),
                Vec4::new(0.0, scaling.y, 0.0, 0.0),
                Vec4::new(0.0, 0.0, scaling.z, 0.0),
                Vec4::new(0.0, 0.0, 0.0, 1.0),
            ],
        }
    }

    pub fn rotation_x(angle: f32) -> Self {
        let cos = angle.cos();
        let sin = angle.sin();
        Mat4 {
            e: [
                Vec4::new(1.0, 0.0, 0.0, 0.0),
                Vec4::new(0.0, cos, sin, 0.0),
                Vec4::new(0.0, -sin, cos, 0.0),
                Vec4::new(0.0, 0.0, 0.0, 1.0),
            ],
        }
    }

    pub fn rotation_y(angle: f32) -> Self {
        let cos = angle.cos();
        let sin = angle.sin();
        Mat4 {
            e: [
                Vec4::new(cos, 0.0, -sin, 0.0),
                Vec4::new(0.0, 1.0, 0.0, 0.0),
                Vec4::new(sin, 0.0, cos, 0.0),
                Vec4::new(0.0, 0.0, 0.0, 1.0),
            ],
        }
    }

    pub fn rotation_z(angle: f32) -> Self {
        let cos = angle.cos();
        let sin = angle.sin();
        Mat4 {
            e: [
                Vec4::new(cos, sin, 0.0, 0.0),
                Vec4::new(-sin, cos, 0.0, 0.0),
                Vec4::new(0.0, 0.0, 1.0, 0.0),
                Vec4::new(0.0, 0.0, 0.0, 1.0),
            ],
        }
    }

    pub fn rotation(axis: Vec, angle: f32) -> Self {
        let cos = angle.cos();
        let sin = angle.sin();
        let one_minus_cos = 1.0 - cos;

        let x = axis.x;
        let y = axis.y;
        let z = axis.z;

        Mat4 {
            e: [
                Vec4::new(cos + x * x * one_minus_cos, x * y * one_minus_cos + z * sin, x * z * one_minus_cos - y * sin, 0.0),
                Vec4::new(y * x * one_minus_cos - z * sin, cos + y * y * one_minus_cos, y * z * one_minus_cos + x * sin, 0.0),
                Vec4::new(z * x * one_minus_cos + y * sin, z * y * one_minus_cos - x * sin, cos + z * z * one_minus_cos, 0.0),
                Vec4::new(0.0, 0.0, 0.0, 1.0),
            ],
        }
    }

    pub fn from_look_at(eye: Vec, target: Vec, up: Vec) -> Self {
        let z_axis = (eye - target).normalized();
        let x_axis = up.cross(z_axis).normalized();
        let y_axis = z_axis.cross(x_axis);

        Mat4 {
            e: [
                Vec4::new(x_axis.x, x_axis.y, x_axis.z, 0.0),
                Vec4::new(y_axis.x, y_axis.y, y_axis.z, 0.0),
                Vec4::new(z_axis.x, z_axis.y, z_axis.z, 0.0),
                Vec4::new(-x_axis.dot(eye), -y_axis.dot(eye), -z_axis.dot(eye), 1.0),
            ],
        }
    }

/// Creates a perspective projection matrix.
    ///
    /// # Arguments
    /// * `fov_y` - Vertical field of view in radians
    /// * `aspect` - Aspect ratio (width / height)
    /// * `near` - Near clipping plane distance (positive)
    /// * `far` - Far clipping plane distance (positive)
    pub fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> Self {
        assert!(fov_y > 0.0, "Field of view must be positive");
        assert!(aspect > 0.0, "Aspect ratio must be positive");
        assert!(near > 0.0, "Near plane must be positive");
        assert!(far > near, "Far plane must be greater than near plane");

        let f = 1.0 / (fov_y / 2.0).tan();

        let z_range = near - far;  // This will be negative
        let a = (far + near) / z_range;
        let b = (2.0 * far * near) / z_range;

        Mat4 {
            e: [
                Vec4::new(f / aspect, 0.0, 0.0, 0.0),
                Vec4::new(0.0, f, 0.0, 0.0),
                Vec4::new(0.0, 0.0, a, b),
                Vec4::new(0.0, 0.0, -1.0, 0.0),
            ],
        }
    }
}

// Constants

impl Mat4 {
    pub const IDENTITY: Self = Mat4 {
        e: [
            Vec4 { x: 1.0, y: 0.0, z: 0.0, w: 0.0 },
            Vec4 { x: 0.0, y: 1.0, z: 0.0, w: 0.0 },
            Vec4 { x: 0.0, y: 0.0, z: 1.0, w: 0.0 },
            Vec4 { x: 0.0, y: 0.0, z: 0.0, w: 1.0 }
        ]
    };

    pub const ZERO: Self = Mat4 {
        e: [Vec4::ZERO, Vec4::ZERO, Vec4::ZERO, Vec4::ZERO],
    };

    pub const FLIP_X: Self = Mat4 {
        e: [
            Vec4 { x:-1.0, y:0.0, z:0.0, w:0.0 },
            Vec4 { x: 0.0, y:1.0, z:0.0, w:0.0 },
            Vec4 { x: 0.0, y:0.0, z:1.0, w:0.0 },
            Vec4 { x: 0.0, y:0.0, z:0.0, w:1.0 }
        ]
    };

    pub const FLIP_Y: Self = Mat4 {
        e: [
            Vec4 { x:1.0, y:0.0, z:0.0, w:0.0 },
            Vec4 { x:0.0, y:-1.0, z:0.0, w:0.0 },
            Vec4 { x:0.0, y:0.0, z:1.0, w:0.0 },
            Vec4 { x:0.0, y:0.0, z:0.0, w:1.0 }
        ]
    };

    pub const FLIP_Z: Self = Mat4 {
        e: [
            Vec4 { x:1.0, y:0.0, z:0.0, w:0.0 },
            Vec4 { x:0.0, y:1.0, z:0.0, w:0.0 },
            Vec4 { x:0.0, y:0.0, z:-1.0, w:0.0 },
            Vec4 { x:0.0, y:0.0, z:0.0, w:1.0 }
        ]
    };

    pub const RH_TO_LH: Self = Mat4::FLIP_Z;

    pub const Y_UP_TO_Z_UP: Self = Mat4 {
        e: [
            Vec4 { x:1.0, y:0.0, z:0.0, w:0.0 },
            Vec4 { x:0.0, y:0.0, z:-1.0, w:0.0 },
            Vec4 { x:0.0, y:1.0, z:0.0, w:0.0 },
            Vec4 { x:0.0, y:0.0, z:0.0, w:1.0 }
        ]
    };
}

// Conversion Implementations

impl From<f32> for Mat4 {
    fn from(value: f32) -> Self {
        Mat4 {
            e: [
                Vec4::from(value),
                Vec4::from(value),
                Vec4::from(value),
                Vec4::from(value),
            ],
        }
    }
}

impl From<[f32; 16]> for Mat4 {
    /// Assumes column-major order (OpenGL style)
    fn from(arr: [f32; 16]) -> Self {
        Mat4 {
            e: [
                Vec4::new(arr[0], arr[1], arr[2], arr[3]),    // col 0
                Vec4::new(arr[4], arr[5], arr[6], arr[7]),    // col 1
                Vec4::new(arr[8], arr[9], arr[10], arr[11]),  // col 2
                Vec4::new(arr[12], arr[13], arr[14], arr[15]), // col 3
            ],
        }
    }
}

impl From<[[f32; 4]; 4]> for Mat4 {
    /// Assumes arr[col][row] - column-major 2D array
    fn from(arr: [[f32; 4]; 4]) -> Self {
        Mat4 {
            e: [
                Vec4::new(arr[0][0], arr[0][1], arr[0][2], arr[0][3]), // col 0
                Vec4::new(arr[1][0], arr[1][1], arr[1][2], arr[1][3]), // col 1
                Vec4::new(arr[2][0], arr[2][1], arr[2][2], arr[2][3]), // col 2
                Vec4::new(arr[3][0], arr[3][1], arr[3][2], arr[3][3]), // col 3
            ],
        }
    }
}

impl From<[Vec4; 4]> for Mat4 {
    fn from(arr: [Vec4; 4]) -> Self {
        Mat4 { e: arr }
    }
}

impl Into<[f32; 16]> for Mat4 {
    /// Outputs in column-major order (OpenGL style)
    fn into(self) -> [f32; 16] {
        [
            // Column 0
            self.e[0].x, self.e[0].y, self.e[0].z, self.e[0].w,
            // Column 1
            self.e[1].x, self.e[1].y, self.e[1].z, self.e[1].w,
            // Column 2
            self.e[2].x, self.e[2].y, self.e[2].z, self.e[2].w,
            // Column 3
            self.e[3].x, self.e[3].y, self.e[3].z, self.e[3].w,
        ]
    }
}

impl Into<[[f32; 4]; 4]> for Mat4 {
    /// Outputs as [col][row] - column-major 2D array
    fn into(self) -> [[f32; 4]; 4] {
        [
            [self.e[0].x, self.e[0].y, self.e[0].z, self.e[0].w], // col 0
            [self.e[1].x, self.e[1].y, self.e[1].z, self.e[1].w], // col 1
            [self.e[2].x, self.e[2].y, self.e[2].z, self.e[2].w], // col 2
            [self.e[3].x, self.e[3].y, self.e[3].z, self.e[3].w], // col 3
        ]
    }
}

impl Into<[Vec4; 4]> for Mat4 {
    fn into(self) -> [Vec4; 4] {
        self.e
    }
}

// Accessors
impl Mat4 {
    pub fn cols(&self) -> [Vec4; 4] {
        self.e
    }

    pub fn col(&self, i: usize) -> Vec4 {
        self.e[i]
    }

    pub fn rows(&self) -> [Vec4; 4] {
        [
            Vec4::new(self.e[0].x, self.e[1].x, self.e[2].x, self.e[3].x),
            Vec4::new(self.e[0].y, self.e[1].y, self.e[2].y, self.e[3].y),
            Vec4::new(self.e[0].z, self.e[1].z, self.e[2].z, self.e[3].z),
            Vec4::new(self.e[0].w, self.e[1].w, self.e[2].w, self.e[3].w),
        ]
    }

    pub fn row(&self, i: usize) -> Vec4 {
        match i {
            0 => Vec4::new(self.e[0].x, self.e[1].x, self.e[2].x, self.e[3].x),
            1 => Vec4::new(self.e[0].y, self.e[1].y, self.e[2].y, self.e[3].y),
            2 => Vec4::new(self.e[0].z, self.e[1].z, self.e[2].z, self.e[3].z),
            3 => Vec4::new(self.e[0].w, self.e[1].w, self.e[2].w, self.e[3].w),
            _ => panic!("Column index {} out of bounds (0..4)", i),
        }
    }
}

// Setters

impl Mat4 {
    pub fn set_col(&mut self, i: usize, col: Vec4) {
        if i < 4 {
            self.e[i] = col;
        } else {
            panic!("Col index {} out of bounds (0..4)", i);
        }
    }

    pub fn set_row(&mut self, i: usize, row: Vec4) {
        match i {
            0 => {
                self.e[0].x = row.x;
                self.e[1].x = row.y;
                self.e[2].x = row.z;
                self.e[3].x = row.w;
            }
            1 => {
                self.e[0].y = row.x;
                self.e[1].y = row.y;
                self.e[2].y = row.z;
                self.e[3].y = row.w;
            }
            2 => {
                self.e[0].z = row.x;
                self.e[1].z = row.y;
                self.e[2].z = row.z;
                self.e[3].z = row.w;
            }
            3 => {
                self.e[0].w = row.x;
                self.e[1].w = row.y;
                self.e[2].w = row.z;
                self.e[3].w = row.w;
            }
            _ => panic!("Row index {} out of bounds (0..4)", i),
        }
    }

    /// Sets matrix[row][col] = value
    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        if row < 4 && col < 4 {
            self.e[col][row] = value;  // column-major: [col][row]
        } else {
            panic!("Index out of bounds for Matrix4: ({}, {})", row, col);
        }
    }
}

impl Index<usize> for Mat4 {
    type Output = Vec4;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.e[0],
            1 => &self.e[1],
            2 => &self.e[2],
            3 => &self.e[3],
            _ => panic!("Index {} out of bounds", index),
        }
    }
}

impl IndexMut<usize> for Mat4 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.e[0],
            1 => &mut self.e[1],
            2 => &mut self.e[2],
            3 => &mut self.e[3],
            _ => panic!("Index {} out of bounds", index),
        }
    }
}

// Matrix - Matrix Operations

impl Add for Mat4 {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        Mat4 {
            e: [
                self.e[0] + other.e[0],
                self.e[1] + other.e[1],
                self.e[2] + other.e[2],
                self.e[3] + other.e[3],
            ],
        }
    }
}

impl AddAssign for Mat4 {
    fn add_assign(&mut self, other: Self) {
        self.e[0] += other.e[0];
        self.e[1] += other.e[1];
        self.e[2] += other.e[2];
        self.e[3] += other.e[3];
    }
}

impl Sub for Mat4 {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        Mat4 {
            e: [
                self.e[0] - other.e[0],
                self.e[1] - other.e[1],
                self.e[2] - other.e[2],
                self.e[3] - other.e[3],
            ],
        }
    }
}

impl SubAssign for Mat4 {
    fn sub_assign(&mut self, other: Self) {
        self.e[0] -= other.e[0];
        self.e[1] -= other.e[1];
        self.e[2] -= other.e[2];
        self.e[3] -= other.e[3];
    }
}

impl Mul<Mat4> for Mat4 {
    type Output = Self;
    fn mul(self, other: Mat4) -> Self::Output {
        Mat4::from_mul(self, other)
    }
}

impl MulAssign<Mat4> for Mat4 {
    fn mul_assign(&mut self, other: Mat4) {
        *self = Mat4::from_mul(self.clone(), other);
    }
}

impl Mul<Vec4> for Mat4 {
    type Output = Vec4;

    fn mul(self, vec: Vec4) -> Self::Output {
        let rows = self.rows();
        Vec4::new(
            rows[0].dot(vec),
            rows[1].dot(vec),
            rows[2].dot(vec),
            rows[3].dot(vec),
        )
    }
}

impl MulAssign<Mat4> for Vec4 {
    fn mul_assign(&mut self, matrix: Mat4) {
        *self = matrix * *self;
    }
}

impl Mul<Vec> for Mat4 {
    type Output = Vec;

    fn mul(self, vec: Vec) -> Self::Output {
        let vec4 = Vec4::from_vec3(vec, 1.0);
        let result = self * vec4;
        result.xyz()
    }
}

impl MulAssign<Mat4> for Vec {
    fn mul_assign(&mut self, matrix: Mat4) {
        *self = matrix * *self;
    }
}

impl Mul<f32> for Mat4 {
    type Output = Self;

    fn mul(self, scalar: f32) -> Self::Output {
        Mat4 {
            e: [
                self.e[0] * scalar,
                self.e[1] * scalar,
                self.e[2] * scalar,
                self.e[3] * scalar,
            ],
        }
    }
}

impl Mul<Mat4> for f32 {
    type Output = Mat4;

    fn mul(self, matrix: Mat4) -> Self::Output {
        matrix * self
    }
}

impl MulAssign<f32> for Mat4 {
    fn mul_assign(&mut self, scalar: f32) {
        self.e[0] *= scalar;
        self.e[1] *= scalar;
        self.e[2] *= scalar;
        self.e[3] *= scalar;
    }
}

impl Div<f32> for Mat4 {
    type Output = Self;

    fn div(self, scalar: f32) -> Self::Output {
        if scalar == 0.0 {
            panic!("Division by zero in Matrix4 division");
        }
        Mat4 {
            e: [
                self.e[0] / scalar,
                self.e[1] / scalar,
                self.e[2] / scalar,
                self.e[3] / scalar,
            ],
        }
    }
}

impl DivAssign<f32> for Mat4 {
    fn div_assign(&mut self, scalar: f32) {
        if scalar == 0.0 {
            panic!("Division by zero in Matrix4 division");
        }
        self.e[0] /= scalar;
        self.e[1] /= scalar;
        self.e[2] /= scalar;
        self.e[3] /= scalar;
    }
}

impl Div<Mat4> for Mat4 {
    type Output = Self;

    fn div(self, other: Self) -> Self::Output {
        if other.is_zero() {
            panic!("Division by zero in Matrix4 division");
        }
        self * other.inverse().unwrap()
    }
}

impl DivAssign<Mat4> for Mat4 {
    fn div_assign(&mut self, other: Self) {
        if other.is_zero() {
            panic!("Division by zero in Matrix4 division");
        }
        *self = *self * other.inverse().unwrap();
    }
}

// Matrix - Vector Operations

impl Mat4 {
    pub fn transform_point(&self, point: Vec) -> Vec {
        let vec4 = Vec4::from_vec3(point, 1.0);
        let result = *self * vec4;
        result.xyz()
    }

    pub fn transform_direction(&self, direction: Vec) -> Vec {
        let vec4 = Vec4::from_vec3(direction, 0.0);
        let result = *self * vec4;
        result.xyz()
    }
}

// Matrix unary operations

impl Neg for Mat4 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Mat4 {
            e: [
                -self.e[0],
                -self.e[1],
                -self.e[2],
                -self.e[3],
            ],
        }
    }
}

// Matrix - Operations

impl Mat4 {
    fn minor(&self, row: usize, col: usize) -> Mat3 {
        let mut minor = Mat3::from([Vec::ZERO; 3]);
        let mut minor_row = 0;
        for i in 0..4 {
            if i == row { continue; }
            let mut minor_col = 0;
            for j in 0..4 {
                if j == col { continue; }
                // Fix: access as [col][row] for column-major
                minor[minor_row][minor_col] = self[j][i];
                minor_col += 1;
            }
            minor_row += 1;
        }
        minor
    }

    fn determinant(&self) -> f32 {
        // Expand along first row (row 0)
        let row0 = self.row(0);
        row0.x * self.minor(0, 0).determinant() -
            row0.y * self.minor(0, 1).determinant() +
            row0.z * self.minor(0, 2).determinant() -
            row0.w * self.minor(0, 3).determinant()
    }

    fn cofactor_matrix(&self) -> Mat4 {
        let mut cofactor = Mat4::ZERO;
        for i in 0..4 {
            for j in 0..4 {
                let minor = self.minor(i, j);
                let sign = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
                cofactor.set(i, j, sign * minor.determinant());
            }
        }
        cofactor
    }

    pub fn adjoint(&self) -> Self {
        self.cofactor_matrix().transpose()
    }

    pub fn inverse(&self) -> Option<Self> {
        let det = self.determinant();
        if det.abs() < 1e-6 {
            None
        } else {
            Some(self.adjoint() / det)
        }
    }

    pub fn transpose(&self) -> Self {
        Mat4 {
            e: [
                Vec4::new(self.e[0].x, self.e[1].x, self.e[2].x, self.e[3].x),
                Vec4::new(self.e[0].y, self.e[1].y, self.e[2].y, self.e[3].y),
                Vec4::new(self.e[0].z, self.e[1].z, self.e[2].z, self.e[3].z),
                Vec4::new(self.e[0].w, self.e[1].w, self.e[2].w, self.e[3].w),
            ],
        }
    }

    pub fn transpose_mut(&mut self) -> &mut Self {
        let orig = *self;

        self.e[0] = Vec4::new(orig.e[0].x, orig.e[1].x, orig.e[2].x, orig.e[3].x);
        self.e[1] = Vec4::new(orig.e[0].y, orig.e[1].y, orig.e[2].y, orig.e[3].y);
        self.e[2] = Vec4::new(orig.e[0].z, orig.e[1].z, orig.e[2].z, orig.e[3].z);
        self.e[3] = Vec4::new(orig.e[0].w, orig.e[1].w, orig.e[2].w, orig.e[3].w);

        self
    }

    pub fn trace(&self) -> f32 {
        self.e[0].x + self.e[1].y + self.e[2].z + self.e[3].w
    }
}

// Matrix - Utility Operations

impl Mat4 {
    pub fn is_identity(&self) -> bool {
        *self == Mat4::IDENTITY
    }

    pub fn is_zero(&self) -> bool {
        self.e[0].is_zero() && self.e[1].is_zero() &&
            self.e[2].is_zero() && self.e[3].is_zero()
    }

    pub fn is_invertible(&self) -> bool {
        self.determinant().abs() > 1e-6
    }
}

// Matrix - Composition Helpers, chainable operations

impl Mat4 {
    pub fn translate(&self, translation: Vec) -> Self {
        *self * Mat4::translation(translation)
    }

    pub fn scale(&self, scaling: Vec) -> Self {
        *self * Mat4::scaling(scaling)
    }

    pub fn rotate_x(&self, angle: f32) -> Self {
        *self * Mat4::rotation_x(angle)
    }

    pub fn rotate_y(&self, angle: f32) -> Self {
        *self * Mat4::rotation_y(angle)
    }

    pub fn rotate_z(&self, angle: f32) -> Self {
        *self * Mat4::rotation_z(angle)
    }

    pub fn rotate(&self, axis: Vec, angle: f32) -> Self {
        *self * Mat4::rotation(axis, angle)
    }

    pub fn look_at(&self, eye: Vec, target: Vec, up: Vec) -> Self {
        *self * Mat4::from_look_at(eye, target, up)
    }

    pub fn p_project(&self, fov_y: f32, aspect: f32, near: f32, far: f32) -> Self {
        *self * Mat4::perspective(fov_y, aspect, near, far)
    }
}

#[cfg(test)]
mod tests;