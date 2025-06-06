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

use crate::math::{Vec4};
use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};
use crate::Vec;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat4 {
    e: [Vec4; 4],
}

// Constructors

impl Mat4 {
    pub fn new(
        m00: f32,
        m01: f32,
        m02: f32,
        m03: f32,
        m10: f32,
        m11: f32,
        m12: f32,
        m13: f32,
        m20: f32,
        m21: f32,
        m22: f32,
        m23: f32,
        m30: f32,
        m31: f32,
        m32: f32,
        m33: f32,
    ) -> Self {
        let v1 = Vec4::new(m00, m01, m02, m03);
        let v2 = Vec4::new(m10, m11, m12, m13);
        let v3 = Vec4::new(m20, m21, m22, m23);
        let v4 = Vec4::new(m30, m31, m32, m33);

        Mat4 {
            e: [v1, v2, v3, v4],
        }
    }

    pub fn from_rows(row1: Vec4, row2: Vec4, row3: Vec4, row4: Vec4) -> Self {
        Mat4 {
            e: [row1, row2, row3, row4],
        }
    }

    pub fn from_cols(col1: Vec4, col2: Vec4, col3: Vec4, col4: Vec4) -> Self {
        Mat4::new(
            col1.x, col2.x, col3.x, col4.x,
            col1.y, col2.y, col3.y, col4.y,
            col1.z, col2.z, col3.z, col4.z,
            col1.w, col2.w, col3.w, col4.w,
        )
    }

    pub fn from_mul(one: Mat4, other: Mat4) -> Self {
        let rows = one.rows();
        let cols = other.cols();
        Mat4 {
            e: [
                Vec4::new(
                    rows[0].dot(cols[0]),
                    rows[0].dot(cols[1]),
                    rows[0].dot(cols[2]),
                    rows[0].dot(cols[3]),
                ),
                Vec4::new(
                    rows[1].dot(cols[0]),
                    rows[1].dot(cols[1]),
                    rows[1].dot(cols[2]),
                    rows[1].dot(cols[3]),
                ),
                Vec4::new(
                    rows[2].dot(cols[0]),
                    rows[2].dot(cols[1]),
                    rows[2].dot(cols[2]),
                    rows[2].dot(cols[3]),
                ),
                Vec4::new(
                    rows[3].dot(cols[0]),
                    rows[3].dot(cols[1]),
                    rows[3].dot(cols[2]),
                    rows[3].dot(cols[3]),
                ),
            ],
        }
    }

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
    fn from(arr: [f32; 16]) -> Self {
        Mat4::new(
            arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7], arr[8], arr[9],
            arr[10], arr[11], arr[12], arr[13], arr[14], arr[15],
        )
    }
}

impl From<[[f32; 4]; 4]> for Mat4 {
    fn from(arr: [[f32; 4]; 4]) -> Self {
        Mat4::new(
            arr[0][0], arr[0][1], arr[0][2], arr[0][3], arr[1][0], arr[1][1], arr[1][2], arr[1][3],
            arr[2][0], arr[2][1], arr[2][2], arr[2][3], arr[3][0], arr[3][1], arr[3][2], arr[3][3],
        )
    }
}

impl From<[Vec4; 4]> for Mat4 {
    fn from(arr: [Vec4; 4]) -> Self {
        Mat4 { e: arr }
    }
}

impl Into<[f32; 16]> for Mat4 {
    fn into(self) -> [f32; 16] {
        [
            self.e[0].x,
            self.e[0].y,
            self.e[0].z,
            self.e[0].w,
            self.e[1].x,
            self.e[1].y,
            self.e[1].z,
            self.e[1].w,
            self.e[2].x,
            self.e[2].y,
            self.e[2].z,
            self.e[2].w,
            self.e[3].x,
            self.e[3].y,
            self.e[3].z,
            self.e[3].w,
        ]
    }
}

impl Into<[Vec4; 4]> for Mat4 {
    fn into(self) -> [Vec4; 4] {
        self.e
    }
}

impl Into<[[f32; 4]; 4]> for Mat4 {
    fn into(self) -> [[f32; 4]; 4] {
        [
            [self.e[0].x, self.e[0].y, self.e[0].z, self.e[0].w],
            [self.e[1].x, self.e[1].y, self.e[1].z, self.e[1].w],
            [self.e[2].x, self.e[2].y, self.e[2].z, self.e[2].w],
            [self.e[3].x, self.e[3].y, self.e[3].z, self.e[3].w],
        ]
    }
}

// Accessors
impl Mat4 {
    pub fn rows(&self) -> [Vec4; 4] {
        self.e
    }

    pub fn row(&self, i: usize) -> Vec4 {
        self.e[i]
    }

    pub fn cols(&self) -> [Vec4; 4] {
        [
            Vec4::new(self.e[0].x, self.e[1].x, self.e[2].x, self.e[3].x),
            Vec4::new(self.e[0].y, self.e[1].y, self.e[2].y, self.e[3].y),
            Vec4::new(self.e[0].z, self.e[1].z, self.e[2].z, self.e[3].z),
            Vec4::new(self.e[0].w, self.e[1].w, self.e[2].w, self.e[3].w),
        ]
    }

    pub fn col(&self, i: usize) -> Vec4 {
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
    pub fn set_row(&mut self, i: usize, row: Vec4) {
        if i < 4 {
            self.e[i] = row;
        } else {
            panic!("Row index {} out of bounds (0..4)", i);
        }
    }

    pub fn set_col(&mut self, i: usize, col: Vec4) {
        match i {
            0 => {
                self.e[0].x = col.x;
                self.e[1].x = col.y;
                self.e[2].x = col.z;
                self.e[3].x = col.w;
            }
            1 => {
                self.e[0].y = col.x;
                self.e[1].y = col.y;
                self.e[2].y = col.z;
                self.e[3].y = col.w;
            }
            2 => {
                self.e[0].z = col.x;
                self.e[1].z = col.y;
                self.e[2].z = col.z;
                self.e[3].z = col.w;
            }
            3 => {
                self.e[0].w = col.x;
                self.e[1].w = col.y;
                self.e[2].w = col.z;
                self.e[3].w = col.w;
            }
            _ => panic!("Column index {} out of bounds (0..4)", i),
        }
    }

    pub fn set (&mut self, i: usize, j: usize, value: f32) {
        if i < 4 && j < 4 {
            self[i][j] = value;
        } else {
            panic!("Index out of bounds for Matrix4: ({}, {})", i, j);
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
        Vec4::new(
            self.e[0].dot(vec),
            self.e[1].dot(vec),
            self.e[2].dot(vec),
            self.e[3].dot(vec),
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

struct Mat3 {
    e: [Vec; 3],
}

impl Mat3 {
    pub fn determinant(&self) -> f32 {
        self.e[0].x * (self.e[1].y * self.e[2].z - self.e[1].z * self.e[2].y) -
            self.e[0].y * (self.e[1].x * self.e[2].z - self.e[1].z * self.e[2].x) +
            self.e[0].z * (self.e[1].x * self.e[2].y - self.e[1].y * self.e[2].x)
    }
}


impl Mat4 {
    fn minor(&self, row: usize, col: usize) -> Mat3 {
        let mut minor = Mat3 { e: [Vec::ZERO; 3] };
        let mut minor_row = 0;
        for i in 0..4 {
            if i == row { continue; }
            let mut minor_col = 0;
            for j in 0..4 {
                if j == col { continue; }
                minor.e[minor_row][minor_col] = self[i][j];
                minor_col += 1;
            }
            minor_row += 1;
        }
        minor
    }

    fn determinant(&self) -> f32 {
        self.e[0].x * self.minor(0, 0).determinant() -
            self.e[0].y * self.minor(0, 1).determinant() +
            self.e[0].z * self.minor(0, 2).determinant() -
            self.e[0].w * self.minor(0, 3).determinant()
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
mod tests {
    use super::*;
    use std::f32::consts::PI;

    // Helper function for floating point comparisons
    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-6
    }

    fn approx_eq_mat4(a: Mat4, b: Mat4) -> bool {
        for i in 0..4 {
            for j in 0..4 {
                if !approx_eq(a[i][j], b[i][j]) {
                    return false;
                }
            }
        }
        true
    }

    #[test]
    fn test_mat4_new() {
        let m = Mat4::new(
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        );

        assert_eq!(m[0][0], 1.0);
        assert_eq!(m[0][3], 4.0);
        assert_eq!(m[3][3], 16.0);
    }

    #[test]
    fn test_mat4_from_rows() {
        let row1 = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let row2 = Vec4::new(5.0, 6.0, 7.0, 8.0);
        let row3 = Vec4::new(9.0, 10.0, 11.0, 12.0);
        let row4 = Vec4::new(13.0, 14.0, 15.0, 16.0);

        let m = Mat4::from_rows(row1, row2, row3, row4);

        assert_eq!(m.row(0), row1);
        assert_eq!(m.row(1), row2);
        assert_eq!(m.row(2), row3);
        assert_eq!(m.row(3), row4);
    }

    #[test]
    fn test_mat4_from_cols() {
        let col1 = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let col2 = Vec4::new(5.0, 6.0, 7.0, 8.0);
        let col3 = Vec4::new(9.0, 10.0, 11.0, 12.0);
        let col4 = Vec4::new(13.0, 14.0, 15.0, 16.0);

        let m = Mat4::from_cols(col1, col2, col3, col4);

        assert_eq!(m.col(0), col1);
        assert_eq!(m.col(1), col2);
        assert_eq!(m.col(2), col3);
        assert_eq!(m.col(3), col4);
    }

    #[test]
    fn test_mat4_constants() {
        assert_eq!(Mat4::IDENTITY[0][0], 1.0);
        assert_eq!(Mat4::IDENTITY[1][1], 1.0);
        assert_eq!(Mat4::IDENTITY[2][2], 1.0);
        assert_eq!(Mat4::IDENTITY[3][3], 1.0);
        assert_eq!(Mat4::IDENTITY[0][1], 0.0);

        assert!(Mat4::ZERO.is_zero());

        assert_eq!(Mat4::FLIP_X[0][0], -1.0);
        assert_eq!(Mat4::FLIP_Y[1][1], -1.0);
        assert_eq!(Mat4::FLIP_Z[2][2], -1.0);
    }

    #[test]
    fn test_mat4_translation() {
        let t = Vec::new(5.0, 10.0, 15.0);
        let m = Mat4::translation(t);

        assert_eq!(m[3][0], 5.0);
        assert_eq!(m[3][1], 10.0);
        assert_eq!(m[3][2], 15.0);
        assert_eq!(m[3][3], 1.0);

        // Test translation of a point
        let point = Vec::new(1.0, 2.0, 3.0);
        let transformed = m.transform_point(point);
        assert_eq!(transformed, Vec::new(6.0, 12.0, 18.0));
    }

    #[test]
    fn test_mat4_scaling() {
        let s = Vec::new(2.0, 3.0, 4.0);
        let m = Mat4::scaling(s);

        assert_eq!(m[0][0], 2.0);
        assert_eq!(m[1][1], 3.0);
        assert_eq!(m[2][2], 4.0);

        // Test scaling of a point
        let point = Vec::new(1.0, 2.0, 3.0);
        let transformed = m.transform_point(point);
        assert_eq!(transformed, Vec::new(2.0, 6.0, 12.0));
    }

    #[test]
    fn test_mat4_rotation_x() {
        let m = Mat4::rotation_x(PI / 2.0);
        let point = Vec::new(0.0, 1.0, 0.0);
        let transformed = m.transform_point(point);

        assert!(approx_eq(transformed.x, 0.0));
        assert!(approx_eq(transformed.y, 0.0));
        assert!(approx_eq(transformed.z, 1.0));
    }

    #[test]
    fn test_mat4_rotation_y() {
        let m = Mat4::rotation_y(PI / 2.0);
        let point = Vec::new(1.0, 0.0, 0.0);
        let transformed = m.transform_point(point);

        assert!(approx_eq(transformed.x, 0.0));
        assert!(approx_eq(transformed.y, 0.0));
        assert!(approx_eq(transformed.z, -1.0));
    }

    #[test]
    fn test_mat4_rotation_z() {
        let m = Mat4::rotation_z(PI / 2.0);
        let point = Vec::new(1.0, 0.0, 0.0);
        let transformed = m.transform_point(point);

        assert!(approx_eq(transformed.x, 0.0));
        assert!(approx_eq(transformed.y, 1.0));
        assert!(approx_eq(transformed.z, 0.0));
    }

    #[test]
    fn test_mat4_rotation_arbitrary_axis() {
        let axis = Vec::new(0.0, 0.0, 1.0); // Z axis
        let m = Mat4::rotation(axis, PI / 2.0);
        let point = Vec::new(1.0, 0.0, 0.0);
        let transformed = m.transform_point(point);

        assert!(approx_eq(transformed.x, 0.0));
        assert!(approx_eq(transformed.y, 1.0));
        assert!(approx_eq(transformed.z, 0.0));
    }

    #[test]
    fn test_mat4_look_at() {
        let eye = Vec::new(0.0, 0.0, 5.0);
        let target = Vec::new(0.0, 0.0, 0.0);
        let up = Vec::new(0.0, 1.0, 0.0);

        let m = Mat4::from_look_at(eye, target, up);

        // The z-axis should point from target to eye (normalized)
        let z_axis = m.col(2).xyz();
        let expected_z = Vec::new(0.0, 0.0, 1.0);
        assert!(approx_eq(z_axis.x, expected_z.x));
        assert!(approx_eq(z_axis.y, expected_z.y));
        assert!(approx_eq(z_axis.z, expected_z.z));
    }

    #[test]
    fn test_mat4_perspective() {
        let fov = PI / 4.0; // 45 degrees
        let aspect = 16.0 / 9.0;
        let near = 0.1;
        let far = 100.0;

        let m = Mat4::perspective(fov, aspect, near, far);

        // Check that w coordinate becomes -z (perspective divide)
        assert!(approx_eq(m[3][2], -1.0));
        assert!(approx_eq(m[3][3], 0.0));
    }

    #[test]
    #[should_panic(expected = "Field of view must be positive")]
    fn test_mat4_perspective_invalid_fov() {
        Mat4::perspective(-1.0, 1.0, 0.1, 100.0);
    }

    #[test]
    #[should_panic(expected = "Aspect ratio must be positive")]
    fn test_mat4_perspective_invalid_aspect() {
        Mat4::perspective(PI / 4.0, -1.0, 0.1, 100.0);
    }

    #[test]
    #[should_panic(expected = "Near plane must be positive")]
    fn test_mat4_perspective_invalid_near() {
        Mat4::perspective(PI / 4.0, 1.0, -0.1, 100.0);
    }

    #[test]
    #[should_panic(expected = "Far plane must be greater than near plane")]
    fn test_mat4_perspective_invalid_far() {
        Mat4::perspective(PI / 4.0, 1.0, 100.0, 50.0);
    }

    #[test]
    fn test_mat4_conversions() {
        let arr = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
        let m: Mat4 = arr.into();
        let back: [f32; 16] = m.into();

        for i in 0..16 {
            assert_eq!(arr[i], back[i]);
        }
    }

    #[test]
    fn test_mat4_from_scalar() {
        let m: Mat4 = 5.0.into();
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(m[i][j], 5.0);
            }
        }
    }

    #[test]
    fn test_mat4_indexing() {
        let mut m = Mat4::IDENTITY;

        // Test read access
        assert_eq!(m[0][0], 1.0);
        assert_eq!(m[1][1], 1.0);

        // Test write access
        m[0][1] = 5.0;
        assert_eq!(m[0][1], 5.0);
    }

    #[test]
    #[should_panic]
    fn test_mat4_index_out_of_bounds() {
        let m = Mat4::IDENTITY;
        let _ = m[4];
    }

    #[test]
    fn test_mat4_rows_and_cols() {
        let m = Mat4::new(
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        );

        // Test row access
        assert_eq!(m.row(0), Vec4::new(1.0, 2.0, 3.0, 4.0));
        assert_eq!(m.row(3), Vec4::new(13.0, 14.0, 15.0, 16.0));

        // Test column access
        assert_eq!(m.col(0), Vec4::new(1.0, 5.0, 9.0, 13.0));
        assert_eq!(m.col(3), Vec4::new(4.0, 8.0, 12.0, 16.0));
    }

    #[test]
    fn test_mat4_set_row_and_col() {
        let mut m = Mat4::ZERO;

        let new_row = Vec4::new(1.0, 2.0, 3.0, 4.0);
        m.set_row(0, new_row);
        assert_eq!(m.row(0), new_row);

        let new_col = Vec4::new(5.0, 6.0, 7.0, 8.0);
        m.set_col(1, new_col);
        assert_eq!(m.col(1), new_col);
    }

    #[test]
    fn test_mat4_set_element() {
        let mut m = Mat4::ZERO;
        m.set(2, 3, 42.0);
        assert_eq!(m[2][3], 42.0);
    }

    #[test]
    fn test_mat4_addition() {
        let a = Mat4::IDENTITY;
        let b = Mat4::IDENTITY;
        let c = a + b;

        for i in 0..4 {
            assert_eq!(c[i][i], 2.0);
        }
    }

    #[test]
    fn test_mat4_addition_assign() {
        let mut a = Mat4::IDENTITY;
        a += Mat4::IDENTITY;

        for i in 0..4 {
            assert_eq!(a[i][i], 2.0);
        }
    }

    #[test]
    fn test_mat4_subtraction() {
        let a = Mat4::IDENTITY;
        let b = Mat4::IDENTITY;
        let c = a - b;

        assert!(c.is_zero());
    }

    #[test]
    fn test_mat4_multiplication_identity() {
        let a = Mat4::translation(Vec::new(1.0, 2.0, 3.0));
        let result = a * Mat4::IDENTITY;

        assert_eq!(a, result);
    }

    #[test]
    fn test_mat4_multiplication_transforms() {
        let translate = Mat4::translation(Vec::new(5.0, 0.0, 0.0));
        let scale = Mat4::scaling(Vec::new(2.0, 2.0, 2.0));

        // Scale then translate
        let combined = translate * scale;
        let point = Vec::new(1.0, 1.0, 1.0);
        let result = combined.transform_point(point);

        assert_eq!(result, Vec::new(7.0, 2.0, 2.0)); // scaled to (2,2,2) then translated by (5,0,0)
    }

    #[test]
    fn test_mat4_vector_multiplication() {
        let m = Mat4::translation(Vec::new(5.0, 10.0, 15.0));
        let v = Vec4::new(1.0, 2.0, 3.0, 1.0);
        let result = m * v;

        assert_eq!(result, Vec4::new(6.0, 12.0, 18.0, 1.0));
    }

    #[test]
    fn test_mat4_scalar_multiplication() {
        let m = Mat4::IDENTITY;
        let scaled = m * 3.0;

        for i in 0..4 {
            assert_eq!(scaled[i][i], 3.0);
        }

        // Test commutative property
        let scaled2 = 3.0 * m;
        assert_eq!(scaled, scaled2);
    }

    #[test]
    fn test_mat4_scalar_division() {
        let m = Mat4::IDENTITY * 6.0;
        let divided = m / 2.0;

        for i in 0..4 {
            assert_eq!(divided[i][i], 3.0);
        }
    }

    #[test]
    #[should_panic(expected = "Division by zero")]
    fn test_mat4_division_by_zero() {
        let m = Mat4::IDENTITY;
        let _ = m / 0.0;
    }

    #[test]
    fn test_mat4_transform_point_vs_direction() {
        let translate = Mat4::translation(Vec::new(5.0, 0.0, 0.0));

        let point = Vec::new(1.0, 0.0, 0.0);
        let direction = Vec::new(1.0, 0.0, 0.0);

        let transformed_point = translate.transform_point(point);
        let transformed_direction = translate.transform_direction(direction);

        // Point should be translated
        assert_eq!(transformed_point, Vec::new(6.0, 0.0, 0.0));
        // Direction should not be affected by translation
        assert_eq!(transformed_direction, Vec::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn test_mat4_negation() {
        let m = Mat4::IDENTITY;
        let neg = -m;

        for i in 0..4 {
            assert_eq!(neg[i][i], -1.0);
        }
    }

    #[test]
    fn test_mat4_determinant() {
        // Test identity determinant
        assert!(approx_eq(Mat4::IDENTITY.determinant(), 1.0));

        // Test scaling matrix determinant
        let scale = Mat4::scaling(Vec::new(2.0, 3.0, 4.0));
        assert!(approx_eq(scale.determinant(), 24.0)); // 2 * 3 * 4 * 1
    }

    #[test]
    fn test_mat4_transpose() {
        let m = Mat4::new(
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        );

        let transposed = m.transpose();

        assert_eq!(transposed[0][1], 5.0);
        assert_eq!(transposed[1][0], 2.0);
        assert_eq!(transposed[2][3], 16.0);
        assert_eq!(transposed[3][2], 12.0);
    }

    #[test]
    fn test_mat4_transpose_mut() {
        let mut m = Mat4::new(
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        );

        m.transpose_mut();

        assert_eq!(m[0][1], 5.0);
        assert_eq!(m[1][0], 2.0);
    }

    #[test]
    fn test_mat4_inverse() {
        // Test identity inverse
        let inv = Mat4::IDENTITY.inverse().unwrap();
        assert!(approx_eq_mat4(inv, Mat4::IDENTITY));

        // Test translation inverse
        let translate = Mat4::translation(Vec::new(5.0, 10.0, 15.0));
        let inv_translate = translate.inverse().unwrap();
        let should_be_identity = translate * inv_translate;
        assert!(approx_eq_mat4(should_be_identity, Mat4::IDENTITY));
    }

    #[test]
    fn test_mat4_inverse_non_invertible() {
        let non_invertible = Mat4::ZERO;
        assert!(non_invertible.inverse().is_none());
    }

    #[test]
    fn test_mat4_trace() {
        let m = Mat4::new(
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        );

        assert_eq!(m.trace(), 34.0); // 1 + 6 + 11 + 16
    }

    #[test]
    fn test_mat4_utility_checks() {
        assert!(Mat4::IDENTITY.is_identity());
        assert!(!Mat4::ZERO.is_identity());

        assert!(Mat4::ZERO.is_zero());
        assert!(!Mat4::IDENTITY.is_zero());

        assert!(Mat4::IDENTITY.is_invertible());
        assert!(!Mat4::ZERO.is_invertible());
    }

    #[test]
    fn test_mat4_chainable_operations() {
        let result = Mat4::IDENTITY
            .translate(Vec::new(5.0, 0.0, 0.0))
            .scale(Vec::new(2.0, 2.0, 2.0))
            .rotate_z(PI / 2.0);

        let point = Vec::new(1.0, 0.0, 0.0);
        let transformed = result.transform_point(point);

        // After scaling by 2: (2, 0, 0)
        // After rotating 90° around Z: (0, 2, 0)  
        // After translating by (5, 0, 0): (5, 2, 0)
        assert!(approx_eq(transformed.x, 5.0));
        assert!(approx_eq(transformed.y, 2.0));
        assert!(approx_eq(transformed.z, 0.0));
    }

    #[test]
    fn test_mat4_composition_order() {
        // Matrix multiplication is not commutative
        let translate = Mat4::translation(Vec::new(5.0, 0.0, 0.0));
        let scale = Mat4::scaling(Vec::new(2.0, 1.0, 1.0));

        let translate_then_scale = scale * translate;
        let scale_then_translate = translate * scale;

        let point = Vec::new(1.0, 0.0, 0.0);

        let result1 = translate_then_scale.transform_point(point);
        let result2 = scale_then_translate.transform_point(point);

        // Should produce different results
        assert!(!approx_eq(result1.x, result2.x));
    }

    #[test]
    fn test_mat4_from_mul() {
        let a = Mat4::translation(Vec::new(1.0, 2.0, 3.0));
        let b = Mat4::scaling(Vec::new(2.0, 2.0, 2.0));

        let c1 = Mat4::from_mul(a, b);
        let c2 = a * b;

        assert_eq!(c1, c2);
    }

    #[test]
    fn test_mat4_column_access_bounds() {
        let m = Mat4::IDENTITY;

        // Valid access
        let _ = m.col(0);
        let _ = m.col(3);

        // Should panic for invalid access
        std::panic::catch_unwind(|| m.col(4)).expect_err("Should panic for col index 4");
    }

    #[test]
    fn test_mat4_row_column_setters_bounds() {
        let mut m = Mat4::IDENTITY;

        // Should panic for out of bounds
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            m.set_row(4, Vec4::ZERO);
        })).expect_err("Should panic for row index 4");

        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            m.set_col(4, Vec4::ZERO);
        })).expect_err("Should panic for col index 4");
    }

    #[test]
    fn test_mat4_comprehensive_transform_chain() {
        // Test a realistic transform chain: scale -> rotate -> translate
        let point = Vec::new(1.0, 0.0, 0.0);

        let transform = Mat4::IDENTITY
            .scale(Vec::new(2.0, 2.0, 2.0))           // Scale to (2, 0, 0)
            .rotate_z(PI / 2.0)                        // Rotate to (0, 2, 0)
            .translate(Vec::new(10.0, 20.0, 30.0));    // Translate to (10, 22, 30)

        let result = transform.transform_point(point);

        assert!(approx_eq(result.x, 10.0));
        assert!(approx_eq(result.y, 22.0));
        assert!(approx_eq(result.z, 30.0));
    }
}