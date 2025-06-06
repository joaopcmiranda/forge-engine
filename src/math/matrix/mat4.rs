/*
Constructors

from_translation(Vector) - Translation matrix
from_rotation_x/y/z(f32) - Single-axis rotation matrices
from_rotation(Vector, f32) - Arbitrary axis rotation
from_scale(Vector) - Scale matrix
from_quaternion(Quaternion) - When you add quaternions later
look_at(eye, target, up) - View matrix
perspective(fov, aspect, near, far) - Projection matrix
orthographic(left, right, bottom, top, near, far) - Ortho projection

Decomposition

extract_translation() - Get translation component
extract_scale() - Get scale component (approximate)
extract_rotation() - Get rotation component (future quaternion)

Utility Operations

lerp(other, t) - Linear interpolation

Composition Helpers

translate(Vector) - Multiply by translation
rotate_x/y/z(f32) - Multiply by rotation
scale(Vector) - Multiply by scale
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
            col1.x, col2.x, col3.x, col4.x,  // Row 0: x components of each column
            col1.y, col2.y, col3.y, col4.y,  // Row 1: y components of each column
            col1.z, col2.z, col3.z, col4.z,  // Row 2: z components of each column
            col1.w, col2.w, col3.w, col4.w,  // Row 3: w components of each column
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
}

// Constants

impl Mat4 {
    pub const IDENTITY: Self = Mat4 {
        e: [Vec4::X, Vec4::Y, Vec4::Z, Vec4::W],
    };

    pub const ZERO: Self = Mat4 {
        e: [Vec4::ZERO, Vec4::ZERO, Vec4::ZERO, Vec4::ZERO],
    };

    // Standard transformations
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
            Vec4 { x:0.0, y:0.0, z:1.0, w:0.0 },
            Vec4 { x:0.0, y:-1.0, z:0.0, w:0.0 },
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

    pub  fn set (&mut self, i: usize, j: usize, value: f32) {
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
    pub  fn determinant(&self) -> f32 {
        // Calculate the determinant of the 3x3 matrix
        self.e[0].x * (self.e[1].y * self.e[2].z - self.e[1].z * self.e[2].y) -
        self.e[0].y * (self.e[1].x * self.e[2].z - self.e[1].z * self.e[2].x) +
        self.e[0].z * (self.e[1].x * self.e[2].y - self.e[1].y * self.e[2].x)
    }
}


impl Mat4 {   
    fn minor(&self, row: usize, col: usize) -> Mat3 {
        // Calculate the minor of the matrix by removing the specified row and column
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
        // Calculate the determinant using the first row and minors
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
    
    pub  fn inverse(&self) -> Option<Self> {
        let det = self.determinant();
        if det.abs() < 1e-6 {
            None // Matrix is not invertible
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
    
    pub  fn trace(&self) -> f32 {
        self.e[0].x + self.e[1].y + self.e[2].z + self.e[3].w
    }
}

// Matrix - Utility Operations

impl Mat4 {
    pub fn is_identity(&self) -> bool {
        self.e[0] == Vec4::X && self.e[1] == Vec4::Y &&
        self.e[2] == Vec4::Z && self.e[3] == Vec4::W
    }

    pub fn is_zero(&self) -> bool {
        self.e[0].is_zero() && self.e[1].is_zero() &&
        self.e[2].is_zero() && self.e[3].is_zero()
    }
    
    pub fn is_invertible(&self) -> bool {
        self.determinant().abs() > 1e-6
    }
}
