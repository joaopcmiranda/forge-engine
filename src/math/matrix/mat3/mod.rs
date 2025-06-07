use crate::Vec;
use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat3 {
    e: [Vec; 3], // Column-major storage
}

// Constructors
impl Mat3 {
    pub fn new() -> Self {
        Mat3::IDENTITY
    }

    pub fn from_rows(row1: Vec, row2: Vec, row3: Vec) -> Self {
        Mat3 {
            e: [
                Vec::new(row1.x, row2.x, row3.x), // col 0
                Vec::new(row1.y, row2.y, row3.y), // col 1
                Vec::new(row1.z, row2.z, row3.z), // col 2
            ],
        }
    }

    pub fn from_cols(col1: Vec, col2: Vec, col3: Vec) -> Self {
        Mat3 {
            e: [col1, col2, col3],
        }
    }

    pub fn from_mul(one: Mat3, other: Mat3) -> Self {
        let rows = one.rows();
        let cols = other.cols();
        Mat3 {
            e: [
                // Column 0: each row of 'one' dotted with column 0 of 'other'
                Vec::new(
                    rows[0].dot(cols[0]),
                    rows[1].dot(cols[0]),
                    rows[2].dot(cols[0]),
                ),
                // Column 1: each row of 'one' dotted with column 1 of 'other'
                Vec::new(
                    rows[0].dot(cols[1]),
                    rows[1].dot(cols[1]),
                    rows[2].dot(cols[1]),
                ),
                // Column 2: each row of 'one' dotted with column 2 of 'other'
                Vec::new(
                    rows[0].dot(cols[2]),
                    rows[1].dot(cols[2]),
                    rows[2].dot(cols[2]),
                ),
            ],
        }
    }
}

// Transformation Constructors
impl Mat3 {
    pub fn scaling(scaling: Vec) -> Self {
        Mat3 {
            e: [
                Vec::new(scaling.x, 0.0, 0.0),
                Vec::new(0.0, scaling.y, 0.0),
                Vec::new(0.0, 0.0, scaling.z),
            ],
        }
    }

    pub fn rotation_x(angle: f32) -> Self {
        let cos = angle.cos();
        let sin = angle.sin();
        Mat3 {
            e: [
                Vec::new(1.0, 0.0, 0.0),
                Vec::new(0.0, cos, sin),
                Vec::new(0.0, -sin, cos),
            ],
        }
    }

    pub fn rotation_y(angle: f32) -> Self {
        let cos = angle.cos();
        let sin = angle.sin();
        Mat3 {
            e: [
                Vec::new(cos, 0.0, -sin),
                Vec::new(0.0, 1.0, 0.0),
                Vec::new(sin, 0.0, cos),
            ],
        }
    }

    pub fn rotation_z(angle: f32) -> Self {
        let cos = angle.cos();
        let sin = angle.sin();
        Mat3 {
            e: [
                Vec::new(cos, sin, 0.0),
                Vec::new(-sin, cos, 0.0),
                Vec::new(0.0, 0.0, 1.0),
            ],
        }
    }

    pub fn rotation(axis: Vec, angle: f32) -> Self {
        let axis = axis.normalized();
        let cos = angle.cos();
        let sin = angle.sin();
        let one_minus_cos = 1.0 - cos;

        let x = axis.x;
        let y = axis.y;
        let z = axis.z;

        Mat3 {
            e: [
                Vec::new(
                    cos + x * x * one_minus_cos,
                    x * y * one_minus_cos + z * sin,
                    x * z * one_minus_cos - y * sin,
                ),
                Vec::new(
                    y * x * one_minus_cos - z * sin,
                    cos + y * y * one_minus_cos,
                    y * z * one_minus_cos + x * sin,
                ),
                Vec::new(
                    z * x * one_minus_cos + y * sin,
                    z * y * one_minus_cos - x * sin,
                    cos + z * z * one_minus_cos,
                ),
            ],
        }
    }
}

// Constants
impl Mat3 {
    pub const IDENTITY: Self = Mat3 {
        e: [
            Vec { x: 1.0, y: 0.0, z: 0.0 },
            Vec { x: 0.0, y: 1.0, z: 0.0 },
            Vec { x: 0.0, y: 0.0, z: 1.0 },
        ]
    };

    pub const ZERO: Self = Mat3 {
        e: [Vec::ZERO, Vec::ZERO, Vec::ZERO],
    };

    pub const FLIP_X: Self = Mat3 {
        e: [
            Vec { x: -1.0, y: 0.0, z: 0.0 },
            Vec { x: 0.0, y: 1.0, z: 0.0 },
            Vec { x: 0.0, y: 0.0, z: 1.0 },
        ]
    };

    pub const FLIP_Y: Self = Mat3 {
        e: [
            Vec { x: 1.0, y: 0.0, z: 0.0 },
            Vec { x: 0.0, y: -1.0, z: 0.0 },
            Vec { x: 0.0, y: 0.0, z: 1.0 },
        ]
    };

    pub const FLIP_Z: Self = Mat3 {
        e: [
            Vec { x: 1.0, y: 0.0, z: 0.0 },
            Vec { x: 0.0, y: 1.0, z: 0.0 },
            Vec { x: 0.0, y: 0.0, z: -1.0 },
        ]
    };
}

// Conversion Implementations
impl From<f32> for Mat3 {
    fn from(value: f32) -> Self {
        Mat3 {
            e: [
                Vec::new(value, value, value),
                Vec::new(value, value, value),
                Vec::new(value, value, value),
            ],
        }
    }
}

impl From<[f32; 9]> for Mat3 {
    /// Assumes column-major order
    fn from(arr: [f32; 9]) -> Self {
        Mat3 {
            e: [
                Vec::new(arr[0], arr[1], arr[2]),    // col 0
                Vec::new(arr[3], arr[4], arr[5]),    // col 1
                Vec::new(arr[6], arr[7], arr[8]),    // col 2
            ],
        }
    }
}

impl From<[[f32; 3]; 3]> for Mat3 {
    /// Assumes arr[col][row] - column-major 2D array
    fn from(arr: [[f32; 3]; 3]) -> Self {
        Mat3 {
            e: [
                Vec::new(arr[0][0], arr[0][1], arr[0][2]), // col 0
                Vec::new(arr[1][0], arr[1][1], arr[1][2]), // col 1
                Vec::new(arr[2][0], arr[2][1], arr[2][2]), // col 2
            ],
        }
    }
}

impl From<[Vec; 3]> for Mat3 {
    fn from(arr: [Vec; 3]) -> Self {
        Mat3 { e: arr }
    }
}

impl Into<[f32; 9]> for Mat3 {
    /// Outputs in column-major order
    fn into(self) -> [f32; 9] {
        [
            // Column 0
            self.e[0].x, self.e[0].y, self.e[0].z,
            // Column 1
            self.e[1].x, self.e[1].y, self.e[1].z,
            // Column 2
            self.e[2].x, self.e[2].y, self.e[2].z,
        ]
    }
}

impl Into<[[f32; 3]; 3]> for Mat3 {
    /// Outputs as [col][row] - column-major 2D array
    fn into(self) -> [[f32; 3]; 3] {
        [
            [self.e[0].x, self.e[0].y, self.e[0].z], // col 0
            [self.e[1].x, self.e[1].y, self.e[1].z], // col 1
            [self.e[2].x, self.e[2].y, self.e[2].z], // col 2
        ]
    }
}

impl Into<[Vec; 3]> for Mat3 {
    fn into(self) -> [Vec; 3] {
        self.e
    }
}

// Accessors
impl Mat3 {
    pub fn cols(&self) -> [Vec; 3] {
        self.e
    }

    pub fn col(&self, i: usize) -> Vec {
        self.e[i]
    }

    pub fn rows(&self) -> [Vec; 3] {
        [
            Vec::new(self.e[0].x, self.e[1].x, self.e[2].x),
            Vec::new(self.e[0].y, self.e[1].y, self.e[2].y),
            Vec::new(self.e[0].z, self.e[1].z, self.e[2].z),
        ]
    }

    pub fn row(&self, i: usize) -> Vec {
        match i {
            0 => Vec::new(self.e[0].x, self.e[1].x, self.e[2].x),
            1 => Vec::new(self.e[0].y, self.e[1].y, self.e[2].y),
            2 => Vec::new(self.e[0].z, self.e[1].z, self.e[2].z),
            _ => panic!("Row index {} out of bounds (0..3)", i),
        }
    }
}

// Setters
impl Mat3 {
    pub fn set_col(&mut self, i: usize, col: Vec) {
        if i < 3 {
            self.e[i] = col;
        } else {
            panic!("Col index {} out of bounds (0..3)", i);
        }
    }

    pub fn set_row(&mut self, i: usize, row: Vec) {
        match i {
            0 => {
                self.e[0].x = row.x;
                self.e[1].x = row.y;
                self.e[2].x = row.z;
            }
            1 => {
                self.e[0].y = row.x;
                self.e[1].y = row.y;
                self.e[2].y = row.z;
            }
            2 => {
                self.e[0].z = row.x;
                self.e[1].z = row.y;
                self.e[2].z = row.z;
            }
            _ => panic!("Row index {} out of bounds (0..3)", i),
        }
    }

    /// Sets matrix[row][col] = value
    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        if row < 3 && col < 3 {
            self.e[col][row] = value;  // column-major: [col][row]
        } else {
            panic!("Index out of bounds for Mat3: ({}, {})", row, col);
        }
    }
}

impl Index<usize> for Mat3 {
    type Output = Vec;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.e[0],
            1 => &self.e[1],
            2 => &self.e[2],
            _ => panic!("Index {} out of bounds", index),
        }
    }
}

impl IndexMut<usize> for Mat3 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.e[0],
            1 => &mut self.e[1],
            2 => &mut self.e[2],
            _ => panic!("Index {} out of bounds", index),
        }
    }
}

// Matrix - Matrix Operations
impl Add for Mat3 {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        Mat3 {
            e: [
                self.e[0] + other.e[0],
                self.e[1] + other.e[1],
                self.e[2] + other.e[2],
            ],
        }
    }
}

impl AddAssign for Mat3 {
    fn add_assign(&mut self, other: Self) {
        self.e[0] += other.e[0];
        self.e[1] += other.e[1];
        self.e[2] += other.e[2];
    }
}

impl Sub for Mat3 {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        Mat3 {
            e: [
                self.e[0] - other.e[0],
                self.e[1] - other.e[1],
                self.e[2] - other.e[2],
            ],
        }
    }
}

impl SubAssign for Mat3 {
    fn sub_assign(&mut self, other: Self) {
        self.e[0] -= other.e[0];
        self.e[1] -= other.e[1];
        self.e[2] -= other.e[2];
    }
}

impl Mul<Mat3> for Mat3 {
    type Output = Self;
    fn mul(self, other: Mat3) -> Self::Output {
        Mat3::from_mul(self, other)
    }
}

impl MulAssign<Mat3> for Mat3 {
    fn mul_assign(&mut self, other: Mat3) {
        *self = Mat3::from_mul(*self, other);
    }
}

impl Mul<Vec> for Mat3 {
    type Output = Vec;

    fn mul(self, vec: Vec) -> Self::Output {
        let rows = self.rows();
        Vec::new(
            rows[0].dot(vec),
            rows[1].dot(vec),
            rows[2].dot(vec),
        )
    }
}

impl MulAssign<Mat3> for Vec {
    fn mul_assign(&mut self, matrix: Mat3) {
        *self = matrix * *self;
    }
}

impl Mul<f32> for Mat3 {
    type Output = Self;

    fn mul(self, scalar: f32) -> Self::Output {
        Mat3 {
            e: [
                self.e[0] * scalar,
                self.e[1] * scalar,
                self.e[2] * scalar,
            ],
        }
    }
}

impl Mul<Mat3> for f32 {
    type Output = Mat3;

    fn mul(self, matrix: Mat3) -> Self::Output {
        matrix * self
    }
}

impl MulAssign<f32> for Mat3 {
    fn mul_assign(&mut self, scalar: f32) {
        self.e[0] *= scalar;
        self.e[1] *= scalar;
        self.e[2] *= scalar;
    }
}

impl Div<f32> for Mat3 {
    type Output = Self;

    fn div(self, scalar: f32) -> Self::Output {
        if scalar == 0.0 {
            panic!("Division by zero in Mat3 division");
        }
        Mat3 {
            e: [
                self.e[0] / scalar,
                self.e[1] / scalar,
                self.e[2] / scalar,
            ],
        }
    }
}

impl DivAssign<f32> for Mat3 {
    fn div_assign(&mut self, scalar: f32) {
        if scalar == 0.0 {
            panic!("Division by zero in Mat3 division");
        }
        self.e[0] /= scalar;
        self.e[1] /= scalar;
        self.e[2] /= scalar;
    }
}

impl Div<Mat3> for Mat3 {
    type Output = Self;

    fn div(self, other: Self) -> Self::Output {
        if other.is_zero() {
            panic!("Division by zero in Mat3 division");
        }
        self * other.inverse().unwrap()
    }
}

impl DivAssign<Mat3> for Mat3 {
    fn div_assign(&mut self, other: Self) {
        if other.is_zero() {
            panic!("Division by zero in Mat3 division");
        }
        *self = *self * other.inverse().unwrap();
    }
}

impl Neg for Mat3 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Mat3 {
            e: [
                -self.e[0],
                -self.e[1],
                -self.e[2],
            ],
        }
    }
}

// Matrix Operations
impl Mat3 {
    pub fn determinant(&self) -> f32 {
        self.e[0].x * (self.e[1].y * self.e[2].z - self.e[1].z * self.e[2].y)
            - self.e[0].y * (self.e[1].x * self.e[2].z - self.e[1].z * self.e[2].x)
            + self.e[0].z * (self.e[1].x * self.e[2].y - self.e[1].y * self.e[2].x)
    }

    pub fn minor(&self, row: usize, col: usize) -> f32 {
        // Create 2x2 matrix by removing row and col
        let mut elements = [0.0f32; 4];
        let mut idx = 0;

        for i in 0..3 {
            if i == row { continue; }
            for j in 0..3 {
                if j == col { continue; }
                elements[idx] = self.e[j][i]; // column-major access
                idx += 1;
            }
        }

        // 2x2 determinant: ad - bc
        elements[0] * elements[3] - elements[1] * elements[2]
    }

    pub fn cofactor_matrix(&self) -> Mat3 {
        let mut cofactor = Mat3::ZERO;
        for i in 0..3 {
            for j in 0..3 {
                let minor = self.minor(i, j);
                let sign = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
                cofactor.set(i, j, sign * minor);
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
        Mat3 {
            e: [
                Vec::new(self.e[0].x, self.e[1].x, self.e[2].x),
                Vec::new(self.e[0].y, self.e[1].y, self.e[2].y),
                Vec::new(self.e[0].z, self.e[1].z, self.e[2].z),
            ],
        }
    }

    pub fn transpose_mut(&mut self) -> &mut Self {
        let orig = *self;
        self.e[0] = Vec::new(orig.e[0].x, orig.e[1].x, orig.e[2].x);
        self.e[1] = Vec::new(orig.e[0].y, orig.e[1].y, orig.e[2].y);
        self.e[2] = Vec::new(orig.e[0].z, orig.e[1].z, orig.e[2].z);
        self
    }

    pub fn trace(&self) -> f32 {
        self.e[0].x + self.e[1].y + self.e[2].z
    }
}

// Utility Operations
impl Mat3 {
    pub fn is_identity(&self) -> bool {
        *self == Mat3::IDENTITY
    }

    pub fn is_zero(&self) -> bool {
        self.e[0].is_zero() && self.e[1].is_zero() && self.e[2].is_zero()
    }

    pub fn is_invertible(&self) -> bool {
        self.determinant().abs() > 1e-6
    }
}

// Composition Helpers - chainable operations
impl Mat3 {
    pub fn scale(&self, scaling: Vec) -> Self {
        *self * Mat3::scaling(scaling)
    }

    pub fn rotate_x(&self, angle: f32) -> Self {
        *self * Mat3::rotation_x(angle)
    }

    pub fn rotate_y(&self, angle: f32) -> Self {
        *self * Mat3::rotation_y(angle)
    }

    pub fn rotate_z(&self, angle: f32) -> Self {
        *self * Mat3::rotation_z(angle)
    }

    pub fn rotate(&self, axis: Vec, angle: f32) -> Self {
        *self * Mat3::rotation(axis, angle)
    }
}

impl Default for Mat3 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

#[cfg(test)]
mod tests;