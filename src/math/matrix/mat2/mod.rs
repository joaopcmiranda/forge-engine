use crate::math::Vec2;
use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat2 {
    e: [Vec2; 2], // Column-major storage
}

// Constructors
impl Mat2 {
    pub fn new() -> Self {
        Mat2::IDENTITY
    }

    pub fn from_rows(row1: Vec2, row2: Vec2) -> Self {
        Mat2 {
            e: [
                Vec2::new(row1.x, row2.x), // col 0
                Vec2::new(row1.y, row2.y), // col 1
            ],
        }
    }

    pub fn from_cols(col1: Vec2, col2: Vec2) -> Self {
        Mat2 {
            e: [col1, col2],
        }
    }

    pub fn from_mul(one: Mat2, other: Mat2) -> Self {
        let rows = one.rows();
        let cols = other.cols();
        Mat2 {
            e: [
                // Column 0: each row of 'one' dotted with column 0 of 'other'
                Vec2::new(
                    rows[0].dot(cols[0]),
                    rows[1].dot(cols[0]),
                ),
                // Column 1: each row of 'one' dotted with column 1 of 'other'
                Vec2::new(
                    rows[0].dot(cols[1]),
                    rows[1].dot(cols[1]),
                ),
            ],
        }
    }
}

// Constants
impl Mat2 {
    pub const IDENTITY: Self = Mat2 {
        e: [
            Vec2 { x: 1.0, y: 0.0 },
            Vec2 { x: 0.0, y: 1.0 },
        ]
    };

    pub const ZERO: Self = Mat2 {
        e: [Vec2::ZERO, Vec2::ZERO],
    };

    pub const FLIP_X: Self = Mat2 {
        e: [
            Vec2 { x: -1.0, y: 0.0 },
            Vec2 { x: 0.0, y: 1.0 },
        ]
    };

    pub const FLIP_Y: Self = Mat2 {
        e: [
            Vec2 { x: 1.0, y: 0.0 },
            Vec2 { x: 0.0, y: -1.0 },
        ]
    };
}

// Conversion Implementations
impl From<f32> for Mat2 {
    fn from(value: f32) -> Self {
        Mat2 {
            e: [
                Vec2::new(value, value),
                Vec2::new(value, value),
            ],
        }
    }
}

impl From<[f32; 4]> for Mat2 {
    /// Assumes column-major order
    fn from(arr: [f32; 4]) -> Self {
        Mat2 {
            e: [
                Vec2::new(arr[0], arr[1]), // col 0
                Vec2::new(arr[2], arr[3]), // col 1
            ],
        }
    }
}

impl From<[[f32; 2]; 2]> for Mat2 {
    /// Assumes arr[col][row] - column-major 2D array
    fn from(arr: [[f32; 2]; 2]) -> Self {
        Mat2 {
            e: [
                Vec2::new(arr[0][0], arr[0][1]), // col 0
                Vec2::new(arr[1][0], arr[1][1]), // col 1
            ],
        }
    }
}

impl From<[Vec2; 2]> for Mat2 {
    fn from(arr: [Vec2; 2]) -> Self {
        Mat2 { e: arr }
    }
}

impl Into<[f32; 4]> for Mat2 {
    /// Outputs in column-major order
    fn into(self) -> [f32; 4] {
        [
            // Column 0
            self.e[0].x, self.e[0].y,
            // Column 1
            self.e[1].x, self.e[1].y,
        ]
    }
}

impl Into<[[f32; 2]; 2]> for Mat2 {
    /// Outputs as [col][row] - column-major 2D array
    fn into(self) -> [[f32; 2]; 2] {
        [
            [self.e[0].x, self.e[0].y], // col 0
            [self.e[1].x, self.e[1].y], // col 1
        ]
    }
}

impl Into<[Vec2; 2]> for Mat2 {
    fn into(self) -> [Vec2; 2] {
        self.e
    }
}

// Accessors
impl Mat2 {
    pub fn cols(&self) -> [Vec2; 2] {
        self.e
    }

    pub fn col(&self, i: usize) -> Vec2 {
        self.e[i]
    }

    pub fn rows(&self) -> [Vec2; 2] {
        [
            Vec2::new(self.e[0].x, self.e[1].x),
            Vec2::new(self.e[0].y, self.e[1].y),
        ]
    }

    pub fn row(&self, i: usize) -> Vec2 {
        match i {
            0 => Vec2::new(self.e[0].x, self.e[1].x),
            1 => Vec2::new(self.e[0].y, self.e[1].y),
            _ => panic!("Row index {} out of bounds (0..2)", i),
        }
    }
}

// Setters
impl Mat2 {
    pub fn set_col(&mut self, i: usize, col: Vec2) {
        if i < 2 {
            self.e[i] = col;
        } else {
            panic!("Col index {} out of bounds (0..2)", i);
        }
    }

    pub fn set_row(&mut self, i: usize, row: Vec2) {
        match i {
            0 => {
                self.e[0].x = row.x;
                self.e[1].x = row.y;
            }
            1 => {
                self.e[0].y = row.x;
                self.e[1].y = row.y;
            }
            _ => panic!("Row index {} out of bounds (0..2)", i),
        }
    }

    /// Sets matrix[row][col] = value
    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        if row < 2 && col < 2 {
            self.e[col][row] = value;  // column-major: [col][row]
        } else {
            panic!("Index out of bounds for Mat2: ({}, {})", row, col);
        }
    }
}

impl Index<usize> for Mat2 {
    type Output = Vec2;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.e[0],
            1 => &self.e[1],
            _ => panic!("Index {} out of bounds", index),
        }
    }
}

impl IndexMut<usize> for Mat2 {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.e[0],
            1 => &mut self.e[1],
            _ => panic!("Index {} out of bounds", index),
        }
    }
}

// Matrix - Matrix Operations
impl Add for Mat2 {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        Mat2 {
            e: [
                self.e[0] + other.e[0],
                self.e[1] + other.e[1],
            ],
        }
    }
}

impl AddAssign for Mat2 {
    fn add_assign(&mut self, other: Self) {
        self.e[0] += other.e[0];
        self.e[1] += other.e[1];
    }
}

impl Sub for Mat2 {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        Mat2 {
            e: [
                self.e[0] - other.e[0],
                self.e[1] - other.e[1],
            ],
        }
    }
}

impl SubAssign for Mat2 {
    fn sub_assign(&mut self, other: Self) {
        self.e[0] -= other.e[0];
        self.e[1] -= other.e[1];
    }
}

impl Mul<Mat2> for Mat2 {
    type Output = Self;
    fn mul(self, other: Mat2) -> Self::Output {
        Mat2::from_mul(self, other)
    }
}

impl MulAssign<Mat2> for Mat2 {
    fn mul_assign(&mut self, other: Mat2) {
        *self = Mat2::from_mul(*self, other);
    }
}

impl Mul<Vec2> for Mat2 {
    type Output = Vec2;

    fn mul(self, vec: Vec2) -> Self::Output {
        let rows = self.rows();
        Vec2::new(
            rows[0].dot(vec),
            rows[1].dot(vec),
        )
    }
}

impl MulAssign<Mat2> for Vec2 {
    fn mul_assign(&mut self, matrix: Mat2) {
        *self = matrix * *self;
    }
}

impl Mul<f32> for Mat2 {
    type Output = Self;

    fn mul(self, scalar: f32) -> Self::Output {
        Mat2 {
            e: [
                self.e[0] * scalar,
                self.e[1] * scalar,
            ],
        }
    }
}

impl Mul<Mat2> for f32 {
    type Output = Mat2;

    fn mul(self, matrix: Mat2) -> Self::Output {
        matrix * self
    }
}

impl MulAssign<f32> for Mat2 {
    fn mul_assign(&mut self, scalar: f32) {
        self.e[0] *= scalar;
        self.e[1] *= scalar;
    }
}

impl Div<f32> for Mat2 {
    type Output = Self;

    fn div(self, scalar: f32) -> Self::Output {
        if scalar == 0.0 {
            panic!("Division by zero in Mat2 division");
        }
        Mat2 {
            e: [
                self.e[0] / scalar,
                self.e[1] / scalar,
            ],
        }
    }
}

impl DivAssign<f32> for Mat2 {
    fn div_assign(&mut self, scalar: f32) {
        if scalar == 0.0 {
            panic!("Division by zero in Mat2 division");
        }
        self.e[0] /= scalar;
        self.e[1] /= scalar;
    }
}

impl Div<Mat2> for Mat2 {
    type Output = Self;

    fn div(self, other: Self) -> Self::Output {
        if other.is_zero() {
            panic!("Division by zero in Mat2 division");
        }
        self * other.inverse().unwrap()
    }
}

impl DivAssign<Mat2> for Mat2 {
    fn div_assign(&mut self, other: Self) {
        if other.is_zero() {
            panic!("Division by zero in Mat2 division");
        }
        *self = *self * other.inverse().unwrap();
    }
}

impl Neg for Mat2 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Mat2 {
            e: [
                -self.e[0],
                -self.e[1],
            ],
        }
    }
}

// Matrix Operations
impl Mat2 {
    pub fn determinant(&self) -> f32 {
        self.e[0].x * self.e[1].y - self.e[1].x * self.e[0].y
    }

    pub fn inverse(&self) -> Option<Self> {
        let det = self.determinant();
        if det.abs() < 1e-6 {
            None
        } else {
            Some(Mat2::from_cols(
                Vec2::new(self.e[1].y, -self.e[0].y),
                Vec2::new(-self.e[1].x, self.e[0].x),
            ) / det)
        }
    }

    pub fn transpose(&self) -> Self {
        Mat2 {
            e: [
                Vec2::new(self.e[0].x, self.e[1].x),
                Vec2::new(self.e[0].y, self.e[1].y),
            ],
        }
    }

    pub fn transpose_mut(&mut self) -> &mut Self {
        let orig = *self;
        self.e[0] = Vec2::new(orig.e[0].x, orig.e[1].x);
        self.e[1] = Vec2::new(orig.e[0].y, orig.e[1].y);
        self
    }

    pub fn trace(&self) -> f32 {
        self.e[0].x + self.e[1].y
    }
}

// Utility Operations
impl Mat2 {
    pub fn is_identity(&self) -> bool {
        *self == Mat2::IDENTITY
    }

    pub fn is_zero(&self) -> bool {
        self.e[0].is_zero() && self.e[1].is_zero()
    }

    pub fn is_invertible(&self) -> bool {
        self.determinant().abs() > 1e-6
    }
}

impl Default for Mat2 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

#[cfg(test)]
mod tests;