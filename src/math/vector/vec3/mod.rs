//! 3D vector mathematics implementation.
//!
//! This module provides a complete 3D vector type with comprehensive mathematical
//! operations, conversions, and utility functions commonly needed in graphics,
//! physics, and game development.

use crate::math::Vec2;
use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};


/// A 3D vector with `x`, `y`, and `z` components.
///
/// `Vector` represents a point or direction in 3D space using single-precision
/// floating-point numbers. It supports all standard mathematical operations and
/// provides many utility functions for common vector operations.
///
/// # Examples
///
/// ```rust
/// use crate::forge_engine::Vec;
///
/// // Create vectors
/// let v1 = Vec::new(1.0, 2.0, 3.0);
/// let v2 = Vec::from((4.0, 5.0, 6.0));
///
/// // Basic arithmetic
/// let sum = v1 + v2;
/// let scaled = v1 * 2.0;
///
/// // Vector operations
/// let dot_product = v1.dot(v2);
/// let cross_product = v1.cross(v2);
/// let magnitude = v1.magnitude();
/// let normalized = v1.normalized();
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    /// The x-component of the vector
    pub x: f32,
    /// The y-component of the vector
    pub y: f32,
    /// The z-component of the vector
    pub z: f32,
}

pub type Vec = Vec3;

// Constructor
impl Vec3 {
    /// Creates a new `Vector` with the specified components.
    ///
    /// # Arguments
    ///
    /// * `x` - The x-component
    /// * `y` - The y-component
    /// * `z` - The z-component
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::Vec;
    ///
    /// let v = Vec::new(1.0, 2.0, 3.0);
    /// assert_eq!(v.x, 1.0);
    /// assert_eq!(v.y, 2.0);
    /// assert_eq!(v.z, 3.0);
    /// ```
    pub fn new(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3 { x, y, z }
    }
}

// Constant constructors
impl Vec3 {
    /// A vector with all components set to zero: `(0, 0, 0)`
    pub const ZERO: Vec3 = Vec3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// A vector with all components set to one: `(1, 1, 1)`
    pub const ONE: Vec3 = Vec3 {
        x: 1.0,
        y: 1.0,
        z: 1.0,
    };

    /// The up direction vector: `(0, 1, 0)`
    pub const UP: Vec3 = Vec3 {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };

    /// The down direction vector: `(0, -1, 0)`
    pub const DOWN: Vec3 = Vec3 {
        x: 0.0,
        y: -1.0,
        z: 0.0,
    };

    /// The right direction vector: `(1, 0, 0)`
    pub const RIGHT: Vec3 = Vec3 {
        x: 1.0,
        y: 0.0,
        z: 0.0,
    };

    /// The left direction vector: `(-1, 0, 0)`
    pub const LEFT: Vec3 = Vec3 {
        x: -1.0,
        y: 0.0,
        z: 0.0,
    };

    /// The forward direction vector: `(0, 0, 1)`
    pub const FORWARD: Vec3 = Vec3 {
        x: 0.0,
        y: 0.0,
        z: 1.0,
    };

    /// The backward direction vector: `(0, 0, -1)`
    pub const BACKWARD: Vec3 = Vec3 {
        x: 0.0,
        y: 0.0,
        z: -1.0,
    };
}

// Accessors
impl Vec3 {
    /// Returns the x component of the vector.
    #[inline]
    pub fn x(&self) -> f32 {
        self.x
    }

    /// Returns the y component of the vector.
    #[inline]
    pub fn y(&self) -> f32 {
        self.y
    }

    /// Returns the z component of the vector.
    #[inline]
    pub fn z(&self) -> f32 {
        self.z
    }
}

/// Converts a tuple `(f32, f32, f32)` into a `Vector`.
///
/// # Examples
///
/// ```rust
/// use crate::forge_engine::Vec;
///
/// let v: Vec = (1.0, 2.0, 3.0).into();
/// assert_eq!(v, Vec::new(1.0, 2.0, 3.0));
/// ```
impl From<(f32, f32, f32)> for Vec3 {
    fn from(t: (f32, f32, f32)) -> Self {
        Vec3::new(t.0, t.1, t.2)
    }
}

/// Converts a `Vector2` into a `Vector` with `z` set to `0.0`.
///
/// # Examples
///
/// ```rust
/// use crate::forge_engine::Vec;
/// use crate::forge_engine::math::Vec2;
///
/// let v2 = Vec2::new(1.0, 2.0);
/// let v3: Vec = v2.into();
/// assert_eq!(v3, Vec::new(1.0, 2.0, 0.0));
/// ```
impl From<Vec2> for Vec3 {
    fn from(v: Vec2) -> Self {
        Vec3::new(v.x, v.y, 0.0)
    }
}

/// Converts an array `[f32; 3]` into a `Vector`.
///
/// # Examples
///
/// ```rust
/// use crate::forge_engine::Vec;
///
/// let v: Vec = [1.0, 2.0, 3.0].into();
/// assert_eq!(v, Vec::new(1.0, 2.0, 3.0));
/// ```
impl From<[f32; 3]> for Vec3 {
    fn from(arr: [f32; 3]) -> Self {
        Vec3::new(arr[0], arr[1], arr[2])
    }
}

/// Converts a `Vector` into a tuple `(f32, f32, f32)`.
impl From<Vec3> for (f32, f32, f32) {
    fn from(v: Vec3) -> Self {
        (v.x, v.y, v.z)
    }
}

/// Converts a `Vector` into an array `[f32; 3]`.
impl From<Vec3> for [f32; 3] {
    fn from(v: Vec3) -> Self {
        [v.x, v.y, v.z]
    }
}

// Operators

impl Index<usize> for Vec3 {
    type Output = f32;

    /// Access vector components by index.
    ///
    /// # Indexing
    /// - `0` returns the x component
    /// - `1` returns the y component
    /// - `2` returns the z component
    ///
    /// # Panics
    /// Panics if the index is greater than 2.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::Vec;
    ///
    /// let v = Vec::new(1.0, 2.0, 3.0);
    /// assert_eq!(v[0], 1.0);  // x component
    /// assert_eq!(v[1], 2.0);  // y component
    /// assert_eq!(v[2], 3.0);  // z component
    /// ```
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Vector index {} out of bounds (0..3)", index),
        }
    }
}

impl IndexMut<usize> for Vec3 {
    /// Mutably access vector components by index.
    ///
    /// # Indexing
    /// - `0` returns the x component
    /// - `1` returns the y component
    /// - `2` returns the z component
    ///
    /// # Panics
    /// Panics if the index is greater than 2.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::Vec;
    ///
    /// let mut v = Vec::new(1.0, 2.0, 3.0);
    /// v[0] = 5.0;  // Set x component
    /// v[1] = 6.0;  // Set y component
    /// v[2] = 7.0;  // Set z component
    /// assert_eq!(v, Vec::new(5.0, 6.0, 7.0));
    /// ```
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("Vector index {} out of bounds (0..3)", index),
        }
    }
}

/// Adds two vectors component-wise.
impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

/// Adds a scalar to each component of the vector.
impl Add<f32> for Vec3 {
    type Output = Vec3;
    fn add(self, other: f32) -> Vec3 {
        Vec3::new(self.x + other, self.y + other, self.z + other)
    }
}

/// Adds a vector to a scalar (commutative addition).
impl Add<Vec3> for f32 {
    type Output = Vec3;
    fn add(self, other: Vec3) -> Vec3 {
        Vec3::new(self + other.x, self + other.y, self + other.z)
    }
}

/// Adds another vector to this vector in place.
impl AddAssign for Vec3 {
    fn add_assign(&mut self, other: Vec3) {
        *self = Vec3::new(self.x + other.x, self.y + other.y, self.z + other.z);
    }
}

/// Subtracts two vectors component-wise.
impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

/// Subtracts a scalar from each component of the vector.
impl Sub<f32> for Vec3 {
    type Output = Vec3;
    fn sub(self, other: f32) -> Vec3 {
        Vec3::new(self.x - other, self.y - other, self.z - other)
    }
}

/// Subtracts a vector from a scalar.
impl Sub<Vec3> for f32 {
    type Output = Vec3;
    fn sub(self, other: Vec3) -> Vec3 {
        Vec3::new(self - other.x, self - other.y, self - other.z)
    }
}

/// Subtracts another vector from this vector in place.
impl SubAssign for Vec3 {
    fn sub_assign(&mut self, other: Vec3) {
        *self = Vec3::new(self.x - other.x, self.y - other.y, self.z - other.z);
    }
}

/// Multiplies two vectors component-wise (Hadamard product).
impl Mul<Vec3> for Vec3 {
    type Output = Vec3;
    fn mul(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x * other.x, self.y * other.y, self.z * other.z)
    }
}

/// Multiplies the vector by a scalar.
impl Mul<f32> for Vec3 {
    type Output = Vec3;
    fn mul(self, other: f32) -> Vec3 {
        Vec3::new(self.x * other, self.y * other, self.z * other)
    }
}

/// Multiplies a scalar by a vector (commutative multiplication).
impl Mul<Vec3> for f32 {
    type Output = Vec3;
    fn mul(self, other: Vec3) -> Vec3 {
        Vec3::new(self * other.x, self * other.y, self * other.z)
    }
}

/// Multiplies this vector by another vector in place (component-wise).
impl MulAssign<Vec3> for Vec3 {
    fn mul_assign(&mut self, other: Vec3) {
        *self = Vec3::new(self.x * other.x, self.y * other.y, self.z * other.z);
    }
}

/// Multiplies this vector by a scalar in place.
impl MulAssign<f32> for Vec3 {
    fn mul_assign(&mut self, other: f32) {
        *self = Vec3::new(self.x * other, self.y * other, self.z * other);
    }
}

/// Divides two vectors component-wise.
impl Div<Vec3> for Vec3 {
    type Output = Vec3;
    fn div(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x / other.x, self.y / other.y, self.z / other.z)
    }
}

/// Divides the vector by a scalar.
impl Div<f32> for Vec3 {
    type Output = Vec3;
    fn div(self, other: f32) -> Vec3 {
        Vec3::new(self.x / other, self.y / other, self.z / other)
    }
}

/// Divides a scalar by a vector component-wise.
impl Div<Vec3> for f32 {
    type Output = Vec3;
    fn div(self, other: Vec3) -> Vec3 {
        Vec3::new(self / other.x, self / other.y, self / other.z)
    }
}

/// Divides this vector by another vector in place (component-wise).
impl DivAssign<Vec3> for Vec3 {
    fn div_assign(&mut self, other: Vec3) {
        *self = Vec3::new(self.x / other.x, self.y / other.y, self.z / other.z);
    }
}

/// Divides this vector by a scalar in place.
impl DivAssign<f32> for Vec3 {
    fn div_assign(&mut self, other: f32) {
        *self = Vec3::new(self.x / other, self.y / other, self.z / other);
    }
}

/// Negates the vector (multiplies each component by -1).
impl Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        Vec3::new(-self.x, -self.y, -self.z)
    }
}

/// Creates a zero vector by default.
impl Default for Vec3 {
    fn default() -> Self {
        Self::ZERO
    }
}

// Other Methods
impl Vec3 {


    /// Checks if two vectors are approximately equal within a default precision of 6 decimal places (micrometers)
    /// This is useful for comparing floating-point vectors where exact equality
    /// is not reliable due to precision issues.
    /// # Arguments
    /// * `other` - The other vector to compare against
    pub fn equals(self, other: Vec3) -> bool {
        self.equals_with_precision(other, 6)
    }


    /// Checks if two vectors are approximately equal within a given precision.
    /// This is useful for comparing floating-point vectors where exact equality
    /// is not reliable due to precision issues.
    /// # Arguments
    /// * `other` - The other vector to compare against
    /// * `precision` - The number of decimal places to consider for equality
    pub fn equals_with_precision(self, other: Vec3, precision: usize) -> bool {
        let factor = 10f32.powi(precision as i32);
        (self.x * factor).round() == (other.x * factor).round() &&
            (self.y * factor).round() == (other.y * factor).round() &&
            (self.z * factor).round() == (other.z * factor).round()
    }

    /// Computes the dot product of two vectors.
    ///
    /// The dot product is a scalar value equal to the sum of the products
    /// of the corresponding components.
    ///
    /// # Arguments
    ///
    /// * `other` - The other vector
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::Vec;
    ///
    /// let a = Vec::new(1.0, 2.0, 3.0);
    /// let b = Vec::new(4.0, 5.0, 6.0);
    /// assert_eq!(a.dot(b), 32.0); // 1*4 + 2*5 + 3*6
    /// ```
    #[inline]
    pub fn dot(self, other: Vec3) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Raises each component of the vector to the given power.
    ///
    /// # Arguments
    ///
    /// * `exp` - The exponent to raise each component to
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::Vec;
    ///
    /// let v = Vec::new(2.0, 3.0, 4.0);
    /// let squared = v.pow(2.0);
    /// assert_eq!(squared, Vec::new(4.0, 9.0, 16.0));
    /// ```
    #[inline]
    pub fn pow(self, exp: f32) -> Vec3 {
        Vec3::new(self.x.powf(exp), self.y.powf(exp), self.z.powf(exp))
    }

    /// Computes the square root of each component.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::Vec;
    ///
    /// let v = Vec::new(4.0, 9.0, 16.0);
    /// let roots = v.sqrt();
    /// assert_eq!(roots, Vec::new(2.0, 3.0, 4.0));
    /// ```
    #[inline]
    pub fn sqrt(&self) -> Vec3 {
        Vec3::new(self.x.sqrt(), self.y.sqrt(), self.z.sqrt())
    }

    /// Computes the cross product of two vectors.
    ///
    /// The cross product produces a vector perpendicular to both input vectors,
    /// with magnitude equal to the area of the parallelogram formed by the vectors.
    ///
    /// # Arguments
    ///
    /// * `other` - The other vector
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::Vec;
    ///
    /// let a = Vec::new(1.0, 0.0, 0.0);
    /// let b = Vec::new(0.0, 1.0, 0.0);
    /// let c = a.cross(b);
    /// assert_eq!(c, Vec::new(0.0, 0.0, 1.0));
    /// ```
    #[inline]
    pub fn cross(self, other: Vec3) -> Vec3 {
        Vec3::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    /// Computes the magnitude (length) of the vector.
    ///
    /// This is the Euclidean distance from the origin to the point
    /// represented by the vector.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::Vec;
    ///
    /// let v = Vec::new(2.0, 3.0, 6.0);
    /// assert_eq!(v.magnitude(), 7.0);
    /// ```
    #[inline]
    pub fn magnitude(self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Computes the length of the vector, which is the same as `magnitude()`.
    #[inline]
    pub  fn length(self) -> f32 {
        self.magnitude()
    }

    /// Computes the squared magnitude of the vector.
    ///
    /// This is more efficient than `magnitude()` when you only need
    /// to compare lengths or don't need the actual distance.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::Vec;
    ///
    /// let v = Vec::new(2.0, 3.0, 6.0);
    /// assert_eq!(v.magnitude_squared(), 49.0);
    /// ```
    #[inline]
    pub fn magnitude_squared(self) -> f32 {
        self.dot(self)
    }

    /// Returns a normalized version of the vector (unit vector).
    ///
    /// If the vector has zero length, returns the zero vector.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::Vec;
    ///
    /// let v = Vec::new(0.0, 3.0, 4.0);
    /// let normalized = v.normalized();
    /// assert_eq!(normalized, Vec::new(0.0, 0.6, 0.8));
    /// assert!((normalized.magnitude() - 1.0).abs() < f32::EPSILON);
    /// ```
    #[inline]
    pub fn normalized(self) -> Vec3 {
        let magnitude = self.magnitude();
        if magnitude == 0.0 {
            Vec3::ZERO
        } else {
            self / magnitude
        }
    }

    /// Checks if the vector is approximately zero.
    ///
    /// Uses `f32::EPSILON` for the comparison threshold.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::Vec;
    ///
    /// assert!(Vec::ZERO.is_zero());
    /// assert!(!Vec::ONE.is_zero());
    /// ```
    #[inline]
    pub fn is_zero(self) -> bool {
        self.magnitude() < f32::EPSILON
    }

    /// Checks if the vector is normalized (has unit length).
    ///
    /// Uses `f32::EPSILON` for the comparison threshold.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::Vec;
    ///
    /// assert!(Vec::RIGHT.is_normalised());
    /// assert!(!Vec::new(2.0, 3.0, 6.0).is_normalised());
    /// ```
    #[inline]
    pub fn is_normalised(self) -> bool {
        (self.magnitude() - 1.0).abs() < f32::EPSILON
    }

    /// Returns a normalized version of the vector, or `None` if the vector is zero.
    ///
    /// This is safer than `normalized()` when you need to handle the zero vector case.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::Vec;
    ///
    /// let v = Vec::new(3.0, 4.0, 0.0);
    /// assert!(v.safe_normal().is_some());
    /// assert!(Vec::ZERO.safe_normal().is_none());
    /// ```
    pub fn safe_normal(self) -> Option<Vec3> {
        if self.is_zero() {
            None
        } else {
            Some(self / self.magnitude())
        }
    }

    /// Linearly interpolates between two vectors.
    ///
    /// # Arguments
    ///
    /// * `other` - The target vector
    /// * `t` - The interpolation factor (0.0 = self, 1.0 = other)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::Vec;
    ///
    /// let a = Vec::new(0.0, 0.0, 0.0);
    /// let b = Vec::new(10.0, 20.0, 30.0);
    /// let mid = a.lerp(b, 0.5);
    /// assert_eq!(mid, Vec::new(5.0, 10.0, 15.0));
    /// ```
    pub fn lerp(self, other: Vec3, t: f32) -> Vec3 {
        self + (other - self) * t
    }

    /// Computes the distance between two points.
    ///
    /// # Arguments
    ///
    /// * `other` - The other point
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::Vec;
    ///
    /// let a = Vec::new(1.0, 2.0, 3.0);
    /// let b = Vec::new(4.0, 6.0, 8.0);
    /// let dist = a.distance(b);
    /// // dist ≈ 7.07
    /// ```
    pub fn distance(self, other: Vec3) -> f32 {
        (other - self).magnitude()
    }

    /// Computes the squared distance between two points.
    ///
    /// More efficient than `distance()` when you only need to compare distances.
    ///
    /// # Arguments
    ///
    /// * `other` - The other point
    pub fn distance_squared(self, other: Vec3) -> f32 {
        let diff = self - other;
        diff.dot(diff)
    }

    /// Returns a vector with the absolute value of each component.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::Vec;
    ///
    /// let v = Vec::new(-1.0, 2.0, -3.0);
    /// assert_eq!(v.abs(), Vec::new(1.0, 2.0, 3.0));
    /// ```
    pub fn abs(self) -> Vec3 {
        Vec3::new(self.x.abs(), self.y.abs(), self.z.abs())
    }

    /// Returns a vector with the minimum component values from two vectors.
    ///
    /// # Arguments
    ///
    /// * `other` - The other vector
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::Vec;
    ///
    /// let a = Vec::new(1.0, 5.0, 3.0);
    /// let b = Vec::new(4.0, 2.0, 6.0);
    /// assert_eq!(a.min(b), Vec::new(1.0, 2.0, 3.0));
    /// ```
    pub fn min(self, other: Vec3) -> Vec3 {
        Vec3::new(
            self.x.min(other.x),
            self.y.min(other.y),
            self.z.min(other.z),
        )
    }

    /// Returns a vector with the maximum component values from two vectors.
    ///
    /// # Arguments
    ///
    /// * `other` - The other vector
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::Vec;
    ///
    /// let a = Vec::new(1.0, 5.0, 3.0);
    /// let b = Vec::new(4.0, 2.0, 6.0);
    /// assert_eq!(a.max(b), Vec::new(4.0, 5.0, 6.0));
    /// ```
    pub fn max(self, other: Vec3) -> Vec3 {
        Vec3::new(
            self.x.max(other.x),
            self.y.max(other.y),
            self.z.max(other.z),
        )
    }

    /// Clamps each component of the vector between corresponding min and max values.
    ///
    /// # Arguments
    ///
    /// * `min` - The minimum values for each component
    /// * `max` - The maximum values for each component
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::Vec;
    ///
    /// let v = Vec::new(-1.0, 5.0, 2.0);
    /// let min = Vec::new(0.0, 0.0, 0.0);
    /// let max = Vec::new(3.0, 3.0, 3.0);
    /// assert_eq!(v.clamp(min, max), Vec::new(0.0, 3.0, 2.0));
    /// ```
    pub fn clamp(self, min: Vec3, max: Vec3) -> Vec3 {
        self.max(min).min(max)
    }

    /// Computes the angle between two vectors in radians.
    ///
    /// Returns a value between 0 and π radians.
    ///
    /// # Arguments
    ///
    /// * `other` - The other vector
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::Vec;
    ///
    /// use std::f32::consts::PI;
    /// let a = Vec::new(1.0, 0.0, 0.0);
    /// let b = Vec::new(0.0, 1.0, 0.0);
    /// let angle = a.angle_between(b);
    /// assert!((angle - PI / 2.0).abs() < f32::EPSILON);
    /// ```
    pub fn angle_between(self, other: Vec3) -> f32 {
        let dot = self.dot(other);
        let mags = self.magnitude() * other.magnitude();
        (dot / mags).acos()
    }

    /// Projects this vector onto another vector.
    ///
    /// Returns the component of this vector in the direction of the `onto` vector.
    ///
    /// # Arguments
    ///
    /// * `onto` - The vector to project onto
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::Vec;
    ///
    /// let v = Vec::new(3.0, 4.0, 5.0);
    /// let onto = Vec::new(1.0, 0.0, 0.0);
    /// let proj = v.project_onto(onto);
    /// assert_eq!(proj, Vec::new(3.0, 0.0, 0.0));
    /// ```
    pub fn project_onto(self, onto: Vec3) -> Vec3 {
        let d = onto.dot(onto);
        if d > 0.0 {
            onto * (self.dot(onto) / d)
        } else {
            Vec3::ZERO
        }
    }

    /// Rejects this vector from another vector (returns the perpendicular component).
    ///
    /// This is the complement of projection: `v = project + reject`.
    ///
    /// # Arguments
    ///
    /// * `from` - The vector to reject from
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::Vec;
    ///
    /// let v = Vec::new(3.0, 4.0, 5.0);
    /// let from = Vec::new(1.0, 0.0, 0.0);
    /// let rej = v.reject_from(from);
    /// assert_eq!(rej, Vec::new(0.0, 4.0, 5.0));
    /// ```
    pub fn reject_from(self, from: Vec3) -> Vec3 {
        self - self.project_onto(from)
    }

    /// Reflects this vector across a surface with the given normal.
    ///
    /// The normal should be normalized for correct results.
    ///
    /// # Arguments
    ///
    /// * `normal` - The surface normal (should be normalized)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::Vec;
    ///
    /// let v = Vec::new(1.0, -1.0, 0.0);
    /// let normal = Vec::new(0.0, 1.0, 0.0);
    /// let reflected = v.reflect(normal);
    /// assert_eq!(reflected, Vec::new(1.0, 1.0, 0.0));
    /// ```
    pub fn reflect(self, normal: Vec3) -> Vec3 {
        self - normal * (2.0 * self.dot(normal))
    }
}

// Swizzling
impl Vec3 {
    /// Converts to a `Vector2` using the x and y components.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::Vec;
    /// use crate::forge_engine::math::Vec2;
    ///
    /// let v3 = Vec::new(1.0, 2.0, 3.0);
    /// let v2 = v3.vec2();
    /// assert_eq!(v2, Vec2::new(1.0, 2.0));
    /// ```
    pub fn vec2(self) -> Vec2 {
        Vec2::new(self.x, self.y)
    }

    /// Returns a `Vector2` with the x and y components.
    pub fn xy(self) -> Vec2 {
        Vec2::new(self.x, self.y)
    }

    /// Returns a `Vector2` with the y and x components (swapped).
    pub fn yx(self) -> Vec2 {
        Vec2::new(self.y, self.x)
    }

    /// Returns a `Vector2` with the x and z components.
    pub fn xz(self) -> Vec2 {
        Vec2::new(self.x, self.z)
    }

    /// Returns a `Vector2` with the z and x components (swapped).
    pub fn zx(self) -> Vec2 {
        Vec2::new(self.z, self.x)
    }

    /// Returns a `Vector2` with the y and z components.
    pub fn yz(self) -> Vec2 {
        Vec2::new(self.y, self.z)
    }

    /// Returns a `Vector2` with the z and y components (swapped).
    pub fn zy(self) -> Vec2 {
        Vec2::new(self.z, self.y)
    }

    /// Returns a `Vector2` with both components set to x.
    pub fn xx(self) -> Vec2 {
        Vec2::new(self.x, self.x)
    }

    /// Returns a `Vector2` with both components set to y.
    pub fn yy(self) -> Vec2 {
        Vec2::new(self.y, self.y)
    }

    /// Returns a `Vector2` with both components set to z.
    pub fn zz(self) -> Vec2 {
        Vec2::new(self.z, self.z)
    }
}

#[cfg(test)]
mod tests;
