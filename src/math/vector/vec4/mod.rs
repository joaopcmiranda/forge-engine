//! 4D vector mathematics implementation.
//!
//! This module provides a complete 4D vector type with comprehensive mathematical
//! operations, conversions, and utility functions commonly needed in graphics,
//! physics, and game development. Vector4 is commonly used for homogeneous
//! coordinates, RGBA colors, quaternions, and other 4D mathematical operations.

use crate::math::{Vec, Vec2};
use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign};

/// A 4D vector with `x`, `y`, `z`, and `w` components.
///
/// `Vector4` represents a point or direction in 4D space using single-precision
/// floating-point numbers. It's commonly used for homogeneous coordinates in
/// 3D graphics, RGBA color values, and quaternion operations. It supports all
/// standard mathematical operations and provides many utility functions.
///
/// # Examples
///
/// ```rust
/// use crate::forge_engine::math::Vec4;
/// use crate::forge_engine::Vec;
///
/// // Create vectors
/// let v1 = Vec4::new(1.0, 2.0, 3.0, 4.0);
/// let v2 = Vec4::from((5.0, 6.0, 7.0, 8.0));
/// let v3 = Vec4::from_vec3(Vec::new(1.0, 2.0, 3.0), 1.0);
///
/// // Basic arithmetic
/// let sum = v1 + v2;
/// let scaled = v1 * 2.0;
///
/// // Vector operations
/// let dot_product = v1.dot(v2);
/// let magnitude = v1.magnitude();
/// let normalized = v1.normalized();
///
/// // Homogeneous coordinate operations
/// let point = Vec4::from_vec3(Vec::new(10.0, 20.0, 30.0), 1.0);
/// let direction = Vec4::from_vec3(Vec::new(1.0, 0.0, 0.0), 0.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec4 {
    /// The x-component of the vector
    pub x: f32,
    /// The y-component of the vector
    pub y: f32,
    /// The z-component of the vector
    pub z: f32,
    /// The w-component of the vector
    pub w: f32,
}

// Constructor
impl Vec4 {
    /// Creates a new `Vector4` with the specified components.
    ///
    /// # Arguments
    ///
    /// * `x` - The x-component
    /// * `y` - The y-component
    /// * `z` - The z-component
    /// * `w` - The w-component
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::math::Vec4;
    ///
    /// let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(v.x, 1.0);
    /// assert_eq!(v.y, 2.0);
    /// assert_eq!(v.z, 3.0);
    /// assert_eq!(v.w, 4.0);
    /// ```
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Vec4 { x, y, z, w }
    }

    /// Creates a `Vector4` from a 3D vector and a w-component.
    ///
    /// This is commonly used in graphics programming to convert 3D points
    /// and directions to homogeneous coordinates.
    ///
    /// # Arguments
    ///
    /// * `v` - The 3D vector providing x, y, and z components
    /// * `w` - The w-component (typically 1.0 for points, 0.0 for directions)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::math::Vec4;
    /// use crate::forge_engine::Vec;
    ///
    /// let point = Vec4::from_vec3(Vec::new(1.0, 2.0, 3.0), 1.0);
    /// let direction = Vec4::from_vec3(Vec::new(0.0, 1.0, 0.0), 0.0);
    /// ```
    pub fn from_vec3(v: Vec, w: f32) -> Self {
        Vec4 {
            x: v.x,
            y: v.y,
            z: v.z,
            w,
        }
    }
}

// Constant constructors
impl Vec4 {
    /// A vector with all components set to zero: `(0, 0, 0, 0)`
    pub const ZERO: Vec4 = Vec4 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: 0.0,
    };

    /// A vector with all components set to one: `(1, 1, 1, 1)`
    pub const ONE: Vec4 = Vec4 {
        x: 1.0,
        y: 1.0,
        z: 1.0,
        w: 1.0,
    };

    /// A unit vector in the x direction: `(1, 0, 0, 0)`
    pub const X: Vec4 = Vec4 {
        x: 1.0,
        y: 0.0,
        z: 0.0,
        w: 0.0,
    };

    /// A unit vector in the y direction: `(0, 1, 0, 0)`
    pub const Y: Vec4 = Vec4 {
        x: 0.0,
        y: 1.0,
        z: 0.0,
        w: 0.0,
    };

    /// A unit vector in the z direction: `(0, 0, 1, 0)`
    pub const Z: Vec4 = Vec4 {
        x: 0.0,
        y: 0.0,
        z: 1.0,
        w: 0.0,
    };

    /// A unit vector in the w direction: `(0, 0, 0, 1)`
    pub const W: Vec4 = Vec4 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: 1.0,
    };
}

// Accessors
impl Vec4 {
    /// Returns the x-component of the vector.
    #[inline]
    pub fn x(&self) -> f32 {
        self.x
    }

    /// Returns the y-component of the vector.
    #[inline]
    pub fn y(&self) -> f32 {
        self.y
    }

    /// Returns the z-component of the vector.
    #[inline]
    pub fn z(&self) -> f32 {
        self.z
    }

    /// Returns the w-component of the vector.
    #[inline]
    pub fn w(&self) -> f32 {
        self.w
    }
}

/// Converts a tuple `(f32, f32, f32, f32)` into a `Vector4`.
///
/// # Examples
///
/// ```rust
/// use crate::forge_engine::math::Vec4;
///
/// let v: Vec4 = (1.0, 2.0, 3.0, 4.0).into();
/// assert_eq!(v, Vec4::new(1.0, 2.0, 3.0, 4.0));
/// ```
impl From<(f32, f32, f32, f32)> for Vec4 {
    fn from(t: (f32, f32, f32, f32)) -> Self {
        Vec4::new(t.0, t.1, t.2, t.3)
    }
}

/// Converts a `Vector` into a `Vector4` with `w` set to `1.0`.
///
/// This is useful for converting 3D points to homogeneous coordinates.
///
/// # Examples
///
/// ```rust
/// use crate::forge_engine::math::Vec4;
/// use crate::forge_engine::Vec;
///
/// let v3 = Vec::new(1.0, 2.0, 3.0);
/// let v4: Vec4 = v3.into();
/// assert_eq!(v4, Vec4::new(1.0, 2.0, 3.0, 1.0));
/// ```
impl From<Vec> for Vec4 {
    fn from(v: Vec) -> Self {
        Vec4::new(v.x, v.y, v.z, 1.0)
    }
}

/// Converts a `Vector2` into a `Vector4` with `z` and `w` set to `0.0` and `1.0`.
///
/// # Examples
///
/// ```rust
/// use crate::forge_engine::math::{Vec2, Vec4};
///
/// let v2 = Vec2::new(1.0, 2.0);
/// let v4: Vec4 = v2.into();
/// assert_eq!(v4, Vec4::new(1.0, 2.0, 0.0, 1.0));
/// ```
impl From<Vec2> for Vec4 {
    fn from(v: Vec2) -> Self {
        Vec4::new(v.x, v.y, 0.0, 1.0)
    }
}

/// Converts an array `[f32; 4]` into a `Vector4`.
///
/// # Examples
///
/// ```rust
/// use crate::forge_engine::math::Vec4;
///
/// let v: Vec4 = [1.0, 2.0, 3.0, 4.0].into();
/// assert_eq!(v, Vec4::new(1.0, 2.0, 3.0, 4.0));
/// ```
impl From<[f32; 4]> for Vec4 {
    fn from(arr: [f32; 4]) -> Self {
        Vec4::new(arr[0], arr[1], arr[2], arr[3])
    }
}

/// Converts a single `f32` into a `Vector4` with all components set to that value.
///
/// # Examples
///
/// ```rust
/// use crate::forge_engine::math::Vec4;
///
/// let v: Vec4 = 3.0.into();
/// assert_eq!(v, Vec4::new(3.0, 3.0, 3.0, 3.0));
/// ```
impl From<f32> for Vec4 {
    fn from(value: f32) -> Self {
        Vec4::new(value, value, value, value)
    }
}

/// Converts a `Vector4` into a tuple `(f32, f32, f32, f32)`.
impl From<Vec4> for (f32, f32, f32, f32) {
    fn from(v: Vec4) -> Self {
        (v.x, v.y, v.z, v.w)
    }
}

/// Converts a `Vector4` into an array `[f32; 4]`.
impl From<Vec4> for [f32; 4] {
    fn from(v: Vec4) -> Self {
        [v.x, v.y, v.z, v.w]
    }
}

/// Converts a `Vector4` into a `Vector` using perspective division.
///
/// If `w` is not zero, the x, y, and z components are divided by `w`.
/// If `w` is zero, returns the x, y, and z components directly.
impl From<Vec4> for Vec {
    fn from(v: Vec4) -> Self {
        if v.w != 0.0 {
            Vec::new(v.x / v.w, v.y / v.w, v.z / v.w)
        } else {
            Vec::new(v.x, v.y, v.z)
        }
    }
}

// Utility setters
impl Vec4 {
    /// Overwrites the vector with new x.
    pub fn with_x(self, x: f32) -> Self {
        Vec4 {
            x,
            y: self.y,
            z: self.z,
            w: self.w,
        }
    }

    /// Overwrites the vector with new y.
    pub fn with_y(self, y: f32) -> Self {
        Vec4 {
            x: self.x,
            y,
            z: self.z,
            w: self.w,
        }
    }

    /// Overwrites the vector with new z.
    pub fn with_z(self, z: f32) -> Self {
        Vec4 {
            x: self.x,
            y: self.y,
            z,
            w: self.w,
        }
    }

    /// Overwrites the vector with new w.
    pub fn with_w(self, w: f32) -> Self {
        Vec4 {
            x: self.x,
            y: self.y,
            z: self.z,
            w,
        }
    }
}

// Operators

impl Index<usize> for Vec4 {
    type Output = f32;

    /// Access vector components by index.
    ///
    /// # Indexing
    /// - `0` returns the x component
    /// - `1` returns the y component
    /// - `2` returns the z component
    /// - `3` returns the w component
    ///
    /// # Panics
    /// Panics if the index is greater than 3.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::math::Vec4;
    ///
    /// let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(v[0], 1.0);  // x component
    /// assert_eq!(v[1], 2.0);  // y component
    /// assert_eq!(v[2], 3.0);  // z component
    /// assert_eq!(v[3], 4.0);  // w component
    /// ```
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            3 => &self.w,
            _ => panic!("Vector4 index {} out of bounds (0..4)", index),
        }
    }
}

impl IndexMut<usize> for Vec4 {
    /// Mutably access vector components by index.
    ///
    /// # Indexing
    /// - `0` returns the x component
    /// - `1` returns the y component
    /// - `2` returns the z component
    /// - `3` returns the w component
    ///
    /// # Panics
    /// Panics if the index is greater than 3.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::math::Vec4;
    ///
    /// let mut v = Vec4::new(1.0, 2.0, 3.0, 4.0);
    /// v[0] = 5.0;  // Set x component
    /// v[1] = 6.0;  // Set y component
    /// v[2] = 7.0;  // Set z component
    /// v[3] = 8.0;  // Set w component
    /// assert_eq!(v, Vec4::new(5.0, 6.0, 7.0, 8.0));
    /// ```
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            3 => &mut self.w,
            _ => panic!("Vector4 index {} out of bounds (0..4)", index),
        }
    }
}

/// Adds two vectors component-wise.
impl Add for Vec4 {
    type Output = Vec4;
    fn add(self, other: Vec4) -> Vec4 {
        Vec4::new(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
            self.w + other.w,
        )
    }
}

/// Adds a scalar to each component of the vector.
impl Add<f32> for Vec4 {
    type Output = Vec4;
    fn add(self, other: f32) -> Vec4 {
        Vec4::new(
            self.x + other,
            self.y + other,
            self.z + other,
            self.w + other,
        )
    }
}

/// Adds a vector to a scalar (commutative addition).
impl Add<Vec4> for f32 {
    type Output = Vec4;
    fn add(self, other: Vec4) -> Vec4 {
        Vec4::new(
            self + other.x,
            self + other.y,
            self + other.z,
            self + other.w,
        )
    }
}

/// Adds another vector to this vector in place.
impl AddAssign for Vec4 {
    fn add_assign(&mut self, other: Vec4) {
        *self = Vec4::new(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
            self.w + other.w,
        );
    }
}

/// Subtracts two vectors component-wise.
impl Sub for Vec4 {
    type Output = Vec4;
    fn sub(self, other: Vec4) -> Vec4 {
        Vec4::new(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
            self.w - other.w,
        )
    }
}

/// Subtracts a scalar from each component of the vector.
impl Sub<f32> for Vec4 {
    type Output = Vec4;
    fn sub(self, other: f32) -> Vec4 {
        Vec4::new(
            self.x - other,
            self.y - other,
            self.z - other,
            self.w - other,
        )
    }
}

/// Subtracts a vector from a scalar.
impl Sub<Vec4> for f32 {
    type Output = Vec4;
    fn sub(self, other: Vec4) -> Vec4 {
        Vec4::new(
            self - other.x,
            self - other.y,
            self - other.z,
            self - other.w,
        )
    }
}

/// Subtracts another vector from this vector in place.
impl SubAssign for Vec4 {
    fn sub_assign(&mut self, other: Vec4) {
        *self = Vec4::new(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
            self.w - other.w,
        );
    }
}

/// Multiplies two vectors component-wise (Hadamard product).
impl Mul<Vec4> for Vec4 {
    type Output = Vec4;
    fn mul(self, other: Vec4) -> Vec4 {
        Vec4::new(
            self.x * other.x,
            self.y * other.y,
            self.z * other.z,
            self.w * other.w,
        )
    }
}

/// Multiplies the vector by a scalar.
impl Mul<f32> for Vec4 {
    type Output = Vec4;
    fn mul(self, other: f32) -> Vec4 {
        Vec4::new(
            self.x * other,
            self.y * other,
            self.z * other,
            self.w * other,
        )
    }
}

/// Multiplies a scalar by a vector (commutative multiplication).
impl Mul<Vec4> for f32 {
    type Output = Vec4;
    fn mul(self, other: Vec4) -> Vec4 {
        Vec4::new(
            self * other.x,
            self * other.y,
            self * other.z,
            self * other.w,
        )
    }
}

/// Multiplies this vector by another vector in place (component-wise).
impl MulAssign<Vec4> for Vec4 {
    fn mul_assign(&mut self, other: Vec4) {
        *self = Vec4::new(
            self.x * other.x,
            self.y * other.y,
            self.z * other.z,
            self.w * other.w,
        );
    }
}

/// Divides this vector by another vector component-wise.
impl MulAssign<f32> for Vec4 {
    fn mul_assign(&mut self, other: f32) {
        *self = Vec4::new(
            self.x * other,
            self.y * other,
            self.z * other,
            self.w * other,
        );
    }
}

/// Divides two vectors component-wise.
impl Div<Vec4> for Vec4 {
    type Output = Vec4;
    fn div(self, other: Vec4) -> Vec4 {
        Vec4::new(
            self.x / other.x,
            self.y / other.y,
            self.z / other.z,
            self.w / other.w,
        )
    }
}

/// Divides the vector by a scalar.
impl Div<f32> for Vec4 {
    type Output = Vec4;
    fn div(self, other: f32) -> Vec4 {
        Vec4::new(
            self.x / other,
            self.y / other,
            self.z / other,
            self.w / other,
        )
    }
}

/// Divides a scalar by a vector component-wise.
impl Div<Vec4> for f32 {
    type Output = Vec4;
    fn div(self, other: Vec4) -> Vec4 {
        Vec4::new(
            self / other.x,
            self / other.y,
            self / other.z,
            self / other.w,
        )
    }
}

/// Divides this vector by another vector in place (component-wise).
impl DivAssign<Vec4> for Vec4 {
    fn div_assign(&mut self, other: Vec4) {
        *self = Vec4::new(
            self.x / other.x,
            self.y / other.y,
            self.z / other.z,
            self.w / other.w,
        );
    }
}

/// Divides this vector by a scalar in place.
impl DivAssign<f32> for Vec4 {
    fn div_assign(&mut self, other: f32) {
        *self = Vec4::new(
            self.x / other,
            self.y / other,
            self.z / other,
            self.w / other,
        );
    }
}

/// Negates the vector (multiplies each component by -1).
impl Neg for Vec4 {
    type Output = Vec4;
    fn neg(self) -> Vec4 {
        Vec4::new(-self.x, -self.y, -self.z, -self.w)
    }
}

/// Creates a zero vector by default.
impl Default for Vec4 {
    fn default() -> Self {
        Self::ZERO
    }
}

// Other Methods
impl Vec4 {
    /// Checks if two vectors are approximately equal within a default precision of 6 decimal places (micrometers)
    /// This is useful for comparing floating-point vectors where exact equality
    /// is not reliable due to precision issues.
    /// # Arguments
    /// * `other` - The other vector to compare against
    pub fn equals(self, other: Vec4) -> bool {
        self.equals_with_precision(other, 6)
    }

    /// Checks if two vectors are approximately equal within a given precision.
    /// This is useful for comparing floating-point vectors where exact equality
    /// is not reliable due to precision issues.
    /// # Arguments
    /// * `other` - The other vector to compare against
    /// * `precision` - The number of decimal places to consider for equality
    pub fn equals_with_precision(self, other: Vec4, precision: usize) -> bool {
        let factor = 10f32.powi(precision as i32);
        (self.x * factor).round() == (other.x * factor).round() &&
            (self.y * factor).round() == (other.y * factor).round() &&
            (self.z * factor).round() == (other.z * factor).round() &&
            (self.w * factor).round() == (other.w * factor).round()
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
    /// use crate::forge_engine::math::Vec4;
    ///
    /// let a = Vec4::new(1.0, 2.0, 3.0, 4.0);
    /// let b = Vec4::new(5.0, 6.0, 7.0, 8.0);
    /// assert_eq!(a.dot(b), 70.0); // 1*5 + 2*6 + 3*7 + 4*8
    /// ```
    #[inline]
    pub fn dot(self, other: Vec4) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
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
    /// use crate::forge_engine::math::Vec4;
    ///
    /// let v = Vec4::new(2.0, 3.0, 4.0, 5.0);
    /// let squared = v.pow(2.0);
    /// assert_eq!(squared, Vec4::new(4.0, 9.0, 16.0, 25.0));
    /// ```
    #[inline]
    pub fn pow(self, exp: f32) -> Vec4 {
        Vec4::new(
            self.x.powf(exp),
            self.y.powf(exp),
            self.z.powf(exp),
            self.w.powf(exp),
        )
    }

    /// Computes the square root of each component.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::math::Vec4;
    ///
    /// let v = Vec4::new(4.0, 9.0, 16.0, 25.0);
    /// let roots = v.sqrt();
    /// assert_eq!(roots, Vec4::new(2.0, 3.0, 4.0, 5.0));
    /// ```
    #[inline]
    pub fn sqrt(&self) -> Vec4 {
        Vec4::new(self.x.sqrt(), self.y.sqrt(), self.z.sqrt(), self.w.sqrt())
    }

    /// Computes the magnitude (length) of the vector.
    ///
    /// This is the Euclidean distance from the origin to the point
    /// represented by the vector in 4D space.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::math::Vec4;
    ///
    /// let v = Vec4::new(1.0, 2.0, 2.0, 4.0);
    /// assert_eq!(v.magnitude(), 5.0);
    /// ```
    #[inline]
    pub fn magnitude(self) -> f32 {
        self.dot(self).sqrt()
    }

    /// Computes the squared magnitude of the vector.
    ///
    /// This is more efficient than `magnitude()` when you only need
    /// to compare lengths or don't need the actual distance.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::math::Vec4;
    ///
    /// let v = Vec4::new(1.0, 2.0, 2.0, 4.0);
    /// assert_eq!(v.magnitude_squared(), 25.0);
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
    /// use crate::forge_engine::math::Vec4;
    ///
    /// let v = Vec4::new(0.0, 3.0, 0.0, 4.0);
    /// let normalized = v.normalized();
    /// assert_eq!(normalized, Vec4::new(0.0, 0.6, 0.0, 0.8));
    /// assert!((normalized.magnitude() - 1.0).abs() < f32::EPSILON);
    /// ```
    #[inline]
    pub fn normalized(self) -> Vec4 {
        let magnitude = self.magnitude();
        if magnitude == 0.0 {
            Vec4::ZERO
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
    /// use crate::forge_engine::math::Vec4;
    ///
    /// assert!(Vec4::ZERO.is_zero());
    /// assert!(!Vec4::ONE.is_zero());
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
    /// use crate::forge_engine::math::Vec4;
    ///
    /// assert!(Vec4::X.is_normalised());
    /// assert!(!Vec4::new(1.0, 2.0, 2.0, 4.0).is_normalised());
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
    /// use crate::forge_engine::math::Vec4;
    ///
    /// let v = Vec4::new(3.0, 4.0, 0.0, 0.0);
    /// assert!(v.safe_normal().is_some());
    /// assert!(Vec4::ZERO.safe_normal().is_none());
    /// ```
    pub fn safe_normal(self) -> Option<Vec4> {
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
    /// use crate::forge_engine::math::Vec4;
    ///
    /// let a = Vec4::new(0.0, 0.0, 0.0, 0.0);
    /// let b = Vec4::new(10.0, 20.0, 30.0, 40.0);
    /// let mid = a.lerp(b, 0.5);
    /// assert_eq!(mid, Vec4::new(5.0, 10.0, 15.0, 20.0));
    /// ```
    pub fn lerp(self, other: Vec4, t: f32) -> Vec4 {
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
    /// use crate::forge_engine::math::Vec4;
    ///
    /// let a = Vec4::new(1.0, 2.0, 3.0, 4.0);
    /// let b = Vec4::new(5.0, 6.0, 7.0, 8.0);
    /// let dist = a.distance(b);
    /// assert_eq!(dist, 8.0);
    /// ```
    pub fn distance(self, other: Vec4) -> f32 {
        (other - self).magnitude()
    }

    /// Computes the squared distance between two points.
    ///
    /// More efficient than `distance()` when you only need to compare distances.
    ///
    /// # Arguments
    ///
    /// * `other` - The other point
    pub fn distance_squared(self, other: Vec4) -> f32 {
        let diff = self - other;
        diff.dot(diff)
    }

    /// Returns a vector with the absolute value of each component.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::math::Vec4;
    ///
    /// let v = Vec4::new(-1.0, 2.0, -3.0, 4.0);
    /// assert_eq!(v.abs(), Vec4::new(1.0, 2.0, 3.0, 4.0));
    /// ```
    pub fn abs(self) -> Vec4 {
        Vec4::new(self.x.abs(), self.y.abs(), self.z.abs(), self.w.abs())
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
    /// use crate::forge_engine::math::Vec4;
    ///
    /// let a = Vec4::new(1.0, 5.0, 3.0, 7.0);
    /// let b = Vec4::new(4.0, 2.0, 6.0, 1.0);
    /// assert_eq!(a.min(b), Vec4::new(1.0, 2.0, 3.0, 1.0));
    /// ```
    pub fn min(self, other: Vec4) -> Vec4 {
        Vec4::new(
            self.x.min(other.x),
            self.y.min(other.y),
            self.z.min(other.z),
            self.w.min(other.w),
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
    /// use crate::forge_engine::math::Vec4;
    ///
    /// let a = Vec4::new(1.0, 5.0, 3.0, 7.0);
    /// let b = Vec4::new(4.0, 2.0, 6.0, 1.0);
    /// assert_eq!(a.max(b), Vec4::new(4.0, 5.0, 6.0, 7.0));
    /// ```
    pub fn max(self, other: Vec4) -> Vec4 {
        Vec4::new(
            self.x.max(other.x),
            self.y.max(other.y),
            self.z.max(other.z),
            self.w.max(other.w),
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
    /// use crate::forge_engine::math::Vec4;
    ///
    /// let v = Vec4::new(-1.0, 5.0, 2.0, 8.0);
    /// let min = Vec4::new(0.0, 0.0, 0.0, 0.0);
    /// let max = Vec4::new(3.0, 3.0, 3.0, 3.0);
    /// assert_eq!(v.clamp(min, max), Vec4::new(0.0, 3.0, 2.0, 3.0));
    /// ```
    pub fn clamp(self, min: Vec4, max: Vec4) -> Vec4 {
        self.max(min).min(max)
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
    /// use crate::forge_engine::math::Vec4;
    ///
    /// let v = Vec4::new(3.0, 4.0, 5.0, 6.0);
    /// let onto = Vec4::new(1.0, 0.0, 0.0, 0.0);
    /// let proj = v.project_onto(onto);
    /// assert_eq!(proj, Vec4::new(3.0, 0.0, 0.0, 0.0));
    /// ```
    pub fn project_onto(self, onto: Vec4) -> Vec4 {
        let d = onto.dot(onto);
        if d > 0.0 {
            onto * (self.dot(onto) / d)
        } else {
            Vec4::ZERO
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
    /// use crate::forge_engine::math::Vec4;
    ///
    /// let v = Vec4::new(3.0, 4.0, 5.0, 6.0);
    /// let from = Vec4::new(1.0, 0.0, 0.0, 0.0);
    /// let rej = v.reject_from(from);
    /// assert_eq!(rej, Vec4::new(0.0, 4.0, 5.0, 6.0));
    /// ```
    pub fn reject_from(self, from: Vec4) -> Vec4 {
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
    /// use crate::forge_engine::math::Vec4;
    ///
    /// let v = Vec4::new(1.0, -1.0, 0.0, 0.0);
    /// let normal = Vec4::new(0.0, 1.0, 0.0, 0.0);
    /// let reflected = v.reflect(normal);
    /// assert_eq!(reflected, Vec4::new(1.0, 1.0, 0.0, 0.0));
    /// ```
    pub fn reflect(self, normal: Vec4) -> Vec4 {
        self - normal * (2.0 * self.dot(normal))
    }
}

// Swizzling and Conversions
impl Vec4 {
    /// Returns the xyz components as a `Vector`, discarding the w component.
    ///
    /// This is useful when you need to extract 3D coordinates from homogeneous coordinates
    /// without performing perspective division.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::math::Vec4;
    /// use crate::forge_engine::Vec;
    ///
    /// let v4 = Vec4::new(1.0, 2.0, 3.0, 4.0);
    /// let v3 = v4.xyz();
    /// assert_eq!(v3, Vec::new(1.0, 2.0, 3.0));
    /// ```
    #[inline]
    pub fn xyz(self) -> Vec {
        Vec::new(self.x, self.y, self.z)
    }

    /// Converts to a `Vector` using perspective division if w != 0.
    ///
    /// If the w component is not zero, the x, y, and z components are divided by w
    /// to perform perspective division. If w is zero, returns the x, y, and z components directly.
    /// This is commonly used to convert from homogeneous coordinates back to 3D coordinates.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::math::Vec4;
    /// use crate::forge_engine::Vec;
    ///
    /// // Point in homogeneous coordinates
    /// let point = Vec4::new(2.0, 4.0, 6.0, 2.0);
    /// let v3 = point.vec3();
    /// assert_eq!(v3, Vec::new(1.0, 2.0, 3.0));
    ///
    /// // Direction vector (w = 0)
    /// let direction = Vec4::new(2.0, 4.0, 6.0, 0.0);
    /// let v3 = direction.vec3();
    /// assert_eq!(v3, Vec::new(2.0, 4.0, 6.0));
    /// ```
    #[inline]
    pub fn vec3(self) -> Vec {
        if self.w != 0.0 {
            Vec::new(self.x / self.w, self.y / self.w, self.z / self.w)
        } else {
            Vec::new(self.x, self.y, self.z)
        }
    }

    /// Returns a `Vector2` with the x and y components.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::math::{Vec2, Vec4};
    ///
    /// let v4 = Vec4::new(1.0, 2.0, 3.0, 4.0);
    /// let v2 = v4.xy();
    /// assert_eq!(v2, Vec2::new(1.0, 2.0));
    /// ```
    pub fn xy(self) -> Vec2 {
        Vec2::new(self.x, self.y)
    }

    /// Returns a `Vector2` with the x and z components.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::math::{Vec2, Vec4};
    ///
    /// let v4 = Vec4::new(1.0, 2.0, 3.0, 4.0);
    /// let v2 = v4.xz();
    /// assert_eq!(v2, Vec2::new(1.0, 3.0));
    /// ```
    pub fn xz(self) -> Vec2 {
        Vec2::new(self.x, self.z)
    }

    /// Returns a `Vector2` with the x and w components.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::math::{Vec2, Vec4};
    ///
    /// let v4 = Vec4::new(1.0, 2.0, 3.0, 4.0);
    /// let v2 = v4.xw();
    /// assert_eq!(v2, Vec2::new(1.0, 4.0));
    /// ```
    pub fn xw(self) -> Vec2 {
        Vec2::new(self.x, self.w)
    }

    /// Returns a `Vector2` with the y and z components.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::math::{Vec2, Vec4};
    ///
    /// let v4 = Vec4::new(1.0, 2.0, 3.0, 4.0);
    /// let v2 = v4.yz();
    /// assert_eq!(v2, Vec2::new(2.0, 3.0));
    /// ```
    pub fn yz(self) -> Vec2 {
        Vec2::new(self.y, self.z)
    }

    /// Returns a `Vector2` with the y and w components.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::math::{Vec2, Vec4};
    ///
    /// let v4 = Vec4::new(1.0, 2.0, 3.0, 4.0);
    /// let v2 = v4.yw();
    /// assert_eq!(v2, Vec2::new(2.0, 4.0));
    /// ```
    pub fn yw(self) -> Vec2 {
        Vec2::new(self.y, self.w)
    }

    /// Returns a `Vector2` with the z and w components.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::math::{Vec2, Vec4};
    ///
    /// let v4 = Vec4::new(1.0, 2.0, 3.0, 4.0);
    /// let v2 = v4.zw();
    /// assert_eq!(v2, Vec2::new(3.0, 4.0));
    /// ```
    pub fn zw(self) -> Vec2 {
        Vec2::new(self.z, self.w)
    }

    /// Returns a `Vector2` with both components set to x.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::math::{Vec2, Vec4};
    ///
    /// let v4 = Vec4::new(1.0, 2.0, 3.0, 4.0);
    /// let v2 = v4.xx();
    /// assert_eq!(v2, Vec2::new(1.0, 1.0));
    /// ```
    pub fn xx(self) -> Vec2 {
        Vec2::new(self.x, self.x)
    }

    /// Returns a `Vector2` with both components set to y.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::math::{Vec2, Vec4};
    ///
    /// let v4 = Vec4::new(1.0, 2.0, 3.0, 4.0);
    /// let v2 = v4.yy();
    /// assert_eq!(v2, Vec2::new(2.0, 2.0));
    /// ```
    pub fn yy(self) -> Vec2 {
        Vec2::new(self.y, self.y)
    }

    /// Returns a `Vector2` with both components set to z.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::math::{Vec2, Vec4};
    ///
    /// let v4 = Vec4::new(1.0, 2.0, 3.0, 4.0);
    /// let v2 = v4.zz();
    /// assert_eq!(v2, Vec2::new(3.0, 3.0));
    /// ```
    pub fn zz(self) -> Vec2 {
        Vec2::new(self.z, self.z)
    }

    /// Returns a `Vector2` with both components set to w.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::math::{Vec2, Vec4};
    ///
    /// let v4 = Vec4::new(1.0, 2.0, 3.0, 4.0);
    /// let v2 = v4.ww();
    /// assert_eq!(v2, Vec2::new(4.0, 4.0));
    /// ```
    pub fn ww(self) -> Vec2 {
        Vec2::new(self.w, self.w)
    }
}

#[cfg(test)]
mod tests;

