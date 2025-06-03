//! 4D vector mathematics implementation.
//!
//! This module provides a complete 4D vector type with comprehensive mathematical
//! operations, conversions, and utility functions commonly needed in graphics,
//! physics, and game development. Vector4 is commonly used for homogeneous
//! coordinates, RGBA colors, quaternions, and other 4D mathematical operations.

use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use crate::math::{Vector, Vector2};

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
/// use crate::forge_engine::math::Vector4;
/// use crate::forge_engine::Vector;
///
/// // Create vectors
/// let v1 = Vector4::new(1.0, 2.0, 3.0, 4.0);
/// let v2 = Vector4::from((5.0, 6.0, 7.0, 8.0));
/// let v3 = Vector4::from_vec3(Vector::new(1.0, 2.0, 3.0), 1.0);
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
/// let point = Vector4::from_vec3(Vector::new(10.0, 20.0, 30.0), 1.0);
/// let direction = Vector4::from_vec3(Vector::new(1.0, 0.0, 0.0), 0.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vector4 {
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
impl Vector4 {
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
    /// use crate::forge_engine::math::Vector4;
    ///
    /// let v = Vector4::new(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(v.x, 1.0);
    /// assert_eq!(v.y, 2.0);
    /// assert_eq!(v.z, 3.0);
    /// assert_eq!(v.w, 4.0);
    /// ```
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Vector4 { x, y, z, w }
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
    /// use crate::forge_engine::math::Vector4;
    /// use crate::forge_engine::Vector;
    ///
    /// let point = Vector4::from_vec3(Vector::new(1.0, 2.0, 3.0), 1.0);
    /// let direction = Vector4::from_vec3(Vector::new(0.0, 1.0, 0.0), 0.0);
    /// ```
    pub fn from_vec3(v: Vector, w: f32) -> Self {
        Vector4 { x: v.x, y: v.y, z: v.z, w }
    }
}

// Constant constructors
impl Vector4 {
    /// A vector with all components set to zero: `(0, 0, 0, 0)`
    pub const ZERO: Vector4 = Vector4 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: 0.0,
    };

    /// A vector with all components set to one: `(1, 1, 1, 1)`
    pub const ONE: Vector4 = Vector4 {
        x: 1.0,
        y: 1.0,
        z: 1.0,
        w: 1.0,
    };

    /// A unit vector in the x direction: `(1, 0, 0, 0)`
    pub const X: Vector4 = Vector4 {
        x: 1.0,
        y: 0.0,
        z: 0.0,
        w: 0.0,
    };

    /// A unit vector in the y direction: `(0, 1, 0, 0)`
    pub const Y: Vector4 = Vector4 {
        x: 0.0,
        y: 1.0,
        z: 0.0,
        w: 0.0,
    };

    /// A unit vector in the z direction: `(0, 0, 1, 0)`
    pub const Z: Vector4 = Vector4 {
        x: 0.0,
        y: 0.0,
        z: 1.0,
        w: 0.0,
    };

    /// A unit vector in the w direction: `(0, 0, 0, 1)`
    pub const W: Vector4 = Vector4 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: 1.0,
    };
}

/// Converts a tuple `(f32, f32, f32, f32)` into a `Vector4`.
///
/// # Examples
///
/// ```rust
/// use crate::forge_engine::math::Vector4;
///
/// let v: Vector4 = (1.0, 2.0, 3.0, 4.0).into();
/// assert_eq!(v, Vector4::new(1.0, 2.0, 3.0, 4.0));
/// ```
impl From<(f32, f32, f32, f32)> for Vector4 {
    fn from(t: (f32, f32, f32, f32)) -> Self {
        Vector4::new(t.0, t.1, t.2, t.3)
    }
}

/// Converts a `Vector` into a `Vector4` with `w` set to `1.0`.
///
/// This is useful for converting 3D points to homogeneous coordinates.
///
/// # Examples
///
/// ```rust
/// use crate::forge_engine::math::Vector4;
/// use crate::forge_engine::Vector;
///
/// let v3 = Vector::new(1.0, 2.0, 3.0);
/// let v4: Vector4 = v3.into();
/// assert_eq!(v4, Vector4::new(1.0, 2.0, 3.0, 1.0));
/// ```
impl From<Vector> for Vector4 {
    fn from(v: Vector) -> Self {
        Vector4::new(v.x, v.y, v.z, 1.0)
    }
}

/// Converts a `Vector2` into a `Vector4` with `z` and `w` set to `0.0` and `1.0`.
///
/// # Examples
///
/// ```rust
/// use crate::forge_engine::math::{Vector2, Vector4};
///
/// let v2 = Vector2::new(1.0, 2.0);
/// let v4: Vector4 = v2.into();
/// assert_eq!(v4, Vector4::new(1.0, 2.0, 0.0, 1.0));
/// ```
impl From<Vector2> for Vector4 {
    fn from(v: Vector2) -> Self {
        Vector4::new(v.x, v.y, 0.0, 1.0)
    }
}

/// Converts an array `[f32; 4]` into a `Vector4`.
///
/// # Examples
///
/// ```rust
/// use crate::forge_engine::math::Vector4;
///
/// let v: Vector4 = [1.0, 2.0, 3.0, 4.0].into();
/// assert_eq!(v, Vector4::new(1.0, 2.0, 3.0, 4.0));
/// ```
impl From<[f32; 4]> for Vector4 {
    fn from(arr: [f32; 4]) -> Self {
        Vector4::new(arr[0], arr[1], arr[2], arr[3])
    }
}

/// Converts a `Vector4` into a tuple `(f32, f32, f32, f32)`.
impl From<Vector4> for (f32, f32, f32, f32) {
    fn from(v: Vector4) -> Self {
        (v.x, v.y, v.z, v.w)
    }
}

/// Converts a `Vector4` into an array `[f32; 4]`.
impl From<Vector4> for [f32; 4] {
    fn from(v: Vector4) -> Self {
        [v.x, v.y, v.z, v.w]
    }
}

/// Converts a `Vector4` into a `Vector` using perspective division.
///
/// If `w` is not zero, the x, y, and z components are divided by `w`.
/// If `w` is zero, returns the x, y, and z components directly.
impl From<Vector4> for Vector {
    fn from(v: Vector4) -> Self {
        if v.w != 0.0 {
            Vector::new(v.x / v.w, v.y / v.w, v.z / v.w)
        } else {
            Vector::new(v.x, v.y, v.z)
        }
    }
}

// Operators

/// Adds two vectors component-wise.
impl Add for Vector4 {
    type Output = Vector4;
    fn add(self, other: Vector4) -> Vector4 {
        Vector4::new(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
            self.w + other.w,
        )
    }
}

/// Adds a scalar to each component of the vector.
impl Add<f32> for Vector4 {
    type Output = Vector4;
    fn add(self, other: f32) -> Vector4 {
        Vector4::new(
            self.x + other,
            self.y + other,
            self.z + other,
            self.w + other,
        )
    }
}

/// Adds a vector to a scalar (commutative addition).
impl Add<Vector4> for f32 {
    type Output = Vector4;
    fn add(self, other: Vector4) -> Vector4 {
        Vector4::new(
            self + other.x,
            self + other.y,
            self + other.z,
            self + other.w,
        )
    }
}

/// Adds another vector to this vector in place.
impl AddAssign for Vector4 {
    fn add_assign(&mut self, other: Vector4) {
        *self = Vector4::new(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
            self.w + other.w,
        );
    }
}

/// Subtracts two vectors component-wise.
impl Sub for Vector4 {
    type Output = Vector4;
    fn sub(self, other: Vector4) -> Vector4 {
        Vector4::new(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
            self.w - other.w,
        )
    }
}

/// Subtracts a scalar from each component of the vector.
impl Sub<f32> for Vector4 {
    type Output = Vector4;
    fn sub(self, other: f32) -> Vector4 {
        Vector4::new(
            self.x - other,
            self.y - other,
            self.z - other,
            self.w - other,
        )
    }
}

/// Subtracts a vector from a scalar.
impl Sub<Vector4> for f32 {
    type Output = Vector4;
    fn sub(self, other: Vector4) -> Vector4 {
        Vector4::new(
            self - other.x,
            self - other.y,
            self - other.z,
            self - other.w,
        )
    }
}

/// Subtracts another vector from this vector in place.
impl SubAssign for Vector4 {
    fn sub_assign(&mut self, other: Vector4) {
        *self = Vector4::new(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
            self.w - other.w,
        );
    }
}

/// Multiplies two vectors component-wise (Hadamard product).
impl Mul<Vector4> for Vector4 {
    type Output = Vector4;
    fn mul(self, other: Vector4) -> Vector4 {
        Vector4::new(
            self.x * other.x,
            self.y * other.y,
            self.z * other.z,
            self.w * other.w,
        )
    }
}

/// Multiplies the vector by a scalar.
impl Mul<f32> for Vector4 {
    type Output = Vector4;
    fn mul(self, other: f32) -> Vector4 {
        Vector4::new(
            self.x * other,
            self.y * other,
            self.z * other,
            self.w * other,
        )
    }
}

/// Multiplies a scalar by a vector (commutative multiplication).
impl Mul<Vector4> for f32 {
    type Output = Vector4;
    fn mul(self, other: Vector4) -> Vector4 {
        Vector4::new(
            self * other.x,
            self * other.y,
            self * other.z,
            self * other.w,
        )
    }
}

/// Multiplies this vector by another vector in place (component-wise).
impl MulAssign<Vector4> for Vector4 {
    fn mul_assign(&mut self, other: Vector4) {
        *self = Vector4::new(
            self.x * other.x,
            self.y * other.y,
            self.z * other.z,
            self.w * other.w,
        );
    }
}

/// Divides two vectors component-wise.
impl Div<Vector4> for Vector4 {
    type Output = Vector4;
    fn div(self, other: Vector4) -> Vector4 {
        Vector4::new(
            self.x / other.x,
            self.y / other.y,
            self.z / other.z,
            self.w / other.w,
        )
    }
}

/// Divides the vector by a scalar.
impl Div<f32> for Vector4 {
    type Output = Vector4;
    fn div(self, other: f32) -> Vector4 {
        Vector4::new(
            self.x / other,
            self.y / other,
            self.z / other,
            self.w / other,
        )
    }
}

/// Divides a scalar by a vector component-wise.
impl Div<Vector4> for f32 {
    type Output = Vector4;
    fn div(self, other: Vector4) -> Vector4 {
        Vector4::new(
            self / other.x,
            self / other.y,
            self / other.z,
            self / other.w,
        )
    }
}

/// Divides this vector by another vector in place (component-wise).
impl DivAssign<Vector4> for Vector4 {
    fn div_assign(&mut self, other: Vector4) {
        *self = Vector4::new(
            self.x / other.x,
            self.y / other.y,
            self.z / other.z,
            self.w / other.w,
        );
    }
}

/// Negates the vector (multiplies each component by -1).
impl Neg for Vector4 {
    type Output = Vector4;
    fn neg(self) -> Vector4 {
        Vector4::new(-self.x, -self.y, -self.z, -self.w)
    }
}

/// Creates a zero vector by default.
impl Default for Vector4 {
    fn default() -> Self {
        Self::ZERO
    }
}

// Other Methods
impl Vector4 {
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
    /// use crate::forge_engine::math::Vector4;
    ///
    /// let a = Vector4::new(1.0, 2.0, 3.0, 4.0);
    /// let b = Vector4::new(5.0, 6.0, 7.0, 8.0);
    /// assert_eq!(a.dot(b), 70.0); // 1*5 + 2*6 + 3*7 + 4*8
    /// ```
    #[inline]
    pub fn dot(self, other: Vector4) -> f32 {
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
    /// use crate::forge_engine::math::Vector4;
    ///
    /// let v = Vector4::new(2.0, 3.0, 4.0, 5.0);
    /// let squared = v.pow(2.0);
    /// assert_eq!(squared, Vector4::new(4.0, 9.0, 16.0, 25.0));
    /// ```
    #[inline]
    pub fn pow(self, exp: f32) -> Vector4 {
        Vector4::new(
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
    /// use crate::forge_engine::math::Vector4;
    ///
    /// let v = Vector4::new(4.0, 9.0, 16.0, 25.0);
    /// let roots = v.sqrt();
    /// assert_eq!(roots, Vector4::new(2.0, 3.0, 4.0, 5.0));
    /// ```
    #[inline]
    pub fn sqrt(&self) -> Vector4 {
        Vector4::new(
            self.x.sqrt(),
            self.y.sqrt(),
            self.z.sqrt(),
            self.w.sqrt(),
        )
    }

    /// Computes the magnitude (length) of the vector.
    ///
    /// This is the Euclidean distance from the origin to the point
    /// represented by the vector in 4D space.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::math::Vector4;
    ///
    /// let v = Vector4::new(1.0, 2.0, 2.0, 4.0);
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
    /// use crate::forge_engine::math::Vector4;
    ///
    /// let v = Vector4::new(1.0, 2.0, 2.0, 4.0);
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
    /// use crate::forge_engine::math::Vector4;
    ///
    /// let v = Vector4::new(0.0, 3.0, 0.0, 4.0);
    /// let normalized = v.normalized();
    /// assert_eq!(normalized, Vector4::new(0.0, 0.6, 0.0, 0.8));
    /// assert!((normalized.magnitude() - 1.0).abs() < f32::EPSILON);
    /// ```
    #[inline]
    pub fn normalized(self) -> Vector4 {
        let magnitude = self.magnitude();
        if magnitude == 0.0 {
            Vector4::ZERO
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
    /// use crate::forge_engine::math::Vector4;
    ///
    /// assert!(Vector4::ZERO.is_zero());
    /// assert!(!Vector4::ONE.is_zero());
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
    /// use crate::forge_engine::math::Vector4;
    ///
    /// assert!(Vector4::X.is_normalised());
    /// assert!(!Vector4::new(1.0, 2.0, 2.0, 4.0).is_normalised());
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
    /// use crate::forge_engine::math::Vector4;
    ///
    /// let v = Vector4::new(3.0, 4.0, 0.0, 0.0);
    /// assert!(v.safe_normal().is_some());
    /// assert!(Vector4::ZERO.safe_normal().is_none());
    /// ```
    pub fn safe_normal(self) -> Option<Vector4> {
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
    /// use crate::forge_engine::math::Vector4;
    ///
    /// let a = Vector4::new(0.0, 0.0, 0.0, 0.0);
    /// let b = Vector4::new(10.0, 20.0, 30.0, 40.0);
    /// let mid = a.lerp(b, 0.5);
    /// assert_eq!(mid, Vector4::new(5.0, 10.0, 15.0, 20.0));
    /// ```
    pub fn lerp(self, other: Vector4, t: f32) -> Vector4 {
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
    /// use crate::forge_engine::math::Vector4;
    ///
    /// let a = Vector4::new(1.0, 2.0, 3.0, 4.0);
    /// let b = Vector4::new(5.0, 6.0, 7.0, 8.0);
    /// let dist = a.distance(b);
    /// assert_eq!(dist, 8.0);
    /// ```
    pub fn distance(self, other: Vector4) -> f32 {
        (other - self).magnitude()
    }

    /// Computes the squared distance between two points.
    ///
    /// More efficient than `distance()` when you only need to compare distances.
    ///
    /// # Arguments
    ///
    /// * `other` - The other point
    pub fn distance_squared(self, other: Vector4) -> f32 {
        let diff = self - other;
        diff.dot(diff)
    }

    /// Returns a vector with the absolute value of each component.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::math::Vector4;
    ///
    /// let v = Vector4::new(-1.0, 2.0, -3.0, 4.0);
    /// assert_eq!(v.abs(), Vector4::new(1.0, 2.0, 3.0, 4.0));
    /// ```
    pub fn abs(self) -> Vector4 {
        Vector4::new(self.x.abs(), self.y.abs(), self.z.abs(), self.w.abs())
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
    /// use crate::forge_engine::math::Vector4;
    ///
    /// let a = Vector4::new(1.0, 5.0, 3.0, 7.0);
    /// let b = Vector4::new(4.0, 2.0, 6.0, 1.0);
    /// assert_eq!(a.min(b), Vector4::new(1.0, 2.0, 3.0, 1.0));
    /// ```
    pub fn min(self, other: Vector4) -> Vector4 {
        Vector4::new(
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
    /// use crate::forge_engine::math::Vector4;
    ///
    /// let a = Vector4::new(1.0, 5.0, 3.0, 7.0);
    /// let b = Vector4::new(4.0, 2.0, 6.0, 1.0);
    /// assert_eq!(a.max(b), Vector4::new(4.0, 5.0, 6.0, 7.0));
    /// ```
    pub fn max(self, other: Vector4) -> Vector4 {
        Vector4::new(
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
    /// use crate::forge_engine::math::Vector4;
    ///
    /// let v = Vector4::new(-1.0, 5.0, 2.0, 8.0);
    /// let min = Vector4::new(0.0, 0.0, 0.0, 0.0);
    /// let max = Vector4::new(3.0, 3.0, 3.0, 3.0);
    /// assert_eq!(v.clamp(min, max), Vector4::new(0.0, 3.0, 2.0, 3.0));
    /// ```
    pub fn clamp(self, min: Vector4, max: Vector4) -> Vector4 {
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
    /// use crate::forge_engine::math::Vector4;
    ///
    /// let v = Vector4::new(3.0, 4.0, 5.0, 6.0);
    /// let onto = Vector4::new(1.0, 0.0, 0.0, 0.0);
    /// let proj = v.project_onto(onto);
    /// assert_eq!(proj, Vector4::new(3.0, 0.0, 0.0, 0.0));
    /// ```
    pub fn project_onto(self, onto: Vector4) -> Vector4 {
        let d = onto.dot(onto);
        if d > 0.0 {
            onto * (self.dot(onto) / d)
        } else {
            Vector4::ZERO
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
    /// use crate::forge_engine::math::Vector4;
    ///
    /// let v = Vector4::new(3.0, 4.0, 5.0, 6.0);
    /// let from = Vector4::new(1.0, 0.0, 0.0, 0.0);
    /// let rej = v.reject_from(from);
    /// assert_eq!(rej, Vector4::new(0.0, 4.0, 5.0, 6.0));
    /// ```
    pub fn reject_from(self, from: Vector4) -> Vector4 {
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
    /// use crate::forge_engine::math::Vector4;
    ///
    /// let v = Vector4::new(1.0, -1.0, 0.0, 0.0);
    /// let normal = Vector4::new(0.0, 1.0, 0.0, 0.0);
    /// let reflected = v.reflect(normal);
    /// assert_eq!(reflected, Vector4::new(1.0, 1.0, 0.0, 0.0));
    /// ```
    pub fn reflect(self, normal: Vector4) -> Vector4 {
        self - normal * (2.0 * self.dot(normal))
    }
}

// Swizzling and Conversions
impl Vector4 {
    /// Returns the xyz components as a `Vector`, discarding the w component.
    ///
    /// This is useful when you need to extract 3D coordinates from homogeneous coordinates
    /// without performing perspective division.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::math::Vector4;
    /// use crate::forge_engine::Vector;
    ///
    /// let v4 = Vector4::new(1.0, 2.0, 3.0, 4.0);
    /// let v3 = v4.xyz();
    /// assert_eq!(v3, Vector::new(1.0, 2.0, 3.0));
    /// ```
    #[inline]
    pub fn xyz(self) -> Vector {
        Vector::new(self.x, self.y, self.z)
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
    /// use crate::forge_engine::math::Vector4;
    /// use crate::forge_engine::Vector;
    ///
    /// // Point in homogeneous coordinates
    /// let point = Vector4::new(2.0, 4.0, 6.0, 2.0);
    /// let v3 = point.vec3();
    /// assert_eq!(v3, Vector::new(1.0, 2.0, 3.0));
    ///
    /// // Direction vector (w = 0)
    /// let direction = Vector4::new(2.0, 4.0, 6.0, 0.0);
    /// let v3 = direction.vec3();
    /// assert_eq!(v3, Vector::new(2.0, 4.0, 6.0));
    /// ```
    #[inline]
    pub fn vec3(self) -> Vector {
        if self.w != 0.0 {
            Vector::new(self.x / self.w, self.y / self.w, self.z / self.w)
        } else {
            Vector::new(self.x, self.y, self.z)
        }
    }

    /// Returns a `Vector2` with the x and y components.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::math::{Vector2, Vector4};
    ///
    /// let v4 = Vector4::new(1.0, 2.0, 3.0, 4.0);
    /// let v2 = v4.xy();
    /// assert_eq!(v2, Vector2::new(1.0, 2.0));
    /// ```
    pub fn xy(self) -> Vector2 {
        Vector2::new(self.x, self.y)
    }

    /// Returns a `Vector2` with the x and z components.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::math::{Vector2, Vector4};
    ///
    /// let v4 = Vector4::new(1.0, 2.0, 3.0, 4.0);
    /// let v2 = v4.xz();
    /// assert_eq!(v2, Vector2::new(1.0, 3.0));
    /// ```
    pub fn xz(self) -> Vector2 {
        Vector2::new(self.x, self.z)
    }

    /// Returns a `Vector2` with the x and w components.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::math::{Vector2, Vector4};
    ///
    /// let v4 = Vector4::new(1.0, 2.0, 3.0, 4.0);
    /// let v2 = v4.xw();
    /// assert_eq!(v2, Vector2::new(1.0, 4.0));
    /// ```
    pub fn xw(self) -> Vector2 {
        Vector2::new(self.x, self.w)
    }

    /// Returns a `Vector2` with the y and z components.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::math::{Vector2, Vector4};
    ///
    /// let v4 = Vector4::new(1.0, 2.0, 3.0, 4.0);
    /// let v2 = v4.yz();
    /// assert_eq!(v2, Vector2::new(2.0, 3.0));
    /// ```
    pub fn yz(self) -> Vector2 {
        Vector2::new(self.y, self.z)
    }

    /// Returns a `Vector2` with the y and w components.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::math::{Vector2, Vector4};
    ///
    /// let v4 = Vector4::new(1.0, 2.0, 3.0, 4.0);
    /// let v2 = v4.yw();
    /// assert_eq!(v2, Vector2::new(2.0, 4.0));
    /// ```
    pub fn yw(self) -> Vector2 {
        Vector2::new(self.y, self.w)
    }

    /// Returns a `Vector2` with the z and w components.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::math::{Vector2, Vector4};
    ///
    /// let v4 = Vector4::new(1.0, 2.0, 3.0, 4.0);
    /// let v2 = v4.zw();
    /// assert_eq!(v2, Vector2::new(3.0, 4.0));
    /// ```
    pub fn zw(self) -> Vector2 {
        Vector2::new(self.z, self.w)
    }

    /// Returns a `Vector2` with both components set to x.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::math::{Vector2, Vector4};
    ///
    /// let v4 = Vector4::new(1.0, 2.0, 3.0, 4.0);
    /// let v2 = v4.xx();
    /// assert_eq!(v2, Vector2::new(1.0, 1.0));
    /// ```
    pub fn xx(self) -> Vector2 {
        Vector2::new(self.x, self.x)
    }

    /// Returns a `Vector2` with both components set to y.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::math::{Vector2, Vector4};
    ///
    /// let v4 = Vector4::new(1.0, 2.0, 3.0, 4.0);
    /// let v2 = v4.yy();
    /// assert_eq!(v2, Vector2::new(2.0, 2.0));
    /// ```
    pub fn yy(self) -> Vector2 {
        Vector2::new(self.y, self.y)
    }

    /// Returns a `Vector2` with both components set to z.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::math::{Vector2, Vector4};
    ///
    /// let v4 = Vector4::new(1.0, 2.0, 3.0, 4.0);
    /// let v2 = v4.zz();
    /// assert_eq!(v2, Vector2::new(3.0, 3.0));
    /// ```
    pub fn zz(self) -> Vector2 {
        Vector2::new(self.z, self.z)
    }

    /// Returns a `Vector2` with both components set to w.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::math::{Vector2, Vector4};
    ///
    /// let v4 = Vector4::new(1.0, 2.0, 3.0, 4.0);
    /// let v2 = v4.ww();
    /// assert_eq!(v2, Vector2::new(4.0, 4.0));
    /// ```
    pub fn ww(self) -> Vector2 {
        Vector2::new(self.w, self.w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper function for floating point comparisons
    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-6
    }

    // ============ Vector4 Tests ============
    #[cfg(test)]
    mod vector4_tests {
        use super::*;

        #[test]
        fn test_vector4_construction() {
            let v = Vector4::new(1.0, 2.0, 3.0, 4.0);
            assert_eq!(v.x, 1.0);
            assert_eq!(v.y, 2.0);
            assert_eq!(v.z, 3.0);
            assert_eq!(v.w, 4.0);
        }

        #[test]
        fn test_vector4_from_vec3() {
            let v3 = Vector::new(1.0, 2.0, 3.0);
            let v4 = Vector4::from_vec3(v3, 4.0);
            assert_eq!(v4, Vector4::new(1.0, 2.0, 3.0, 4.0));
        }

        #[test]
        fn test_vector4_constants() {
            assert_eq!(Vector4::ZERO, Vector4::new(0.0, 0.0, 0.0, 0.0));
            assert_eq!(Vector4::ONE, Vector4::new(1.0, 1.0, 1.0, 1.0));
            assert_eq!(Vector4::X, Vector4::new(1.0, 0.0, 0.0, 0.0));
            assert_eq!(Vector4::Y, Vector4::new(0.0, 1.0, 0.0, 0.0));
            assert_eq!(Vector4::Z, Vector4::new(0.0, 0.0, 1.0, 0.0));
            assert_eq!(Vector4::W, Vector4::new(0.0, 0.0, 0.0, 1.0));
        }

        #[test]
        fn test_vector4_from_tuple() {
            let v: Vector4 = (1.0, 2.0, 3.0, 4.0).into();
            assert_eq!(v, Vector4::new(1.0, 2.0, 3.0, 4.0));
        }

        #[test]
        fn test_vector4_from_array() {
            let v: Vector4 = [1.0, 2.0, 3.0, 4.0].into();
            assert_eq!(v, Vector4::new(1.0, 2.0, 3.0, 4.0));
        }

        #[test]
        fn test_vector4_from_vector3() {
            let v3 = Vector::new(1.0, 2.0, 3.0);
            let v4: Vector4 = v3.into();
            assert_eq!(v4, Vector4::new(1.0, 2.0, 3.0, 1.0));
        }

        #[test]
        fn test_vector4_from_vector2() {
            let v2 = Vector2::new(1.0, 2.0);
            let v4: Vector4 = v2.into();
            assert_eq!(v4, Vector4::new(1.0, 2.0, 0.0, 1.0));
        }

        #[test]
        fn test_vector4_to_tuple() {
            let v = Vector4::new(1.0, 2.0, 3.0, 4.0);
            let t: (f32, f32, f32, f32) = v.into();
            assert_eq!(t, (1.0, 2.0, 3.0, 4.0));
        }

        #[test]
        fn test_vector4_to_array() {
            let v = Vector4::new(1.0, 2.0, 3.0, 4.0);
            let arr: [f32; 4] = v.into();
            assert_eq!(arr, [1.0, 2.0, 3.0, 4.0]);
        }

        #[test]
        fn test_vector4_to_vector3() {
            let v = Vector4::new(2.0, 4.0, 6.0, 2.0);
            let v3: Vector = v.into();
            assert_eq!(v3, Vector::new(1.0, 2.0, 3.0));

            let v = Vector4::new(2.0, 4.0, 6.0, 0.0);
            let v3: Vector = v.into();
            assert_eq!(v3, Vector::new(2.0, 4.0, 6.0));
        }

        #[test]
        fn test_vector4_add() {
            let a = Vector4::new(1.0, 2.0, 3.0, 4.0);
            let b = Vector4::new(5.0, 6.0, 7.0, 8.0);
            assert_eq!(a + b, Vector4::new(6.0, 8.0, 10.0, 12.0));
            assert_eq!(a + 10.0, Vector4::new(11.0, 12.0, 13.0, 14.0));
            assert_eq!(10.0 + a, Vector4::new(11.0, 12.0, 13.0, 14.0));
        }

        #[test]
        fn test_vector4_add_assign() {
            let mut v = Vector4::new(1.0, 2.0, 3.0, 4.0);
            v += Vector4::new(5.0, 6.0, 7.0, 8.0);
            assert_eq!(v, Vector4::new(6.0, 8.0, 10.0, 12.0));
        }

        #[test]
        fn test_vector4_sub() {
            let a = Vector4::new(5.0, 6.0, 7.0, 8.0);
            let b = Vector4::new(1.0, 2.0, 3.0, 4.0);
            assert_eq!(a - b, Vector4::new(4.0, 4.0, 4.0, 4.0));
            assert_eq!(a - 2.0, Vector4::new(3.0, 4.0, 5.0, 6.0));
            assert_eq!(10.0 - a, Vector4::new(5.0, 4.0, 3.0, 2.0));
        }

        #[test]
        fn test_vector4_sub_assign() {
            let mut v = Vector4::new(5.0, 6.0, 7.0, 8.0);
            v -= Vector4::new(1.0, 2.0, 3.0, 4.0);
            assert_eq!(v, Vector4::new(4.0, 4.0, 4.0, 4.0));
        }

        #[test]
        fn test_vector4_mul() {
            let a = Vector4::new(2.0, 3.0, 4.0, 5.0);
            let b = Vector4::new(6.0, 7.0, 8.0, 9.0);
            assert_eq!(a * b, Vector4::new(12.0, 21.0, 32.0, 45.0));
            assert_eq!(a * 2.0, Vector4::new(4.0, 6.0, 8.0, 10.0));
            assert_eq!(2.0 * a, Vector4::new(4.0, 6.0, 8.0, 10.0));
        }

        #[test]
        fn test_vector4_mul_assign() {
            let mut v = Vector4::new(2.0, 3.0, 4.0, 5.0);
            v *= Vector4::new(6.0, 7.0, 8.0, 9.0);
            assert_eq!(v, Vector4::new(12.0, 21.0, 32.0, 45.0));
        }

        #[test]
        fn test_vector4_div() {
            let a = Vector4::new(12.0, 21.0, 32.0, 45.0);
            let b = Vector4::new(6.0, 7.0, 8.0, 9.0);
            assert_eq!(a / b, Vector4::new(2.0, 3.0, 4.0, 5.0));
            assert_eq!(a / 2.0, Vector4::new(6.0, 10.5, 16.0, 22.5));
            assert_eq!(60.0 / a, Vector4::new(5.0, 60.0/21.0, 60.0/32.0, 60.0/45.0));
        }

        #[test]
        fn test_vector4_div_assign() {
            let mut v = Vector4::new(12.0, 21.0, 32.0, 45.0);
            v /= Vector4::new(6.0, 7.0, 8.0, 9.0);
            assert_eq!(v, Vector4::new(2.0, 3.0, 4.0, 5.0));
        }

        #[test]
        fn test_vector4_neg() {
            let v = Vector4::new(1.0, -2.0, 3.0, -4.0);
            assert_eq!(-v, Vector4::new(-1.0, 2.0, -3.0, 4.0));
        }

        #[test]
        fn test_vector4_dot() {
            let a = Vector4::new(1.0, 2.0, 3.0, 4.0);
            let b = Vector4::new(5.0, 6.0, 7.0, 8.0);
            assert_eq!(a.dot(b), 70.0);
        }

        #[test]
        fn test_vector4_magnitude() {
            let v = Vector4::new(1.0, 2.0, 2.0, 4.0);
            assert_eq!(v.magnitude(), 5.0);
            assert_eq!(v.magnitude_squared(), 25.0);
        }

        #[test]
        fn test_vector4_normalized() {
            let v = Vector4::new(0.0, 3.0, 0.0, 4.0);
            let n = v.normalized();
            assert!(approx_eq(n.magnitude(), 1.0));
            assert_eq!(n, Vector4::new(0.0, 0.6, 0.0, 0.8));
        }

        #[test]
        fn test_vector4_normalized_zero() {
            let v = Vector4::ZERO;
            let n = v.normalized();
            assert_eq!(n, Vector4::ZERO);
        }

        #[test]
        fn test_vector4_is_zero() {
            assert!(Vector4::ZERO.is_zero());
            assert!(!Vector4::ONE.is_zero());
        }

        #[test]
        fn test_vector4_is_normalised() {
            assert!(Vector4::X.is_normalised());
            assert!(Vector4::new(0.0, 0.6, 0.0, 0.8).is_normalised());
            assert!(!Vector4::new(1.0, 2.0, 2.0, 4.0).is_normalised());
        }

        #[test]
        fn test_vector4_safe_normal() {
            let v = Vector4::new(3.0, 4.0, 0.0, 0.0);
            assert_eq!(v.safe_normal(), Some(v.normalized()));
            assert_eq!(Vector4::ZERO.safe_normal(), None);
        }

        #[test]
        fn test_vector4_lerp() {
            let a = Vector4::new(0.0, 0.0, 0.0, 0.0);
            let b = Vector4::new(10.0, 20.0, 30.0, 40.0);
            assert_eq!(a.lerp(b, 0.0), a);
            assert_eq!(a.lerp(b, 1.0), b);
            assert_eq!(a.lerp(b, 0.5), Vector4::new(5.0, 10.0, 15.0, 20.0));
        }

        #[test]
        fn test_vector4_distance() {
            let a = Vector4::new(1.0, 2.0, 3.0, 4.0);
            let b = Vector4::new(5.0, 6.0, 7.0, 8.0);
            assert_eq!(a.distance(b), 8.0);
            assert_eq!(a.distance_squared(b), 64.0);
        }

        #[test]
        fn test_vector4_abs() {
            let v = Vector4::new(-1.0, 2.0, -3.0, 4.0);
            assert_eq!(v.abs(), Vector4::new(1.0, 2.0, 3.0, 4.0));
        }

        #[test]
        fn test_vector4_min_max() {
            let a = Vector4::new(1.0, 5.0, 3.0, 7.0);
            let b = Vector4::new(4.0, 2.0, 6.0, 1.0);
            assert_eq!(a.min(b), Vector4::new(1.0, 2.0, 3.0, 1.0));
            assert_eq!(a.max(b), Vector4::new(4.0, 5.0, 6.0, 7.0));
        }

        #[test]
        fn test_vector4_clamp() {
            let v = Vector4::new(-1.0, 5.0, 2.0, 8.0);
            let min = Vector4::new(0.0, 0.0, 0.0, 0.0);
            let max = Vector4::new(3.0, 3.0, 3.0, 3.0);
            assert_eq!(v.clamp(min, max), Vector4::new(0.0, 3.0, 2.0, 3.0));
        }

        #[test]
        fn test_vector4_project_reject() {
            let v = Vector4::new(3.0, 4.0, 5.0, 6.0);
            let onto = Vector4::new(1.0, 0.0, 0.0, 0.0);
            let proj = v.project_onto(onto);
            let rej = v.reject_from(onto);
            assert_eq!(proj, Vector4::new(3.0, 0.0, 0.0, 0.0));
            assert_eq!(rej, Vector4::new(0.0, 4.0, 5.0, 6.0));
            assert!(approx_eq((proj + rej).x, v.x));
            assert!(approx_eq((proj + rej).y, v.y));
            assert!(approx_eq((proj + rej).z, v.z));
            assert!(approx_eq((proj + rej).w, v.w));
        }

        #[test]
        fn test_vector4_reflect() {
            let v = Vector4::new(1.0, -1.0, 0.0, 0.0);
            let normal = Vector4::new(0.0, 1.0, 0.0, 0.0);
            let reflected = v.reflect(normal);
            assert_eq!(reflected, Vector4::new(1.0, 1.0, 0.0, 0.0));
        }

        #[test]
        fn test_vector4_pow_sqrt() {
            let v = Vector4::new(4.0, 9.0, 16.0, 25.0);
            assert_eq!(v.pow(2.0), Vector4::new(16.0, 81.0, 256.0, 625.0));
            assert_eq!(v.sqrt(), Vector4::new(2.0, 3.0, 4.0, 5.0));
        }

        #[test]
        fn test_vector4_xyz() {
            let v = Vector4::new(1.0, 2.0, 3.0, 4.0);
            assert_eq!(v.xyz(), Vector::new(1.0, 2.0, 3.0));
        }

        #[test]
        fn test_vector4_to_vec3() {
            let v = Vector4::new(2.0, 4.0, 6.0, 2.0);
            assert_eq!(v.vec3(), Vector::new(1.0, 2.0, 3.0));

            let v = Vector4::new(2.0, 4.0, 6.0, 0.0);
            assert_eq!(v.vec3(), Vector::new(2.0, 4.0, 6.0));
        }

        #[test]
        fn test_vector4_swizzle() {
            let v = Vector4::new(1.0, 2.0, 3.0, 4.0);
            assert_eq!(v.xy(), Vector2::new(1.0, 2.0));
            assert_eq!(v.xz(), Vector2::new(1.0, 3.0));
            assert_eq!(v.xw(), Vector2::new(1.0, 4.0));
            assert_eq!(v.yz(), Vector2::new(2.0, 3.0));
            assert_eq!(v.yw(), Vector2::new(2.0, 4.0));
            assert_eq!(v.zw(), Vector2::new(3.0, 4.0));
            assert_eq!(v.xx(), Vector2::new(1.0, 1.0));
            assert_eq!(v.yy(), Vector2::new(2.0, 2.0));
            assert_eq!(v.zz(), Vector2::new(3.0, 3.0));
            assert_eq!(v.ww(), Vector2::new(4.0, 4.0));
        }

        #[test]
        fn test_vector4_default() {
            let v: Vector4 = Default::default();
            assert_eq!(v, Vector4::ZERO);
        }
    }
}