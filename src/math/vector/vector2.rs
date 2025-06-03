//! 2D vector mathematics implementation.
//!
//! This module provides a complete 2D vector type with comprehensive mathematical
//! operations, conversions, and utility functions commonly needed in graphics,
//! physics, and game development.

use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use crate::math::Vector;

/// A 2D vector with `x` and `y` components.
///
/// `Vector2` represents a point or direction in 2D space using single-precision
/// floating-point numbers. It supports all standard mathematical operations and
/// provides many utility functions for common vector operations.
///
/// # Examples
///
/// ```rust
/// use crate::forge_engine::math::Vector2;
///
/// // Create vectors
/// let v1 = Vector2::new(1.0, 2.0);
/// let v2 = Vector2::from((3.0, 4.0));
/// let v3 = Vector2::from_angle(std::f32::consts::PI / 4.0);
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
/// let angle = v1.angle();
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vector2 {
    /// The x-component of the vector
    pub x: f32,
    /// The y-component of the vector
    pub y: f32,
}

// Constructor
impl Vector2 {
    /// Creates a new `Vector2` with the specified components.
    ///
    /// # Arguments
    ///
    /// * `x` - The x-component
    /// * `y` - The y-component
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::math::Vector2;
    ///
    /// let v = Vector2::new(3.0, 4.0);
    /// assert_eq!(v.x, 3.0);
    /// assert_eq!(v.y, 4.0);
    /// ```
    pub fn new(x: f32, y: f32) -> Vector2 {
        Vector2 { x, y }
    }

    /// Creates a unit vector pointing in the direction of the given angle.
    ///
    /// # Arguments
    ///
    /// * `angle` - The angle in radians (0 = right, π/2 = up)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::math::Vector2;
    /// use std::f32::consts::PI;
    ///
    /// let right = Vector2::from_angle(0.0);
    /// let up = Vector2::from_angle(PI / 2.0);
    /// ```
    pub fn from_angle(angle: f32) -> Vector2 {
        Vector2::new(angle.cos(), angle.sin())
    }
}

// Constant constructors
impl Vector2 {
    /// A vector with all components set to zero: `(0, 0)`
    pub const ZERO: Vector2 = Vector2 { x: 0.0, y: 0.0 };

    /// A vector with all components set to one: `(1, 1)`
    pub const ONE: Vector2 = Vector2 { x: 1.0, y: 1.0 };

    /// The up direction vector: `(0, 1)`
    pub const UP: Vector2 = Vector2 { x: 0.0, y: 1.0 };

    /// The down direction vector: `(0, -1)`
    pub const DOWN: Vector2 = Vector2 { x: 0.0, y: -1.0 };

    /// The right direction vector: `(1, 0)`
    pub const RIGHT: Vector2 = Vector2 { x: 1.0, y: 0.0 };

    /// The left direction vector: `(-1, 0)`
    pub const LEFT: Vector2 = Vector2 { x: -1.0, y: 0.0 };
}

/// Converts a tuple `(f32, f32)` into a `Vector2`.
///
/// # Examples
///
/// ```rust
/// use crate::forge_engine::math::Vector2;
///
/// let v: Vector2 = (3.0, 4.0).into();
/// assert_eq!(v, Vector2::new(3.0, 4.0));
/// ```
impl From<(f32, f32)> for Vector2 {
    fn from(t: (f32, f32)) -> Self {
        Vector2::new(t.0, t.1)
    }
}

/// Converts an array `[f32; 2]` into a `Vector2`.
///
/// # Examples
///
/// ```rust
/// use crate::forge_engine::math::Vector2;
///
/// let v: Vector2 = [3.0, 4.0].into();
/// assert_eq!(v, Vector2::new(3.0, 4.0));
/// ```
impl From<[f32; 2]> for Vector2 {
    fn from(arr: [f32; 2]) -> Self {
        Vector2::new(arr[0], arr[1])
    }
}

/// Converts a `Vector2` into a tuple `(f32, f32)`.
impl From<Vector2> for (f32, f32) {
    fn from(v: Vector2) -> Self {
        (v.x, v.y)
    }
}

/// Converts a `Vector2` into an array `[f32; 2]`.
impl From<Vector2> for [f32; 2] {
    fn from(v: Vector2) -> Self {
        [v.x, v.y]
    }
}

// Operators

/// Adds two vectors component-wise.
impl Add for Vector2 {
    type Output = Vector2;
    fn add(self, other: Vector2) -> Vector2 {
        Vector2::new(self.x + other.x, self.y + other.y)
    }
}

/// Adds a scalar to each component of the vector.
impl Add<f32> for Vector2 {
    type Output = Vector2;
    fn add(self, other: f32) -> Vector2 {
        Vector2::new(self.x + other, self.y + other)
    }
}

/// Adds a vector to a scalar (commutative addition).
impl Add<Vector2> for f32 {
    type Output = Vector2;
    fn add(self, other: Vector2) -> Vector2 {
        Vector2::new(self + other.x, self + other.y)
    }
}

/// Adds another vector to this vector in place.
impl AddAssign for Vector2 {
    fn add_assign(&mut self, other: Vector2) {
        *self = Vector2::new(self.x + other.x, self.y + other.y);
    }
}

/// Subtracts two vectors component-wise.
impl Sub for Vector2 {
    type Output = Vector2;
    fn sub(self, other: Vector2) -> Vector2 {
        Vector2::new(self.x - other.x, self.y - other.y)
    }
}

/// Subtracts a scalar from each component of the vector.
impl Sub<f32> for Vector2 {
    type Output = Vector2;
    fn sub(self, other: f32) -> Vector2 {
        Vector2::new(self.x - other, self.y - other)
    }
}

/// Subtracts a vector from a scalar.
impl Sub<Vector2> for f32 {
    type Output = Vector2;
    fn sub(self, other: Vector2) -> Vector2 {
        Vector2::new(self - other.x, self - other.y)
    }
}

/// Subtracts another vector from this vector in place.
impl SubAssign for Vector2 {
    fn sub_assign(&mut self, other: Vector2) {
        *self = Vector2::new(self.x - other.x, self.y - other.y);
    }
}

/// Multiplies two vectors component-wise (Hadamard product).
impl Mul<Vector2> for Vector2 {
    type Output = Vector2;
    fn mul(self, other: Vector2) -> Vector2 {
        Vector2::new(self.x * other.x, self.y * other.y)
    }
}

/// Multiplies the vector by a scalar.
impl Mul<f32> for Vector2 {
    type Output = Vector2;
    fn mul(self, other: f32) -> Vector2 {
        Vector2::new(self.x * other, self.y * other)
    }
}

/// Multiplies a scalar by a vector (commutative multiplication).
impl Mul<Vector2> for f32 {
    type Output = Vector2;
    fn mul(self, other: Vector2) -> Vector2 {
        Vector2::new(self * other.x, self * other.y)
    }
}

/// Multiplies this vector by another vector in place (component-wise).
impl MulAssign<Vector2> for Vector2 {
    fn mul_assign(&mut self, other: Vector2) {
        *self = Vector2::new(self.x * other.x, self.y * other.y);
    }
}

/// Divides two vectors component-wise.
impl Div<Vector2> for Vector2 {
    type Output = Vector2;
    fn div(self, other: Vector2) -> Vector2 {
        Vector2::new(self.x / other.x, self.y / other.y)
    }
}

/// Divides the vector by a scalar.
impl Div<f32> for Vector2 {
    type Output = Vector2;
    fn div(self, other: f32) -> Vector2 {
        Vector2::new(self.x / other, self.y / other)
    }
}

/// Divides a scalar by a vector component-wise.
impl Div<Vector2> for f32 {
    type Output = Vector2;
    fn div(self, other: Vector2) -> Vector2 {
        Vector2::new(self / other.x, self / other.y)
    }
}

/// Divides this vector by another vector in place (component-wise).
impl DivAssign<Vector2> for Vector2 {
    fn div_assign(&mut self, other: Vector2) {
        *self = Vector2::new(self.x / other.x, self.y / other.y);
    }
}

/// Negates the vector (multiplies each component by -1).
impl Neg for Vector2 {
    type Output = Vector2;
    fn neg(self) -> Vector2 {
        Vector2::new(-self.x, -self.y)
    }
}

/// Creates a zero vector by default.
impl Default for Vector2 {
    fn default() -> Self {
        Self::ZERO
    }
}

// Other Methods
impl Vector2 {
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
    /// use crate::forge_engine::math::Vector2;
    ///
    /// let a = Vector2::new(3.0, 4.0);
    /// let b = Vector2::new(2.0, 1.0);
    /// assert_eq!(a.dot(b), 10.0); // 3*2 + 4*1
    /// ```
    #[inline]
    pub fn dot(self, other: Vector2) -> f32 {
        self.x * other.x + self.y * other.y
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
    /// use crate::forge_engine::math::Vector2;
    ///
    /// let v = Vector2::new(3.0, 4.0);
    /// let squared = v.pow(2.0);
    /// assert_eq!(squared, Vector2::new(9.0, 16.0));
    /// ```
    #[inline]
    pub fn pow(self, exp: f32) -> Vector2 {
        Vector2::new(self.x.powf(exp), self.y.powf(exp))
    }

    /// Computes the square root of each component.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::math::Vector2;
    ///
    /// let v = Vector2::new(9.0, 16.0);
    /// let roots = v.sqrt();
    /// assert_eq!(roots, Vector2::new(3.0, 4.0));
    /// ```
    #[inline]
    pub fn sqrt(&self) -> Vector2 {
        Vector2::new(self.x.sqrt(), self.y.sqrt())
    }

    /// Computes the 2D cross product (determinant) of two vectors.
    ///
    /// The 2D cross product returns a scalar representing the magnitude
    /// of the vector that would result from a 3D cross product (z-component).
    ///
    /// # Arguments
    ///
    /// * `other` - The other vector
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::math::Vector2;
    ///
    /// let a = Vector2::new(3.0, 4.0);
    /// let b = Vector2::new(2.0, 1.0);
    /// assert_eq!(a.cross(b), -5.0); // 3*1 - 4*2
    /// ```
    #[inline]
    pub fn cross(self, other: Vector2) -> f32 {
        self.x * other.y - self.y * other.x
    }

    /// Computes the magnitude (length) of the vector.
    ///
    /// This is the Euclidean distance from the origin to the point
    /// represented by the vector.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::math::Vector2;
    ///
    /// let v = Vector2::new(3.0, 4.0);
    /// assert_eq!(v.magnitude(), 5.0);
    /// ```
    #[inline]
    pub fn magnitude(self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    /// Computes the squared magnitude of the vector.
    ///
    /// This is more efficient than `magnitude()` when you only need
    /// to compare lengths or don't need the actual distance.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::math::Vector2;
    ///
    /// let v = Vector2::new(3.0, 4.0);
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
    /// use crate::forge_engine::math::Vector2;
    ///
    /// let v = Vector2::new(3.0, 4.0);
    /// let normalized = v.normalized();
    /// assert_eq!(normalized, Vector2::new(0.6, 0.8));
    /// assert!((normalized.magnitude() - 1.0).abs() < f32::EPSILON);
    /// ```
    #[inline]
    pub fn normalized(self) -> Vector2 {
        let magnitude = self.magnitude();
        if magnitude == 0.0 {
            Vector2::ZERO
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
    /// use crate::forge_engine::math::Vector2;
    ///
    /// assert!(Vector2::ZERO.is_zero());
    /// assert!(!Vector2::ONE.is_zero());
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
    /// use crate::forge_engine::math::Vector2;
    ///
    /// assert!(Vector2::RIGHT.is_normalised());
    /// assert!(!Vector2::new(3.0, 4.0).is_normalised());
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
    /// use crate::forge_engine::math::Vector2;
    ///
    /// let v = Vector2::new(3.0, 4.0);
    /// assert!(v.safe_normal().is_some());
    /// assert!(Vector2::ZERO.safe_normal().is_none());
    /// ```
    pub fn safe_normal(self) -> Option<Vector2> {
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
    /// use crate::forge_engine::math::Vector2;
    ///
    /// let a = Vector2::new(0.0, 0.0);
    /// let b = Vector2::new(10.0, 20.0);
    /// let mid = a.lerp(b, 0.5);
    /// assert_eq!(mid, Vector2::new(5.0, 10.0));
    /// ```
    pub fn lerp(self, other: Vector2, t: f32) -> Vector2 {
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
    /// use crate::forge_engine::math::Vector2;
    ///
    /// let a = Vector2::new(1.0, 2.0);
    /// let b = Vector2::new(4.0, 6.0);
    /// assert_eq!(a.distance(b), 5.0);
    /// ```
    pub fn distance(self, other: Vector2) -> f32 {
        (other - self).magnitude()
    }

    /// Computes the squared distance between two points.
    ///
    /// More efficient than `distance()` when you only need to compare distances.
    ///
    /// # Arguments
    ///
    /// * `other` - The other point
    pub fn distance_squared(self, other: Vector2) -> f32 {
        let diff = self - other;
        diff.dot(diff)
    }

    /// Returns the angle of the vector in radians.
    ///
    /// The angle is measured from the positive x-axis, counter-clockwise.
    /// Returns a value in the range [-π, π].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::math::Vector2;
    /// use std::f32::consts::PI;
    ///
    /// let right = Vector2::new(1.0, 0.0);
    /// assert_eq!(right.angle(), 0.0);
    ///
    /// let up = Vector2::new(0.0, 1.0);
    /// assert!((up.angle() - PI / 2.0).abs() < f32::EPSILON);
    /// ```
    #[inline]
    pub fn angle(self) -> f32 {
        self.y.atan2(self.x)
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
    /// use crate::forge_engine::math::Vector2;
    /// use std::f32::consts::PI;
    ///
    /// let a = Vector2::new(1.0, 0.0);
    /// let b = Vector2::new(0.0, 1.0);
    /// let angle = a.angle_between(b);
    /// assert!((angle - PI / 2.0).abs() < f32::EPSILON);
    /// ```
    pub fn angle_between(self, other: Vector2) -> f32 {
        let dot = self.dot(other);
        let mags = self.magnitude() * other.magnitude();
        (dot / mags).acos()
    }

    /// Rotates the vector by the given angle in radians.
    ///
    /// Positive angles rotate counter-clockwise.
    ///
    /// # Arguments
    ///
    /// * `angle` - The rotation angle in radians
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::math::Vector2;
    /// use std::f32::consts::PI;
    ///
    /// let v = Vector2::new(1.0, 0.0);
    /// let rotated = v.rotate(PI / 2.0);
    /// // Should be approximately (0, 1)
    /// assert!((rotated.x).abs() < f32::EPSILON);
    /// assert!((rotated.y - 1.0).abs() < f32::EPSILON);
    /// ```
    pub fn rotate(self, angle: f32) -> Vector2 {
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        Vector2::new(
            self.x * cos_a - self.y * sin_a,
            self.x * sin_a + self.y * cos_a,
        )
    }

    /// Returns a vector perpendicular to this one.
    ///
    /// The perpendicular vector is rotated 90 degrees counter-clockwise.
    /// If this vector is (x, y), the perpendicular is (-y, x).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::math::Vector2;
    ///
    /// let v = Vector2::new(3.0, 4.0);
    /// let perp = v.perpendicular();
    /// assert_eq!(perp, Vector2::new(-4.0, 3.0));
    /// // Perpendicular vectors have zero dot product
    /// assert!((v.dot(perp)).abs() < f32::EPSILON);
    /// ```
    #[inline]
    pub fn perpendicular(self) -> Vector2 {
        Vector2::new(-self.y, self.x)
    }

    /// Returns a vector with the absolute value of each component.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::math::Vector2;
    ///
    /// let v = Vector2::new(-3.0, 4.0);
    /// assert_eq!(v.abs(), Vector2::new(3.0, 4.0));
    /// ```
    pub fn abs(self) -> Vector2 {
        Vector2::new(self.x.abs(), self.y.abs())
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
    /// use crate::forge_engine::math::Vector2;
    ///
    /// let a = Vector2::new(1.0, 4.0);
    /// let b = Vector2::new(3.0, 2.0);
    /// assert_eq!(a.min(b), Vector2::new(1.0, 2.0));
    /// ```
    pub fn min(self, other: Vector2) -> Vector2 {
        Vector2::new(self.x.min(other.x), self.y.min(other.y))
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
    /// use crate::forge_engine::math::Vector2;
    ///
    /// let a = Vector2::new(1.0, 4.0);
    /// let b = Vector2::new(3.0, 2.0);
    /// assert_eq!(a.max(b), Vector2::new(3.0, 4.0));
    /// ```
    pub fn max(self, other: Vector2) -> Vector2 {
        Vector2::new(self.x.max(other.x), self.y.max(other.y))
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
    /// use crate::forge_engine::math::Vector2;
    ///
    /// let v = Vector2::new(5.0, -2.0);
    /// let min = Vector2::new(0.0, 0.0);
    /// let max = Vector2::new(3.0, 3.0);
    /// assert_eq!(v.clamp(min, max), Vector2::new(3.0, 0.0));
    /// ```
    pub fn clamp(self, min: Vector2, max: Vector2) -> Vector2 {
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
    /// use crate::forge_engine::math::Vector2;
    ///
    /// let v = Vector2::new(3.0, 4.0);
    /// let onto = Vector2::new(1.0, 0.0);
    /// let proj = v.project_onto(onto);
    /// assert_eq!(proj, Vector2::new(3.0, 0.0));
    /// ```
    pub fn project_onto(self, onto: Vector2) -> Vector2 {
        let d = onto.dot(onto);
        if d > 0.0 {
            onto * (self.dot(onto) / d)
        } else {
            Vector2::ZERO
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
    /// use crate::forge_engine::math::Vector2;
    ///
    /// let v = Vector2::new(3.0, 4.0);
    /// let from = Vector2::new(1.0, 0.0);
    /// let rej = v.reject_from(from);
    /// assert_eq!(rej, Vector2::new(0.0, 4.0));
    /// ```
    pub fn reject_from(self, from: Vector2) -> Vector2 {
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
    /// use crate::forge_engine::math::Vector2;
    ///
    /// let v = Vector2::new(1.0, -1.0);
    /// let normal = Vector2::new(0.0, 1.0);
    /// let reflected = v.reflect(normal);
    /// assert_eq!(reflected, Vector2::new(1.0, 1.0));
    /// ```
    pub fn reflect(self, normal: Vector2) -> Vector2 {
        self - normal * (2.0 * self.dot(normal))
    }
}

// Conversions
impl Vector2 {
    /// Converts to a `Vector` using the x and y components, with z set to 0.0.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::math::Vector2;
    /// use crate::forge_engine::Vector;
    ///
    /// let v2 = Vector2::new(3.0, 4.0);
    /// let v3 = v2.vec3();
    /// assert_eq!(v3, Vector::new(3.0, 4.0, 0.0));
    /// ```
    pub fn vec3(self) -> Vector {
        Vector::new(self.x, self.y, 0.0)
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

    // ============ Vector2 Tests ============
    #[cfg(test)]
    mod vector2_tests {
        use super::*;

        #[test]
        fn test_vector2_construction() {
            let v = Vector2::new(3.0, 4.0);
            assert_eq!(v.x, 3.0);
            assert_eq!(v.y, 4.0);
        }

        #[test]
        fn test_vector2_from_angle() {
            let v = Vector2::from_angle(0.0);
            assert!(approx_eq(v.x, 1.0));
            assert!(approx_eq(v.y, 0.0));

            let v = Vector2::from_angle(PI / 2.0);
            assert!(approx_eq(v.x, 0.0));
            assert!(approx_eq(v.y, 1.0));
        }

        #[test]
        fn test_vector2_constants() {
            assert_eq!(Vector2::ZERO, Vector2::new(0.0, 0.0));
            assert_eq!(Vector2::ONE, Vector2::new(1.0, 1.0));
            assert_eq!(Vector2::UP, Vector2::new(0.0, 1.0));
            assert_eq!(Vector2::DOWN, Vector2::new(0.0, -1.0));
            assert_eq!(Vector2::RIGHT, Vector2::new(1.0, 0.0));
            assert_eq!(Vector2::LEFT, Vector2::new(-1.0, 0.0));
        }

        #[test]
        fn test_vector2_from_tuple() {
            let v: Vector2 = (3.0, 4.0).into();
            assert_eq!(v, Vector2::new(3.0, 4.0));
        }

        #[test]
        fn test_vector2_add() {
            let a = Vector2::new(1.0, 2.0);
            let b = Vector2::new(3.0, 4.0);
            assert_eq!(a + b, Vector2::new(4.0, 6.0));
            assert_eq!(a + 5.0, Vector2::new(6.0, 7.0));
            assert_eq!(5.0 + a, Vector2::new(6.0, 7.0));
        }

        #[test]
        fn test_vector2_add_assign() {
            let mut v = Vector2::new(1.0, 2.0);
            v += Vector2::new(3.0, 4.0);
            assert_eq!(v, Vector2::new(4.0, 6.0));
        }

        #[test]
        fn test_vector2_sub() {
            let a = Vector2::new(5.0, 7.0);
            let b = Vector2::new(2.0, 3.0);
            assert_eq!(a - b, Vector2::new(3.0, 4.0));
            assert_eq!(a - 2.0, Vector2::new(3.0, 5.0));
            assert_eq!(10.0 - a, Vector2::new(5.0, 3.0));
        }

        #[test]
        fn test_vector2_sub_assign() {
            let mut v = Vector2::new(5.0, 7.0);
            v -= Vector2::new(2.0, 3.0);
            assert_eq!(v, Vector2::new(3.0, 4.0));
        }

        #[test]
        fn test_vector2_mul() {
            let a = Vector2::new(2.0, 3.0);
            let b = Vector2::new(4.0, 5.0);
            assert_eq!(a * b, Vector2::new(8.0, 15.0));
            assert_eq!(a * 3.0, Vector2::new(6.0, 9.0));
            assert_eq!(3.0 * a, Vector2::new(6.0, 9.0));
        }

        #[test]
        fn test_vector2_mul_assign() {
            let mut v = Vector2::new(2.0, 3.0);
            v *= Vector2::new(4.0, 5.0);
            assert_eq!(v, Vector2::new(8.0, 15.0));
        }

        #[test]
        fn test_vector2_div() {
            let a = Vector2::new(8.0, 15.0);
            let b = Vector2::new(4.0, 5.0);
            assert_eq!(a / b, Vector2::new(2.0, 3.0));
            assert_eq!(a / 2.0, Vector2::new(4.0, 7.5));
            assert_eq!(24.0 / a, Vector2::new(3.0, 1.6));
        }

        #[test]
        fn test_vector2_div_assign() {
            let mut v = Vector2::new(8.0, 15.0);
            v /= Vector2::new(4.0, 5.0);
            assert_eq!(v, Vector2::new(2.0, 3.0));
        }

        #[test]
        fn test_vector2_neg() {
            let v = Vector2::new(3.0, -4.0);
            assert_eq!(-v, Vector2::new(-3.0, 4.0));
        }

        #[test]
        fn test_vector2_dot() {
            let a = Vector2::new(3.0, 4.0);
            let b = Vector2::new(2.0, 1.0);
            assert_eq!(a.dot(b), 10.0);
        }

        #[test]
        fn test_vector2_cross() {
            let a = Vector2::new(3.0, 4.0);
            let b = Vector2::new(2.0, 1.0);
            assert_eq!(a.cross(b), -5.0);
        }

        #[test]
        fn test_vector2_magnitude() {
            let v = Vector2::new(3.0, 4.0);
            assert_eq!(v.magnitude(), 5.0);
            assert_eq!(v.magnitude_squared(), 25.0);
        }

        #[test]
        fn test_vector2_normalized() {
            let v = Vector2::new(3.0, 4.0);
            let n = v.normalized();
            assert!(approx_eq(n.magnitude(), 1.0));
            assert!(approx_eq(n.x, 0.6));
            assert!(approx_eq(n.y, 0.8));
        }

        #[test]
        fn test_vector2_is_zero() {
            assert!(Vector2::ZERO.is_zero());
            assert!(!Vector2::ONE.is_zero());
        }

        #[test]
        fn test_vector2_is_normalised() {
            assert!(Vector2::new(1.0, 0.0).is_normalised());
            assert!(Vector2::new(0.6, 0.8).is_normalised());
            assert!(!Vector2::new(3.0, 4.0).is_normalised());
        }

        #[test]
        fn test_vector2_safe_normal() {
            let v = Vector2::new(3.0, 4.0);
            assert_eq!(v.safe_normal(), Some(v.normalized()));
            assert_eq!(Vector2::ZERO.safe_normal(), None);
        }

        #[test]
        fn test_vector2_lerp() {
            let a = Vector2::new(0.0, 0.0);
            let b = Vector2::new(10.0, 20.0);
            assert_eq!(a.lerp(b, 0.0), a);
            assert_eq!(a.lerp(b, 1.0), b);
            assert_eq!(a.lerp(b, 0.5), Vector2::new(5.0, 10.0));
        }

        #[test]
        fn test_vector2_distance() {
            let a = Vector2::new(1.0, 2.0);
            let b = Vector2::new(4.0, 6.0);
            assert_eq!(a.distance(b), 5.0);
            assert_eq!(a.distance_squared(b), 25.0);
        }

        #[test]
        fn test_vector2_angle() {
            let v = Vector2::new(1.0, 0.0);
            assert!(approx_eq(v.angle(), 0.0));

            let v = Vector2::new(0.0, 1.0);
            assert!(approx_eq(v.angle(), PI / 2.0));
        }

        #[test]
        fn test_vector2_angle_between() {
            let a = Vector2::new(1.0, 0.0);
            let b = Vector2::new(0.0, 1.0);
            assert!(approx_eq(a.angle_between(b), PI / 2.0));
        }

        #[test]
        fn test_vector2_rotate() {
            let v = Vector2::new(1.0, 0.0);
            let rotated = v.rotate(PI / 2.0);
            assert!(approx_eq(rotated.x, 0.0));
            assert!(approx_eq(rotated.y, 1.0));
        }

        #[test]
        fn test_vector2_perpendicular() {
            let v = Vector2::new(3.0, 4.0);
            let perp = v.perpendicular();
            assert!(approx_eq(v.dot(perp), 0.0));
            assert_eq!(perp, Vector2::new(-4.0, 3.0));
        }

        #[test]
        fn test_vector2_abs() {
            let v = Vector2::new(-3.0, 4.0);
            assert_eq!(v.abs(), Vector2::new(3.0, 4.0));
        }

        #[test]
        fn test_vector2_min_max() {
            let a = Vector2::new(1.0, 4.0);
            let b = Vector2::new(3.0, 2.0);
            assert_eq!(a.min(b), Vector2::new(1.0, 2.0));
            assert_eq!(a.max(b), Vector2::new(3.0, 4.0));
        }

        #[test]
        fn test_vector2_clamp() {
            let v = Vector2::new(5.0, -2.0);
            assert_eq!(
                v.clamp(Vector2::new(0.0, 0.0), Vector2::new(3.0, 3.0)),
                Vector2::new(3.0, 0.0)
            );
        }

        #[test]
        fn test_vector2_project_reject() {
            let v = Vector2::new(3.0, 4.0);
            let onto = Vector2::new(1.0, 0.0);
            let proj = v.project_onto(onto);
            let rej = v.reject_from(onto);
            assert_eq!(proj, Vector2::new(3.0, 0.0));
            assert_eq!(rej, Vector2::new(0.0, 4.0));
            assert!(approx_eq((proj + rej).x, v.x));
            assert!(approx_eq((proj + rej).y, v.y));
        }

        #[test]
        fn test_vector2_reflect() {
            let v = Vector2::new(1.0, -1.0);
            let normal = Vector2::new(0.0, 1.0);
            let reflected = v.reflect(normal);
            assert_eq!(reflected, Vector2::new(1.0, 1.0));
        }

        #[test]
        fn test_vector2_pow_sqrt() {
            let v = Vector2::new(4.0, 9.0);
            assert_eq!(v.pow(2.0), Vector2::new(16.0, 81.0));
            assert_eq!(v.sqrt(), Vector2::new(2.0, 3.0));
        }

        #[test]
        fn test_vector2_conversions() {
            let v2 = Vector2::new(3.0, 4.0);
            let v3 = v2.vec3();
            assert_eq!(v3, Vector::new(3.0, 4.0, 0.0));
        }

        #[test]
        fn test_vector2_default() {
            let v: Vector2 = Default::default();
            assert_eq!(v, Vector2::ZERO);
        }
    }
}
