use crate::math::Vec2;
use std::ops::{Deref, DerefMut};

/// A 2D point in space.
///
/// `Point2D` represents a position in 2D space and inherits all vector operations
/// from `Vec2` through deref coercion. This allows points to be used in vector
/// calculations while maintaining semantic distinction.
///
/// # Examples
///
/// ```rust
/// use crate::forge_engine::math::{Vec2, Point2D};
///
/// // Create points
/// let p1 = Point2D::new(1.0, 2.0);
/// let p2 = Point2D::from((3.0, 4.0));
/// let origin = Point2D::ORIGIN;
///
/// // Use vector operations (inherited from Vec2)
/// let distance = p1.distance(*p2);
/// let midpoint = p1.lerp(*p2, 0.5);
/// let offset = p1 + Vec2::new(10.0, 5.0);
///
/// // Point-specific operations
/// let translation = p2 - p1;  // Returns Vec2
/// let moved_point = p1.translate(Vec2::new(5.0, 3.0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2D(pub Vec2);

impl Point2D {
    /// Creates a new point with the specified coordinates.
    ///
    /// # Arguments
    ///
    /// * `x` - The x-coordinate
    /// * `y` - The y-coordinate
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::math::Point2D;
    ///
    /// let point = Point2D::new(3.0, 4.0);
    /// assert_eq!(point.x(), 3.0);
    /// assert_eq!(point.y(), 4.0);
    /// ```
    pub fn new(x: f32, y: f32) -> Self {
        Self(Vec2::new(x, y))
    }

    /// Returns the x-coordinate of the point.
    #[inline]
    pub fn x(&self) -> f32 {
        self.0.x
    }

    /// Returns the y-coordinate of the point.
    #[inline]
    pub fn y(&self) -> f32 {
        self.0.y
    }

    /// Sets the x-coordinate of the point.
    #[inline]
    pub fn set_x(&mut self, x: f32) {
        self.0.x = x;
    }

    /// Sets the y-coordinate of the point.
    #[inline]
    pub fn set_y(&mut self, y: f32) {
        self.0.y = y;
    }

    /// Translates the point by the given vector.
    ///
    /// # Arguments
    ///
    /// * `offset` - The translation vector
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::math::{Point2D, Vec2};
    ///
    /// let point = Point2D::new(1.0, 2.0);
    /// let offset = Vec2::new(3.0, 4.0);
    /// let moved = point.translate(offset);
    /// assert_eq!(moved, Point2D::new(4.0, 6.0));
    /// ```
    pub fn translate(self, offset: Vec2) -> Self {
        Self(self.0 + offset)
    }

    /// Translates the point by the given vector in place.
    ///
    /// # Arguments
    ///
    /// * `offset` - The translation vector
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::math::{Point2D, Vec2};
    ///
    /// let mut point = Point2D::new(1.0, 2.0);
    /// point.translate_mut(Vec2::new(3.0, 4.0));
    /// assert_eq!(point, Point2D::new(4.0, 6.0));
    /// ```
    pub fn translate_mut(&mut self, offset: Vec2) {
        self.0 += offset;
    }

    /// Returns the vector from this point to another point.
    ///
    /// # Arguments
    ///
    /// * `other` - The target point
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::math::{Point2D, Vec2};
    ///
    /// let p1 = Point2D::new(1.0, 2.0);
    /// let p2 = Point2D::new(4.0, 6.0);
    /// let vector = p1.vector_to(p2);
    /// assert_eq!(vector, Vec2::new(3.0, 4.0));
    /// ```
    pub fn vector_to(self, other: Self) -> Vec2 {
        other.0 - self.0
    }

    /// Returns the midpoint between this point and another point.
    ///
    /// # Arguments
    ///
    /// * `other` - The other point
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::math::Point2D;
    ///
    /// let p1 = Point2D::new(0.0, 0.0);
    /// let p2 = Point2D::new(10.0, 20.0);
    /// let midpoint = p1.midpoint(p2);
    /// assert_eq!(midpoint, Point2D::new(5.0, 10.0));
    /// ```
    pub fn midpoint(self, other: Self) -> Self {
        Self(self.0.lerp(other.0, 0.5))
    }

    /// Converts the point to a Vec2.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::math::{Point2D, Vec2};
    ///
    /// let point = Point2D::new(3.0, 4.0);
    /// let vec: Vec2 = point.to_vec2();
    /// assert_eq!(vec, Vec2::new(3.0, 4.0));
    /// ```
    pub fn to_vec2(self) -> Vec2 {
        self.0
    }

    /// Creates a point from a Vec2.
    ///
    /// # Arguments
    ///
    /// * `vec` - The vector to convert
    ///
    /// # Examples
    ///
    /// ```rust
    /// use crate::forge_engine::math::{Point2D, Vec2};
    ///
    /// let vec = Vec2::new(3.0, 4.0);
    /// let point = Point2D::from_vec2(vec);
    /// assert_eq!(point, Point2D::new(3.0, 4.0));
    /// ```
    pub fn from_vec2(vec: Vec2) -> Self {
        Self(vec)
    }
}

// Constant constructors
impl Point2D {
    /// The origin point: `(0, 0)`
    pub const ORIGIN: Point2D = Point2D(Vec2::ZERO);

    /// A point at `(1, 1)`
    pub const ONE: Point2D = Point2D(Vec2::ONE);
}

// Deref implementation - this gives us "inheritance" from Vec2
impl Deref for Point2D {
    type Target = Vec2;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Point2D {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

// Default implementation
impl Default for Point2D {
    fn default() -> Self {
        Self::ORIGIN
    }
}

// From/Into conversions
impl From<Vec2> for Point2D {
    fn from(vec: Vec2) -> Self {
        Self(vec)
    }
}

impl From<Point2D> for Vec2 {
    fn from(point: Point2D) -> Self {
        point.0
    }
}

impl From<(f32, f32)> for Point2D {
    fn from(tuple: (f32, f32)) -> Self {
        Self(Vec2::from(tuple))
    }
}

impl From<[f32; 2]> for Point2D {
    fn from(array: [f32; 2]) -> Self {
        Self(Vec2::from(array))
    }
}

impl From<Point2D> for (f32, f32) {
    fn from(point: Point2D) -> Self {
        point.0.into()
    }
}

impl From<Point2D> for [f32; 2] {
    fn from(point: Point2D) -> Self {
        point.0.into()
    }
}

// Arithmetic operations specific to points
use std::ops::{Add, AddAssign, Sub, SubAssign};

/// Adding a vector to a point gives a new point (translation)
impl Add<Vec2> for Point2D {
    type Output = Point2D;

    fn add(self, offset: Vec2) -> Point2D {
        Point2D(self.0 + offset)
    }
}

/// Adding a vector to a point in place
impl AddAssign<Vec2> for Point2D {
    fn add_assign(&mut self, offset: Vec2) {
        self.0 += offset;
    }
}

/// Subtracting a vector from a point gives a new point (translation)
impl Sub<Vec2> for Point2D {
    type Output = Point2D;

    fn sub(self, offset: Vec2) -> Point2D {
        Point2D(self.0 - offset)
    }
}

/// Subtracting a vector from a point in place
impl SubAssign<Vec2> for Point2D {
    fn sub_assign(&mut self, offset: Vec2) {
        self.0 -= offset;
    }
}

/// Subtracting two points gives the vector between them
impl Sub<Point2D> for Point2D {
    type Output = Vec2;

    fn sub(self, other: Point2D) -> Vec2 {
        self.0 - other.0
    }
}


#[cfg(test)]
mod tests;