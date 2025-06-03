use std::ops::{Add, Div, Mul, Neg, Sub};
use crate::math::{Vector, Vector2};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vector4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

// Constructor
impl Vector4 {
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Vector4 { x, y, z, w }
    }

    pub fn from_vec3(v: Vector, w: f32) -> Self {
        Vector4 { x: v.x, y: v.y, z: v.z, w }
    }
}

// Constants
impl Vector4 {
    pub const ZERO: Vector4 = Vector4 { x: 0.0, y: 0.0, z: 0.0, w: 0.0 };
    pub const ONE: Vector4 = Vector4 { x: 1.0, y: 1.0, z: 1.0, w: 1.0 };
}

// Basic operations
impl Vector4 {
    #[inline]
    pub fn dot(self, other: Vector4) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }

    #[inline]
    pub fn magnitude(self) -> f32 {
        self.dot(self).sqrt()
    }

    #[inline]
    pub fn magnitude_squared(self) -> f32 {
        self.dot(self)
    }

    #[inline]
    pub fn normalized(self) -> Vector4 {
        let mag = self.magnitude();
        if mag > 0.0 {
            self / mag
        } else {
            Vector4::ZERO
        }
    }

    #[inline]
    pub fn lerp(self, other: Vector4, t: f32) -> Vector4 {
        self + (other - self) * t
    }
}

// Conversions
impl Vector4 {
    #[inline]
    pub fn xyz(self) -> Vector {
        Vector::new(self.x, self.y, self.z)
    }

    #[inline]
    pub fn vec3(self) -> Vector {
        if self.w != 0.0 {
            Vector::new(self.x / self.w, self.y / self.w, self.z / self.w)
        } else {
            Vector::new(self.x, self.y, self.z)
        }
    }
}

// Operators
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

impl Mul<f32> for Vector4 {
    type Output = Vector4;
    fn mul(self, scalar: f32) -> Vector4 {
        Vector4::new(
            self.x * scalar,
            self.y * scalar,
            self.z * scalar,
            self.w * scalar,
        )
    }
}

impl Mul<Vector4> for f32 {
    type Output = Vector4;
    fn mul(self, vec: Vector4) -> Vector4 {
        vec * self
    }
}

impl Div<f32> for Vector4 {
    type Output = Vector4;
    fn div(self, scalar: f32) -> Vector4 {
        Vector4::new(
            self.x / scalar,
            self.y / scalar,
            self.z / scalar,
            self.w / scalar,
        )
    }
}

impl Neg for Vector4 {
    type Output = Vector4;
    fn neg(self) -> Vector4 {
        Vector4::new(-self.x, -self.y, -self.z, -self.w)
    }
}

// Traits
impl Default for Vector4 {
    fn default() -> Self {
        Self::ZERO
    }
}

impl From<(f32, f32, f32, f32)> for Vector4 {
    fn from(t: (f32, f32, f32, f32)) -> Self {
        Vector4::new(t.0, t.1, t.2, t.3)
    }
}

impl From<Vector> for Vector4 {
    fn from(v: Vector) -> Self {
        Vector4::new(v.x, v.y, v.z, 1.0)  // Default w=1 for points
    }
}

impl From<[f32; 4]> for Vector4 {
    fn from(arr: [f32; 4]) -> Self {
        Vector4::new(arr[0], arr[1], arr[2], arr[3])
    }
}

impl From<Vector4> for (f32, f32, f32, f32) {
    fn from(v: Vector4) -> Self {
        (v.x, v.y, v.z, v.w)
    }
}

impl From<Vector4> for [f32; 4] {
    fn from(v: Vector4) -> Self {
        [v.x, v.y, v.z, v.w]
    }
}

// Optional: From Vector4 to Vector3 (though you already have to_vec3())
impl From<Vector4> for Vector {
    fn from(v: Vector4) -> Self {
        if v.w != 0.0 {
            Vector::new(v.x / v.w, v.y / v.w, v.z / v.w)
        } else {
            Vector::new(v.x, v.y, v.z)
        }
    }
}

// Optional: Direct Vector2 to Vector4 conversion
impl From<Vector2> for Vector4 {
    fn from(v: Vector2) -> Self {
        Vector4::new(v.x, v.y, 0.0, 1.0)
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
        }

        #[test]
        fn test_vector4_from_tuple() {
            let v: Vector4 = (1.0, 2.0, 3.0, 4.0).into();
            assert_eq!(v, Vector4::new(1.0, 2.0, 3.0, 4.0));
        }

        #[test]
        fn test_vector4_from_vector3() {
            let v3 = Vector::new(1.0, 2.0, 3.0);
            let v4: Vector4 = v3.into();
            assert_eq!(v4, Vector4::new(1.0, 2.0, 3.0, 1.0));
        }

        #[test]
        fn test_vector4_add() {
            let a = Vector4::new(1.0, 2.0, 3.0, 4.0);
            let b = Vector4::new(5.0, 6.0, 7.0, 8.0);
            assert_eq!(a + b, Vector4::new(6.0, 8.0, 10.0, 12.0));
        }

        #[test]
        fn test_vector4_sub() {
            let a = Vector4::new(5.0, 6.0, 7.0, 8.0);
            let b = Vector4::new(1.0, 2.0, 3.0, 4.0);
            assert_eq!(a - b, Vector4::new(4.0, 4.0, 4.0, 4.0));
        }

        #[test]
        fn test_vector4_mul() {
            let v = Vector4::new(1.0, 2.0, 3.0, 4.0);
            assert_eq!(v * 2.0, Vector4::new(2.0, 4.0, 6.0, 8.0));
            assert_eq!(2.0 * v, Vector4::new(2.0, 4.0, 6.0, 8.0));
        }

        #[test]
        fn test_vector4_div() {
            let v = Vector4::new(2.0, 4.0, 6.0, 8.0);
            assert_eq!(v / 2.0, Vector4::new(1.0, 2.0, 3.0, 4.0));
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
        fn test_vector4_lerp() {
            let a = Vector4::new(0.0, 0.0, 0.0, 0.0);
            let b = Vector4::new(10.0, 20.0, 30.0, 40.0);
            assert_eq!(a.lerp(b, 0.0), a);
            assert_eq!(a.lerp(b, 1.0), b);
            assert_eq!(a.lerp(b, 0.5), Vector4::new(5.0, 10.0, 15.0, 20.0));
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
        fn test_vector4_default() {
            let v: Vector4 = Default::default();
            assert_eq!(v, Vector4::ZERO);
        }
    }
}