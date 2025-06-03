use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use crate::math::Vector2;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vector3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

// Constructor
impl Vector3 {
    pub fn new(x: f32, y: f32, z: f32) -> Vector3 {
        Vector3 { x, y, z }
    }
}

// Constant constructors
impl Vector3 {
    pub const ZERO: Vector3 = Vector3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    pub const ONE: Vector3 = Vector3 {
        x: 1.0,
        y: 1.0,
        z: 1.0,
    };
    pub const UP: Vector3 = Vector3 {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };
    pub const DOWN: Vector3 = Vector3 {
        x: 0.0,
        y: -1.0,
        z: 0.0,
    };
    pub const RIGHT: Vector3 = Vector3 {
        x: 1.0,
        y: 0.0,
        z: 0.0,
    };
    pub const LEFT: Vector3 = Vector3 {
        x: -1.0,
        y: 0.0,
        z: 0.0,
    };
    pub const FORWARD: Vector3 = Vector3 {
        x: 0.0,
        y: 0.0,
        z: 1.0,
    };
    pub const BACKWARD: Vector3 = Vector3 {
        x: 0.0,
        y: 0.0,
        z: -1.0,
    };
}

impl From<(f32, f32, f32)> for Vector3 {
    fn from(t: (f32, f32, f32)) -> Self {
        Vector3::new(t.0, t.1, t.2)
    }
}

impl From<Vector2> for Vector3 {
    fn from(v: Vector2) -> Self {
        Vector3::new(v.x, v.y, 0.0)
    }    
}

impl From<[f32; 3]> for Vector3 {
    fn from(arr: [f32; 3]) -> Self {
        Vector3::new(arr[0], arr[1], arr[2])
    }
}

impl From<Vector3> for (f32, f32, f32) {
    fn from(v: Vector3) -> Self {
        (v.x, v.y, v.z)
    }
}

impl From<Vector3> for [f32; 3] {
    fn from(v: Vector3) -> Self {
        [v.x, v.y, v.z]
    }
}

// Operators
impl Add for Vector3 {
    type Output = Vector3;
    fn add(self, other: Vector3) -> Vector3 {
        Vector3::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

impl Add<f32> for Vector3 {
    type Output = Vector3;
    fn add(self, other: f32) -> Vector3 {
        Vector3::new(self.x + other, self.y + other, self.z + other)
    }
}

impl Add<Vector3> for f32 {
    type Output = Vector3;
    fn add(self, other: Vector3) -> Vector3 {
        Vector3::new(self + other.x, self + other.y, self + other.z)
    }
}

impl AddAssign for Vector3 {
    fn add_assign(&mut self, other: Vector3) {
        *self = Vector3::new(self.x + other.x, self.y + other.y, self.z + other.z);
    }
}

impl Sub for Vector3 {
    type Output = Vector3;
    fn sub(self, other: Vector3) -> Vector3 {
        Vector3::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

impl Sub<f32> for Vector3 {
    type Output = Vector3;
    fn sub(self, other: f32) -> Vector3 {
        Vector3::new(self.x - other, self.y - other, self.z - other)
    }
}

impl Sub<Vector3> for f32 {
    type Output = Vector3;
    fn sub(self, other: Vector3) -> Vector3 {
        Vector3::new(self - other.x, self - other.y, self - other.z)
    }
}

impl SubAssign for Vector3 {
    fn sub_assign(&mut self, other: Vector3) {
        *self = Vector3::new(self.x - other.x, self.y - other.y, self.z - other.z);
    }
}

impl Mul<Vector3> for Vector3 {
    type Output = Vector3;
    fn mul(self, other: Vector3) -> Vector3 {
        Vector3::new(self.x * other.x, self.y * other.y, self.z * other.z)
    }
}

impl Mul<f32> for Vector3 {
    type Output = Vector3;
    fn mul(self, other: f32) -> Vector3 {
        Vector3::new(self.x * other, self.y * other, self.z * other)
    }
}

impl Mul<Vector3> for f32 {
    type Output = Vector3;
    fn mul(self, other: Vector3) -> Vector3 {
        Vector3::new(self * other.x, self * other.y, self * other.z)
    }
}

impl MulAssign<Vector3> for Vector3 {
    fn mul_assign(&mut self, other: Vector3) {
        *self = Vector3::new(self.x * other.x, self.y * other.y, self.z * other.z);
    }
}

impl Div<Vector3> for Vector3 {
    type Output = Vector3;
    fn div(self, other: Vector3) -> Vector3 {
        Vector3::new(self.x / other.x, self.y / other.y, self.z / other.z)
    }
}

impl Div<f32> for Vector3 {
    type Output = Vector3;
    fn div(self, other: f32) -> Vector3 {
        Vector3::new(self.x / other, self.y / other, self.z / other)
    }
}

impl Div<Vector3> for f32 {
    type Output = Vector3;
    fn div(self, other: Vector3) -> Vector3 {
        Vector3::new(self / other.x, self / other.y, self / other.z)
    }
}

impl DivAssign<Vector3> for Vector3 {
    fn div_assign(&mut self, other: Vector3) {
        *self = Vector3::new(self.x / other.x, self.y / other.y, self.z / other.z);
    }
}

impl Neg for Vector3 {
    type Output = Vector3;
    fn neg(self) -> Vector3 {
        Vector3::new(-self.x, -self.y, -self.z)
    }
}

impl Default for Vector3 {
    fn default() -> Self {
        Self::ZERO
    }
}

// Other Methods
impl Vector3 {
    #[inline]
    pub fn dot(self, other: Vector3) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    #[inline]
    pub fn pow(self, exp: f32) -> Vector3 {
        Vector3::new(self.x.powf(exp), self.y.powf(exp), self.z.powf(exp))
    }

    #[inline]
    pub fn sqrt(&self) -> Vector3 {
        Vector3::new(self.x.sqrt(), self.y.sqrt(), self.z.sqrt())
    }

    #[inline]
    pub fn cross(self, other: Vector3) -> Vector3 {
        Vector3::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    #[inline]
    pub fn magnitude(self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    #[inline]
    pub fn magnitude_squared(self) -> f32 {
        self.dot(self)
    }

    #[inline]
    pub fn normalized(self) -> Vector3 {
        let magnitude = self.magnitude();
        if magnitude == 0.0 {
            Vector3::ZERO
        } else {
            self / magnitude
        }
    }

    #[inline]
    pub fn is_zero(self) -> bool {
        self.magnitude() < f32::EPSILON
    }

    #[inline]
    pub fn is_normalised(self) -> bool {
        (self.magnitude() - 1.0).abs() < f32::EPSILON
    }

    pub fn safe_normal(self) -> Option<Vector3> {
        if self.is_zero() {
            None
        } else {
            Some(self / self.magnitude())
        }
    }

    pub fn lerp(self, other: Vector3, t: f32) -> Vector3 {
        self + (other - self) * t
    }

    pub fn distance(self, other: Vector3) -> f32 {
        (other - self).magnitude()
    }

    pub fn distance_squared(self, other: Vector3) -> f32 {
        let diff = self - other;
        diff.dot(diff)
    }
    
    pub fn abs(self) -> Vector3 {
        Vector3::new(self.x.abs(), self.y.abs(), self.z.abs())
    }

    pub fn min(self, other: Vector3) -> Vector3 {
        Vector3::new(
            self.x.min(other.x),
            self.y.min(other.y),
            self.z.min(other.z),
        )
    }

    pub fn max(self, other: Vector3) -> Vector3 {
        Vector3::new(
            self.x.max(other.x),
            self.y.max(other.y),
            self.z.max(other.z),
        )
    }

    pub fn clamp(self, min: Vector3, max: Vector3) -> Vector3 {
        self.max(min).min(max)
    }

    pub fn angle_between(self, other: Vector3) -> f32 {
        let dot = self.dot(other);
        let mags = self.magnitude() * other.magnitude();
        (dot / mags).acos()
    }

    pub fn project_onto(self, onto: Vector3) -> Vector3 {
        let d = onto.dot(onto);
        if d > 0.0 {
            onto * (self.dot(onto) / d)
        } else {
            Vector3::ZERO
        }
    }

    pub fn reject_from(self, from: Vector3) -> Vector3 {
        self - self.project_onto(from)
    }

    pub fn reflect(self, normal: Vector3) -> Vector3 {
        self - normal * (2.0 * self.dot(normal))
    }
}

// Swizzling

impl Vector3 {
    pub fn vec2(self) -> Vector2 {
        Vector2::new(self.x, self.y)
    }
    
    pub fn xy(self) -> Vector2 {
        Vector2::new(self.x, self.y)
    }

    pub fn yx(self) -> Vector2 {
        Vector2::new(self.y, self.x)
    }

    pub fn xz(self) -> Vector2 {
        Vector2::new(self.x, self.z)
    }

    pub fn zx(self) -> Vector2 {
        Vector2::new(self.z, self.x)
    }

    pub fn yz(self) -> Vector2 {
        Vector2::new(self.y, self.z)
    }

    pub fn zy(self) -> Vector2 {
        Vector2::new(self.z, self.y)
    }

    pub  fn xx(self) -> Vector2 {
        Vector2::new(self.x, self.x)
    }

    pub fn yy(self) -> Vector2 {
        Vector2::new(self.y, self.y)
    }

    pub fn zz(self) -> Vector2 {
        Vector2::new(self.z, self.z)
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
    
    // ============ Vector3 Tests ============
    #[cfg(test)]
    mod vector3_tests {
        use super::*;

        #[test]
        fn test_vector3_construction() {
            let v = Vector3::new(1.0, 2.0, 3.0);
            assert_eq!(v.x, 1.0);
            assert_eq!(v.y, 2.0);
            assert_eq!(v.z, 3.0);
        }

        #[test]
        fn test_vector3_constants() {
            assert_eq!(Vector3::ZERO, Vector3::new(0.0, 0.0, 0.0));
            assert_eq!(Vector3::ONE, Vector3::new(1.0, 1.0, 1.0));
            assert_eq!(Vector3::UP, Vector3::new(0.0, 1.0, 0.0));
            assert_eq!(Vector3::DOWN, Vector3::new(0.0, -1.0, 0.0));
            assert_eq!(Vector3::RIGHT, Vector3::new(1.0, 0.0, 0.0));
            assert_eq!(Vector3::LEFT, Vector3::new(-1.0, 0.0, 0.0));
            assert_eq!(Vector3::FORWARD, Vector3::new(0.0, 0.0, 1.0));
            assert_eq!(Vector3::BACKWARD, Vector3::new(0.0, 0.0, -1.0));
        }

        #[test]
        fn test_vector3_from_tuple() {
            let v: Vector3 = (1.0, 2.0, 3.0).into();
            assert_eq!(v, Vector3::new(1.0, 2.0, 3.0));
        }

        #[test]
        fn test_vector3_add() {
            let a = Vector3::new(1.0, 2.0, 3.0);
            let b = Vector3::new(4.0, 5.0, 6.0);
            assert_eq!(a + b, Vector3::new(5.0, 7.0, 9.0));
            assert_eq!(a + 10.0, Vector3::new(11.0, 12.0, 13.0));
            assert_eq!(10.0 + a, Vector3::new(11.0, 12.0, 13.0));
        }

        #[test]
        fn test_vector3_add_assign() {
            let mut v = Vector3::new(1.0, 2.0, 3.0);
            v += Vector3::new(4.0, 5.0, 6.0);
            assert_eq!(v, Vector3::new(5.0, 7.0, 9.0));
        }

        #[test]
        fn test_vector3_sub() {
            let a = Vector3::new(5.0, 7.0, 9.0);
            let b = Vector3::new(1.0, 2.0, 3.0);
            assert_eq!(a - b, Vector3::new(4.0, 5.0, 6.0));
            assert_eq!(a - 2.0, Vector3::new(3.0, 5.0, 7.0));
            assert_eq!(10.0 - a, Vector3::new(5.0, 3.0, 1.0));
        }

        #[test]
        fn test_vector3_sub_assign() {
            let mut v = Vector3::new(5.0, 7.0, 9.0);
            v -= Vector3::new(1.0, 2.0, 3.0);
            assert_eq!(v, Vector3::new(4.0, 5.0, 6.0));
        }

        #[test]
        fn test_vector3_mul() {
            let a = Vector3::new(2.0, 3.0, 4.0);
            let b = Vector3::new(5.0, 6.0, 7.0);
            assert_eq!(a * b, Vector3::new(10.0, 18.0, 28.0));
            assert_eq!(a * 2.0, Vector3::new(4.0, 6.0, 8.0));
            assert_eq!(2.0 * a, Vector3::new(4.0, 6.0, 8.0));
        }

        #[test]
        fn test_vector3_mul_assign() {
            let mut v = Vector3::new(2.0, 3.0, 4.0);
            v *= Vector3::new(5.0, 6.0, 7.0);
            assert_eq!(v, Vector3::new(10.0, 18.0, 28.0));
        }

        #[test]
        fn test_vector3_div() {
            let a = Vector3::new(10.0, 18.0, 28.0);
            let b = Vector3::new(5.0, 6.0, 7.0);
            assert_eq!(a / b, Vector3::new(2.0, 3.0, 4.0));
            assert_eq!(a / 2.0, Vector3::new(5.0, 9.0, 14.0));
            assert_eq!(60.0 / a, Vector3::new(6.0, 60.0/18.0, 60.0/28.0));
        }

        #[test]
        fn test_vector3_div_assign() {
            let mut v = Vector3::new(10.0, 18.0, 28.0);
            v /= Vector3::new(5.0, 6.0, 7.0);
            assert_eq!(v, Vector3::new(2.0, 3.0, 4.0));
        }

        #[test]
        fn test_vector3_neg() {
            let v = Vector3::new(1.0, -2.0, 3.0);
            assert_eq!(-v, Vector3::new(-1.0, 2.0, -3.0));
        }

        #[test]
        fn test_vector3_dot() {
            let a = Vector3::new(1.0, 2.0, 3.0);
            let b = Vector3::new(4.0, 5.0, 6.0);
            assert_eq!(a.dot(b), 32.0);
        }

        #[test]
        fn test_vector3_cross() {
            let a = Vector3::new(1.0, 0.0, 0.0);
            let b = Vector3::new(0.0, 1.0, 0.0);
            assert_eq!(a.cross(b), Vector3::new(0.0, 0.0, 1.0));

            let a = Vector3::new(2.0, 3.0, 4.0);
            let b = Vector3::new(5.0, 6.0, 7.0);
            assert_eq!(a.cross(b), Vector3::new(-3.0, 6.0, -3.0));
        }

        #[test]
        fn test_vector3_magnitude() {
            let v = Vector3::new(2.0, 3.0, 6.0);
            assert_eq!(v.magnitude(), 7.0);
        }

        #[test]
        fn test_vector3_normalized() {
            let v = Vector3::new(0.0, 3.0, 4.0);
            let n = v.normalized();
            assert!(approx_eq(n.magnitude(), 1.0));
            assert_eq!(n, Vector3::new(0.0, 0.6, 0.8));
        }

        #[test]
        fn test_vector3_is_zero() {
            assert!(Vector3::ZERO.is_zero());
            assert!(!Vector3::ONE.is_zero());
        }

        #[test]
        fn test_vector3_is_normalised() {
            assert!(Vector3::RIGHT.is_normalised());
            assert!(Vector3::new(0.0, 0.6, 0.8).is_normalised());
            assert!(!Vector3::new(2.0, 3.0, 6.0).is_normalised());
        }

        #[test]
        fn test_vector3_safe_normal() {
            let v = Vector3::new(3.0, 4.0, 0.0);
            assert_eq!(v.safe_normal(), Some(v.normalized()));
            assert_eq!(Vector3::ZERO.safe_normal(), None);
        }

        #[test]
        fn test_vector3_lerp() {
            let a = Vector3::new(0.0, 0.0, 0.0);
            let b = Vector3::new(10.0, 20.0, 30.0);
            assert_eq!(a.lerp(b, 0.0), a);
            assert_eq!(a.lerp(b, 1.0), b);
            assert_eq!(a.lerp(b, 0.5), Vector3::new(5.0, 10.0, 15.0));
        }

        #[test]
        fn test_vector3_distance() {
            let a = Vector3::new(1.0, 2.0, 3.0);
            let b = Vector3::new(4.0, 6.0, 8.0);
            assert!(approx_eq(a.distance(b), 50.0_f32.sqrt()));
            assert_eq!(a.distance_squared(b), 50.0);
        }

        #[test]
        fn test_vector3_abs() {
            let v = Vector3::new(-1.0, 2.0, -3.0);
            assert_eq!(v.abs(), Vector3::new(1.0, 2.0, 3.0));
        }

        #[test]
        fn test_vector3_min_max() {
            let a = Vector3::new(1.0, 5.0, 3.0);
            let b = Vector3::new(4.0, 2.0, 6.0);
            assert_eq!(a.min(b), Vector3::new(1.0, 2.0, 3.0));
            assert_eq!(a.max(b), Vector3::new(4.0, 5.0, 6.0));
        }

        #[test]
        fn test_vector3_clamp() {
            let v = Vector3::new(-1.0, 5.0, 2.0);
            let min = Vector3::new(0.0, 0.0, 0.0);
            let max = Vector3::new(3.0, 3.0, 3.0);
            assert_eq!(v.clamp(min, max), Vector3::new(0.0, 3.0, 2.0));
        }

        #[test]
        fn test_vector3_angle_between() {
            let a = Vector3::new(1.0, 0.0, 0.0);
            let b = Vector3::new(0.0, 1.0, 0.0);
            assert!(approx_eq(a.angle_between(b), PI / 2.0));
        }

        #[test]
        fn test_vector3_project_reject() {
            let v = Vector3::new(3.0, 4.0, 5.0);
            let onto = Vector3::new(1.0, 0.0, 0.0);
            let proj = v.project_onto(onto);
            let rej = v.reject_from(onto);
            assert_eq!(proj, Vector3::new(3.0, 0.0, 0.0));
            assert_eq!(rej, Vector3::new(0.0, 4.0, 5.0));
            assert!(approx_eq((proj + rej).x, v.x));
            assert!(approx_eq((proj + rej).y, v.y));
            assert!(approx_eq((proj + rej).z, v.z));
        }

        #[test]
        fn test_vector3_reflect() {
            let v = Vector3::new(1.0, -1.0, 0.0);
            let normal = Vector3::new(0.0, 1.0, 0.0);
            let reflected = v.reflect(normal);
            assert_eq!(reflected, Vector3::new(1.0, 1.0, 0.0));
        }

        #[test]
        fn test_vector3_pow_sqrt() {
            let v = Vector3::new(4.0, 9.0, 16.0);
            assert_eq!(v.pow(2.0), Vector3::new(16.0, 81.0, 256.0));
            assert_eq!(v.sqrt(), Vector3::new(2.0, 3.0, 4.0));
        }

        #[test]
        fn test_vector3_swizzle() {
            let v = Vector3::new(1.0, 2.0, 3.0);
            assert_eq!(v.xy(), Vector2::new(1.0, 2.0));
            assert_eq!(v.yx(), Vector2::new(2.0, 1.0));
            assert_eq!(v.xz(), Vector2::new(1.0, 3.0));
            assert_eq!(v.zx(), Vector2::new(3.0, 1.0));
            assert_eq!(v.yz(), Vector2::new(2.0, 3.0));
            assert_eq!(v.zy(), Vector2::new(3.0, 2.0));
            assert_eq!(v.xx(), Vector2::new(1.0, 1.0));
            assert_eq!(v.yy(), Vector2::new(2.0, 2.0));
            assert_eq!(v.zz(), Vector2::new(3.0, 3.0));
        }

        #[test]
        fn test_vector3_conversions() {
            let v3 = Vector3::new(3.0, 4.0, 5.0);
            let v2 = v3.vec2();
            assert_eq!(v2, Vector2::new(3.0, 4.0));
        }

        #[test]
        fn test_vector3_default() {
            let v: Vector3 = Default::default();
            assert_eq!(v, Vector3::ZERO);
        }
    }

}