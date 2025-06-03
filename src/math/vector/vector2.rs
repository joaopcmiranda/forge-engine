use crate::math::Vector;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vector2 {
    pub x: f32,
    pub y: f32,
}

// Constructor
impl Vector2 {
    pub fn new(x: f32, y: f32) -> Vector2 {
        Vector2 { x, y }
    }

    pub fn from_angle(angle: f32) -> Vector2 {
        Vector2::new(angle.cos(), angle.sin())
    }
}

// Constant constructors
impl Vector2 {
    pub const ZERO: Vector2 = Vector2 { x: 0.0, y: 0.0 };
    pub const ONE: Vector2 = Vector2 { x: 1.0, y: 1.0 };
    pub const UP: Vector2 = Vector2 { x: 0.0, y: 1.0 };
    pub const DOWN: Vector2 = Vector2 { x: 0.0, y: -1.0 };
    pub const RIGHT: Vector2 = Vector2 { x: 1.0, y: 0.0 };
    pub const LEFT: Vector2 = Vector2 { x: -1.0, y: 0.0 };
}

impl From<(f32, f32)> for Vector2 {
    fn from(t: (f32, f32)) -> Self {
        Vector2::new(t.0, t.1)
    }
}

impl From<[f32; 2]> for Vector2 {
    fn from(arr: [f32; 2]) -> Self {
        Vector2::new(arr[0], arr[1])
    }
}

impl From<Vector2> for (f32, f32) {
    fn from(v: Vector2) -> Self {
        (v.x, v.y)
    }
}

impl From<Vector2> for [f32; 2] {
    fn from(v: Vector2) -> Self {
        [v.x, v.y]
    }
}

// Operators
impl Add for Vector2 {
    type Output = Vector2;
    fn add(self, other: Vector2) -> Vector2 {
        Vector2::new(self.x + other.x, self.y + other.y)
    }
}

impl Add<f32> for Vector2 {
    type Output = Vector2;
    fn add(self, other: f32) -> Vector2 {
        Vector2::new(self.x + other, self.y + other)
    }
}

impl Add<Vector2> for f32 {
    type Output = Vector2;
    fn add(self, other: Vector2) -> Vector2 {
        Vector2::new(self + other.x, self + other.y)
    }
}

impl AddAssign for Vector2 {
    fn add_assign(&mut self, other: Vector2) {
        *self = Vector2::new(self.x + other.x, self.y + other.y);
    }
}

impl Sub for Vector2 {
    type Output = Vector2;
    fn sub(self, other: Vector2) -> Vector2 {
        Vector2::new(self.x - other.x, self.y - other.y)
    }
}

impl Sub<f32> for Vector2 {
    type Output = Vector2;
    fn sub(self, other: f32) -> Vector2 {
        Vector2::new(self.x - other, self.y - other)
    }
}

impl Sub<Vector2> for f32 {
    type Output = Vector2;
    fn sub(self, other: Vector2) -> Vector2 {
        Vector2::new(self - other.x, self - other.y)
    }
}

impl SubAssign for Vector2 {
    fn sub_assign(&mut self, other: Vector2) {
        *self = Vector2::new(self.x - other.x, self.y - other.y);
    }
}

impl Mul<Vector2> for Vector2 {
    type Output = Vector2;
    fn mul(self, other: Vector2) -> Vector2 {
        Vector2::new(self.x * other.x, self.y * other.y)
    }
}

impl Mul<f32> for Vector2 {
    type Output = Vector2;
    fn mul(self, other: f32) -> Vector2 {
        Vector2::new(self.x * other, self.y * other)
    }
}

impl Mul<Vector2> for f32 {
    type Output = Vector2;
    fn mul(self, other: Vector2) -> Vector2 {
        Vector2::new(self * other.x, self * other.y)
    }
}

impl MulAssign<Vector2> for Vector2 {
    fn mul_assign(&mut self, other: Vector2) {
        *self = Vector2::new(self.x * other.x, self.y * other.y);
    }
}

impl Div<Vector2> for Vector2 {
    type Output = Vector2;
    fn div(self, other: Vector2) -> Vector2 {
        Vector2::new(self.x / other.x, self.y / other.y)
    }
}

impl Div<f32> for Vector2 {
    type Output = Vector2;
    fn div(self, other: f32) -> Vector2 {
        Vector2::new(self.x / other, self.y / other)
    }
}

impl Div<Vector2> for f32 {
    type Output = Vector2;
    fn div(self, other: Vector2) -> Vector2 {
        Vector2::new(self / other.x, self / other.y)
    }
}

impl DivAssign<Vector2> for Vector2 {
    fn div_assign(&mut self, other: Vector2) {
        *self = Vector2::new(self.x / other.x, self.y / other.y);
    }
}

impl Neg for Vector2 {
    type Output = Vector2;
    fn neg(self) -> Vector2 {
        Vector2::new(-self.x, -self.y)
    }
}

// Other Methods
impl Vector2 {
    #[inline]
    pub fn dot(self, other: Vector2) -> f32 {
        self.x * other.x + self.y * other.y
    }

    #[inline]
    pub fn pow(self, exp: f32) -> Vector2 {
        Vector2::new(self.x.powf(exp), self.y.powf(exp))
    }

    #[inline]
    pub fn sqrt(&self) -> Vector2 {
        Vector2::new(self.x.sqrt(), self.y.sqrt())
    }

    #[inline]
    pub fn cross(self, other: Vector2) -> f32 {
        self.x * other.y - self.y * other.x
    }

    #[inline]
    pub fn magnitude(self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    #[inline]
    pub fn normalized(self) -> Vector2 {
        let magnitude = self.magnitude();
        if magnitude == 0.0 {
            Vector2::ZERO
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

    pub fn safe_normal(self) -> Option<Vector2> {
        if self.is_zero() {
            None
        } else {
            Some(self / self.magnitude())
        }
    }

    pub fn lerp(self, other: Vector2, t: f32) -> Vector2 {
        self + (other - self) * t
    }

    pub fn distance(self, other: Vector2) -> f32 {
        (other - self).magnitude()
    }

    pub fn distance_squared(self, other: Vector2) -> f32 {
        let diff = self - other;
        diff.dot(diff)
    }

    #[inline]
    pub fn magnitude_squared(self) -> f32 {
        self.dot(self)
    }

    #[inline]
    pub fn angle(self) -> f32 {
        self.y.atan2(self.x)
    }

    #[inline]
    pub fn angle_between(self, other: Vector2) -> f32 {
        (self.dot(other) / (self.magnitude() * other.magnitude())).acos()
    }

    pub fn rotate(self, angle: f32) -> Vector2 {
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        Vector2::new(
            self.x * cos_a - self.y * sin_a,
            self.x * sin_a + self.y * cos_a,
        )
    }

    #[inline]
    pub fn perpendicular(self) -> Vector2 {
        Vector2::new(-self.y, self.x)
    }

    #[inline]
    pub fn abs(self) -> Vector2 {
        Vector2 {
            x: self.x.abs(),
            y: self.y.abs(),
        }
    }

    #[inline]
    pub fn min(self, other: Vector2) -> Vector2 {
        Vector2::new(self.x.min(other.x), self.y.min(other.y))
    }

    #[inline]
    pub fn max(self, other: Vector2) -> Vector2 {
        Vector2::new(self.x.max(other.x), self.y.max(other.y))
    }

    pub fn project_onto(self, onto: Vector2) -> Vector2 {
        let d = onto.dot(onto);
        if d > 0.0 {
            onto * (self.dot(onto) / d)
        } else {
            Vector2::ZERO
        }
    }

    pub fn reject_from(self, from: Vector2) -> Vector2 {
        self - self.project_onto(from)
    }

    pub fn reflect(self, normal: Vector2) -> Vector2 {
        self - normal * (2.0 * self.dot(normal))
    }

    pub fn clamp(self, min: Vector2, max: Vector2) -> Vector2 {
        self.max(min).min(max)
    }
}

// Conversions
impl Vector2 {
    pub fn vec3(self) -> Vector {
        Vector::new(self.x, self.y, 0.0)
    }
}

// Defaults

impl Default for Vector2 {
    fn default() -> Self {
        Self::ZERO
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
