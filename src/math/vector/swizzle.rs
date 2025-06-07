use crate::math::{Vec2, Vec3, Vec4};

#[macro_export]
macro_rules! s {
    ($vec:expr, $($comp:tt)+) => {{
        let v = $vec;
        s!(@build v, $($comp)+)
    }};
    
    // Count components and build appropriate vector
    (@build $v:ident, $c1:tt $c2:tt) => {
        Vec2::new(s!(@get $v, $c1), s!(@get $v, $c2))
    };
    
    (@build $v:ident, $c1:tt $c2:tt $c3:tt) => {
        Vec3::new(s!(@get $v, $c1), s!(@get $v, $c2), s!(@get $v, $c3))
    };
    
    (@build $v:ident, $c1:tt $c2:tt $c3:tt $c4:tt) => {
        Vec4::new(
            s!(@get $v, $c1), 
            s!(@get $v, $c2), 
            s!(@get $v, $c3), 
            s!(@get $v, $c4)
        )
    };
    
    // Get component value
    (@get $v:ident, 0) => { 0.0 };
    (@get $v:ident, _) => { 0.0 };
    (@get $v:ident, x) => { $v.x() };
    (@get $v:ident, y) => { $v.y() };
    (@get $v:ident, z) => { $v.z() };
    (@get $v:ident, w) => { $v.w() };
    (@get $v:ident, r) => { $v.x() };  // Color aliases
    (@get $v:ident, g) => { $v.y() };
    (@get $v:ident, b) => { $v.z() };
    (@get $v:ident, a) => { $v.w() };
}

// Usage examples:
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_szle() {
        let v = Vec4::new(1.0, 2.0, 3.0, 4.0);

        // Using s! macro
        let v2 = s!(v, x y);        // Vec2(1.0, 2.0)
        assert_eq!(v2, Vec2::new(1.0, 2.0));
        let v3 = s!(v, x y z);       // Vec3(1.0, 2.0, 3.0)
        assert_eq!(v3, Vec3::new(1.0, 2.0, 3.0));
        let v4 = s!(v, w z y x);      // Vec4(4.0, 3.0, 2.0, 1.0)
        assert_eq!(v4, Vec4::new(4.0, 3.0, 2.0, 1.0));
        let zero = s!(v, 0 w 0);     // Vec3(0.0, 4.0, 0.0)
        assert_eq!(zero, Vec3::new(0.0, 4.0, 0.0));

        // Works with colors too
        let color = Vec4::new(0.5, 0.7, 0.9, 1.0);
        let rgb = s!(color, r g b);   // Vec3(0.5, 0.7, 0.9)
        assert_eq!(rgb, Vec3::new(0.5, 0.7, 0.9));
        let rg0a = s!(color, r g 0 a); // Vec4(0.5, 0.7, 0.0, 1.0)
        assert_eq!(rg0a, Vec4::new(0.5, 0.7, 0.0, 1.0));
    }
}
