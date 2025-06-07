
#[cfg(test)]
    use super::*;

    #[test]
    fn test_point_creation() {
        let p = Point2D::new(3.0, 4.0);
        assert_eq!(p.x(), 3.0);
        assert_eq!(p.y(), 4.0);
    }

    #[test]
    fn test_point_translation() {
        let p = Point2D::new(1.0, 2.0);
        let offset = Vec2::new(3.0, 4.0);
        let moved = p + offset;
        assert_eq!(moved, Point2D::new(4.0, 6.0));
    }

    #[test]
    fn test_point_subtraction() {
        let p1 = Point2D::new(5.0, 7.0);
        let p2 = Point2D::new(2.0, 3.0);
        let vec = p1 - p2;
        assert_eq!(vec, Vec2::new(3.0, 4.0));
    }

    #[test]
    fn test_inherited_vec2_methods() {
        let p1 = Point2D::new(0.0, 0.0);
        let p2 = Point2D::new(3.0, 4.0);

        // These methods are inherited from Vec2 via Deref
        assert_eq!(p1.distance(*p2), 5.0);
        assert_eq!(p1.lerp(*p2, 0.5), *Point2D::new(1.5, 2.0));
        assert_eq!(p2.magnitude(), 5.0);
    }

    #[test]
    fn test_conversions() {
        let tuple = (3.0, 4.0);
        let point: Point2D = tuple.into();
        assert_eq!(point, Point2D::new(3.0, 4.0));

        let back: (f32, f32) = point.into();
        assert_eq!(back, tuple);
    }
