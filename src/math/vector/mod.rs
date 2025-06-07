mod vec3;
mod vec4;
mod vec2;

// Re-export the main types for convenience
pub use vec3::Vec3;
pub use vec3::Vec;
pub use vec2::Vec2;
pub use vec4::Vec4;

#[cfg(test)]
mod tests;
mod swizzle;
