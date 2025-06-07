use crate::Vec;

pub struct Mat3 {
    e: [Vec; 3],
}

impl Mat3 {
    pub fn determinant(&self) -> f32 {
        self.e[0].x * (self.e[1].y * self.e[2].z - self.e[1].z * self.e[2].y)
            - self.e[0].y * (self.e[1].x * self.e[2].z - self.e[1].z * self.e[2].x)
            + self.e[0].z * (self.e[1].x * self.e[2].y - self.e[1].y * self.e[2].x)
    }
}
