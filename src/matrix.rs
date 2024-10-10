use rand::{thread_rng, Rng};
use std::{arch::aarch64::*};

#[derive(Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    pub fn index(&self, row: usize, col: usize) -> usize {
        row * self.cols + col // Calculate the index in the flat vector
    }

    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[self.index(row, col)] // Access element using flat index
    }

    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        let idx = self.index(row, col);
        self.data[idx] = value; // Set element using flat index
    }

    pub fn zeros(rows: usize, cols: usize) -> Matrix {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn random(rows: usize, cols: usize) -> Matrix {
        let mut rng = thread_rng();
        let mut res = Matrix::zeros(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                let idx = res.index(i, j); // Store index in a local variable
                res.data[idx] = rng.gen::<f64>() * 2.0 - 1.0;
            }
        }
        res
    }

    pub fn from(data: Vec<Vec<f64>>) -> Matrix {
        let rows = data.len();
        let cols = if rows > 0 { data[0].len() } else { 0 };

        for row in &data {
            assert_eq!(
                row.len(),
                cols,
                "All rows must have the same number of columns"
            );
        }

        let mut flat_data = Vec::with_capacity(rows * cols);
        for row in data {
            flat_data.extend(row);
        }

        Matrix {
            rows,
            cols,
            data: flat_data,
        }
    }

    pub fn multiply(&self, other: &Matrix) -> Matrix {
        if self.cols != other.rows {
            panic!("Attempted to multiply by matrix of incorrect dimensions");
        }

        let other_transposed = other.transpose();
        let mut res = Matrix::zeros(self.rows, other.cols);

        unsafe {
            for i in 0..self.rows {
                for j in 0..other.cols {
                    let mut sum = vdupq_n_f64(0.0);
                    let mut k = 0;
                    while k + 4 <= self.cols {
                        let idx_a = self.index(i, k);
                        let idx_b = other_transposed.index(j, k);
                        let a = vld1q_f64(&self.data[idx_a] as *const f64);
                        let b = vld1q_f64(&other_transposed.data[idx_b] as *const f64);

                        let product = vmulq_f64(a, b);
                        sum = vaddq_f64(sum, product);
                        k += 4;
                    }

                    let mut final_sum = vaddvq_f64(sum);

                    while k < self.cols {
                        final_sum += self.data[self.index(i, k)] * other_transposed.data[other_transposed.index(j, k)];
                        k += 1;
                    }

                    let res_idx = res.index(i, j);
                    res.data[res_idx] = final_sum;
                }
            }
        }

        res
    }

    pub fn add(&self, other: &Matrix) -> Matrix {
        if self.cols != other.cols || self.rows != other.rows {
            panic!("Attempted to add matrices with incorrect dimensions");
        }

        let mut res = Matrix::zeros(self.rows, self.cols);

        unsafe {
            for i in 0..self.rows {
                let mut j = 0;

                // SIMD: Process 4 elements at a time
                while j + 4 <= self.cols {
                    let idx_a = self.index(i, j);
                    let idx_b = other.index(i, j);
                    let a = vld1q_f64(&self.data[idx_a] as *const f64);
                    let b = vld1q_f64(&other.data[idx_b] as *const f64);

                    let sum = vaddq_f64(a, b);

                    let res_idx = res.index(i, j);
                    vst1q_f64(&mut res.data[res_idx] as *mut f64, sum);

                    j += 4;
                }

                // Handle remaining elements (non-multiples of 4)
                while j < self.cols {
                    let res_idx = res.index(i, j); // Store the result of res.index(i, j)
                    res.data[res_idx] = self.data[self.index(i, j)] + other.data[other.index(i, j)];
                    j += 1;
                }
            }
        }

        res
    }




    pub fn dot_multiply(&self, other: &Matrix) -> Matrix {
        let mut res = Matrix::zeros(self.rows, self.cols);

        unsafe {
            for i in 0..self.rows {
                let mut j = 0;

                while j + 4 <= self.cols {
                    let idx_a = self.index(i, j);
                    let idx_b = other.index(i, j);
                    let a = vld1q_f64(&self.data[idx_a] as *const f64);
                    let b = vld1q_f64(&other.data[idx_b] as *const f64);

                    let product = vmulq_f64(a, b);

                    let res_idx = res.index(i, j);
                    vst1q_f64(&mut res.data[res_idx] as *mut f64, product);

                    j += 4;
                }

                while j < self.cols {
                    let res_idx = res.index(i, j);
                    res.data[res_idx] = self.data[self.index(i, j)] * other.data[other.index(i, j)];
                    j += 1;
                }
            }
        }

        res
    }

    pub fn subtract(&self, other: &Matrix) -> Matrix {
    if self.cols != other.cols || self.rows != other.rows {
        panic!("Attempted to subtract matrices with incorrect dimensions");
    }

    let mut res = Matrix::zeros(self.rows, self.cols);

    unsafe {
        for i in 0..self.rows {
            let mut j = 0;

            // SIMD: Process 4 elements at a time
            while j + 4 <= self.cols {
                let idx_a = self.index(i, j);
                let idx_b = other.index(i, j);
                let a = vld1q_f64(&self.data[idx_a] as *const f64);
                let b = vld1q_f64(&other.data[idx_b] as *const f64);

                let difference = vsubq_f64(a, b);

                let res_idx = res.index(i, j);
                vst1q_f64(&mut res.data[res_idx] as *mut f64, difference);

                j += 4;
            }

            // Handle remaining elements (non-multiples of 4)
            while j < self.cols {
                let res_idx = res.index(i, j); // Store the result of res.index(i, j)
                res.data[res_idx] = self.data[self.index(i, j)] - other.data[other.index(i, j)];
                j += 1;
            }
        }
    }

    res
}


    pub fn map(&self, function: &dyn Fn(f64) -> f64) -> Matrix {
        let mapped_data: Vec<f64> = self.data.iter().map(|&value| function(value)).collect();

        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: mapped_data,
        }
    }

    pub fn transpose(&self) -> Matrix {
        let mut res = Matrix::zeros(self.cols, self.rows);

        for i in 0..self.rows {
            for j in 0..self.cols {
                let res_idx = res.index(j, i);
                res.data[res_idx] = self.data[self.index(i, j)];
            }
        }

        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let a = Matrix::from(vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
        ]);
        let b = Matrix::from(vec![
            vec![2.0, 4.0, 6.0, 8.0],
            vec![1.0, 3.0, 5.0, 7.0],
        ]);

        let result = a.add(&b);
        let expected = Matrix::from(vec![
            vec![3.0, 6.0, 9.0, 12.0],
            vec![6.0, 9.0, 12.0, 15.0],
        ]);

        assert_eq!(result.data, expected.data);
    }

    #[test]
    fn test_subtract() {
        let a = Matrix::from(vec![
            vec![5.0, 6.0, 7.0, 8.0],
            vec![10.0, 11.0, 12.0, 13.0],
        ]);
        let b = Matrix::from(vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
        ]);

        let result = a.subtract(&b);
        let expected = Matrix::from(vec![
            vec![4.0, 4.0, 4.0, 4.0],
            vec![5.0, 5.0, 5.0, 5.0],
        ]);

        assert_eq!(result.data, expected.data);
    }

    #[test]
    fn test_multiply() {
        let a = Matrix::from(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ]);
        let b = Matrix::from(vec![
            vec![7.0, 8.0],
            vec![9.0, 10.0],
            vec![11.0, 12.0],
        ]);

        let result = a.multiply(&b);
        let expected = Matrix::from(vec![
            vec![58.0, 64.0],
            vec![139.0, 154.0],
        ]);

        assert_eq!(result.data, expected.data);
    }

    #[test]
    fn test_dot_multiply() {
        let a = Matrix::from(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ]);
        let b = Matrix::from(vec![
            vec![7.0, 8.0, 9.0],
            vec![10.0, 11.0, 12.0],
        ]);

        let result = a.dot_multiply(&b);
        let expected = Matrix::from(vec![
            vec![7.0, 16.0, 27.0],
            vec![40.0, 55.0, 72.0],
        ]);

        assert_eq!(result.data, expected.data);
    }

    #[test]
    fn test_transpose() {
        let a = Matrix::from(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ]);

        let result = a.transpose();
        let expected = Matrix::from(vec![
            vec![1.0, 4.0],
            vec![2.0, 5.0],
            vec![3.0, 6.0],
        ]);

        assert_eq!(result.data, expected.data);
    }
}
