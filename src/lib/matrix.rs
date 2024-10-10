use rand::{thread_rng, Rng};
use std::{arch::aarch64::*, env::consts};

#[derive(Clone)]
pub struct Matrix{
    pub rows : usize,
    pub cols : usize,
    pub data : Vec<Vec<f64>>,
}


impl Matrix{
    pub fn zeros(rows: usize, cols: usize) ->Matrix{
        Matrix{
            rows,
            cols,
            data: vec![vec![0.0; cols]; rows]
        }
    }

    pub fn random(rows: usize, cols: usize) ->Matrix{
        let mut rng = thread_rng();
        let mut res = Matrix::zeros(rows,cols);
        for i in 0..rows{
            for j in 0..cols{
                res.data[i][j] = rng.gen::<f64>() * 2.0 - 1.0;
            }
        }
        res
    }

    pub fn from(data: Vec<Vec<f64>>) ->Matrix{
        Matrix{
            rows: data.len(),
            cols: data[0].len(),
            data: data
        }
    }

    pub fn multiply(&self, other : &Matrix) -> Matrix{
        if self.cols != other.rows{
            panic!("Attempted to multiply by matrix of incorrect dimensions");
        }

        let mut res = Matrix::zeros(self.rows,other.cols);

        unsafe {
            for i in 0..self.rows{
                for j in 0.. other.cols {
                    let mut sum = vdupq_n_f64(0.0);
                    let mut k = 0 ;
                    while k + 4 <= self.cols {
                        let a = vld1q_f64(&self.data[i][k] as *const f64);
                        let b = vld1q_f64(&other.data[k][j] as *const f64);

                        let product = vmulq_f64(a, b);
                        sum = vaddq_f64(sum, product);
                        k+=4;
                    }

                    let mut final_sum = vaddvq_f64(sum);

                    while k< self.cols {
                        final_sum += self.data[i][k] * other.data[k][j];
                        k+=1;
                    }
                    res.data[i][j] = final_sum;

                }
            }
        }

        res
    }




    pub fn add(&self, other : &Matrix) -> Matrix{
        if self.cols != other.cols || self.rows != other.rows {
            panic!("Attempted to add matrices with incorrect dimensions");
        }

        let mut res = Matrix::zeros(self.rows, self.cols);

        unsafe {
            for i in 0..self.rows {
                let mut j = 0;

                // SIMD: process 4 elements at a time
                while j + 4 <= self.cols {
                    // Load 4 elements from self and other into SIMD vectors
                    let a = vld1q_f64(&self.data[i][j] as *const f64); // Load 4 elements from self
                    let b = vld1q_f64(&other.data[i][j] as *const f64); // Load 4 elements from other

                    // Add the two vectors
                    let sum = vaddq_f64(a, b);

                    // Store result back into the result matrix
                    vst1q_f64(&mut res.data[i][j] as *mut f64, sum);

                    j += 4; // Move to the next 4 elements
                }

                // Handle the remaining elements (if the number of columns is not a multiple of 4)
                while j < self.cols {
                    res.data[i][j] = self.data[i][j] + other.data[i][j];
                    j += 1;
                }
            }
        }

        res
    }


    pub fn dot_multiply(&self, other : &Matrix) -> Matrix{
        
        let mut res = Matrix::zeros(self.rows, self.cols);

        unsafe {
            for i in 0..self.rows {
                let mut j = 0;

                // SIMD: process 4 elements at a time
                while j + 4 <= self.cols {
                    // Load 4 elements from self and other into SIMD vectors
                    let a = vld1q_f64(&self.data[i][j] as *const f64); // Load 4 elements from self
                    let b = vld1q_f64(&other.data[i][j] as *const f64); // Load 4 elements from other

                    // Add the two vectors
                    let sum = vmulq_f64(a, b);

                    // Store result back into the result matrix
                    vst1q_f64(&mut res.data[i][j] as *mut f64, sum);

                    j += 4; // Move to the next 4 elements
                }

                // Handle the remaining elements (if the number of columns is not a multiple of 4)
                while j < self.cols {
                    res.data[i][j] = self.data[i][j] * other.data[i][j];
                    j += 1;
                }
            }
        }

        res
    }



    pub fn subtract(&self, other : &Matrix) -> Matrix{
        if(self.cols != other.cols && self.rows != other.rows){
            panic!("Attempted to subtract by matrix of incorrect dimensions");
        }

        
        let mut res = Matrix::zeros(self.rows, self.cols);

        unsafe {
            for i in 0..self.rows {
                let mut j = 0;

                // SIMD: process 4 elements at a time
                while j + 4 <= self.cols {
                    // Load 4 elements from self and other into SIMD vectors
                    let a = vld1q_f64(&self.data[i][j] as *const f64); // Load 4 elements from self
                    let b = vld1q_f64(&other.data[i][j] as *const f64); // Load 4 elements from other

                    // Add the two vectors
                    let sum = vsubq_f64(a, b);

                    // Store result back into the result matrix
                    vst1q_f64(&mut res.data[i][j] as *mut f64, sum);

                    j += 4; // Move to the next 4 elements
                }

                // Handle the remaining elements (if the number of columns is not a multiple of 4)
                while j < self.cols {
                    res.data[i][j] = self.data[i][j] - other.data[i][j];
                    j += 1;
                }
            }
        }

        res
    }


    pub fn map(&self, function : &dyn Fn(f64)-> f64) -> Matrix{
            Matrix::from(
                    (self.data)
                        .clone()
                        .into_iter()
                        .map(|row| row.into_iter().map(|value| function(value)).collect())
                        .collect(),
                    )
            }

    pub fn transpose(&self) ->Matrix{
        let mut res = Matrix::zeros(self.cols,self.rows);

        for i in 0..self.rows{
            for j in 0..self.cols{
                res.data[j][i] = self.data[i][j] 
            }
        }
        res
    }

    
}



#[cfg(test)]
mod tests {
    use super::*; // Import everything from the parent module

    // Test for matrix addition
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

    // Test for matrix subtraction
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

    // Test for matrix multiplication
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

    // Test for matrix dot multiplication (Hadamard product)
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

    // Test for matrix transpose
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
