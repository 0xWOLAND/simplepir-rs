use crate::{mat_vec_mul, Matrix, Vector};

pub mod gauss;
use gauss::gauss_sample;

// Generates the secret key used for encryption.
pub fn gen_secret_key(secret_dimension: usize, seed: Option<u64>) -> Vector {
    Vector::new_random(secret_dimension, 0..=u64::MAX, seed)
}

// Generates the A-matrix (the left hand side of the systems of equations) with dimension m x n.
pub fn gen_a_matrix(num_samples: usize, secret_dimension: usize, seed: Option<u64>) -> Matrix {
    Matrix::new_random(num_samples, secret_dimension, 0..=u64::MAX, seed)
}

// Encrypts an array of bits with symmetrical Learning With Errors (LWE) encryption using the
// secret key.
pub fn encrypt(
    secret_key: &Vector,
    a_matrix: &Matrix,
    bits: &Vector,
    plaintext_mod: u64,
) -> (Matrix, Vector) {
    let ciphertext_mod = 2_u128.pow(64);
    let q_over_p = (ciphertext_mod / plaintext_mod as u128) as u64;
    // the number of columns of the a matrix should be the same as the secret dimension
    assert_eq!(
        secret_key.len(),
        a_matrix.ncols(),
        "The number of columns in the a-matrix must match the length of the secret key!"
    );
    // the length of bits and the number of rows in the a matrix should be equal to the number of
    // samples
    assert_eq!(
        bits.len(),
        a_matrix.nrows(),
        "The number of rows in the a-matrix must match the number of samples!"
    );
    let num_samples = bits.len();
    let error_vector = Vector::from_vec((0..num_samples).map(|_| gauss_sample() as u64).collect());
    (
        a_matrix.clone(),
        mat_vec_mul(secret_key, a_matrix)
            .add(&error_vector)
            .add(&bits.mul_scalar(q_over_p)),
    )
}
