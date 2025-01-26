use nalgebra::{DMatrix, DVector};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, Normal};

#[derive(Debug, Clone)]
pub struct SimplePIRParams {
    pub n: usize,       // LWE dimension
    pub m: usize,       // Matrix dimension
    pub q: u64,         // LWE modulus
    pub p: u64,         // Plaintext modulus
    std_dev: f64,   // Standard deviation for error
    seed: u64,      // Random seed for reproducibility
}

pub fn gen_params(m: usize, n: usize, mod_power: u32) -> SimplePIRParams {
    let mut rng = rand::thread_rng();
    SimplePIRParams {
        n,
        m,
        q: u64::MAX,  // 2^64 - 1
        p: 1u64 << mod_power,
        std_dev: 3.2,
        seed: rng.gen(),
    }
}

pub fn gen_matrix_a(seed: u64, m: usize, n: usize, q: u64) -> DMatrix<u64> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let data: Vec<u64> = (0..m * n)
        .map(|_| {
            let a = rng.gen::<u32>() as u64;
            let b = rng.gen::<u32>() as u64;
            ((a as u128 * b as u128) % q as u128) as u64
        })
        .collect();
    DMatrix::from_vec(m, n, data)
}

pub fn gen_secret(q: u64, n: usize, seed: Option<u64>) -> DVector<u64> {
    let mut rng = match seed {
        Some(s) => ChaCha20Rng::seed_from_u64(s),
        None => ChaCha20Rng::from_entropy(),
    };
    
    let data: Vec<u64> = (0..n)
        .map(|_| {
            let a = rng.gen::<u32>() as u64;
            let b = rng.gen::<u32>() as u64;
            ((a as u128 * b as u128) % q as u128) as u64
        })
        .collect();
    DVector::from_vec(data)
}

// Helper function for safe modular multiplication
fn safe_mod_mul(a: u64, b: u64, q: u64) -> u64 {
    ((a as u128 * b as u128) % q as u128) as u64
}

// Helper function for safe modular addition
fn safe_mod_add(a: u64, b: u64, q: u64) -> u64 {
    let sum = a as u128 + b as u128;
    (sum % q as u128) as u64
}

pub fn gen_hint(params: &SimplePIRParams, db: &DMatrix<u64>) -> (DMatrix<u64>, DMatrix<u64>) {
    let a = gen_matrix_a(params.seed, params.m, params.n, params.q);
    
    // Matrix multiplication with modulo
    let mut hint = DMatrix::zeros(db.nrows(), a.ncols());
    for i in 0..db.nrows() {
        for j in 0..a.ncols() {
            let mut sum = 0u64;
            for k in 0..db.ncols() {
                sum = safe_mod_add(sum, safe_mod_mul(db[(i, k)], a[(k, j)], params.q), params.q);
            }
            hint[(i, j)] = sum;
        }
    }
    
    (hint, a)
}

pub fn encrypt(params: &SimplePIRParams, v: &DVector<u64>, a: &DMatrix<u64>, s: &DVector<u64>) -> DVector<u64> {
    let delta = params.q / params.p;
    
    // Generate Gaussian error
    let normal = Normal::new(0.0, params.std_dev).unwrap();
    let mut rng = rand::thread_rng();
    let e: DVector<u64> = DVector::from_iterator(
        params.m,
        (0..params.m).map(|_| {
            let err = (normal.sample(&mut rng) * params.p as f64).round();
            let err_mod = ((err % params.q as f64) + params.q as f64) % params.q as f64;
            err_mod as u64
        })
    );
    
    // Compute As
    let mut as_prod = DVector::zeros(params.m);
    for i in 0..params.m {
        let mut sum = 0u64;
        for j in 0..params.n {
            sum = safe_mod_add(sum, safe_mod_mul(a[(i, j)], s[j], params.q), params.q);
        }
        as_prod[i] = sum;
    }
    
    // Compute final result with modular arithmetic
    let mut result = DVector::zeros(params.m);
    for i in 0..params.m {
        let mut sum = as_prod[i];
        sum = safe_mod_add(sum, e[i], params.q);
        sum = safe_mod_add(sum, safe_mod_mul(delta, v[i], params.q), params.q);
        result[i] = sum;
    }
    
    result
}

pub fn generate_query(params: &SimplePIRParams, v: &DVector<u64>, a: &DMatrix<u64>) -> (DVector<u64>, DVector<u64>) {
    assert_eq!(v.len(), params.m, "Vector dimension mismatch");
    
    let s = gen_secret(params.q, params.n, None);
    let query = encrypt(params, v, a, &s);
    
    (s, query)
}

pub fn process_query(db: &DMatrix<u64>, query: &DVector<u64>, q: u64) -> DVector<u64> {
    let mut result = DVector::zeros(db.nrows());
    for i in 0..db.nrows() {
        let mut sum = 0u64;
        for j in 0..db.ncols() {
            sum = safe_mod_add(sum, safe_mod_mul(db[(i, j)], query[j], q), q);
        }
        result[i] = sum;
    }
    result
}

pub fn recover(hint: &DMatrix<u64>, s: &DVector<u64>, answer: &DVector<u64>, params: &SimplePIRParams) -> DVector<u64> {
    let delta = params.q / params.p;
    
    // Compute hint * s with modulo
    let mut hint_s = DVector::zeros(answer.len());
    for i in 0..answer.len() {
        let mut sum = 0u64;
        for j in 0..s.len() {
            sum = safe_mod_add(sum, safe_mod_mul(hint[(i, j)], s[j], params.q), params.q);
        }
        hint_s[i] = sum;
    }
    
    // Compute decrypted = answer - hint*s mod q
    let mut decrypted = DVector::zeros(answer.len());
    for i in 0..answer.len() {
        let diff = if answer[i] >= hint_s[i] {
            answer[i] - hint_s[i]
        } else {
            params.q - (hint_s[i] - answer[i])
        };
        decrypted[i] = diff % params.q;
    }
    
    // Round to nearest multiple of delta and scale down
    decrypted.map(|x| {
        let rounded = ((x as f64 / delta as f64).round() as u64) % params.p;
        rounded
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pir() {
        let matrix_height = 10;
        let matrix_width = 10;
        let max_val = (1u64 << 17) - 1;
        
        // Create random test data
        let mut rng = rand::thread_rng();
        let d_data: Vec<u64> = (0..matrix_height * matrix_width)
            .map(|_| rng.gen_range(0..=max_val))
            .collect();
        let d = DMatrix::from_vec(matrix_height, matrix_width, d_data);
        
        let v_data: Vec<u64> = (0..matrix_width)
            .map(|_| rng.gen_range(0..=max_val))
            .collect();
        let v = DVector::from_vec(v_data);
        
        // Expected result
        let expected = {
            let mut result = DVector::zeros(matrix_height);
            for i in 0..matrix_height {
                let mut sum = 0u64;
                for j in 0..matrix_width {
                    sum = (sum + d[(i, j)] * v[j]) % (1u64 << 17);
                }
                result[i] = sum;
            }
            result
        };
        
        // Test system
        let params = gen_params(matrix_height, 2048, 17);
        let (hint, a) = gen_hint(&params, &d);
        let (s, query) = generate_query(&params, &v, &a);
        let answer = process_query(&d, &query, params.q);
        let result = recover(&hint, &s, &answer, &params);
        
        // Compare results
        assert_eq!(expected, result, "Test failed: Results don't match");
        println!("Success: Test passed!");
    }

    #[test]
    fn test_pir_row_retrieval() {
        let matrix_height = 10;
        let matrix_width = 10;
        let max_val = (1u64 << 17) - 1;
        
        // Create random test data
        let mut rng = rand::thread_rng();
        let d_data: Vec<u64> = (0..matrix_height * matrix_width)
            .map(|_| rng.gen_range(0..=max_val))
            .collect();
        let d = DMatrix::from_vec(matrix_height, matrix_width, d_data);
        
        // Create query vector (all zeros except 1 at a random position)
        let target_row = rng.gen_range(0..matrix_width);
        let mut v_data = vec![0u64; matrix_width];
        v_data[target_row] = 1;
        let v = DVector::from_vec(v_data);
        
        // Expected result - simply the target row from the matrix
        let expected = {
            let mut result = DVector::zeros(matrix_height);
            for i in 0..matrix_height {
                result[i] = d[(i, target_row)];
            }
            result
        };
        
        // Test system
        let params = gen_params(matrix_height, 2048, 17);
        let (hint, a) = gen_hint(&params, &d);
        let (s, query) = generate_query(&params, &v, &a);
        let answer = process_query(&d, &query, params.q);
        let result = recover(&hint, &s, &answer, &params);
        
        // Compare results
        assert_eq!(expected, result, 
            "Test failed: Results don't match for target row {}", target_row);
        println!("Success: Row retrieval test passed! Retrieved row {}", target_row);
    }
}
