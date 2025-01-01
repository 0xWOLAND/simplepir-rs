//! A fast and efficient implementation of SimplePIR in Rust.
//!
//! # Background
//! [Private information retrieval](https://en.wikipedia.org/wiki/Private_information_retrieval)
//! (PIR) is a protocol for accessing information from a database hosted on a server without the
//! server knowing anything about the information that was accessed, including the index of the
//! information within the database. SimplePIR is a fast and efficient implementation of PIR that
//! provides square-root communication costs and linear computation costs. To learn more about
//! SimplePIR, check out [this paper](https://eprint.iacr.org/2022/949) by Alexandra Henzinger,
//! Matthew M. Hong, Henry Corrigan-Gibbs, Sarah Meiklejohn, and Vinod Vaikuntanathan.
//!
//! # Getting Started
//! We'll start by specifying some basic parameters for the SimplePIR scheme. For
//! good security and performance, a secret-key dimension (the length of the
//! encryption key) of 2,048 is recommended. We'll also specify the plaintext
//! modulus, which tells us the range of numbers that can be accurately accessed
//! and decrypted. In this example, we'll use 2^17.
//! ```
//! use simplepir::{Matrix, Database};
//! let secret_dimension = 2048;
//! let mod_power = 17;
//! let plaintext_mod = 2_u64.pow(mod_power);
//! ```
//! We'll then create a simple database to store our data. Databases can be created
//! at random or from an existing Matrix. This crate provides basic [Matrix] and [Vector] types for
//! convenience.
//! ```
//! let matrix = Matrix::from_data(
//!     vec![
//!         vec![1, 2, 3, 4],
//!         vec![5, 6, 7, 8],
//!         vec![9, 10, 11, 12],
//!         vec![13, 14, 15, 16],
//!     ]
//! );
//! let db = Database::from_matrix(matrix, mod_power);
//! ```
//! To increase performance while also decreasing memory consumption, the database can be
//! compressed by packing three data records (numbers) into a single record.
//! ```
//! let compressed_db = db.compress();
//! ```
//!
//! Now for the fun parts! There are four main functions of the SimplePIR protocol:
//!
//! ### Offline Phase
//!
//! ## [`setup()`]
//! Takes the database as input and outputs a hint for the client and for the
//! server. This is called by the **server** separately and prior to the other functions. It's very
//! computationally heavy, but massively speeds up the "online" portion of the protocol.
//!
//! ```
//! let (server_hint, client_hint) = setup(&compressed_db, secret_dimension);
//! ```
//!
//! ### Online Phase
//!
//! ## [`query()`]
//! Takes an index into the database and outputs an encrypted query. This is called
//! by the **client**.
//!
//! ```
//! let index = 0;
//! let (client_state, query_cipher) = query(index, 4, secret_dimension, server_hint,
//! plaintext_mod);
//!
//! ```
//! The `client_state` variable just stores some basic information about the
//! client's query.
//!
//! ## [`answer()`]
//! Takes the matrix-vector product between the encrypted query and the entire database and outputs
//! an encrypted answer vector. This is called by the **server** and is the most computationally
//! intense part of the online phase.
//!
//! ```
//! let answer_cipher = answer(&compressed_db, &query_cipher);
//! ```
//!
//! ## [`recover()`]
//! Takes the encrypted answer vector, decrypts it, and returns the desired record.
//!
//! ```
//! let record = recover(&client_state, &client_hint, &answer_cipher, &query_cipher, plaintext_mod);
//! ```
//!
//! ## [`recover_row()`]
//! Takes the encrypted answer vector and decrypts it to return an entire row of records. This is more
//! efficient than calling [`recover()`] multiple times when you need all records in a row.
//!
//! ```
//! let row = recover_row(&client_state, &client_hint, &answer_cipher, &query_cipher, plaintext_mod);
//! // Access individual records in the row
//! for col in 0..db_side_len {
//!     let record = row.get_unchecked(col);
//!     // Use the record...
//! }
//! ```
//!
//! Now if we did everything right, this assert shouldn't fail!
//! ```
//! assert_eq!(database.get(index).unwrap(), record);
//! ```

mod matrix;
mod regev;
use matrix::{a_matrix_mul_db, mat_vec_mul, packed_mat_vec_mul};
pub use matrix::{Matrix, Vector};
use rand::prelude::*;
use rand_chacha::ChaCha20Rng;
use rand_distr::Uniform;
use regev::{encrypt, gen_a_matrix, gen_secret_key};
use thiserror::Error;

/// A square matrix that contains `u64` data records.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd)]
pub struct Database {
    pub data: Matrix,
    pub modulus: u64,
}

/// An error type for database creation errors.
#[derive(Error, Debug)]
pub enum DatabaseError {
    /// An error resulting from the number of rows and columns in an inputted matrix not being
    /// equal.
    #[error("The number of rows and columns in the matrix must be equal")]
    NonSquareMatrixError,
    /// An error resulting from the modulus of the database being too large to sucessfully pack
    /// 3 records into 1.
    #[error("The modulus of the database must be less than 21")]
    CompressionModulusError,
}

impl Database {
    /// Creates a new Database from an existing square Matrix. If the supplied modulus is greater
    /// than 2^32, it will be reduced to 2^32.
    pub fn from_matrix(data: Matrix, mod_power: u8) -> Result<Database, DatabaseError> {
        let mod_power = if mod_power > 32 { 32 } else { mod_power };
        if data.nrows() != data.ncols() {
            return Err(DatabaseError::NonSquareMatrixError);
        }
        let modulus = 2_u64.pow(mod_power as u32);
        Ok(Database { data, modulus })
    }
    /// Creates a new Database of size `side_len` × `side_len` populated by random data. The data is
    /// sampled from a uniform distribution over the range from 0 to the plaintext modulus
    /// (`0..2^mod_power`). The plaintext modulus cannot be greater than 2^32. If the supplied
    /// modulus is larger, it will be reduced to 2^32.
    pub fn new_random(side_len: usize, mod_power: u8) -> Database {
        let mod_power = if mod_power > 32 { 32 } else { mod_power };
        let modulus = 2_u64.pow(mod_power as u32);
        let range = 0..=modulus - 1;

        Database {
            data: Matrix::new_random(side_len, side_len, range, None),
            modulus,
        }
    }
    /// Creates a new Database of size `side_len`×`side_len` populated by random data generated
    /// using a seed. The data is sampled from a uniform distribution over the range from 0 to the
    /// plaintext modulus (`0..2^mod_power`). The plaintext modulus cannot be greater than 2^32. If
    /// the supplied modulus is larger, it will be reduced to 2^32.
    pub fn new_random_seed(side_len: usize, mod_power: u8, seed: u64) -> Database {
        let mod_power = if mod_power > 32 { 32 } else { mod_power };
        let modulus = 2_u64.pow(mod_power as u32);
        let range = 0..=modulus - 1;
        Database {
            data: Matrix::new_random(side_len, side_len, range, Some(seed)),
            modulus,
        }
    }
    /// Creates a new Database from a `Vec<u64>` of data and resizes it into a square matrix. Panics
    /// if the number of entries cannot be evenly resized into a square matrix. If the plaintext
    /// modulus is greater than 2^32, the modulus is reduced to 2^32.
    pub fn from_vector(data: Vec<u64>, mod_power: u8) -> Database {
        let mod_power = if mod_power > 32 { 32 } else { mod_power };
        let modulus = 2_u64.pow(mod_power as u32);
        let db_side_len = (data.len() as f32).sqrt().ceil() as usize;
        Database {
            data: Matrix::from_vec(data, db_side_len, db_side_len),
            modulus,
        }
    }
    /// Creates a new Database populated entirely by zeros. An optional modulus can be provided,
    /// however if it's larger than 2^32, the modulus is reduced to 2^32.
    pub fn zeros(side_len: usize, mod_power: Option<u8>) -> Database {
        let mod_power = if let Some(num) = mod_power { num } else { 1 };
        let mod_power = if mod_power > 32 { 32 } else { mod_power };
        let modulus = 2_u64.pow(mod_power as u32);
        Database {
            data: Matrix::zeros(side_len, side_len),
            modulus,
        }
    }

    /// Gets the length of one side of the square Matrix within the Database.
    pub fn side_len(&self) -> usize {
        self.data.nrows()
    }

    /// Get a record at an index. The index is as if the square Matrix was resized into a vector
    /// according to row-major order.
    pub fn get(&self, index: usize) -> Option<u64> {
        let row_index = index / self.data.nrows();
        let col_index = index % self.data.ncols();
        self.data.get(row_index, col_index)
    }

    /// Compresses the database by packing three records into one 64-bit integer. The compression takes
    /// place along each row, meaning there'll be one third the number of columns in the new
    /// database compared to the old one.
    pub fn compress(&self) -> Result<CompressedDatabase, DatabaseError> {
        // let mod_power = (self.modulus as f32).log2().ceil() as u32;
        if self.modulus > 2_u64.pow(21) {
            return Err(DatabaseError::CompressionModulusError);
        }

        let mod_power = self.modulus.ilog2();
        let mask = self.modulus - 1;
        let data: Vec<u64> = self
            .data
            .data
            .iter()
            .map(move |row| {
                (0..row.len().div_ceil(3)).map(move |i| {
                    row.get(i * 3).unwrap_or(&0) & mask
                        | (row.get(i * 3 + 1).unwrap_or(&0) & mask) << mod_power
                        | (row.get(i * 3 + 2).unwrap_or(&0) & mask) << (mod_power * 2)
                })
            })
            .flatten()
            .collect();
        Ok(CompressedDatabase {
            data: Matrix::from_vec(data, self.data.nrows(), self.data.ncols().div_ceil(3)),
            nrows: self.data.nrows(),
            ncols: self.data.ncols().div_ceil(3),
            mod_power,
        })
    }
}

/// A compressed version of a regular Database with 3 records packed into one.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd)]
pub struct CompressedDatabase {
    data: Matrix,
    nrows: usize,
    ncols: usize,
    mod_power: u32,
}

/// A struct that contains information about the client's query, including the row and column index,
/// the a-matrix of the database, the side length of the database, the client's secret key, and the
/// key's length.
#[derive(Clone, Debug, Eq, PartialEq, PartialOrd)]
pub struct ClientState {
    row_index: usize,
    column_index: usize,
    a_matrix_seed: u64,
    db_side_len: usize,
    secret_key: Vector,
    secret_dimension: usize,
}

impl ClientState {
    fn new(
        row_index: usize,
        column_index: usize,
        a_matrix_seed: u64,
        db_side_len: usize,
        secret_key: &Vector,
    ) -> ClientState {
        ClientState {
            row_index,
            column_index,
            a_matrix_seed,
            db_side_len,
            secret_key: secret_key.clone(),
            secret_dimension: secret_key.len(),
        }
    }
}

/// Outputs one hint for the server and one hint for the client. The server hint is the seed to
/// generate the a-matrix for the query, which since it stays constant, can be generated ahead of
/// time. The server hint is generated at random. The client hint is the matrix multiplication of this a-matrix with the data in the
/// database. This also stays constant and can be generated ahead of time to save on computation.
pub fn setup(database: &Database, secret_dimension: usize) -> (u64, Matrix) {
    let mut rng = ChaCha20Rng::from_entropy();
    let server_hint = Uniform::from(0..=u64::MAX).sample(&mut rng);
    let data = database.data.add_scalar(u64::MAX * (database.modulus / 2));
    let a_matrix = gen_a_matrix(database.side_len(), secret_dimension, Some(server_hint));
    let client_hint = a_matrix_mul_db(&a_matrix, &data);
    (server_hint, client_hint)
}

/// Outputs one hint for the server and one hint for the client. The server hint is the seed to
/// generate the a-matrix for the query, which since it stays constant, can be generated ahead of
/// time. The server hint is generated at random according to a specified seed. The client hint is the matrix multiplication of this a-matrix with the data in the
/// database. This also stays constant and can be generated ahead of time to save on computation.
pub fn setup_seeded(database: &Database, secret_dimension: usize, seed: [u8; 32]) -> (u64, Matrix) {
    let mut rng = ChaCha20Rng::from_seed(seed);
    let server_hint = Uniform::from(0..=u64::MAX).sample(&mut rng);
    let data = database.data.add_scalar(u64::MAX * (database.modulus / 2));
    let a_matrix = gen_a_matrix(database.side_len(), secret_dimension, Some(server_hint));
    let client_hint = a_matrix_mul_db(&a_matrix, &data);
    (server_hint, client_hint)
}
/// Takes an index in the length-N database and outputs an encrypted vector with all 0s except for
/// a 1 at the column index.
pub fn query(
    index: usize,
    db_side_len: usize,
    secret_dimension: usize,
    a_matrix_seed: u64,
    plain_mod: u64,
) -> (ClientState, Vector) {
    let secret_key = gen_secret_key(secret_dimension, None);
    let a_matrix = gen_a_matrix(db_side_len, secret_dimension, Some(a_matrix_seed));
    let row_index = index % db_side_len;
    let column_index = index / db_side_len;
    let mut query_vector = Vector::zeros(db_side_len);
    query_vector.data[row_index] = 1;
    let client_state = ClientState::new(
        row_index,
        column_index,
        a_matrix_seed,
        db_side_len,
        &secret_key,
    );
    (
        client_state,
        encrypt(&secret_key, &a_matrix, &query_vector, plain_mod).1,
    )
}

/// Computes the matrix-vector product of the **packed** database and the encrypted query. The output is an
/// encrypted vector that can be decrypted to reveal the records along the column indicated in the
/// query.
pub fn answer(database: &CompressedDatabase, query_cipher: &Vector) -> Vector {
    packed_mat_vec_mul(&query_cipher, &database.data, database.mod_power)
}
/// Computes the matrix-vector product of the **non-packed** database and the encrypted query. The
/// output is an encrypted vector that can be decrypted to reveal the records along the column
/// indicated in the query.
pub fn answer_uncompressed(database: &Database, query_cipher: &Vector) -> Vector {
    mat_vec_mul(&query_cipher, &database.data)
}

/// Takes the encrypted vector of records along the column specified in the query, decrypts it using
/// the secret key, and returns the record at the row and column that was specified in the query.
pub fn recover(
    client_state: &ClientState,
    client_hint: &Matrix,
    answer_cipher: &Vector,
    query_cipher: &Vector,
    plaintext_mod: u64,
) -> u64 {
    let ciphertext_mod = 2u128.pow(64);
    let q_over_p = (ciphertext_mod / plaintext_mod as u128) as u64;

    let secret_key = &client_state.secret_key;
    let column_index = client_state.column_index;

    let ratio = plaintext_mod / 2;
    let noised = answer_cipher.get_unchecked(column_index)
        - ratio * query_cipher.sum()
        - Vector::from_vec(client_hint.row_unchecked(column_index)).dot(secret_key);
    let denoised = (noised + q_over_p / 2) / q_over_p;

    (denoised - ratio).rem_euclid(plaintext_mod)
}

/// Takes the encrypted vector of records along the column specified in the query, decrypts it using
/// the secret key, and returns a vector containing all records in the row.
pub fn recover_row(
    client_state: &ClientState,
    client_hint: &Matrix,
    answer_cipher: &Vector,
    query_cipher: &Vector,
    plaintext_mod: u64,
) -> Vector {
    let ciphertext_mod = 2u128.pow(64);
    let q_over_p = (ciphertext_mod / plaintext_mod as u128) as u64;
    let secret_key = &client_state.secret_key;
    let ratio = plaintext_mod / 2;
    let db_side_len = client_state.db_side_len;

    let mut records = Vec::with_capacity(db_side_len);
    for i in 0..db_side_len {
        let noised = answer_cipher.get_unchecked(i)
            - ratio * query_cipher.sum()
            - Vector::from_vec(client_hint.row_unchecked(i)).dot(secret_key);
        let denoised = (noised + q_over_p / 2) / q_over_p;
        records.push((denoised - ratio).rem_euclid(plaintext_mod));
    }
    Vector::from_vec(records)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test1() {
        const SEED: [u8; 32] = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        ];

        let secret_dimension = 2048;
        let db_side_len = 40;
        let mod_power = 3;
        let plain_mod = 2_u64.pow(mod_power as u32);
        let index = 0;

        let database = Database::new_random_seed(db_side_len, mod_power, 42);
        let compressed_db = database.compress().unwrap();
        let (server_hint, client_hint) = setup_seeded(&database, secret_dimension, SEED);
        let (client_state, query_cipher) =
            query(index, db_side_len, secret_dimension, server_hint, plain_mod);
        let answer_cipher = answer(&compressed_db, &query_cipher);
        let record = recover(
            &client_state,
            &client_hint,
            &answer_cipher,
            &query_cipher,
            plain_mod,
        );
        assert_eq!(record, database.get(index).unwrap())
    }

    #[test]
    fn test2() {
        const SEED: [u8; 32] = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        ];

        let secret_dimension = 2048;
        let db_side_len = 1000;
        let mod_power = 17;
        let plain_mod = 2u64.pow(mod_power as u32);

        let database = Database::new_random_seed(db_side_len, mod_power, 42);
        let compressed_database = database.compress().unwrap();
        let (server_hint, client_hint) = setup_seeded(&database, secret_dimension, SEED);
        for index in 0..100 {
            let (client_state, query_cipher) =
                query(index, db_side_len, secret_dimension, server_hint, plain_mod);
            let answer_cipher = answer(&compressed_database, &query_cipher);
            let record = recover(
                &client_state,
                &client_hint,
                &answer_cipher,
                &query_cipher,
                plain_mod,
            );
            assert_eq!(record, database.get(index).unwrap())
        }
    }

    #[test]
    fn test3() {
        const SEED: [u8; 32] = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        ];
        let secret_dimension = 10;
        let db_side_len = 1000;
        let mod_power = 3;
        let plain_mod = 2u64.pow(mod_power as u32);

        let database = Database::new_random_seed(db_side_len, mod_power, 42);
        println!("Database Modulus: {}", database.modulus);
        let (server_hint, client_hint) = setup_seeded(&database, secret_dimension, SEED);
        for index in 0..100 {
            let (client_state, query_cipher) =
                query(index, db_side_len, secret_dimension, server_hint, plain_mod);
            let answer_cipher = answer_uncompressed(&database, &query_cipher);
            let record = recover(
                &client_state,
                &client_hint,
                &answer_cipher,
                &query_cipher,
                plain_mod,
            );
            assert_eq!(record, database.get(index).unwrap())
        }
    }

    #[test]
    fn test4() {
        const SEED: [u8; 32] = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        ];

        let secret_dimension = 1000;
        let db_side_len = 38;
        let mod_power = 17;
        let plain_mod = 2u64.pow(mod_power as u32);

        let database = Database::new_random_seed(db_side_len, mod_power, 42);
        let database_compressed = database.compress().unwrap();
        let (server_hint, client_hint) = setup_seeded(&database, secret_dimension, SEED);
        for index in 0..1444 {
            let (client_state, query_cipher) =
                query(index, db_side_len, secret_dimension, server_hint, plain_mod);
            let answer_cipher = answer(&database_compressed, &query_cipher);
            let record = recover(
                &client_state,
                &client_hint,
                &answer_cipher,
                &query_cipher,
                plain_mod,
            );
            assert_eq!(record, database.get(index).unwrap())
        }
    }

    #[test]
    fn test_recover_row_basic() {
        const SEED: [u8; 32] = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        ];

        let secret_dimension = 2048;
        let db_side_len = 40;
        let mod_power = 3;
        let plain_mod = 2_u64.pow(mod_power as u32);

        let test_rows = vec![0, 1, 5, db_side_len/2, db_side_len-2, db_side_len-1];

        let database = Database::new_random_seed(db_side_len, mod_power, 42);
        
        let compressed_db = database.compress().unwrap();
        let (server_hint, client_hint) = setup_seeded(&database, secret_dimension, SEED);

        for &row_index in test_rows.iter() {
            let index = row_index;
            let (client_state, query_cipher) =
                query(index, db_side_len, secret_dimension, server_hint, plain_mod);
            
            let answer_cipher = answer(&compressed_db, &query_cipher);
            let recovered_row = recover_row(
                &client_state,
                &client_hint,
                &answer_cipher,
                &query_cipher,
                plain_mod,
            );

            let answer_cipher_uncompressed = answer_uncompressed(&database, &query_cipher);
            let recovered_row_uncompressed = recover_row(
                &client_state,
                &client_hint,
                &answer_cipher_uncompressed,
                &query_cipher,
                plain_mod,
            );

            for col in 0..db_side_len {
                let index = row_index + col * db_side_len;
                let expected = database.get(index).unwrap();
                assert_eq!(recovered_row.get_unchecked(col), expected, 
                    "Compressed DB: Mismatch at row {}, col {}", row_index, col);
                assert_eq!(recovered_row_uncompressed.get_unchecked(col), expected,
                    "Uncompressed DB: Mismatch at row {}, col {}", row_index, col);
            }
        }
    }

    #[test]
    fn test_recover_row_large() {
        const SEED: [u8; 32] = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        ];

        let secret_dimension = 2048;
        let db_side_len = 100;
        let mod_power = 17;
        let plain_mod = 2_u64.pow(mod_power as u32);

        let database = Database::new_random_seed(db_side_len, mod_power, 42);
        let compressed_db = database.compress().unwrap();
        let (server_hint, client_hint) = setup_seeded(&database, secret_dimension, SEED);

        let mut rng = ChaCha20Rng::from_seed(SEED);
        let test_rows: Vec<usize> = (0..10)
            .map(|_| rng.gen_range(0..db_side_len))
            .collect();

        for &row_index in test_rows.iter() {
            let (client_state, query_cipher) =
                query(row_index, db_side_len, secret_dimension, server_hint, plain_mod);
            let answer_cipher = answer(&compressed_db, &query_cipher);
            let recovered_row = recover_row(
                &client_state,
                &client_hint,
                &answer_cipher,
                &query_cipher,
                plain_mod,
            );

            for col in 0..db_side_len {
                let index = row_index + col * db_side_len;
                assert_eq!(recovered_row.get_unchecked(col), database.get(index).unwrap(),
                    "Large DB: Mismatch at row {}, col {}", row_index, col);
            }
        }
    }

    #[test]
    fn test_recover_row_small() {
        const SEED: [u8; 32] = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        ];

        let secret_dimension = 10;
        let db_side_len = 4;
        let mod_power = 2;
        let plain_mod = 2_u64.pow(mod_power as u32);

        let database = Database::new_random_seed(db_side_len, mod_power, 42);
        let (server_hint, client_hint) = setup_seeded(&database, secret_dimension, SEED);

        for row_index in 0..db_side_len {
            let (client_state, query_cipher) =
                query(row_index, db_side_len, secret_dimension, server_hint, plain_mod);
            let answer_cipher = answer_uncompressed(&database, &query_cipher);
            let recovered_row = recover_row(
                &client_state,
                &client_hint,
                &answer_cipher,
                &query_cipher,
                plain_mod,
            );

            for col in 0..db_side_len {
                let index = row_index + col * db_side_len;
                assert_eq!(recovered_row.get_unchecked(col), database.get(index).unwrap(),
                    "Small DB: Mismatch at row {}, col {}", row_index, col);
            }
        }
    }
}
