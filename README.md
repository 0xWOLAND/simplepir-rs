# simplepir-rs
A fast and efficient implementation of SimplePIR in Rust. 

## Background
[Private information retrieval](https://en.wikipedia.org/wiki/Private_information_retrieval)
(PIR) is a protocol for accessing information from a database hosted on a
server without the server knowing anything about the information that was
accessed, including the index of the information within the database. SimplePIR
is a fast and efficient implementation of PIR that provides square-root
communication costs and linear computation costs. To learn more about
SimplePIR, check out [this paper](https://eprint.iacr.org/2022/949) by
Alexandra Henzinger, Matthew M. Hong, Henry Corrigan-Gibbs, Sarah Meiklejohn,
and Vinod Vaikuntanathan.

## Getting Started
We'll start by specifying some basic parameters for the SimplePIR scheme. For
good security and performance, a secret-key dimension (the length of the
encryption key) of 2,048 is recommended. We'll also specify the plaintext
modulus, which tells us the range of numbers that can be accurately accessed
and decrypted. In this example, we'll use 2^17.
```rust
use simplepir::{Matrix, Database};
let secret_dimension = 2048;
let mod_power = 17;
let plaintext_mod = 2_u64.pow(mod_power);
```
We'll then create a simple database to store our data. Databases can be created
at random or from an existing Matrix. This crate provides simple Matrix and Vector types for convenience.
```rust
let matrix = Matrix::from_data(
    vec![
        vec![1, 2, 3, 4],
        vec![5, 6, 7, 8],
        vec![9, 10, 11, 12],
        vec![13, 14, 15, 16],
    ]
);
let db = Database::from_matrix(matrix, mod_power);
```
To increase performance while also decreasing memory consumption, the database can be
compressed by packing three data records (numbers) into a single record.
```rust
let compressed_db = db.compress();
```
Now for the fun parts!

There are four main functions of the SimplePIR protocol:


#### The first function runs during the "offline" phase.

### `setup()`
Takes the database as input and outputs a hint for the client and for the
server. This is called by the **server** separately and prior to the other functions. It's very
computationally heavy, but massively speeds up the "online" portion of the protocol.

```rust
let (server_hint, client_hint) = setup(&compressed_db, secret_dimension);
```

#### The next three functions run during the "online" phase.

### `query()`
Takes an index into the database and outputs an encrypted query. This is called
by the **client**.


```rust
let index = 0;
let (client_state, query_cipher) = query(index, 4, secret_dimension, server_hint, plaintext_mod);

```
The `client_state` variable just stores some basic information about the
client's query. 


### `answer()`
Takes the matrix-vector product between the encrypted query and the entire
database and outputs an encrypted answer vector. This is called by **server**
and is the most computationally intense part of the online phase.

```rust
let answer_cipher = answer(&compressed_db, &query_cipher);
```

### `recover()`
Takes the encrypted answer vector, decrypts it, and returns the desired record.

```rust
let record = recover(&client_state, &client_hint, &answer_cipher, &query_cipher, plaintext_mod);
```

### `recover_row()`
Takes the encrypted answer vector and decrypts it to return an entire row of records. This is more efficient than calling `recover()` multiple times when you need all records in a row.

```rust
let row = recover_row(&client_state, &client_hint, &answer_cipher, &query_cipher, plaintext_mod);
```

Now if we did everything right, these asserts shouldn't fail!
```rust
assert_eq!(database.get(index).unwrap(), record);
// For recover_row, we can check any column in the row
for col in 0..db_side_len {
    assert_eq!(row.get_unchecked(col), database.get(row_index + col * db_side_len).unwrap());
}
```


## But is it fast?
Yup.

SimplePIR is a very efficient PIR protocol. The `answer()` function, the most
performance critical part of the online phase has a linear time-complexity
and runs at memory bandwidth speeds on measured hardware. The `query()` and
`recover()` functions also run very fast.

## Benchmarks
The following benchmarks were recorded on a Lenovo Thinkpad X1 Carbon 6th Gen
with an Intel Core i7-8650U @ 4.2 GHz with 16GB of RAM running Manjaro Linux.
Obviously, these will vary considerably depending on the hardware you use.
The database size used  was 3600×3600 (about 104 MB) and the secret-key
dimension was 2,048.

|Function   |Time        |Throughput          |
|-----------|------------|--------------------|
|setup()    |22.8 s      |7 MB/s              |
|query()    |56.0 ms     |N/A                 |
|answer()   |4.8 ms      |21.6 GB/s           |
|recover()  |1.4 μs      |2.5 GB/s            |

Recorded memory bandwidth was around 10-12 GB/s. The `answer()` function was
actually able to exceed memory bandwidth by around 2x thanks to the efficient
packing implementation discussed earlier.

## Acknowledgments

This repository is a fork of [XiXinping/simplepir-rs](https://github.com/XiXinping/simplepir-rs) with added functionality for efficient row recovery.

## Roadmap
As lovely as this library is, there's definitely room for improvement. I'm not
sure if I'll have time to add in new features. If you feel inclined to
implement a new feature, feel free to make a pull request!

- [ ] Support for `u16`, `u32`, and `u128`
- [ ] Implementing the packing optimization in `setup()`
- [ ] GPU support
