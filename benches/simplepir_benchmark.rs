use criterion::{criterion_group, criterion_main, Criterion};
use simplepir::{answer, query, recover, setup, Database};

pub fn setup_benchmark(c: &mut Criterion) {
    let secret_dimension = 2048;
    let db_side_len = 3_600;
    let mod_power = 1;
    const SEED: Option<u64> = Some(42);
    let database = Database::new_random(db_side_len, mod_power);
    let mut group = c.benchmark_group("setup");
    group.sample_size(10);
    group.bench_function("setup", |b| b.iter(|| setup(&database, secret_dimension)));
    group.finish();
}

pub fn retrieval_benchmark(c: &mut Criterion) {
    let secret_dimension = 2048;
    let db_side_len = 3_600;
    let mod_power = 17;
    let plain_mod = 2_u64.pow(mod_power as u32);
    let database = Database::new_random(db_side_len, mod_power);
    let database_compressed = database.compress().unwrap();
    let (server_hint, client_hint) = setup(&database, secret_dimension);
    let (client_state, query_cipher) =
        query(0, db_side_len, secret_dimension, server_hint, plain_mod);
    let answer_cipher = answer(&database_compressed, &query_cipher);
    let mut group = c.benchmark_group("retrieval");
    group.sample_size(100);
    group.bench_function("query", |b| {
        b.iter(|| query(0, db_side_len, secret_dimension, server_hint, plain_mod))
    });
    group.bench_function("answer", |b| {
        b.iter(|| answer(&database_compressed, &query_cipher))
    });
    group.bench_function("recover", |b| {
        b.iter(|| {
            recover(
                &client_state,
                &client_hint,
                &answer_cipher,
                &query_cipher,
                plain_mod,
            )
        })
    });
    group.finish();
}

criterion_group!(benches, setup_benchmark);
criterion_main!(benches);
