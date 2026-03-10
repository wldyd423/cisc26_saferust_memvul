RUSTFLAGS="-Z instrument-mcount" cargo build --release
uftrace ./target/release/trustmebro
