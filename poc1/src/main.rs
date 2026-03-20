//! Safe Rust Buffer Overflow PoC
//!
//! Demonstrates return address overwrite using HRTB lifetime unsoundness.
//! Goal: Overflow stack buffer to redirect execution to `win()`.

use std::io::{self, Read, Write, stdin, stdout};

// ============================================================================
// WIN FUNCTION - Target for return address overwrite
// ============================================================================

/// Target function for the exploit. Overwrite return address with this address.
/// Find address with: objdump -d target/debug/poc1 | grep "<poc1::win>"
#[inline(never)]
#[allow(dead_code)]
pub fn win() {
    use std::os::unix::process::CommandExt;
    println!("[!] You Win! Spawning shell...");
    // exec() replaces current process with shell, inheriting stdin/stdout
    std::process::Command::new("/bin/sh").exec();
}

// ============================================================================
// HRTB LIFETIME UNSOUNDNESS PRIMITIVES
// ============================================================================

const UNIT: &'static &'static () = &&();

#[inline(never)]
fn swap_mut<'a, 'b, T>(_: &'a &'b (), x: &'b mut T) -> &'a mut T {
    x
}

fn extend_mut<'a, 'b, T>(x: &'a mut T) -> &'b mut T {
    let f: for<'c> fn(_, &'c mut T) -> &'b mut T = swap_mut;
    f(UNIT, x)
}

// ============================================================================
// TYPE TRANSMUTATION (safe code, unsound behavior)
// ============================================================================

fn transmute<A, B>(obj: A) -> B {
    use std::hint::black_box;

    #[allow(dead_code)]
    enum DummyEnum<A, B> {
        A(Option<Box<A>>),
        B(Option<Box<B>>),
    }

    #[inline(never)]
    fn transmute_inner<A, B>(dummy: &mut DummyEnum<A, B>, obj: A) -> B {
        let DummyEnum::B(ref_to_b) = dummy else {
            unreachable!()
        };
        let ref_to_b = extend_mut(ref_to_b);
        *dummy = DummyEnum::A(Some(Box::new(obj)));
        black_box(dummy);
        *ref_to_b.take().unwrap()
    }

    transmute_inner(black_box(&mut DummyEnum::B(None)), obj)
}

// ============================================================================
// FAKE SLICE CONSTRUCTION
// ============================================================================

/// Creates a fake &mut [u8] slice pointing to arbitrary memory.
#[inline(always)]
fn construct_fake_slice(ptr: *mut u8, len: usize) -> &'static mut [u8] {
    // A fat pointer slice is 2 words: (ptr, len)
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

fn buffer_overflow() -> io::Result<()> {
    use std::hint::black_box;

    // Small stack buffer - overflow extends past this to return address
    let mut buf = black_box([0u8; 16]);

    // Create fake slice with large length to reach return address
    let fake_slice = construct_fake_slice(buf.as_mut_ptr(), 1024);

    print!("Input: ");
    stdout().flush()?;

    // Read raw bytes directly - no UTF-8 validation
    stdin().read(fake_slice)?;

    Ok(())
}

// ============================================================================
// ENTRY POINT
// ============================================================================

fn main() {
    // Print win() address for exploit development
    println!("win() @ {:p}", win as *const ());
    println!("Enter payload to overflow return address:");

    let _ = buffer_overflow();
}
