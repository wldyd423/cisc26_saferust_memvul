use std::io::{self, Read, Write, stdin, stdout};
const UNIT: &'static &'static () = &&();

#[inline(never)]
fn swap_mut<'a, 'b, T>(_anchor: &'a &'b (), x: &'b mut T) -> &'a mut T {
    x
}

fn extend_mut<'a, 'b, T>(x: &'a mut T) -> &'b mut T {
    let f: for<'c> fn(_, &'c mut T) -> &'b mut T = swap_mut;
    f(UNIT, x)
}

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

#[inline(always)]
fn construct_fake_string(ptr: *mut u8, cap: usize, len: usize) -> String {
    let sentinel_string = transmute::<_, String>([0usize, 1usize, 2usize]);
    let mut actual_buf = [0usize; 3];
    actual_buf[sentinel_string.as_ptr() as usize] = ptr as usize;
    actual_buf[sentinel_string.capacity()] = cap;
    actual_buf[sentinel_string.len()] = len;

    std::mem::forget(sentinel_string);
    transmute::<_, String>(actual_buf)
}

fn buffer_overflow() -> io::Result<()> {
    use std::hint::black_box;

    // 16-byte stack buffer
    let mut buf = black_box([0u8; 16]);

    // Create fake slice with inflated length to reach return address
    let mut fake_string = construct_fake_string(buf.as_mut_ptr(), 1024usize, 0usize);

    print!("Input: ");
    stdout().flush()?;

    // Read directly into fake slice - no bounds checking!
    stdin().read_line(&mut fake_string)?;

    Ok(())
}

fn main() {
    let _ = buffer_overflow();
}
