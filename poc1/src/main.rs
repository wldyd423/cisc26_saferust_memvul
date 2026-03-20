use std::io::{self, Read, Write, stdin, stdout};
const UNIT: &'static &'static () = &&();

#[inline(never)]
fn swap<'a, 'b, T>(_: &'a &'b (), x: &'b T) -> &'a T {
    x
}
#[allow(dead_code)]
fn extend<'a, 'b, T>(x: &'a T) -> &'b T {
    let f: for<'c> fn(_, &'c T) -> &'b T = swap;
    f(UNIT, x)
}

#[inline(never)]
fn swap_mut<'a, 'b, T>(_: &'a &'b (), x: &'b mut T) -> &'a mut T {
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

#[allow(dead_code)]
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

fn buffer_overflow() -> io::Result<()> {
    use std::hint::black_box;

    // Stack buffer - overflow will extend past this
    let mut buf = black_box([0u8; 16]);

    // Create fake slice with large length to reach return address
    // Length should be: 16 (buf) + padding + 8 (saved rbp) + 8 (ret addr)
    let fake_slice = construct_fake_slice(buf.as_mut_ptr(), 1024);

    print!("Input: ");
    stdout().flush()?;

    // Read raw bytes - NO UTF-8 VALIDATION!
    let bytes_read = stdin().read(fake_slice)?;

    println!("Read {} bytes", bytes_read);

    // Function returns here - ret addr should be overwritten
    Ok(())
}

fn main() {
    println!("Hello, world!");
    buffer_overflow();
}
