//! cve-rs buffer_overflow 핵심 로직 PoC (100% safe Rust, #![deny(unsafe_code)])
//!
//! cve-rs의 lifetime soundness hole을 이용한 safe transmute로
//! fake String을 만들어 BOF를 일으킨다.

#![deny(unsafe_code)]

use std::io::{stdin, stdout, Write};
use std::mem;

// ── lifetime expansion (from cve-rs) ──

#[inline(never)]
const fn lifetime_translator<'a, 'b, T: ?Sized>(_val_a: &'a &'b (), val_b: &'b T) -> &'a T {
    val_b
}

#[inline(never)]
fn lifetime_translator_mut<'a, 'b, T: ?Sized>(
    _val_a: &'a &'b (),
    val_b: &'b mut T,
) -> &'a mut T {
    val_b
}

const STATIC_UNIT: &&() = &&();

fn expand_mut<'a, 'b, T: ?Sized>(x: &'a mut T) -> &'b mut T {
    let f: for<'x> fn(_, &'x mut T) -> &'b mut T = lifetime_translator_mut;
    f(STATIC_UNIT, x)
}

// ── safe transmute (from cve-rs) ──

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
        let ref_to_b = expand_mut(ref_to_b);
        *dummy = DummyEnum::A(Some(Box::new(obj)));
        black_box(dummy);
        *ref_to_b.take().unwrap()
    }

    transmute_inner(black_box(&mut DummyEnum::B(None)), obj)
}

// ── fake String (from cve-rs) ──

fn construct_fake_string(ptr: *mut u8, cap: usize, len: usize) -> String {
    let sentinel: String = transmute::<_, String>([0usize, 1usize, 2usize]);

    let mut buf = [0usize; 3];
    buf[sentinel.as_ptr() as usize] = ptr as usize;
    buf[sentinel.capacity()] = cap;
    buf[sentinel.len()] = len;

    mem::forget(sentinel);
    transmute::<_, String>(buf)
}

// ── vuln ──

#[repr(C)]
#[derive(Default)]
struct Vuln {
    buf: [u8; 16],
}

/// hijack 대상
#[inline(never)]
fn win() {
    println!("== RIP hijacked! You landed in win()! lol! ==");
    use std::os::unix::process::CommandExt;
    let err = std::process::Command::new("/bin/sh").exec();
    eprintln!("exec failed: {err}");
}

/// ROP용: null-terminated "/bin/sh\0" 주소를 출력
#[inline(never)]
fn bin_sh_addr() -> *const u8 {
    let s = std::ffi::CStr::from_bytes_with_nul(b"/bin/sh\0").unwrap();
    s.as_ptr() as *const u8
}

#[inline(never)]
fn buffer_overflow() {
    use std::hint::black_box;

    let mut v = black_box(Vuln::default());

    // 16바이트 buf를 capacity 1024인 fake String으로 위장
    // Box로 감싸서 String 메타데이터를 힙에 둔다 → 스택 overflow가 String 구조체를 안 깨뜨림
    let mut name = Box::new(construct_fake_string(v.buf.as_mut_ptr(), 1024, 0));

    print!("Name? > ");
    stdout().flush().unwrap();
    // read_line은 먼저 스택에 쓴 뒤 UTF-8 검증 → 에러 무시해도 데이터는 이미 스택에 있음
    let _ = stdin().read_line(&mut *name);

    mem::forget(name);
    black_box(v);
}

fn main() {
    println!("[*] win @ 0x{:x}", win as usize);
    println!("[*] binsh @ 0x{:x}", bin_sh_addr() as usize);
    buffer_overflow();
}
