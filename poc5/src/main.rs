fn run(input: &String) -> &String {
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
    fn construct_fake_slice(ptr: *mut u8, len: usize) -> &'static mut [u8] {
        // Slice fat pointer layout: [ptr: usize, len: usize]
        let sentinel: &mut [u8] = transmute::<_, &mut [u8]>([0usize, 1usize]);
        let mut actual = [0usize; 2];
        actual[sentinel.as_ptr() as usize] = ptr as usize;
        actual[sentinel.len()] = len;
        std::mem::forget(sentinel);
        transmute::<_, &mut [u8]>(actual)
    }
    use std::hint::black_box;
    let mut tmp = black_box([0u8; 16]);
    let mut _dmp = construct_fake_slice(tmp.as_mut_ptr(), 2048);
    println!("tmp: {:p} tmp:{:?}", tmp.as_ptr(), tmp);
    println!("_dmp: {:p}", _dmp.as_ptr());
    //std::mem::forget(_dmp);
    _dmp[..input.len()].copy_from_slice(input.as_bytes());

    println!("tmp: {:p} tmp:{:?}", tmp.as_ptr(), tmp);
    println!("_dmp: {:p}", _dmp.as_ptr());

    input
}
fn main() {
    run(&String::from(
        "HELLLLLLLOOOaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    ));
    println!("Hello, world!");
}
