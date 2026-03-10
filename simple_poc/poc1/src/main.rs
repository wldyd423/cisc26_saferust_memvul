//! Demonstration of Rust unsoundness bug (issue #25860)
//!
//! This PoC shows how higher-rank lifetime bounds can be exploited to
//! bypass Rust's borrow checker, creating dangling references in safe code.
//!
//! This variant uses `&'static[&'static str]` (like deserialize_enum's variants
//! parameter) as the anchor instead of `&&()`, showing how this pattern could
//! appear in real-world code.
//!
//! See: https://github.com/rust-lang/rust/issues/25860

/// This function is SOUND on its own.
///
/// The constraint `&'a [&'b T]` requires 'b to outlive 'a (the inner references
/// must live at least as long as the slice). Therefore returning `&'a K` from
/// `&'b K` is valid because 'b >= 'a.
#[inline(never)]
fn lifetime_translator<'a, 'b, T: ?Sized, K: ?Sized>(_anchor: &'a [&'b T], val: &'b K) -> &'a K {
    val
}

/// Expands lifetime 'a to arbitrary lifetime 'b using a static slice as anchor.
///
/// The trick: we coerce `lifetime_translator` to a higher-rank type
/// `for<'x> fn(_, &'x K) -> &'b K`. The compiler should reject this
/// because 'x might not outlive 'b, but due to the soundness hole,
/// it accepts the coercion when we pass a &'static[&'static T].
fn extend<'a, 'b, T: ?Sized, K: ?Sized>(x: &'a K, var: &'static [&'static T]) -> &'b K {
    // Coerce to higher-rank type - this is where the unsoundness occurs
    let f: for<'x> fn(&'static [&'static T], &'x K) -> &'b K = lifetime_translator;
    f(var, x)
}

/// Same as extend but for mutable references
fn extend_mut<'a, 'b, T: ?Sized, K: ?Sized>(x: &'a mut K, var: &'static [&'static T]) -> &'b mut K {
    #[inline(never)]
    fn lifetime_translator_mut<'a, 'b, T: ?Sized, K: ?Sized>(
        _anchor: &'a [&'b T],
        val: &'b mut K,
    ) -> &'a mut K {
        val
    }
    let f: for<'x> fn(&'static [&'static T], &'x mut K) -> &'b mut K = lifetime_translator_mut;
    f(var, x)
}

/// Static variants - simulating what deserialize_enum receives
const VARIANTS: &[&str] = &["Variant1", "Variant2", "Variant3"];

/// Safe transmute using lifetime expansion + enum layout abuse
fn transmute<A, B>(obj: A, var: &'static [&'static str]) -> B {
    use std::hint::black_box;

    #[allow(dead_code)]
    enum DummyEnum<A, B> {
        A(Option<Box<A>>),
        B(Option<Box<B>>),
    }

    #[inline(never)]
    fn transmute_inner<A, B>(
        dummy: &mut DummyEnum<A, B>,
        obj: A,
        var: &'static [&'static str],
    ) -> B {
        let DummyEnum::B(ref_to_b) = dummy else {
            unreachable!()
        };
        // Extend the lifetime of ref_to_b beyond the enum mutation
        // Using var (the variants slice) as our anchor!
        let ref_to_b = extend_mut(ref_to_b, var);
        // Overwrite the enum - ref_to_b now points to the new data
        *dummy = DummyEnum::A(Some(Box::new(obj)));
        black_box(dummy);
        // Extract the data through the dangling reference
        *ref_to_b.take().unwrap()
    }

    transmute_inner(black_box(&mut DummyEnum::B(None)), obj, var)
}

/// Create a null mutable reference - completely "safe" code
fn null_mut<'a, T: 'static>(var: &'static [&'static str]) -> &'a mut T {
    transmute(0usize, var)
}

/// Trigger a segmentation fault by dereferencing null
fn segfault(var: &'static [&'static str]) -> ! {
    let null: &mut u8 = null_mut(var);
    *null = 42; // SEGFAULT: writing to address 0x0
    unreachable!("If you see this, null deref didn't crash")
}

/// Demonstrate use-after-free with the extend function
fn demonstrate_use_after_free(var: &'static [&'static str]) {
    println!("=== Use-After-Free Demonstration ===\n");

    let dangling: &String = {
        let local = String::from("Hello from stack!");
        println!("Local string created: {:?} at {:p}", local, &local);
        // extend() gives us a 'static reference to stack-allocated data
        // Using var as our anchor - this is the key!
        extend(&local, var)
        // local is dropped here, but we still have a reference!
    };

    println!("After scope ends, dangling ref points to: {:p}", dangling);
    println!("Attempting to read freed memory...");

    // This is UB - might print garbage, might crash, might "work"
    println!("Dangling value: {:?}", dangling);
    println!();
}

// This function simulates deserialize_enum - the var parameter enables the exploit
//    fn deserialize_enum<V>(
//        self,
//        _name: &str,
//        _variants: &'static [&'static str],  // <-- This is our anchor!
//        visitor: V,
//    ) -> Result<V::Value, Error>

/// This function demonstrates how a deserialize_enum-like signature
/// can be exploited. The `var` parameter provides the static anchor
/// needed for lifetime expansion.
fn fun_things_can_happen(_name: &str, var: &'static [&'static str]) {
    println!("=== fun_things_can_happen ===");
    println!("Using variants {:?} as lifetime anchor\n", var);

    // Use the var parameter to trigger the vulnerability
    demonstrate_use_after_free(var);

    println!("=== Triggering Segmentation Fault ===");
    println!("Creating null reference using variants as anchor...\n");

    segfault(var);
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  Rust Unsoundness PoC - Issue #25860                     ║");
    println!("║  Using &'static[&'static str] as lifetime anchor         ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // Call with static variants - simulating deserialize_enum pattern
    fun_things_can_happen("TestEnum", VARIANTS);
}
