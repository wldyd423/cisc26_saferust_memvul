
const UNIT: &'static &'static () = &&();

#[inline(never)]
fn swap_mut<'a, 'b, T>(_anchor: &'a &'b (), x: &'b mut T) -> &'a mut T {
    x
}

fn extend_mut<'a, 'b, T>(x: &'a mut T) -> &'b mut T {
    let f: for<'c> fn(_, &'c mut T) -> &'b mut T = swap_mut;
    f(UNIT, x)
}
fn main() {
    let mut dangling = {
        let mut inner = String::from("fjdksalfjakdl");
        extend_mut(&mut inner)
    };
    println!("Creating of dangling pointer:");
    println!("{:p}    dangling: {}", dangling.as_ptr(), dangling);

    println!("\n\nCreation of test on same memory:");
    let test = String::from("fjdksalfjakdl");
    println!("{:p}    test: {}", test.as_ptr(), test);
    println!("{:p}    dangling: {}", dangling.as_ptr(), dangling);

    println!("\n\nAfter changing value pointed to by dangling pointer:");
    *dangling = "Thi".to_string();
    println!("{:p}    dangling: {}", dangling.as_ptr(), dangling);
    println!("{:p}    test: {}", test.as_ptr(), test);
}
