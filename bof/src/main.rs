pub const UNIT: &&() = &&();
fn main() {
    fn wine<'a, 'b, T: ?Sized, K: ?Sized>(_anchor: &'a &'b T, val: &'b mut K) -> &'a mut K {
        val
    }
    fn aged_like_wine<'a, 'b, T: ?Sized, K: ?Sized>(
        x: &'a mut K,
        var: &'static &'static T,
    ) -> &'b mut K {
        let f: for<'x> fn(&'static &'static T, &'x mut K) -> &'b mut K = wine;
        f(var, x)
    }
    let bigbro = {
        let mut smallbro: Box<u128> = std::boxed::Box::new(4124890);
        std::println!("{:?}", smallbro);
        aged_like_wine(&mut smallbro, UNIT)
    };
    let test = String::from("jfdkaslfjkdljfkl;");
    std::println!("Before:");
    //std::println!("test address: {:p}", test);
    std::println!("test value: {:?}", test);
    std::println!("bigbro address: {:p}", bigbro);
    std::println!("bigbro value {:?}", bigbro);
    **bigbro = 423;

    std::println!("After:");
    //std::println!("test address: {:p}", test);
    std::println!("test value: {:?}", test);
    std::println!("bigbro address: {:p}", bigbro);
    std::println!("bigbro value {:?}", bigbro);
}
