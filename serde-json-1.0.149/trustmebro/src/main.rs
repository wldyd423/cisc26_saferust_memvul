use serde::Deserialize;
use serde_json;
#[derive(Deserialize, Debug)]
struct Person {
    name: String,
    age: i32,
}

fn main() {
    let json_str = r#"{"name": "test", "age": 42}"#;
    let john: Person = serde_json::from_str(json_str).unwrap();

    println!("Parsed JSON: {:#?}", john);
}
