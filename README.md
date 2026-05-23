## FIBpeTokenizer 🚀

A blazing fast Byte Pair Encoding (BPE) tokenizer library written in Rust with Python bindings.

📦 **13,000+ downloads on PyPI**

## Performance Highlights

To validate efficiency, **FIBpeTokenizer** was benchmarked against **Hugging Face’s tokenizer** across vocabulary sizes from **4,000** to **90,000** tokens.

## Scaling Behavior

**FIBpeTokenizer** shows highly stable **near-linear scaling** due to Rust-level memory control and parallel processing.

At a vocabulary size of **4,000**, training completes in **5.15** seconds. Even when scaled by **22x** to **90,000** tokens, training time only increases to **14.53** seconds, demonstrating strong computational efficiency under load.

⚔️ Comparison vs Hugging Face Tokenizer
Standard range (**4k – 20k vocab**):
Performance is extremely close. At **10,000** vocab, **FIBpeTokenizer** runs in **5.89s** vs **HF’s** **5.62s**, effectively matching mature C++ backend performance.
Large scale (**30k – 90k vocab**):
Hugging Face benefits from highly optimized memory management and shows better sub-linear scaling (**9.32s at 90k**).
**FIBpeTokenizer** reaches **14.53s,** but remains strongly competitive and production-viable, especially considering it is a lightweight Rust-first implementation.

## Visual Benchmarks

<img width="2484" height="1478" alt="Screenshot 2026-05-23 221735" src="https://github.com/user-attachments/assets/599eaa14-7649-4302-bb36-6f4a57cac2e6" />
<img width="3604" height="988" alt="Screenshot 2026-05-23 221822" src="https://github.com/user-attachments/assets/b528f9a5-3cf9-48f6-859e-dae143d1601a" />
<img width="2721" height="1483" alt="Screenshot 2026-05-23 221522" src="https://github.com/user-attachments/assets/277cd90c-6a42-4feb-9bd8-f876389315a0" />


## Features ✨

- 🔥 **Blazing Fast**: Written in Rust with parallel processing support
- 🐍 **Python Support**: Use it in Python via PyO3 bindings
- 🎯 **Flexible Pre-tokenization**: Choose between whitespace or punctuation-based splitting
- 🔖 **Special Token Handling**: Built-in support for special tokens like `<pad>`, `<mask>`, etc.
- 💾 **Save/Load Models**: Train once, reuse anywhere
- 🔧 **Customizable**: Configure vocabulary size, special tokens, and more

## Installation

### Rust

Add this to your `Cargo.toml`:

```toml
[dependencies]
fibpetokenizer = "0.1.0"
```

### Python

```bash
pip install fibpetokenizer
```

## Quick Start

### Rust Usage

```rust
use fibpetokenizer::{BpeTokenizer, PreTokenization, SpecialTokenRemovalMethod};

fn main() {
    // Define special tokens
    let special_tokens = vec![
        "<pad>".to_string(),
        "<mask>".to_string(),
        "<unk>".to_string()
    ];

    // Create and train tokenizer
    let mut tokenizer = BpeTokenizer::new(
        "corpus.txt",                           // Input text file
        10000,                                   // Target vocabulary size
        PreTokenization::Punctuation,           // Pre-tokenization strategy
        special_tokens,                          // Special tokens
        SpecialTokenRemovalMethod::AhoCorasick, // Special token removal method
        true,                                    // Save model after training
        Some("output_dir")                      // Output directory
    );

    // Train the tokenizer
    tokenizer.train().unwrap();

    // Encode text
    let text = "Hello, world! This is a test.";
    let encoder = tokenizer.encode(text).unwrap();
    
    println!("Tokens: {:?}", encoder.tokens);
    println!("Token IDs: {:?}", encoder.ids);
    println!("Token Types: {:?}", encoder.token_types);

    // Decode back to text
    let decoded = tokenizer.decode(&encoder.ids).unwrap();
    println!("Decoded: {}", decoded);

    // Load a pretrained tokenizer
    let loaded_tokenizer = BpeTokenizer::new_from_pretrained("output_dir");
}
```

### Python Usage

```python
from fibpetokenizer import (
    BpeTokenizer,
    PreTokenization,
    SpecialTokenRemovalMethod
)

# Define special tokens
special_tokens = ["<pad>", "<mask>", "<unk>"]

# Create tokenizer
tokenizer = BpeTokenizer(
    input_path="corpus.txt",
    target_vocab_size=10000,
    pretokenization_type=PreTokenization.punctuation(),
    special_tokens=special_tokens,
    special_token_removal_method=SpecialTokenRemovalMethod.aho_corasick(),
    save_model=True,
    output_dir="output_dir"
)

# Train the tokenizer
tokenizer.train()

# Encode text
text = "Hello, world! This is a test."
encoder = tokenizer.encode(text)

print("Tokens:", encoder.tokens)
print("Token IDs:", encoder.ids)
print("Token Types:", encoder.token_types)

# Decode back to text
decoded = tokenizer.decode(encoder.ids)
print("Decoded:", decoded)

# Load a pretrained tokenizer
loaded_tokenizer = BpeTokenizer.from_pretrained("output_dir")
```

## API Reference

### `BpeTokenizer`

The main tokenizer class.

#### Constructor

```rust
BpeTokenizer::new(
    input_path: &str,
    target_vocab_size: usize,
    pretokenization_type: PreTokenization,
    special_tokens: Vec<String>,
    special_token_removal_method: SpecialTokenRemovalMethod,
    save_model: bool,
    output_dir: Option<&str>
) -> Self
```

#### Methods

- **`train(&mut self) -> Result<(), TokenizerError>`**: Train the tokenizer on the corpus
- **`encode(&self, text: &str) -> Result<Encoder, TokenizerError>`**: Encode text into tokens and IDs
- **`decode(&self, ids: &Vec<u32>) -> Result<String, TokenizerError>`**: Decode token IDs back to text
- **`new_from_pretrained(files_path: &str) -> Self`**: Load a pretrained tokenizer
- **`get_id_by_token(&self, token: String) -> Result<u32, TokenizerError>`**: Get ID for a token
- **`get_token_by_id(&self, id: u32) -> Result<String, TokenizerError>`**: Get token for an ID

### `PreTokenization`

Pre-tokenization strategies:
- **`PreTokenization::Whitespace`**: Split on whitespace
- **`PreTokenization::Punctuation`**: Split on whitespace and punctuation

### `SpecialTokenRemovalMethod`

Methods for removing special tokens from the training corpus:
- **`SpecialTokenRemovalMethod::Simple`**: Simple string replacement
- **`SpecialTokenRemovalMethod::AhoCorasick`**: Fast multi-pattern search using Aho-Corasick algorithm

### `Encoder`

The result of encoding text.

#### Fields

- **`original_text: String`**: The original input text
- **`tokens: Vec<String>`**: The tokenized representation
- **`ids: Vec<u32>`**: Token IDs
- **`token_types: Vec<TokenType>`**: Type of each token (WORD, SUBWORD, or SPECIALTOKEN)

#### Methods

- **`get_token_type(&self, token: &str) -> Result<TokenType, TokenizerError>`**: Get the type of a specific token

## How It Works

BPE (Byte Pair Encoding) is a data compression technique that iteratively merges the most frequent pair of bytes (or characters) in a sequence. This library:

1. **Pre-tokenizes** the input text based on the selected strategy
2. **Builds** an initial vocabulary from individual characters
3. **Iteratively merges** the most frequent adjacent token pairs
4. **Stops** when the target vocabulary size is reached
5. **Saves** the vocabulary, merge rules, and configuration for later use

## Performance

- **Parallel processing** using Rayon for fast training
- **Efficient special token removal** using Aho-Corasick algorithm
- **Optimized data structures** for merge operations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.

## Credits

Developed with ❤️ using Rust and PyO3.
