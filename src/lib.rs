//! # FIBpeTokenizer
//! 
//! A blazing fast Byte Pair Encoding (BPE) tokenizer library for Rust with Python bindings.
//! 
//! This library provides efficient BPE tokenization functionality with support for:
//! - Various pre-tokenization strategies (whitespace, punctuation)
//! - Special token handling
//! - Training from scratch or loading pretrained models
//! - Encoding and decoding text
//! - Python bindings via PyO3
//! 
//! ## Example
//! 
//! ```rust
//! use fibpetokenizer::{BpeTokenizer, PreTokenization, SpecialTokenRemovalMethod};
//! 
//! let special_tokens = vec!["<pad>".to_string(), "<mask>".to_string()];
//! let mut tokenizer = BpeTokenizer::new(
//!     "corpus.txt",
//!     10000,
//!     PreTokenization::Whitespace,
//!     special_tokens,
//!     SpecialTokenRemovalMethod::Simple,
//!     true,
//!     Some("output_dir")
//! );
//! 
//! // Train the tokenizer
//! tokenizer.train().unwrap();
//! 
//! // Encode text
//! let encoder = tokenizer.encode("Hello world!").unwrap();
//! println!("Tokens: {:?}", encoder.tokens);
//! println!("IDs: {:?}", encoder.ids);
//! 
//! // Decode back
//! let decoded = tokenizer.decode(&encoder.ids).unwrap();
//! println!("Decoded: {}", decoded);
//! ```

pub mod bpe_tokenizer;
pub mod pretokenizer;
pub mod tokenizer_tables;

#[cfg(feature = "python")]
pub mod python_wrapper;

pub use bpe_tokenizer::{BpeTokenizer, Encoder, SpecialTokenRemovalMethod, TokenType};
pub use pretokenizer::PreTokenization;
