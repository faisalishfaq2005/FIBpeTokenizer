//! A BPE tokenizer library for Rust
//! 
//! This library provides Byte Pair Encoding (BPE) tokenization functionality which is blazing fast
//! with support for various pre-tokenization strategies and special token handling.


pub mod bpe_tokenizer;
pub mod pretokenizer;
pub mod tokenizer_tables;


pub use bpe_tokenizer::{BpeTokenizer,Encoder,SpecialTokenRemovalMethod};
pub use pretokenizer::PreTokenization;
