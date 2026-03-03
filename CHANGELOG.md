# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-03-03

### Added
- Initial release of FIBpeTokenizer
- Core BPE tokenization algorithm with parallel processing
- Pre-tokenization strategies (Whitespace, Punctuation)
- Special token handling with two removal methods (Simple, AhoCorasick)
- Model save/load functionality
- Encoding and decoding capabilities
- Token type tracking (WORD, SUBWORD, SPECIALTOKEN)
- Python bindings via PyO3
- Comprehensive documentation and examples
- Dual licensing (MIT OR Apache-2.0)

### Features
- Fast training with Rayon parallelization
- Efficient token table with Arc-based string sharing
- JSON serialization for trained models
- Configurable vocabulary size
- Support for special tokens like `<pad>`, `<mask>`, etc.

[Unreleased]: https://github.com/faisalishfaq2005/FIBpeTokenizer/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/faisalishfaq2005/FIBpeTokenizer/releases/tag/v0.1.0
