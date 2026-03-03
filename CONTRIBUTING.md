# Contributing to FIBpeTokenizer

Thank you for your interest in contributing to FIBpeTokenizer! This document provides guidelines for contributing to the project.

## Code of Conduct

Be respectful, inclusive, and constructive in all interactions.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue on GitHub with:
- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior vs actual behavior
- Your environment (OS, Rust version, Python version if applicable)
- Any relevant code samples or error messages

### Suggesting Features

Feature requests are welcome! Please open an issue describing:
- The problem you're trying to solve
- Your proposed solution
- Why this feature would be useful to others

### Pull Requests

1. **Fork the repository** and create a branch from `main`
2. **Make your changes** following the coding standards below
3. **Add tests** for any new functionality
4. **Update documentation** including doc comments and README if needed
5. **Run tests and checks**:
   ```bash
   cargo test
   cargo clippy
   cargo fmt
   ```
6. **Commit your changes** with clear, descriptive commit messages
7. **Push to your fork** and submit a pull request

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fibpetokenizer.git
   cd fibpetokenizer
   ```

2. Build the project:
   ```bash
   cargo build
   ```

3. Run tests:
   ```bash
   cargo test
   ```

4. For Python development:
   ```bash
   pip install maturin
   maturin develop --features python
   ```

## Coding Standards

### Rust Code

- Follow the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use `cargo fmt` to format code
- Use `cargo clippy` to catch common mistakes
- Add documentation comments (`///`) for all public items
- Write descriptive variable names
- Keep functions focused and reasonably sized

### Python Bindings

- Follow [PEP 8](https://peps.python.org/pep-0008/) style guide
- Provide type hints where applicable
- Document all public API functions

### Documentation

- Use clear, concise language
- Include examples in doc comments
- Update README.md for user-facing changes
- Add inline comments for complex logic

### Testing

- Write unit tests for new functions
- Add integration tests for new features
- Ensure all tests pass before submitting PR
- Aim for good test coverage

## Project Structure

```
fibpetokenizer/
├── src/
│   ├── lib.rs              # Library entry point
│   ├── bpe_tokenizer.rs    # Main tokenizer implementation
│   ├── pretokenizer.rs     # Pre-tokenization strategies
│   ├── tokenizer_tables.rs # Token table data structure
│   ├── python_wrapper.rs   # PyO3 Python bindings
│   └── main.rs             # Binary entry point (examples)
├── examples/               # Usage examples
├── Cargo.toml             # Rust dependencies and metadata
├── pyproject.toml         # Python package configuration
└── README.md              # User documentation
```

## Areas for Contribution

Here are some areas where contributions are especially welcome:

### High Priority
- Additional pre-tokenization strategies
- Performance optimizations
- More comprehensive tests
- Better error handling

### Medium Priority
- Additional examples
- Serialization format improvements
- CLI tool for training tokenizers
- Benchmarking suite

### Documentation
- Tutorial documentation
- API reference improvements
- Translation of documentation

## Questions?

If you have questions about contributing, feel free to:
- Open an issue for discussion
- Reach out to maintainers

## License

By contributing to FIBpeTokenizer, you agree that your contributions will be licensed under the same license as the project (MIT OR Apache-2.0).

---

Thank you for contributing to FIBpeTokenizer! 🎉
