# Contributing to TPT GPU

Thank you for your interest in contributing to TPT GPU! We welcome contributions from the community.

---

## 📋 Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [How to Contribute](#how-to-contribute)
4. [Development Setup](#development-setup)
5. [Coding Standards](#coding-standards)
6. [Testing](#testing)
7. [Pull Request Process](#pull-request-process)

---

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

---

## Getting Started

### Prerequisites

- **Rust toolchain ≥ 1.75** — Install via [rustup](https://rustup.rs/)
- **Git** — For version control
- **Cargo** — Rust's package manager (included with rustup)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/tpt-gpu.git
   cd tpt-gpu
   ```
3. Add the upstream remote:
   ```bash
   git remote add upstream https://github.com/tpt-gpu/tpt-gpu.git
   ```

### Branch Naming

- `feature/description` — New features
- `bugfix/description` — Bug fixes
- `docs/description` — Documentation updates
- `refactor/description` — Code refactoring

---

## How to Contribute

### Reporting Bugs

When reporting a bug, include:
1. Clear title and description
2. Steps to reproduce
3. Expected vs actual behavior
4. Environment details (OS, Rust version)
5. Code samples or error messages

### Suggesting Enhancements

Provide:
1. Clear use case
2. Proposed solution
3. Alternatives considered
4. Additional context

### Contributing Code

1. Find an issue to work on
2. Comment on the issue to claim it
3. Create a branch for your changes
4. Make changes following coding standards
5. Write tests for new functionality
6. Update documentation
7. Submit a pull request

---

## Development Setup

### Building

```bash
# Build all Rust layers
cargo build --release

# Build specific components
cd layer7_tptb
cargo build --release -p tptb-cli
cargo build --release -p tptb-lsp
cargo build --release -p tptb-format
```

### Running Tests

```bash
# Run all tests
cargo test --workspace

# Run tests for specific package
cargo test -p tptb-core

# Run specific test
cargo test -p tptb-core -- test_name
```

### Development Tools

```bash
# Format code
cargo fmt

# Run linter
cargo clippy

# Check code
cargo check

# Generate docs
cargo doc --open
```

---

## Coding Standards

### Rust Code

1. Follow Rust idioms and API Guidelines
2. Use `cargo fmt` for formatting
3. Use `cargo clippy` for linting
4. Document all public APIs with doc comments
5. Write meaningful commit messages
6. Keep functions small and focused
7. Handle errors properly with `Result` and `Option`
8. Avoid unsafe code unless necessary

### Naming Conventions

- **Types/traits:** `PascalCase` (e.g., `Tensor`, `KernelConfig`)
- **Functions/methods:** `snake_case` (e.g., `compile_str`, `type_check`)
- **Variables:** `snake_case` (e.g., `input_tensor`)
- **Constants:** `SCREAMING_SNAKE_CASE` (e.g., `MAX_ITERATIONS`)

### Documentation

- All public items must have doc comments (`///`)
- Include examples in doc comments
- Keep documentation up-to-date
- Use markdown formatting in doc comments

---

## Testing

### Test Organization

- **Unit tests:** In the same file as the code
- **Integration tests:** In `tests/` directory
- **Documentation tests:** In doc comments
- **Benchmarks:** In `benches/` directory

### Writing Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_function_name() {
        // Arrange
        let input = create_test_input();

        // Act
        let result = function_under_test(input);

        // Assert
        assert_eq!(result, expected_value);
    }
}
```

### Test Coverage

- Aim for high test coverage on critical paths
- Test edge cases and error conditions
- Run tests before submitting PRs

---

## Pull Request Process

### Before Submitting

1. Update your branch with upstream:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. Run all tests:
   ```bash
   cargo test --workspace
   ```

3. Format code:
   ```bash
   cargo fmt
   ```

4. Run linter:
   ```bash
   cargo clippy -- -D warnings
   ```

5. Update documentation

### Submitting the PR

1. Push your branch:
   ```bash
   git push origin feature/your-feature
   ```

2. Create a pull request with:
   - Clear title
   - Detailed description
   - Reference to related issues
   - Examples if applicable

### PR Review Process

1. Automated checks must pass
2. Code review by at least one maintainer
3. Address feedback promptly
4. Approval from required reviewers
5. Merge by maintainers

### PR Guidelines

- Keep PRs focused on a single change
- Avoid mixing refactoring with features
- Write descriptive commit messages
- Be respectful and professional

---

## Questions?

If you have questions:

1. Check the [documentation](docs/user-guide.md)
2. Search [existing issues](https://github.com/tpt-gpu/tpt-gpu/issues)
3. Ask in [GitHub Discussions](https://github.com/tpt-gpu/tpt-gpu/discussions)

---

## Thank You!

Thank you for contributing to TPT GPU!