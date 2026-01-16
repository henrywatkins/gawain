# Contributing to Gawain

Thank you for your interest in contributing to Gawain! This document provides guidelines for contributing to the project.

## Getting Started

### Setting Up Development Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/henrywatkins/gawain.git
   cd gawain
   ```

2. Install in development mode with all dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

3. For GPU development, additionally install:
   ```bash
   pip install -e ".[gpu]"
   ```

## Development Workflow

### Code Style

Gawain uses standard Python code formatting tools:

- **Black** for code formatting (line length: 88 characters)
- **isort** for import sorting
- **Type hints** where appropriate

Format your code before committing:
```bash
black src/gawain/
isort src/gawain/
```

### Running Tests

Run the test suite before submitting changes:

```bash
# Run all tests
pytest src/gawain/tests/

# Run specific test file
pytest src/gawain/tests/test_validation.py

# Run with verbose output
pytest -v src/gawain/tests/

# Run with coverage report
pytest --cov=src/gawain src/gawain/tests/
```

All tests must pass before a pull request can be merged.

### Testing Your Changes

For physics code changes (fluxes, numerics, integrators):
1. Run relevant example scripts from `examples/` directory
2. Verify results against known solutions
3. Check that existing validation cases still pass
4. Add new test cases if implementing new features

For configuration/validation changes:
1. Run the full test suite
2. Test with various configuration combinations
3. Ensure error messages are clear and helpful

## Types of Contributions

### Bug Reports

When reporting bugs, please include:
- Clear description of the problem
- Minimal code example that reproduces the issue
- Expected vs. actual behavior
- Python version and relevant package versions
- Operating system

### Feature Requests

For feature requests, please describe:
- The problem you're trying to solve
- Proposed solution or implementation approach
- Whether you're willing to contribute the implementation
- Any relevant physics or numerical methods references

### Code Contributions

#### Pull Request Process

1. **Create a feature branch** from `master`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the code style guidelines

3. **Add tests** for new functionality:
   - Unit tests in `src/gawain/tests/`
   - Integration tests using example problems if appropriate
   - Validation tests against analytical solutions when available

4. **Update documentation**:
   - Add docstrings to new functions/classes
   - Update README.md if adding user-facing features
   - Add example scripts if appropriate

5. **Run tests and formatting**:
   ```bash
   black src/gawain/
   isort src/gawain/
   pytest src/gawain/tests/
   ```

6. **Commit your changes** with clear, descriptive commit messages:
   ```bash
   git commit -m "Add feature: brief description"
   ```

7. **Push to your fork** and create a pull request

8. **Respond to review feedback** - maintainers will review your PR and may request changes

#### Commit Message Guidelines

- Use present tense ("Add feature" not "Added feature")
- Keep first line under 50 characters
- Provide detailed description in the body if needed
- Reference issue numbers when applicable

Example:
```
Add HLL flux solver for MHD

Implements the Harten-Lax-van Leer approximate Riemann solver
for magnetohydrodynamics. Includes validation against Brio-Wu
shock tube test case.

Closes #123
```

## Code Architecture

Understanding the code structure will help with contributions:

- **`src/gawain/main.py`**: Entry point - `run_gawain(config)` function
- **`src/gawain/config.py`**: Pydantic validation models
- **`src/gawain/io.py`**: Parameters class and I/O utilities
- **`src/gawain/numerics.py`**: SolutionVector classes (state representation)
- **`src/gawain/fluxes.py`**: Flux calculator implementations
- **`src/gawain/integrators.py`**: Time integration schemes
- **`src/gawain/backend.py`**: NumPy/CuPy abstraction layer

### Adding New Features

#### New Flux Scheme

1. Create subclass of `FluxCalculator` in `fluxes.py`
2. Implement `calculate_flux_divergence()` method
3. Add enum entry to `FluxerType` in `config.py`
4. Add validation test case
5. Document in README.md

#### New Boundary Condition

1. Extend boundary condition logic in `numerics.py`
2. Add enum entry to `BoundaryType` in `config.py`
3. Add validation tests
4. Document behavior

#### GPU Backend Code

When modifying code that uses the backend abstraction:
- Always use `xp` (from `backend.py`) instead of `numpy`
- Test both CPU and GPU paths when possible
- Use `xp.array()`, `xp.zeros()`, etc. for array creation
- Avoid NumPy-specific functions not available in CuPy

## Documentation

Good documentation helps users and future contributors:

- **Docstrings**: Use NumPy-style docstrings for all public functions/classes
- **Type hints**: Add type hints to function signatures
- **Comments**: Explain *why*, not *what* - code should be self-documenting
- **Examples**: Provide runnable examples for new features

## Questions or Need Help?

- **Issues**: Open an issue on GitHub for questions
- **Discussions**: Use GitHub Discussions for broader topics
- **Email**: Contact the maintainer for sensitive matters

## Code of Conduct

This project adheres to a code of conduct of respectful collaboration:
- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive criticism
- Assume good intentions

## License

By contributing to Gawain, you agree that your contributions will be licensed under the Apache License 2.0, the same license as the project.
