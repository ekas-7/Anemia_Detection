# Contributing to Anemia Detection System

Thank you for your interest in contributing to this project! 🎉

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version, hardware)
- Error messages and logs

### Suggesting Features

Feature requests are welcome! Please include:
- Clear use case description
- Expected behavior
- Why this feature would be useful
- Possible implementation approaches (optional)

### Pull Requests

1. **Fork** the repository
2. **Clone** your fork: `git clone https://github.com/your-username/Anemia_Detection.git`
3. **Create** a branch: `git checkout -b feature/your-feature-name`
4. **Make** your changes
5. **Test** thoroughly
6. **Commit** with clear messages: `git commit -m "Add: feature description"`
7. **Push**: `git push origin feature/your-feature-name`
8. **Open** a Pull Request

### Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy

# Run tests
pytest

# Format code
black .

# Lint code
flake8 .
```

## Code Style

- Follow [PEP 8](https://pep8.org/) style guide
- Use meaningful variable and function names
- Add type hints for function parameters and returns
- Write docstrings for all public functions and classes
- Keep functions focused and single-purpose
- Maximum line length: 100 characters

### Example Function

```python
def classify_image(
    image_path: str,
    use_rag: bool = True,
    n_similar: int = 5
) -> Dict[str, Any]:
    """
    Classify an image for anemia detection.
    
    Args:
        image_path: Path to the conjunctiva image
        use_rag: Whether to use RAG enhancement
        n_similar: Number of similar cases to retrieve
        
    Returns:
        Dictionary containing classification results with keys:
        - anemia_classification: 'anemic' or 'non-anemic'
        - confidence_score: float between 0 and 1
        - key_observations: list of observations
        
    Raises:
        FileNotFoundError: If image_path doesn't exist
        ValueError: If n_similar < 1
    """
    # Implementation here
    pass
```

## Testing

- Write unit tests for new features
- Ensure all tests pass before submitting PR
- Aim for >80% code coverage
- Test edge cases and error conditions

```python
# Example test
def test_classify_image():
    pipeline = AnemiaRAGPipeline("./test_data")
    result = pipeline.classify_image("test_image.jpg")
    assert "anemia_classification" in result
    assert result["confidence_score"] >= 0.0
    assert result["confidence_score"] <= 1.0
```

## Documentation

- Update README.md if adding new features
- Add inline comments for complex logic
- Update API reference for new functions
- Include usage examples

## Commit Message Guidelines

Use clear, descriptive commit messages:

- `Add: new feature description`
- `Fix: bug description`
- `Update: what was updated`
- `Refactor: what was refactored`
- `Docs: documentation changes`
- `Test: test additions or changes`

## Areas We'd Love Help With

### High Priority
- 🐛 Bug fixes and stability improvements
- 📚 Documentation and examples
- 🧪 Test coverage expansion
- ⚡ Performance optimizations
- 🌐 Multi-language support

### Medium Priority
- 📱 Mobile interface development
- 🎨 UI/UX improvements
- 📊 Analytics and visualization
- 🔧 Configuration management
- 🎯 Model fine-tuning

### Low Priority (but still welcome!)
- 🌍 Internationalization
- 🎨 Themes and styling
- 📖 Tutorials and guides
- 🔌 Plugin system
- 🚀 Deployment tools

## Code Review Process

All submissions require review:

1. Automated checks must pass (linting, tests)
2. Code must follow style guidelines
3. Changes must be well-documented
4. At least one maintainer approval needed
5. All discussions must be resolved

## Community Guidelines

- Be respectful and inclusive
- Provide constructive feedback
- Help others learn and grow
- Celebrate contributions, big and small
- Follow the [Code of Conduct](CODE_OF_CONDUCT.md)

## Questions?

- Open an issue with the `question` label
- Start a discussion in GitHub Discussions
- Check existing issues and documentation first

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT License).

---

Thank you for contributing to making healthcare more accessible through AI! 🙏
