# Contributing to quantum-retrocausality-ai

Thank you for your interest in contributing! This project explores retrocausal
interpretations of quantum mechanics through computational verification.

## How to Contribute

### Reporting Issues
- Use GitHub Issues for bug reports and feature requests
- Include your Python version and OS

### Code Contributions
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Write tests for new functionality
4. Ensure all tests pass: `python -m pytest tests/ -p no:recording -v`
5. Submit a pull request

### Development Setup
```bash
git clone https://github.com/akaiHuang/quantum-retrocausality-ai.git
cd quantum-retrocausality-ai
pip install -e ".[dev]"
python -m pytest tests/ -p no:recording -v
```

### Code Standards
- Follow PEP 8 style guidelines
- Add docstrings to all public functions
- Include unit tests for new physics calculations
- Verify numerical results against known analytical solutions

### Areas of Interest
- Additional retrocausal hidden variable models
- Quantum circuit implementations (Qiskit/Cirq)
- Web-based interactive visualizations
- Educational content and tutorials

## Code of Conduct
Be respectful, constructive, and inclusive. We welcome researchers at all levels.

## License
By contributing, you agree that your contributions will be licensed under the MIT License.
