# MatLLMSearch

LLM-assisted crystal structure discovery using evolutionary algorithms.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 📖 Overview

MatLLMSearch implements an evolutionary framework for discovering novel crystal structures using Large Language Models (LLMs). The system combines:

- **Pre-trained LLMs** (GPT-4, Claude) for intelligent structure generation
- **Machine Learning Interatomic Potentials** (CHGNet) for rapid evaluation
- **Evolutionary algorithms** for guided search
- **Agentic architecture** (ACE loop) for strategy optimization

Based on the paper: *Gan et al., "MatLLMSearch: Crystal Structure Discovery with Evolution-Guided Large Language Models" (2025)*

## 🚀 Features

- ✅ **Training-free**: Uses pre-trained LLMs without fine-tuning
- ✅ **Multi-objective**: Optimize stability, bulk modulus, and other properties
- ✅ **Flexible**: Supports CSG, CSP, and property-driven design
- ✅ **Modular**: Easy to extend with new agents and objectives
- ✅ **Scalable**: Parallel evaluation with MLIPs

## 📦 Installation

### Prerequisites

- Python 3.8+
- OpenAI API key or Anthropic API key

### Setup
```bash
# Clone repository
git clone https://github.com/yourusername/MatLLMSearch.git
cd MatLLMSearch

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configure API Keys
```bash
# Set your API key
export OPENAI_API_KEY="your-api-key-here"

# Or create .env file
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

## 🎯 Quick Start

### 1. Run Simple Demo
```bash
python demo/simple_search.py
```

### 2. Test Components
```bash
python demo/test_components.py
```

### 3. Run Full Search
```bash
python src/main.py --config config/config.yaml --data data/initial_structures/
```

## 📁 Project Structure
```
MatLLMSearch/
├── config/              # Configuration files
├── src/
│   ├── core/           # Core data structures and algorithms
│   ├── agents/         # Agent implementations
│   │   └── workers/    # Worker agents (modifier, parser, evaluator)
│   ├── utils/          # Utilities (LLM client, parsers)
│   └── main.py         # Main entry point
├── demo/               # Demo scripts
├── data/               # Data storage
└── tests/              # Unit tests
```

## ⚙️ Configuration

Edit `config/config.yaml` to customize:
```yaml
evolution:
  population_size: 100
  parent_size: 2
  children_size: 5
  max_iterations: 10

llm:
  provider: "openai"
  model: "gpt-4"
  temperature: 0.95

objective:
  type: "stability"  # or "bulk_modulus", "multi_objective"
  target_ed: 0.0
```

## 🔬 Usage Examples

### Crystal Structure Generation (CSG)
```python
from src.main import main

# Run with default config
main()
```

### Crystal Structure Prediction (CSP)

Modify filters in `main.py`:
```python
filters = {
    'composition': 'Na3AlCl6',  # Specific composition
    'num_elements': (3, 3)       # Ternary compounds
}
```

### Multi-Objective Optimization

Set in `config.yaml`:
```yaml
objective:
  type: "multi_objective"
  weights:
    stability: 0.7
    bulk_modulus: 0.3
```

## 📊 Results

Results are saved to `data/results/`:

- `final_*.json` - Generated structures
- `energy_distribution.png` - Energy histogram
- `convergence.png` - Search convergence plots
- `history_*.json` - Iteration statistics

## 🧪 Testing
```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_structure.py -v

# With coverage
pytest tests/ --cov=src
```

## 📚 Citation

If you use MatLLMSearch in your research, please cite:
```bibtex
@article{gan2025matllmsearch,
  title={MatLLMSearch: Crystal Structure Discovery with Evolution-Guided Large Language Models},
  author={Gan, Jingru and others},
  journal={arXiv preprint arXiv:2502.20933},
  year={2025}
}
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🔗 Links

- Paper: [arXiv:2502.20933](https://arxiv.org/abs/2502.20933)
- Documentation: [Wiki](https://github.com/yourusername/MatLLMSearch/wiki)
- Issues: [GitHub Issues](https://github.com/yourusername/MatLLMSearch/issues)

## 📧 Contact

For questions or collaboration:
- Open an issue on GitHub
- Email: your.email@example.com

## 🙏 Acknowledgments

- CHGNet team for the MLIP
- Materials Project for structure database
- OpenAI/Anthropic for LLM APIs