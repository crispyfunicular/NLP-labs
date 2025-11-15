# NLP Lab - Catalan Language Processing

A command-line toolkit for Catalan natural language processing experiments using state-of-the-art transformer models.

## Features

- **Mask Filling**: Fill masked tokens in Catalan text using RoBERTa
- **POS Tagging**: Part-of-speech tagging for Catalan text
- **Named Entity Recognition (NER)**: Identify persons, locations, and organizations in Catalan text
- **Machine Translation**: Translation to and from Catalan
- **Interactive Mode**: Real-time experimentation with pre-loaded models
- **Optimized Performance**: Efficient model loading and reuse

## Models Used

- **Mask Filling**: `PlanTL-GOB-ES/roberta-base-ca` - Official Catalan RoBERTa model
- **POS Tagging**: `projecte-aina/roberta-base-ca-cased-pos` - Catalan POS tagging model
- **NER**: `projecte-aina/roberta-base-ca-v2-cased-ner` - Catalan NER model
- **TRAD**: `projecte-aina/aina-translator-xx-xx` - Catalan TRAD model

## Installation

0. Python3.11

Some libaries (pyonmttok) require Python3.11 
To install pyenv to manage multiple Python versions:
```
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev \
liblzma-dev python3-openssl

curl https://pyenv.run | bash
```

Add this to ~/.bashrc and reload the terminal
```
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
```

Install Python 3.11 with pyenv
```
pyenv install 3.11
```

Set Python 3.11 for the current repository only
```
pyenv local 3.11
python --version
```

1. Clone the repository:
```bash
git clone <repository-url>
cd nlp-lab
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Navigate to the catalan directory:
```bash
cd catalan
```

### Mask Filling

Fill masked tokens in Catalan text:

```bash
# Default example
python main.py mask

# Custom text
python main.py mask --text "El meu <mask> favorit és el blau"

# Interactive mode
python main.py mask -i
```

**Example Output:**
```
Model loaded! Filling mask for: La llengua catalana és <mask> bonica.
  La llengua catalana és molt bonica.  (score=0.8382)
  La llengua catalana és tan bonica.  (score=0.0718)
  La llengua catalana és ben bonica.  (score=0.0290)
```

### POS Tagging

Analyze part-of-speech tags in Catalan text:

```bash
# Default example
python main.py pos

# Custom text
python main.py pos --text "Els estudiants treballen molt"

# Interactive mode
python main.py pos -i
```

**Example Output:**
```
Model loaded! POS tagging for: El gat negre dorm tranquil·lament.
 El             → DET (score=1.00)
 gat            → NOUN (score=1.00)
 negre          → ADJ (score=0.99)
 dorm           → VERB (score=1.00)
```

### Named Entity Recognition

Identify named entities in Catalan text:

```bash
# Default example
python main.py ner

# Custom text
python main.py ner --text "Antoni Gaudí va dissenyar la Sagrada Família"

# Interactive mode
python main.py ner -i
```

**Example Output:**
```
Model loaded! NER analysis for: Joan Miró va néixer a Barcelona el 1893.
 Joan Miró           → PER        (score=0.97) [0-9]
 Barcelona           → LOC        (score=0.96) [22-31]
```

### Translation
```bash
# Default example
python main.py trad

# Custom text
python main.py trad --text "Bienvenue en Catalogne !"

# Interactive mode
python main.py ner -i

# Language combination
python main.py ner --combination "es-ca" --text "Bienvenidos a Cataluña!"
```

## Entity Types

### POS Tags
- **DET**: Determiner
- **NOUN**: Noun
- **VERB**: Verb
- **ADJ**: Adjective
- **ADV**: Adverb
- **AUX**: Auxiliary verb
- **CCONJ**: Coordinating conjunction
- **ADP**: Adposition

### NER Labels
- **PER**: Person
- **LOC**: Location
- **ORG**: Organization

## Interactive Mode

All experiments support interactive mode with efficient model reuse:

```bash
python main.py <command> -i
```

In interactive mode:
- Model loads once at startup
- Enter text for real-time analysis
- Type `quit`, `exit`, or `q` to exit

## Help

Get help for any command:

```bash
python main.py --help
python main.py mask --help
python main.py pos --help
python main.py ner --help
```

## System Requirements

- Python 3.8 or higher
- 4GB+ RAM (models are loaded in CPU mode)
- Internet connection for initial model downloads

## Architecture

The toolkit follows a modular design:

```
catalan/
├── main.py         # CLI entry point with Click commands
├── mask.py         # Mask filling functionality
├── pos.py          # POS tagging functionality
├── ner.py          # NER functionality
└── __pycache__/    # Python cache files
```

Each module (`mask.py`, `pos.py`, `ner.py`) follows the same pattern:
- `load_model()`: Model initialization
- `<task>_<action>()`: Core processing function
- `format_<task>_output()`: Output formatting
- `main()`: Command orchestration

## Performance Notes

- Models run on CPU for compatibility
- First run downloads models (~500MB each)
- Subsequent runs use cached models
- Interactive mode reuses loaded models for efficiency

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license here]

## Acknowledgments

- [PlanTL-GOB-ES](https://huggingface.co/PlanTL-GOB-ES) for the Catalan RoBERTa model
- [Projecte Aina](https://huggingface.co/projecte-aina) for the POS and NER models
- [Hugging Face](https://huggingface.co/) for the Transformers library