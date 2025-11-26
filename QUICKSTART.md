# Quick Start Guide

Get up and running with the Anemia Detection System in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- 4GB+ RAM
- Camera (optional, for live detection)

## Installation

### 1. Install Ollama

**macOS/Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download from [https://ollama.ai/download](https://ollama.ai/download)

### 2. Clone Repository

```bash
git clone <repository-url>
cd Anemia_Detection
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Vision Model

```bash
ollama pull llava
```

### 5. Start Ollama Service

```bash
ollama serve
```

Leave this terminal running and open a new one for the next steps.

## Usage Examples

### Example 1: Classify a Single Image

```python
from anemia_rag_pipeline import AnemiaRAGPipeline

# Initialize pipeline
pipeline = AnemiaRAGPipeline(
    dataset_path="./dataset anemia",
    ollama_host="http://localhost:11434"
)

# Classify image
result = pipeline.classify_image("path/to/eye_image.jpg")

# Print results
print(f"Classification: {result['anemia_classification']}")
print(f"Confidence: {result['confidence_score']:.2%}")
print(f"Observations: {result['key_observations']}")
```

### Example 2: Launch Web Interface

```bash
streamlit run streamlit_app.py
```

Then open your browser to: `http://localhost:8501`

### Example 3: Live Camera Detection

```bash
python launch_live_detector.py
```

Press SPACE to capture and analyze, Q to quit.

### Example 4: Analyze Dataset

```bash
python anemia_analysis.py
```

## Configuration

### For Raspberry Pi

```python
# Optimize for lower memory usage
pipeline = AnemiaRAGPipeline(
    dataset_path="./dataset anemia",
    ollama_host="http://localhost:11434"
)

# Use fewer similar cases
result = pipeline.classify_image("image.jpg", n_similar=2)
```

### For Remote Ollama Server

```python
# Use Raspberry Pi as Ollama server
pipeline = AnemiaRAGPipeline(
    dataset_path="./dataset anemia",
    ollama_host="http://192.168.1.100:11434"  # Pi's IP
)
```

## Troubleshooting

### "Ollama not found" Error

Make sure Ollama is installed and running:
```bash
ollama serve
```

In another terminal:
```bash
ollama list  # Check installed models
ollama pull llava  # Install if missing
```

### Memory Errors

Reduce similar cases or increase swap space:
```python
result = pipeline.classify_image("image.jpg", n_similar=2)
```

### Camera Not Detected

Try different camera indices:
```python
detector.initialize_camera(camera_index=1)  # Try 1, 2, etc.
```

## Next Steps

- 📖 Read the full [README.md](README.md)
- 🧪 Run evaluation: `python -c "from anemia_rag_pipeline import AnemiaRAGPipeline; p = AnemiaRAGPipeline('./dataset anemia'); p.evaluate_on_dataset(10)"`
- 🎥 Try live detection with your camera
- 🔧 Customize prompts in `ollama_classifier.py`
- 📊 View results in `results/` directory

## Common Commands

```bash
# Start web interface
streamlit run streamlit_app.py

# Run dataset analysis
python anemia_analysis.py

# Launch live detector
python launch_live_detector.py

# Raspberry Pi setup
python raspberry_pi_setup.py

# Check Ollama status
ollama list
```

## Sample Output

```json
{
  "anemia_classification": "anemic",
  "confidence_score": 0.85,
  "key_observations": [
    "Notable pallor of conjunctiva",
    "Reduced vascularity",
    "Lighter color than reference"
  ],
  "conjunctiva_color": "pale pink",
  "reasoning": "Conjunctiva shows characteristic pallor..."
}
```

## Getting Help

- 📖 Check [README.md](README.md) for detailed documentation
- 🐛 Report issues on GitHub
- 💬 Ask questions in Discussions
- 📧 Contact maintainers

---

**Ready to start detecting anemia with AI!** 🚀
