# Text Classification with Hugging Face

A simple sentiment analysis project using pre-trained models from Hugging Face Transformers to classify text based on sentiment (positive, negative, neutral).

## 📋 Features

- **Sentiment Analysis**: Analyze sentiment using pre-trained RoBERTa model
- **Batch Processing**: Process texts in batches for efficiency
- **Visualization**: Charts showing sentiment distribution and confidence scores
- **Data Export**: Save analysis results in CSV format
- **Easy to Use**: Simple and user-friendly interface

## 🚀 Quick Start

### Prerequisites

Make sure you have Python 3.7+ installed on your system.

### Installation

1. Clone this repository:
```bash
git clone https://github.com/username/text-classification-huggingface.git
cd text-classification-huggingface
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install pandas numpy transformers torch scikit-learn matplotlib seaborn
```

### Running the Program

```bash
python main.py
```

## 📊 Output

The program generates:

1. **Console Output**: Detailed analysis results for each text
2. **Visualization**: Charts showing sentiment distribution and confidence scores (`sentiment_analysis_results.png`)
3. **CSV File**: Complete results in CSV format (`sentiment_results.csv`)

### Sample Console Output:

```
Initializing Text Classifier...
Running sentiment analysis...

Detailed Results:
--------------------------------------------------------------------------------
Text: I love this product! It's amazing!...
Sentiment: POSITIVE (Confidence: 0.943)
--------------------------------------------------------------------------------
Text: This is the worst experience ever....
Sentiment: NEGATIVE (Confidence: 0.891)
--------------------------------------------------------------------------------
```

## 🛠️ Code Structure

### `TextClassifier` Class

Main class for performing sentiment analysis:

- `__init__(model_name)`: Initialize with pre-trained model
- `predict_sentiment(texts)`: Predict sentiment for single text or list
- `batch_predict(texts, batch_size)`: Batch prediction with specified size

### Functions

- `create_sample_data()`: Create sample data for demonstration
- `analyze_results(results)`: Analyze and visualize results
- `main()`: Main function that runs the pipeline

## 🎯 Custom Usage

### Using Your Own Text

```python
from main import TextClassifier

# Initialize classifier
classifier = TextClassifier()

# Your custom texts
my_texts = [
    "I'm really happy with this product!",
    "The service was disappointing.",
    "It's okay, nothing special."
]

# Predict sentiment
results = classifier.predict_sentiment(my_texts)

# View results
for result in results:
    print(f"Text: {result['text']}")
    print(f"Sentiment: {result['label']} ({result['confidence']:.3f})")
```

### Using Different Models

```python
# Use a different model
classifier = TextClassifier("nlptown/bert-base-multilingual-uncased-sentiment")
```

## 📈 Model Used

Uses `cardiffnlp/twitter-roberta-base-sentiment-latest` by default, which:

- Trained on Twitter data
- Supports 3 labels: NEGATIVE, NEUTRAL, POSITIVE
- High accuracy for informal English text

## 🔧 Customization

### Changing Batch Size

```python
results = classifier.batch_predict(texts, batch_size=32)  # Default: 16
```

### Adding Preprocessing

```python
def preprocess_text(text):
    # Custom preprocessing
    text = text.lower().strip()
    # Add other preprocessing as needed
    return text

# Use before prediction
preprocessed_texts = [preprocess_text(text) for text in texts]
results = classifier.predict_sentiment(preprocessed_texts)
```

## 📋 Requirements

- pandas >= 1.3.0
- numpy >= 1.21.0
- transformers >= 4.21.0
- torch >= 1.12.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0

## 🐛 Troubleshooting

### ModuleNotFoundError
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### CUDA/GPU Issues
If experiencing CUDA issues, install CPU-only version:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Memory Issues
For very large text datasets, reduce batch size:
```python
results = classifier.batch_predict(texts, batch_size=8)
```

## 📝 TODO

- [ ] Support for Indonesian language
- [ ] Custom model fine-tuning
- [ ] Support for other input formats (PDF, DOCX)
- [ ] Web interface with Streamlit/Flask
- [ ] API endpoint for deployment

## 🤝 Contributing

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Create a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 👏 Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/) for the amazing library
- [Cardiff NLP](https://huggingface.co/cardiffnlp) for the pre-trained model
- The open source community supporting ML development

By AhmadSP 💌
