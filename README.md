# Fake News Detection System with RAG

A comprehensive system for detecting fake news using BERT-based classification enhanced with Retrieval Augmented Generation (RAG) techniques. This system combines deep learning-based text classification with factual verification against retrieved information.

## Features

- BERT-based classifier for initial fake news detection
- RAG enhancement using knowledge base of verified facts
- Natural Language Inference for factual consistency checking
- Vector-based search for efficient information retrieval
- Multiple testing options including prediction inversion
- Interactive command-line interface
- GPU acceleration support

## Requirements

- Python 3.7+
- PyTorch
- Transformers
- Sentence-Transformers
- FAISS
- BeautifulSoup4
- Requests

Install required packages:

```bash
pip install torch transformers sentence-transformers faiss-cpu beautifulsoup4 requests
```

## Project Structure

- `seco.py` - Main training script for the BERT classifier
- `test_model.py` - Simple testing script for the trained model
- `rag_news_detector.py` - Enhanced testing with RAG capabilities

## Usage

### Training the Model

Run `seco.py` to train the BERT-based classifier:

```bash
python seco.py
```

This will:
- Load and preprocess the dataset
- Train the BERT classifier
- Save the best model weights to `c2_new_model_weights.pt`
- Evaluate the model on the test set

### Basic Testing

Use `test_model.py` to test the trained model on headlines:

```bash
python test_model.py
```

Options:
1. Test with headlines
2. Test with full text articles
3. Invert predictions (useful if model has inverted labels)

You can also adjust the temperature parameter to control prediction confidence.

### Enhanced Testing with RAG

For improved accuracy using fact verification, use the RAG detector:

```bash
python rag_news_detector.py
```

Options:
1. Test with base model only
2. Test with RAG enhancement
3. Test with RAG and invert predictions
4. View knowledge base
5. Clear knowledge base

## How RAG Works in This System

1. **Base Prediction**: The BERT classifier provides an initial prediction
2. **Fact Retrieval**: The system retrieves relevant facts from a knowledge base
3. **Factual Consistency**: NLI model evaluates if the facts support or contradict the headline
4. **Combined Prediction**: The final prediction combines the classifier output with factual analysis

## Tips for Best Results

- Use option 3 (invert predictions) if the model consistently classifies true news as fake
- Process headlines individually for more accurate fact retrieval
- Increase temperature values (2-5) to get more nuanced probabilities
- The base model works best on headlines similar to its training data
- RAG enhancement is more useful for headlines where factual verification can be performed

## Limitations

- The system requires internet access for online fact retrieval
- Pre-populated facts are only available for the example headlines
- The base model may show bias toward classifying content as fake
- NLI model entailment scores may be low if the local model isn't properly loaded

## Example

```
Processing: U.S. lawmakers question businessman at 2016 Trump Tower meeting: sources...
Found pre-populated facts for: U.S. lawmakers question businessm...
Fact analysis for: U.S. lawmakers question busine...
Fact: Reuters reported that U.S. lawmakers questioned a businessman in 2018 about a meeting at Trump Tower in 2016. This appears to be legitimate reporting.
Entailment: 0.6827, Contradiction: 0.0421
Base model prediction: Fake
Base probabilities: Fake=95.1%, True=4.9%
Fact analysis: Score=0.6406, Adjustment=0.1797
Adjusted probabilities: Fake=0.1243, True=0.8757
Final prediction: True
``` 