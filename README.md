# Sentence Completion

Next-word prediction using recurrent neural networks (SimpleRNN and LSTM) trained on a quote dataset. Enter a short seed phrase and the model suggests the next words one at a time.

---

## Project Description

This project trains neural language models to **predict the next word** given previous words in a sentence.

It demonstrates:

- **Sentence completion** – e.g. *"life is"* → *"life is a journey..."*
- **Text suggestion** – autocomplete-style next-word prediction
- **Deep learning concepts** – comparison of SimpleRNN and LSTM architectures

---

## Pipeline Overview

```
Raw quotes → preprocessing → tokenization → sequence generation → padding → model training → inference
```

### Steps

1. Convert text to lowercase and remove punctuation
2. Tokenize using top 10,000 words
3. Generate (input sequence → next word) training pairs
4. Pad sequences to fixed length
5. Train models (Embedding + RNN/LSTM + Dense softmax)
6. Save trained model and artifacts
7. Generate text iteratively at inference

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Trained Artifacts

Place the following files in the project root:

| File             | Description                         |
|------------------|-------------------------------------|
| `lstm_model.h5`  | Trained LSTM model                  |
| `rnn_model.h5`   | Trained SimpleRNN model             |
| `tokenizer.pkl`  | Tokenizer (word → index)            |
| `encoder.pkl`    | OneHotEncoder                       |
| `config.pkl`     | max_length, vocab_size, index_to_word |

---

## Run the Project

```bash
python main.py
```

**Example input:** `life is`

**Output:** `life is a beautiful journey that never truly ends...`

---

## Technical Details

### Dataset

- **Source:** `qoute_dataset.csv`
- ~3,038 quotes used for training

### Tokenization

- Vocabulary size: 10,000 words
- Word index starts from 1 (0 reserved for padding)

### Training Setup

- ~85,000+ training samples generated
- Padding: pre (important for sequence learning)
- Loss: categorical cross-entropy
- Optimizer: Adam

### Model Architectures

#### SimpleRNN

- Embedding (10000 → 50)
- SimpleRNN (128 units)
- Dense (softmax output)

#### LSTM (Used in App)

- Embedding (10000 → 50)
- LSTM (128 units)
- Dense (softmax output)

#### Key Difference

- **SimpleRNN:** Faster, weaker memory
- **LSTM:** Better handling of long-term dependencies

---

## Model Performance & Interpretation

| Metric          | SimpleRNN | LSTM     |
|-----------------|-----------|----------|
| Train Accuracy  | ~15%      | ~12%     |
| Val Accuracy    | ~11%      | ~10%     |
| Train Loss      | ↓ steady  | ↓ steady |
| Val Loss        | plateau   | plateau  |

**Interpretation:**

- The model predicts the next word from a 10,000-word vocabulary.
- Random guessing accuracy ≈ 0.01%.
- Achieved accuracy (~10–15%) shows the model has learned:
  - Meaningful word patterns
  - Phrase structures
  - Contextual relationships
- Validation loss flattening indicates mild overfitting, handled using early stopping.

Overall, the model demonstrates successful learning and correct implementation.

---

## Learning Outcomes

This project focuses on understanding and implementing NLP pipelines, not just maximizing accuracy.

**Key takeaways:**

- Built a complete sequence modeling pipeline
- Learned text preprocessing and tokenization
- Understood RNN vs LSTM behavior
- Trained deep learning models with embeddings
- Interpreted NLP metrics beyond raw accuracy
- Implemented real-time text generation
- Deployed a working CLI-based application

---

## Limitations

- Small dataset limits generalization
- Accuracy is constrained by large vocabulary size
- Only predicts one “most likely” word (argmax)
- No pretrained embeddings used

---

## Future Improvements

- Use pretrained embeddings (GloVe / Word2Vec)
- Reduce vocabulary size for better precision
- Implement Bidirectional LSTM or GRU
- Add top-k or temperature sampling for better text generation
- Scale dataset for improved performance

---

## Conclusion

This project successfully demonstrates how neural language models are built, trained, and deployed. While not designed for production-level performance, it effectively captures linguistic patterns and provides a strong foundation in deep learning-based text generation.

---

**Author:** Sahil Bhatti
