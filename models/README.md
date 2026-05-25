# Models

An Artificial Intelligence (AI) model is a mathematical program designed to recognize patterns, make decisions, or generate content based on the data it was trained on.

Different types of AI and machine learning models:

AI is the overarching concept of machines mimicking human intelligence.

---

## Table of Contents

- [Machine Learning (ML) Models](#machine-learning-ml-models)
  - [The Development Lifecycle](#the-development-lifecycle)
  - [Coding Algorithms](#coding-algorithms)
- [Large Language Models (LLMs)](#large-language-models-llms)
  - [A Pipeline to Build a GPT-Style Model](#a-pipeline-to-build-a-gpt-style-model)
  - [Stage 1: Data Preparation](#stage-1-data-preparation)
  - [Stage 2: Model Architecture](#stage-2-model-architecture)
  - [Stage 3: Pre-training](#stage-3-pre-training)
  - [Stage 4: Fine-Tuning](#stage-4-fine-tuning)
- [Building a Large Language Model from Scratch](#building-a-large-language-model-from-scratch)
  - [Why Build from Scratch](#why-build-from-scratch)
  - [The Three Implementations](#the-three-implementations)
  - [Key Concepts Implemented](#key-concepts-implemented)
- [Project Structure](#project-structure)
- [Virtual Environment Setup on Linux](#virtual-environment-setup-on-linux)
  - [Create the Virtual Environment](#create-the-virtual-environment)
  - [Activate the Virtual Environment](#activate-the-virtual-environment)
  - [Install Dependencies](#install-dependencies)
  - [Deactivate the Virtual Environment](#deactivate-the-virtual-environment)
  - [Using the Setup Script](#using-the-setup-script)
- [Running the Scripts](#running-the-scripts)
  - [Step 1: Pre-train the Base Model](#step-1-pre-train-the-base-model)
  - [Step 2: Fine-tune the Text Classifier](#step-2-fine-tune-the-text-classifier)
  - [Step 3: Run the Chatbot](#step-3-run-the-chatbot)
- [References](#references)

---

## Machine Learning (ML) Models

Machine Learning is the method of teaching computers to learn from historical data to make predictions.

Machine learning is a subset of AI where systems learn from data and improve over time without being explicitly programmed for a specific task.

Instead of writing hard-coded rules, engineers feed the model data, and the model figures out the rules itself.

ML algorithms analyze massive datasets to identify underlying patterns, which they then use to make predictions or categorize new information.

Examples:

- Classification Models: Sort data into categories, such as an email filter detecting "Spam" vs. "Not Spam."
- Regression Models: Predict continuous values, such as a real estate algorithm estimating a house's price based on square footage and location.

Artificial intelligence (AI) versus machine learning (ML):

Artificial intelligence is a broad field, which refers to the use of technologies to build machines and computers that have the ability to mimic cognitive functions associated with human intelligence, such as being able to understand, and respond to spoken or written language, and more.

Machine learning is a subset of artificial intelligence that automatically enables a machine or system to learn and improve from experience.

For more information, explore the Google Cloud Machine Learning document: https://cloud.google.com/learn/artificial-intelligence-vs-machine-learning

Implementing a machine learning (ML) model from scratch generally follows two paths: building a systemic pipeline using standard libraries like scikit-learn or PyTorch, or coding the core algorithm itself using only basic math and numerical libraries like NumPy.

### The Development Lifecycle

Whether you are using libraries or coding from scratch, the development process follows these standard stages:

- Problem Definition: Identify your goal (e.g., classification, regression, or clustering) and choose your success metrics, such as accuracy or F1-score.
- Data Collection and Preparation: Gather high-quality data from sources like Kaggle or UCI Machine Learning Repository. Clean the data by handling missing values and removing duplicates.
- Exploratory Data Analysis (EDA): Visualize your data using Matplotlib or Seaborn to understand patterns and relationships between features.
- Feature Engineering: Select or create the most relevant features to improve your model's predictive power.
- Data Splitting: Divide your dataset into training and testing sets (typically an 80/20 split) so you can evaluate the model on unseen data.
- Training and Evaluation: Feed the training data into your algorithm and then test its performance against your metrics.
- Deployment and Monitoring: Once satisfied, deploy the model (using tools like FastAPI) and monitor its performance in a live environment.

### Coding Algorithms

To build from the "math up," you bypass high-level libraries and implement the following components yourself using NumPy:

- Weight Initialization: Set initial values for the model's parameters (weights and biases).
- Prediction Function: Create a function that calculates an output based on inputs and current weights (e.g., formula for linear regression).
- Loss Function: Implement a way to measure how far off predictions are from actual values, such as Mean Squared Error.
- Optimization Loop: Use an algorithm like Gradient Descent to calculate derivatives and update weights iteratively until the loss is minimized.

---

## Large Language Models (LLMs)

LLMs are a specific type of machine learning model trained explicitly to handle and generate human text.

An LLM is a highly advanced, specialized type of ML model known as deep learning. It is specifically designed to understand, process, and generate human language.

LLMs use neural networks (specifically "transformers") to analyze the statistical relationships between words.

By ingesting billions of pages of text from books, articles, and websites, they learn context and can predict what word should come next in a sentence.

Examples:

- Google's Gemini: A multimodal model that processes and generates both text and visual inputs.
- Anthropic's Claude: An AI assistant focused on complex reasoning and summarization.

Implementing a large language model from scratch requires building a transformer architecture, processing datasets into numerical vectors, and training the model using machine learning libraries like PyTorch.

### A Pipeline to Build a GPT-Style Model

- Text Tokenization: Map raw text to integers. Convert words or subwords into numerical IDs using a tokenizer (e.g., Byte-Pair Encoding) so the model can process the text.
- Embedding: Convert token IDs into continuous-valued vectors. Add positional embeddings so the model understands word order within a sentence.
- Attention Mechanisms: Implement multi-head self-attention. This allows the model to weigh the importance of different words in a sequence relative to the target word.
- Transformer Block: Stack self-attention layers with feed-forward neural networks. Add layer normalization and residual connections to stabilize training.
- Pre-training: Train the model on a massive corpus (e.g., using unsupervised learning). The model learns by predicting the next token in a sequence and updating its weights via backpropagation.

Implementing a Large Language Model (LLM) from scratch using Python and PyTorch involves four foundational stages: data preparation, model architecture construction (Transformer or GPT-style), pre-training on text corpora, and fine-tuning for specific tasks.

### Stage 1: Data Preparation

Raw text must be converted into numerical representations.

- Tokenization: Break your text corpus down into individual words or subwords. You can build a simple character-level tokenizer or use Byte-Pair Encoding (BPE).
- Integer Mapping: Assign a unique integer to each token in your vocabulary.
- Encoding and Batching: Convert text strings into lists of integers and organize them into sliding window batches (inputs and corresponding targets) to train the model to predict the next word.

### Stage 2: Model Architecture

At the core of an LLM is the decoder-only Transformer block.

- Embedding Layer: Convert the integer tokens into continuous vector representations. Add positional embeddings so the model understands the order of words.
- Self-Attention Mechanism: Compute "Query", "Key", and "Value" matrices to allow the model to weigh the importance of different words in a sequence relative to one another.
- Feed-Forward Neural Networks: Pass the attention outputs through multi-layer perceptrons utilizing layer normalization and dropout for stability.
- Causal Masking: Apply a mask in the attention block so the model can only look at preceding tokens, not future ones.

### Stage 3: Pre-training

This is where the model "learns" language structure.

- Next-Word Prediction: Feed batch sequences into the model and calculate the difference between its prediction and the actual next word using a loss function (e.g., Cross-Entropy Loss).
- Optimization: Use an optimizer like Adam or AdamW to update the neural network's weights to minimize the loss.

### Stage 4: Fine-Tuning

A pre-trained base model is essentially an autocomplete engine. To make it a conversational assistant, you need to fine-tune it:

- Instruction Fine-Tuning: Train the model on prompt-response pairs so it learns to follow human instructions.
- RLHF (Reinforcement Learning from Human Feedback): Align the model's responses to be helpful and safe by scoring its outputs.

---

## Building a Large Language Model from Scratch

This section describes how to implement a small but functional large language model entirely from scratch using Python and PyTorch, without relying on any existing LLM libraries. The approach mirrors the methodology described in "Build a Large Language Model (From Scratch)" by Sebastian Raschka (https://github.com/rasbt/LLMs-from-scratch).

The implementation progresses through three stages: a base language model that predicts the next token in a sequence, a text classifier that fine-tunes the base model to categorize input text, and an instruction-following chatbot that uses prompt templates to simulate conversational behavior.

### Why Build from Scratch

Building an LLM from scratch provides a deep understanding of:

- How tokenization converts raw text into integers the model can process
- How self-attention mechanisms allow the model to relate words to each other
- How the transformer architecture stacks attention and feed-forward layers
- How gradient descent and backpropagation update weights during training
- How fine-tuning repurposes a pretrained base model for a specific downstream task

### The Three Implementations

The source code in `scripts/llm_from_scratch/` provides three runnable Python programs:

**1. Base Language Model (`train.py`)**

The base model is a GPT-style decoder-only transformer. It learns by being trained to predict the next character in a sequence given all previous characters. The model is entirely coded from primitive PyTorch building blocks: `nn.Embedding`, `nn.Linear`, `nn.LayerNorm`, and manual matrix multiplications. No high-level LLM framework or pretrained weights are used.

Components implemented from scratch:

- `CharTokenizer` in `tokenizer.py`: maps every unique character in the training corpus to an integer index and back.
- `MultiHeadSelfAttention` in `model.py`: projects input vectors into query, key, and value spaces; computes scaled dot-product attention across all heads in parallel; applies a causal mask so position `t` can only attend to positions `0..t`.
- `FeedForward` in `model.py`: a two-layer MLP with GELU activation placed after each attention block.
- `TransformerBlock` in `model.py`: combines attention and feed-forward sub-layers with pre-norm layer normalization and residual connections.
- `GPTModel` in `model.py`: stacks multiple transformer blocks, adds token and positional embeddings, and produces logits over the vocabulary.
- Training loop in `train.py`: AdamW optimizer, cosine learning-rate decay, gradient clipping, and checkpoint saving.

**2. Text Classifier (`text_classifier.py`)**

The text classifier fine-tunes the frozen pretrained base model by attaching a small trainable classification head. The base model's weights are not updated; only the new head is trained. The input sequence is mean-pooled across the sequence dimension to produce a single vector, which is passed through a two-layer MLP to output class logits.

This demonstrates a fundamental pattern in modern NLP: pretrain a general language model on a large corpus, then fine-tune a lightweight head on a small labeled dataset for a specific task.

**3. Instruction-Following Chatbot (`chatbot.py`)**

The chatbot demonstrates instruction fine-tuning via a prompt template approach. The pretrained model is loaded and a structured prompt in the form `### Human: <input> ### Assistant:` is prepended to each user message. The model then generates continuation text, which is extracted as the response.

This is the same "transcript hack" technique described by Giles Thomas in his LLM-from-scratch series (https://www.gilesthomas.com/llm-from-scratch). It works because a well-trained language model learns to complete realistic-looking conversational transcripts.

### Key Concepts Implemented

**Causal (Masked) Self-Attention**

Each token position can only attend to tokens at equal or earlier positions. This is enforced by filling the upper triangle of the attention score matrix with negative infinity before the softmax, so those positions receive zero attention weight after normalization.

```
scores = Q @ K.T / sqrt(d_head)
scores[mask == 0] = -inf
attention_weights = softmax(scores)
context_vectors = attention_weights @ V
```

**Multi-Head Attention**

Instead of one large attention computation, the model splits the embedding dimension into `num_heads` smaller subspaces and runs attention independently in each. The outputs are concatenated and projected back to the original dimension. Multiple heads allow the model to attend to different aspects of context simultaneously.

**Residual Connections and Layer Normalization**

Each transformer block adds its output back to its input (`x = x + sublayer(x)`). This residual (shortcut) connection preserves gradient flow through deep networks. Layer normalization is applied before each sub-layer (pre-norm style) to stabilize training.

**Weight Tying**

The token embedding matrix and the final output projection matrix share the same weights. This reduces the parameter count and often improves training because the model uses the same representation space for both encoding and decoding tokens.

**Next-Token Generation**

At inference time the model autoregressively samples one token at a time. Temperature scaling controls the sharpness of the probability distribution. Top-k filtering retains only the `k` most probable tokens before sampling, preventing very low-probability completions.

---

## Project Structure

```
◈ scripts/
  ▸ llm_from_scratch/
      ▪ tokenizer.py         Character-level tokenizer
      ▪ model.py             GPT-style transformer model (all layers from scratch)
      ▪ train.py             Pre-training script (base language model)
      ▪ text_classifier.py   Fine-tuning script (sentiment classifier)
      ▪ chatbot.py           Instruction-following chatbot
      ▪ requirements.txt     Python package dependencies
      ▪ setup_venv.sh        Virtual environment creation and activation script
      ▪ data/                Training data directory (created automatically)
      ▪ checkpoints/         Saved model checkpoints (created during training)
```

---

## Virtual Environment Setup on Linux

A Python virtual environment isolates the project's dependencies from the system Python installation. All scripts must be run inside the activated virtual environment.

### Create the Virtual Environment

Run the following commands from the `scripts/llm_from_scratch/` directory:

```bash
cd scripts/llm_from_scratch

python3 -m venv venv
```

This creates a `venv/` subdirectory containing an isolated Python interpreter and a private `site-packages` directory.

### Activate the Virtual Environment

```bash
source venv/bin/activate
```

After activation, the shell prompt changes to show `(venv)` at the start, confirming the virtual environment is active. All subsequent `python` and `pip` commands now use the isolated environment.

### Install Dependencies

With the virtual environment active, install the required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Deactivate the Virtual Environment

When you are finished, deactivate the virtual environment to return to the system Python:

```bash
deactivate
```

### Using the Setup Script

A convenience script `setup_venv.sh` automates the creation, activation, and dependency installation steps:

```bash
cd scripts/llm_from_scratch
chmod +x setup_venv.sh
./setup_venv.sh
source venv/bin/activate
```

Note: `./setup_venv.sh` creates and populates the environment, but the `source venv/bin/activate` command must be run separately in the current shell because a subprocess cannot modify the parent shell's environment.

---

## Running the Scripts

All commands below assume the virtual environment is active and you are in the `scripts/llm_from_scratch/` directory.

### Step 1: Pre-train the Base Model

```bash
python train.py
```

This script creates sample training data in `data/train.txt` if none exists, builds and trains a GPT-style model, saves the best checkpoint to `checkpoints/best_model.pt`, and prints a short sample of generated text when training completes.

Training progress is printed every 500 iterations showing train and validation loss. On a CPU-only machine the default configuration (4 layers, 128-dimensional embeddings, 128-token context) completes in a few minutes.

To train on your own text data, replace the contents of `data/train.txt` with any plain-text corpus before running the script.

### Step 2: Fine-tune the Text Classifier

```bash
python text_classifier.py
```

This script loads the checkpoint saved by `train.py`, attaches a classification head, and fine-tunes it on a small sentiment dataset. After training it prints predictions for several example sentences showing "positive" or "negative" labels.

### Step 3: Run the Chatbot

```bash
python chatbot.py
```

This script loads the pretrained model and starts an interactive command-line chat session. Type a message and press Enter to receive a response. Type `exit` or `quit` to stop.

Because the base model is small and trained on limited data, the chatbot's responses will be grammatically plausible but may not be semantically coherent. Training on a larger and more diverse text corpus significantly improves response quality.

---

## References

Large language model (LLMs) Tutorials https://unsloth.ai/docs/models/tutorials

Lecture 10: inference https://cs336.stanford.edu/lectures/?trace=lecture_10

What data should we train on?
https://github.com/stanford-cs336/lectures/blob/main/lecture_15.pdf

How to train a model given data?
https://cs336.stanford.edu/lectures/?trace=lecture_14

Training a causal language model from scratch https://huggingface.co/learn/llm-course/chapter7/6

Data pipeline: transformation, filtering, deduplication, mixing https://cs336.stanford.edu/lectures/?trace=lecture_14

Fine-tuning: Train the pretrained model on instruction-based datasets so it functions as a conversational assistant.

Build a Large Language Model (From Scratch) https://sebastianraschka.com/llms-from-scratch/

Implement a ChatGPT-like LLM in PyTorch from scratch, step by step
https://github.com/rasbt/LLMs-from-scratch

Working with Text Data https://github.com/rasbt/LLMs-from-scratch/tree/main/ch02

Attention Mechanisms https://github.com/rasbt/LLMs-from-scratch/tree/main/ch03

Implementing a GPT Model https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04

Pretraining on Unlabeled Data https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05

Finetuning for Text Classification https://github.com/rasbt/LLMs-from-scratch/tree/main/ch06

Finetuning to Follow Instructions https://github.com/rasbt/LLMs-from-scratch/tree/main/ch07

LLM from scratch https://www.gilesthomas.com/llm-from-scratch

Lecture 12: evaluation https://cs336.stanford.edu/lectures/?trace=lecture_12

given a model, how "good" is it?

What is LLM & How to Build Your Own Large Language Models? https://www.signitysolutions.com/blog/how-to-build-large-language-models


