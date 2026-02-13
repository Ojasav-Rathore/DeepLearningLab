# Text Generation using Recurrent Neural Networks (RNNs)

The aim of this experiment is to explore text generation using Recurrent Neural Networks (RNNs) and to understand the impact of different word representations: One-Hot Encoding and Trainable Word Embeddings.

## Dataset

The dataset consists of multiple lines of poetry, which will be used to generate text sequences.

## Implementation Steps

The project is broken down into three main phases:

### Part 1: Implement RNN From Scratch
* Implement a basic RNN from scratch using NumPy.

### Part 2: RNN with One-Hot Encoding
* Tokenize the text into words. Convert each word into a one-hot vector. Train the RNN model using the one hot encoded dataset. Generate text.

### Part 3: RNN with Trainable Word Embeddings

* Tokenize the text into words. Convert each word into a word embeddings. Train the RNN model using the one hot encoded dataset. Generate text.

## Comparison and Analysis

### 1. Advantages and Disadvantages of Each Approach

* **One-Hot Encoding:**
  * *Advantages:* Simple to understand and deterministic. It is static and requires no training. 
  * *Disadvantages:* One-hot encoding converts categorical data into sparse binary vectors, where each unique category is assigned a unique position in a vector. It becomes inefficient for large vocabularies because the dimensionality grows with the number of categories, leading to sparse, high-dimensional vectors that consume memory and computational resources. Additionally, one-hot vectors treat categories as entirely independent, ignoring any relationships between them.

* **Trainable Word Embeddings:**
  * *Advantages:* Embeddings reduce dimensionality—a 300-dimensional vector might represent tens of thousands of words—and enable models to generalize better by encoding similarities between categories. In a trained embedding layer, words or categories with related meanings occupy closer positions in the vector space. Because a word embedding is a dense vector representation of words, it represents words more efficiently than sparse vector techniques.
  * *Disadvantages:* While one-hot encoding is static and requires no training, embeddings are dynamic and adapt to the data, making them computationally heavier to train initially.

### 2. Training Time and Loss

* **One-Hot Encoding:** Execution time per epoch is generally much higher. Creating dense operations out of highly sparse, vocabulary-sized matrices forces the RNN to perform massive, mostly empty computations, often leading to slower convergence and earlier loss plateaus.
* **Embeddings:** Passing data through the embedding layer is highly optimized. Because the RNN cell processes a much smaller, dense vector, the matrix multiplications are significantly faster. This generally yields a smoother loss curve and a lower final training loss.

### 3. Quality of Generated Text

* **One-Hot Encoding:** The generated poetry tends to be structurally rigid and occasionally nonsensical. Because One-Hot encoding cannot learn semantic similarity, the network struggles to substitute related words and relies entirely on rote memorization of the training sequence.
* **Embeddings:** Outputs are noticeably more fluid and contextually accurate. Since the embedding space clusters similar concepts (e.g., "sun" and "sky") together, the model understands underlying context, allowing it to generate new, grammatically coherent lines that fit the poetic theme.
