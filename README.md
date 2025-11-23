# DL-lab

Deep Learning Lab course for II B.Sc Computer Science (Data Science and Analytics), Semester IV. This repository contains practical lab assignments and Jupyter notebooks implementing core deep learning concepts using TensorFlow and Keras.

---

## 📋 Course Information

| Detail                 | Information          |
| ---------------------- | -------------------- |
| **Course Code**        | 24DCS404 P           |
| **Semester**           | IV                   |
| **Batch**              | 2024-2027            |
| **Part**               | III (Core Practical) |
| **Credits**            | 3                    |
| **Hours/Week**         | 4                    |
| **Total Hours**        | 60                   |
| **CIA Marks**          | 40                   |
| **End Semester Marks** | 60                   |

---

## 🎯 Course Objective

To enable students to learn the fundamental concepts and practical applications of deep learning, including the design, implementation, training, and evaluation of neural networks using modern deep learning frameworks like TensorFlow and Keras.

---

## 🔬 Lab Programs

The course is structured around 9 practical programs designed to progressively build deep learning skills:

### **Program 1: Simple Neural Network for Classification**

- **Objective**: Understand neural network fundamentals with a single hidden layer
- **Topics**: Perceptron, activation functions, forward propagation
- **Dataset**: Basic synthetic or small dataset
- **Notebook**: `Lab1_SimpleNeuralNetwork.ipynb`

### **Program 2: Data Preprocessing**

- **Objective**: Learn essential data preparation techniques
- **Topics**: Data loading, normalization, train-test splitting, handling missing values
- **Tasks**: Prepare real datasets for model training
- **Notebook**: `Lab2_DataPreprocessing.ipynb`

### **Program 3: Activation Functions**

- **Objective**: Understand common activation functions and their impact
- **Topics**: ReLU, Sigmoid, Softmax, Tanh, visualization
- **Tasks**: Implement and compare different activation functions
- **Notebook**: `Lab3_ActivationFunctions.ipynb`

### **Program 4: Training Models with Backpropagation**

- **Objective**: Implement forward and backward propagation from scratch
- **Topics**: Forward pass, backward propagation, weight updates, gradient descent
- **Tasks**: Train a neural network on sample data, analyze convergence
- **Notebook**: `Lab4_BackpropagationTraining.ipynb`

### **Program 5: Image Classification with CNNs**

- **Objective**: Build and train Convolutional Neural Networks
- **Topics**: Convolution, pooling, feature maps, CNN architecture
- **Dataset**: MNIST or CIFAR-10
- **Tasks**: Build CNN, train, evaluate accuracy and loss
- **Notebook**: `Lab5_ImageClassificationCNN.ipynb`

### **Program 6: Transfer Learning with Pre-trained Models**

- **Objective**: Leverage existing pre-trained models for new tasks
- **Topics**: Transfer learning, fine-tuning, feature extraction
- **Models**: VGG, ResNet, or other architectures
- **Tasks**: Load pre-trained model, adapt for custom dataset
- **Notebook**: `Lab6_TransferLearning.ipynb`

### **Program 7: RNNs for Text Classification**

- **Objective**: Process sequential data with Recurrent Neural Networks
- **Topics**: RNN, LSTM, GRU, text embeddings, sequence processing
- **Tasks**: Implement text classifier, evaluate performance
- **Notebook**: `Lab7_RNNTextClassification.ipynb`

### **Program 8: Text Generation with RNNs**

- **Objective**: Generate text sequences using trained RNN models
- **Topics**: Sequence modeling, character-level predictions, sampling
- **Tasks**: Train model on text corpus, generate new text sequences
- **Notebook**: `Lab8_TextGeneration.ipynb`

### **Program 9: Final Project - Custom Deep Learning Model**

- **Objective**: Demonstrate mastery by solving a novel problem
- **Topics**: End-to-end model development, custom datasets, optimization
- **Tasks**:
  - Choose a problem or dataset
  - Preprocess and explore data
  - Design and train appropriate model
  - Evaluate and optimize performance
  - Document process and results
- **Notebook**: `Lab9_FinalProject.ipynb`

---

## ✅ Course Outcomes

After completing this course, students will be able to:

1. **Build a Perceptron model** - Understand and implement the basic building block of neural networks
2. **Build Neural Network models using BP algorithm** - Implement backpropagation and train networks
3. **Finetune Deep Learning models for performance optimization** - Optimize model architectures and hyperparameters
4. **Use TensorFlow to build prediction models** - Leverage industry-standard frameworks for deep learning
5. **Build, Compile, Test, and evaluate models in Keras** - Use high-level APIs for rapid model development

---

## 📋 Prerequisites

Before starting this lab course, ensure you have knowledge of:

- **Python Programming** (essential) - Variables, functions, loops, libraries
- **Data Structures** - Lists, dictionaries, arrays
- **Linear Algebra** - Matrices, vectors, matrix operations
- **Calculus** - Derivatives, gradients, chain rule
- **Probability & Statistics** - Distributions, mean, variance
- **Machine Learning Basics** (recommended) - Classification, regression, evaluation metrics

---

## 💻 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Yugesh-003/DL-lab.git
cd DL-lab
```

### 2. Set Up Python Environment

#### Using Conda (Recommended)

```bash
conda create -n dl-lab python=3.9
conda activate dl-lab
```

#### Using venv

```bash
python -m venv dl-lab
source dl-lab/bin/activate    # On macOS/Linux
# or
dl-lab\Scripts\activate        # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter

```bash
jupyter notebook
# or for JupyterLab
jupyter lab
```

---

## 📐 Lab Structure

Each lab notebook follows a consistent structure:

```
[N] - [Topic Name]
│
├── Objectives
│   └── What you'll learn and accomplish
│
├── Theory & Background
│   └── Conceptual foundation and mathematical concepts
│
├── Concepts Review
│   └── Key definitions and formulas
│
├── Implementation
│   ├── Example Code
│   └── Your Tasks (marked as [TODO])
│
├── Experiments
│   └── Hands-on exercises and parameter exploration
│
├── Results & Analysis
│   └── Document observations and findings
│
└── Reflection Questions
    └── Conceptual questions to reinforce learning
```

---

## 🌐 Resources

### External Learning Resources

---

**Work hard, learn deeply, and enjoy the journey! 🚀🧠**
