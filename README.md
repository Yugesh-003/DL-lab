# DL-lab

Deep Learning Lab course for II B.Sc Computer Science (Data Science and Analytics), Semester IV. This repository contains practical lab assignments and Jupyter notebooks implementing core deep learning concepts using TensorFlow and Keras.

---

## 📋 Course Information

| Detail | Information |
|--------|-------------|
| **Course Code** | 24DCS404 P |
| **Semester** | IV |
| **Batch** | 2024-2027 |
| **Part** | III (Core Practical) |
| **Credits** | 3 |
| **Hours/Week** | 4 |
| **Total Hours** | 60 |
| **CIA Marks** | 40 |
| **End Semester Marks** | 60 |

---

## 🎯 Course Objective

To enable students to learn the fundamental concepts and practical applications of deep learning, including the design, implementation, training, and evaluation of neural networks using modern deep learning frameworks like TensorFlow and Keras.

---

## 📚 Table of Contents

- [Course Outcomes](#course-outcomes)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Lab Programs](#lab-programs)
- [Getting Started](#getting-started)
- [Lab Structure](#lab-structure)
- [Submission Guidelines](#submission-guidelines)
- [Grading](#grading)
- [Resources](#resources)
- [Academic Integrity](#academic-integrity)

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

## 🚀 Getting Started

### Running Your First Lab

1. **Start Jupyter:**
```bash
jupyter notebook
```

2. **Navigate to the Lab1 notebook** and open `Lab1_SimpleNeuralNetwork.ipynb`

3. **Follow this workflow for each lab:**
   - Read the lab objectives
   - Review the theory and background
   - Study the example code
   - Complete the implementation tasks
   - Run experiments with different parameters
   - Document your observations
   - Answer reflection questions

### Tips for Success

- **Execute cells sequentially** - Always run cells from top to bottom
- **Read carefully** - Understand what each section is doing before writing code
- **Experiment** - Modify parameters to see how they affect results
- **Document** - Add comments explaining your code and observations
- **Save often** - Use `Ctrl+S` or `Cmd+S` to save your work
- **Clear outputs** - Periodically use `Kernel → Restart & Clear All Output`
- **Use reproducible seeds** - Set random seeds for consistent results

### Setting Random Seeds for Reproducibility

Add this to the beginning of your notebooks:

```python
import numpy as np
import random
import tensorflow as tf

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)
```

---

## 📐 Lab Structure

Each lab notebook follows a consistent structure:

```
Lab [N] - [Topic Name]
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

## 📤 Submission Guidelines

### Before Submitting Your Lab

**Checklist:**
- ✅ All cells have been executed
- ✅ No error messages in output
- ✅ All TODO sections completed
- ✅ Code includes comments explaining logic
- ✅ Visualizations and plots are displayed
- ✅ All reflection questions answered
- ✅ File saved with correct naming convention

### File Naming Convention

Submit notebooks with this naming format:
```
YourName_Lab[N]_[TopicName].ipynb
```

**Examples:**
- `Yugesh_Lab1_SimpleNeuralNetwork.ipynb`
- `Yugesh_Lab5_ImageClassificationCNN.ipynb`
- `Yugesh_Lab9_FinalProject.ipynb`

### What to Include

Each submitted notebook must contain:

1. **Completed Code** - All implementation tasks finished
2. **Clear Comments** - Explain what your code does
3. **Experiment Results** - Output, visualizations, and metrics
4. **Observations** - What did you learn? What surprised you?
5. **Answers** - All reflection questions answered thoroughly
6. **No Errors** - Code runs without exceptions

### Code Quality Standards

- Use **descriptive variable names** that indicate purpose
- Add **inline comments** for complex logic
- Follow **PEP 8 style guidelines**
- **Test your code** with different inputs
- Avoid **hard-coded values** - use variables instead
- **Document edge cases** and how you handle them

---

## 📊 Grading

### Continuous Internal Assessment (CIA) - 40 Marks

- **Lab Participation**: 5 marks
- **Individual Lab Submissions** (8 labs × 4 marks): 32 marks
- **Lab Conduct & Attendance**: 3 marks

### End Semester Examination - 60 Marks

- **Practical Exam**: Build a model on unseen dataset
- **Viva Voce**: Questions on concepts and your code
- **Report**: Document your approach and results

### Lab Evaluation Criteria

| Criteria | Marks |
|----------|-------|
| Code Implementation | 2 |
| Correctness & Output | 1 |
| Documentation & Comments | 0.5 |
| Analysis & Reflection | 0.5 |

---

## 🌐 Resources

### Course Materials

- Course lecture slides (provided by instructor)
- Recommended textbooks and research papers
- Course discussion forum/forum

### External Learning Resources

#### Deep Learning Foundations
- **Deep Learning** by Goodfellow, Bengio, and Courville
- Stanford CS231n: Convolutional Neural Networks for Visual Recognition
- Stanford CS224N: Natural Language Processing with Deep Learning

#### Practical Frameworks
- [TensorFlow Official Documentation](https://www.tensorflow.org/)
- [Keras API Reference](https://keras.io/)
- [PyTorch Documentation](https://pytorch.org/) (alternative framework)

#### Interactive Learning
- [Fast.ai Practical Deep Learning](https://www.fast.ai/)
- [Coursera Deep Learning Specialization](https://www.deeplearning.ai/)
- [Kaggle Learn Deep Learning Courses](https://www.kaggle.com/learn)

#### Jupyter Notebook Tips
- [Jupyter Notebook Documentation](https://jupyter.org/)
- [Jupyter Keyboard Shortcuts](https://jupyter-notebook.readthedocs.io/en/stable/notebook.html#keyboard-shortcuts)

---

## ⌨️ Jupyter Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Shift + Enter` | Execute cell and move to next |
| `Ctrl + Enter` | Execute current cell |
| `Alt + Enter` | Execute cell and insert new cell below |
| `A` | Insert cell above (Command mode) |
| `B` | Insert cell below (Command mode) |
| `D, D` | Delete cell (Command mode) |
| `M` | Convert to Markdown (Command mode) |
| `Y` | Convert to Code (Command mode) |
| `Ctrl + /` | Toggle comment |
| `Ctrl + Z` | Undo |
| `Ctrl + S` | Save notebook |

---

## 🔧 Troubleshooting

### Common Issues & Solutions

**"ModuleNotFoundError: No module named 'tensorflow'"**
```bash
# Solution: Install TensorFlow
pip install tensorflow
# Verify environment is activated
conda activate dl-lab  # or source dl-lab/bin/activate
```

**"Kernel keeps crashing or running slow"**
- Restart kernel: `Kernel → Restart and Clear All Output`
- Close other resource-intensive applications
- Reduce batch size or dataset size in your code
- For large computations, consider using GPU

**"GPU not being detected"**
```python
# Check if GPU is available
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

**"Different results every time I run the code"**
- Set random seeds at the beginning (see code above)
- Ensure you're loading the same data each time

**"Notebook file size is too large"**
- Clear outputs: `Kernel → Restart & Clear All Output`
- Remove unnecessary visualizations
- Save checkpoint models separately

---

## 📝 Academic Integrity Policy

### Expectations

- **Write your own code** - Don't copy from classmates or internet
- **Understand your work** - Be able to explain every line of code
- **Cite sources** - Reference any external resources or inspiration
- **Collaborate responsibly** - Discuss concepts but write individual solutions
- **Report issues** - Notify instructor of technical problems immediately

### Plagiarism Consequences

Violation of academic integrity may result in:
- Assignment score of zero
- Failure in the course
- Disciplinary action as per institute policy

---

## 📞 Getting Help

### Course Support Channels

1. **Instructor Office Hours** - (Provide schedule)
2. **Lab TA Support** - (Provide timing/email)
3. **Course Discussion Forum** - (Link provided)
4. **Email** - (Instructor email)

### Before Asking for Help

- Check course materials and notebook comments
- Search for similar issues online
- Review related sections in textbooks
- Try debugging your code step by step

### When Asking for Help

- **Be specific** - Show the exact error message
- **Share context** - What were you trying to do?
- **Show your attempt** - What solutions have you tried?
- **Be respectful** - Instructors and TAs are here to help!

---

## 📋 Lab Attendance & Submission Dates

| Lab # | Topic | Target Date | Submission Date |
|-------|-------|-------------|-----------------|
| 1 | Simple Neural Network | Week 1 | - |
| 2 | Data Preprocessing | Week 2 | - |
| 3 | Activation Functions | Week 3 | - |
| 4 | Backpropagation Training | Week 4 | - |
| 5 | Image Classification CNN | Week 5 | - |
| 6 | Transfer Learning | Week 6 | - |
| 7 | RNN Text Classification | Week 7 | - |
| 8 | Text Generation | Week 8 | - |
| 9 | Final Project | Weeks 9-10 | - |

*Note: Exact dates will be communicated by instructor*

---

## 🎓 Final Notes

- **Consistency is key** - Attend labs regularly and complete assignments on time
- **Learning journey** - Deep learning takes practice; don't get discouraged
- **Ask questions** - Curiosity drives learning; ask whenever you're confused
- **Explore further** - Extend labs with your own ideas and experiments
- **Build a portfolio** - Keep your best work for future opportunities

---

## 📜 Important Links

- **GitHub Repository**: https://github.com/Yugesh-003/DL-lab
- **Syllabus**: Deep-Learning-Lab-Syllabus.docx
- **Course Code**: 24DCS404 P

---

**Good luck with your Deep Learning Lab! Work hard, learn deeply, and enjoy the journey! 🚀🧠**
