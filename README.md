# DL-lab

A comprehensive collection of Jupyter notebooks for deep learning coursework, assignments, and lab exercises covering fundamental to advanced deep learning concepts and implementations.

## 📋 Table of Contents

- [About](#about)
- [Course Information](#course-information)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Lab Structure](#lab-structure)
- [Notebooks Overview](#notebooks-overview)
- [Getting Started](#getting-started)
- [Submission Guidelines](#submission-guidelines)
- [Resources](#resources)

## About

**DL-lab** is a repository containing lab assignments and notebooks for deep learning coursework. This collection includes hands-on exercises, implementations, and experiments designed to reinforce theoretical concepts taught in the deep learning course.

### What's Included

- Lab assignment notebooks with structured exercises
- Code implementation assignments
- Jupyter notebooks for practical experimentation
- Dataset handling and preprocessing tutorials
- Model training and evaluation exercises
- Performance analysis and visualization tasks

## Course Information

This repository contains coursework for **Deep Learning Lab** course. Each notebook corresponds to specific lab sessions and assignments focused on building practical deep learning skills.

### Course Objectives

- Understand and implement fundamental deep learning algorithms
- Gain practical experience with deep learning frameworks
- Learn data preprocessing and model evaluation techniques
- Explore various neural network architectures
- Develop problem-solving skills through hands-on coding

## Prerequisites

Before starting this course, ensure you have knowledge of:

- **Python Programming** (essential)
- **Linear Algebra** (matrices, vectors)
- **Probability and Statistics** (basic concepts)
- **Calculus** (derivatives, gradients)
- **Machine Learning Basics** (recommended)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Yugesh-003/DL-lab.git
cd DL-lab
```

### 2. Set Up Python Environment

```bash
# Using Anaconda (recommended)
conda create -n dl-lab python=3.9
conda activate dl-lab

# Or using venv
python -m venv dl-lab
source dl-lab/bin/activate  # On Windows: dl-lab\Scripts\activate
```

### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

Required packages include:
- `jupyter` - Notebook environment
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `matplotlib` - Data visualization
- `scikit-learn` - Machine learning utilities
- `tensorflow` or `pytorch` - Deep learning frameworks

## Lab Structure

Each lab notebook follows a standard structure:

```
Lab [N] - [Topic Name]
├── Objective(s) - What you'll learn
├── Theory - Conceptual background
├── Implementation - Code to write/complete
├── Experiment(s) - Exercises and tasks
├── Results - Analysis and observations
└── Questions - Reflection and understanding checks
```

## Notebooks Overview

Add your lab assignments below:

| Lab # | Topic | Notebook | Status |
|-------|-------|----------|--------|
| 1 | [Add topic] | `lab1.ipynb` | - |
| 2 | [Add topic] | `lab2.ipynb` | - |
| 3 | [Add topic] | `lab3.ipynb` | - |
| 4 | [Add topic] | `lab4.ipynb` | - |

*Note: Update with your actual lab topics and notebook names*

## Getting Started

### Running Your First Lab

1. **Start Jupyter:**

```bash
jupyter notebook
```

2. **Open the first lab notebook** from the Jupyter file browser

3. **Follow the notebook structure:**
   - Read the objective and theory sections
   - Complete coding tasks in designated cells
   - Run all cells sequentially (top to bottom)
   - Document your observations and results

### Tips for Lab Work

- **Read all instructions** before writing code
- **Execute cells sequentially** - don't skip or jump around
- **Run each cell individually** and verify output before moving to the next
- **Document your work** - add comments explaining your code
- **Save frequently** - use `Ctrl+S` or `Cmd+S`
- **Clear outputs** periodically: `Kernel → Restart & Clear All Output`
- **Test different parameters** to understand how they affect results

## Completing Assignments

### Assignment Workflow

1. Read the lab notebook objectives and instructions carefully
2. Complete the theory review section
3. Implement the required code in the designated cells
4. Run all experiments and document observations
5. Answer reflection questions in markdown cells
6. Verify all cells execute without errors
7. Save the notebook before submission

### Code Requirements

- **Use clear variable names** that describe what the variable stores
- **Add comments** explaining complex logic
- **Follow Python naming conventions** (snake_case for variables)
- **Avoid hard-coding values** - use variables instead
- **Test your code** with different inputs
- **Handle edge cases** appropriately

## Submission Guidelines

### Before Submitting

- Ensure all cells have been executed and show output
- Check that there are no error messages
- Review your code for clarity and completeness
- Verify all answers are written in the notebook
- Save the file with proper naming convention

### File Naming

Use this format for submitted notebooks:
```
YourName_Lab[N]_[TopicName].ipynb
```

Example: `Yugesh_Lab1_NeuralNetworks.ipynb`

### What to Include

- All completed code cells
- Clear comments and explanations
- Experiment results and observations
- Answers to reflection questions
- Visualizations and plots

## Troubleshooting

### Common Issues

**Issue: `ModuleNotFoundError: No module named 'tensorflow'` (or other package)**
- Solution: Install missing package with `pip install tensorflow`
- Ensure you're in the correct conda/virtual environment

**Issue: Kernel crashes or notebook becomes very slow**
- Solution: Restart kernel with `Kernel → Restart and Clear All Output`
- Reduce dataset size or batch size in your code
- Close other resource-intensive applications

**Issue: Different results each time I run the notebook**
- Solution: Set random seeds at the beginning:
```python
import numpy as np
import random
import tensorflow as tf

np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)
```

**Issue: GPU not being used (if available)**
- Solution: Check if TensorFlow detects GPU:
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

## Learning Resources

### Recommended Reading

- **Deep Learning** by Goodfellow, Bengio, and Courville
- Official TensorFlow/PyTorch documentation
- Stanford CS231n course notes (Convolutional Neural Networks)
- Fast.ai practical deep learning courses

### Helpful Links

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [PyTorch Documentation](https://pytorch.org/)
- [NumPy Tutorial](https://numpy.org/doc/stable/user/)
- [Jupyter Notebook Tips & Tricks](https://www.datacamp.com/community/tutorials/tutorial-jupyter-notebook)

### Getting Help

- Review course materials and lectures
- Check notebook comments and documentation
- Consult assigned readings
- Discuss with classmates (but write your own code)
- Ask instructor during office hours or via email

## Academic Integrity

- Write your own code and solutions
- Cite any external resources or references used
- Do not copy code from other students
- Understand what your code does - be able to explain it
- Report any issues or concerns to your instructor

## Jupyter Notebook Shortcuts

| Shortcut | Action |
|----------|--------|
| `Shift + Enter` | Execute cell and move to next |
| `Ctrl + Enter` | Execute cell |
| `Alt + Enter` | Execute cell and insert new cell below |
| `A` | Insert cell above (command mode) |
| `B` | Insert cell below (command mode) |
| `D, D` | Delete cell (command mode) |
| `M` | Convert cell to Markdown (command mode) |
| `Y` | Convert cell to Code (command mode) |
| `Ctrl + /` | Comment/uncomment lines |

## Environment Management

### Save Your Environment

```bash
pip freeze > requirements.txt
```

### Recreate Environment on Another Machine

```bash
pip install -r requirements.txt
```

### Update All Packages

```bash
pip install --upgrade -r requirements.txt
```

## Important Notes

- Always keep backup copies of your work
- Don't modify files outside of lab assignments without asking
- Report any issues with notebooks to your instructor immediately
- Respect deadlines for lab submissions
- Academic honesty is essential - write your own solutions

---

## Contact

For questions or issues regarding the labs:
- Reach out to the course instructor
- Check the course discussion forum
- Review course documentation

**Good luck with your deep learning labs! 📚🧠**
