# Artificial Neural Network-Based Detection and Classification of Security Risks in Vibe-Coded Software

## 1. Introduction

The rapid growth of **vibe coding**, where developers use natural-language prompts and AI coding assistants to generate software, has significantly accelerated software development. However, AI-generated code may contain security weaknesses because AI models can reproduce unsafe programming practices present in their training data.

Developers may also accept AI-generated code without performing detailed security reviews. This creates a need for automated systems that can identify potentially vulnerable source code before it is deployed.

This project proposes an **Artificial Neural Network (ANN)** based machine learning system for detecting and classifying security risks in source code. The system analyzes source code using static code features and TF-IDF-based syntactic features and predicts whether a function is secure or potentially vulnerable.

---

## 2. Problem Statement

AI-generated and rapidly developed software can introduce security vulnerabilities because:

* Developers may not manually inspect all AI-generated code.
* AI coding assistants can reproduce unsafe or deprecated programming practices.
* Rapid development cycles can bypass conventional security testing.
* Vulnerabilities may not be obvious from the syntax of the generated code.

Therefore, an automated machine learning system is required to analyze source code and identify the likelihood of security vulnerabilities before deployment.

---

## 3. Objectives

The main objectives of the project are:

1. To acquire and process the **Microsoft CodeXGLUE Defect Detection (Devign) dataset**.
2. To extract meaningful static and syntactic features from source code.
3. To construct an **85-dimensional feature representation** of source code.
4. To develop an Artificial Neural Network for vulnerability classification.
5. To compare the ANN with traditional machine learning algorithms.
6. To evaluate the models using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
7. To develop an interactive application for real-time vulnerability screening.
8. To ensure that submitted source code is analyzed safely without executing it.

---

## 4. Dataset

The project uses the **Microsoft CodeXGLUE Defect Detection Benchmark (Devign Dataset)**.

### Dataset Characteristics

* **Dataset:** CodeXGLUE Defect Detection / Devign
* **Total functions:** 27,318
* **Original split:**

  * Training: 21,854
  * Development: 2,732
  * Testing: 2,732
* **Evaluated subset:** 6,000 functions

  * Training: 4,200
  * Validation: 900
  * Testing: 900

### Target Classes

The problem is treated as a binary classification task:

* **0 – Secure / Defect-free**
* **1 – Vulnerable / Insecure**

The dataset consists primarily of real-world C/C++ software functions from projects such as **QEMU and FFmpeg**.

---

## 5. Proposed Methodology

The proposed system follows the following workflow:

**Source Code → Feature Extraction → Preprocessing → ANN Model → Vulnerability Prediction → Risk Probability**

The source code is treated as passive text. Features are extracted from the code without dynamically executing the submitted program.

The methodology consists of:

1. Dataset collection
2. Data preprocessing
3. Feature extraction
4. Feature normalization
5. ANN model development
6. Baseline model development
7. Model training
8. Performance evaluation
9. Vulnerability prediction

---

## 6. Feature Engineering

An **85-dimensional numerical feature vector** is extracted from each source-code function.

### 6.1 Structural and Lexical Features

These features describe the general structure of the source code:

* Character length
* Number of lines
* Average line length
* Number of blank lines
* Comment ratio
* Maximum indentation depth
* Token count
* Digit count

### 6.2 Control-Flow and Complexity Features

The system analyzes programming constructs such as:

* `if`
* `else`
* `for`
* `while`
* `switch`
* `case`
* `break`
* `return`
* `goto`
* Branching nodes

These features provide information about the complexity of the source code.

### 6.3 Operators and Pointer Features

The system also extracts information related to:

* Pointer dereferencing
* Array subscripting
* Bitwise operations
* Logical operations
* Relational operations
* Arithmetic operations
* Assignment operations

### 6.4 Security-Sensitive Features

Security-related programming patterns are identified, including:

* Dynamic memory allocation
* `malloc`
* `calloc`
* `free`
* `realloc`
* Unsafe buffer operations such as `strcpy`, `strcat`, `gets`, `sprintf`, `scanf`, and `memcpy`
* Format-string patterns
* OS command execution functions
* File input/output operations
* Secret or token-related literals
* Null-pointer checks
* Pointer casts
* `sizeof` operations

### 6.5 TF-IDF Features

The system also uses **50 TF-IDF-based syntactic code-token features**.

These features capture frequently occurring code patterns and syntactic structures.

Therefore:

**Static Code Features + TF-IDF Features = 85-dimensional feature vector**

---

## 7. Data Preprocessing

The extracted features are converted into numerical representations suitable for machine learning.

The preprocessing stage includes:

* Feature extraction
* TF-IDF transformation
* Feature scaling
* Training/validation/test separation

A **StandardScaler** is used for normalization.

To prevent data leakage, the scaler and TF-IDF vectorizer are fitted only using the training data.

---

## 8. Proposed ANN Architecture

The proposed model is a **Multi-Layer Perceptron Artificial Neural Network**.

### Architecture

**Input Layer**

* 85 input features

↓

**Dense Layer**

* 128 neurons
* ReLU activation

↓

**Batch Normalization**

↓

**Dropout**

* Rate: 0.3

↓

**Dense Layer**

* 64 neurons
* ReLU activation

↓

**Batch Normalization**

↓

**Dropout**

* Rate: 0.2

↓

**Dense Layer**

* 32 neurons
* ReLU activation

↓

**Dropout**

* Rate: 0.2

↓

**Output Layer**

* 1 neuron
* Sigmoid activation

The sigmoid output produces the probability that the input source code is vulnerable.

---

## 9. ANN Training

The ANN uses:

* **Loss Function:** Binary Cross-Entropy
* **Optimizer:** Adam
* **Initial Learning Rate:** 0.001
* **Activation Function:** ReLU for hidden layers
* **Output Activation:** Sigmoid

To improve training and reduce overfitting, the model uses:

* Batch Normalization
* Dropout
* Early Stopping
* Learning-rate reduction
* Best-model checkpointing

---

## 10. Baseline Machine Learning Models

The ANN is compared with four traditional machine learning algorithms:

1. Logistic Regression
2. Random Forest
3. Support Vector Machine (SVM)
4. Gradient Boosting

All models are evaluated using the same test dataset so that their performance can be compared fairly.

---

## 11. Performance Evaluation

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC

### Model Comparison

| Model               |   Accuracy |  Precision |     Recall |   F1-Score | ROC-AUC |
| ------------------- | ---------: | ---------: | ---------: | ---------: | ------: |
| Logistic Regression |     55.00% |     51.47% |     33.90% |     0.4088 |  0.5517 |
| Random Forest       |     58.00% |     57.38% |     32.93% |     0.4185 |  0.6048 |
| SVM                 | **58.56%** | **58.55%** |     33.17% |     0.4235 |  0.5790 |
| Gradient Boosting   |     56.56% |     54.20% |     34.38% |     0.4207 |  0.5792 |
| **Proposed ANN**    |     56.44% |     53.35% | **40.44%** | **0.4601** |  0.5864 |

---

## 12. Results and Findings

The proposed ANN achieved:

* **Highest Recall:** 40.44%
* **Highest F1-Score:** 0.4601

Although the SVM achieved the highest accuracy of 58.56%, the ANN performed better in terms of recall and F1-score.

Recall is particularly important in vulnerability detection because a false negative represents a vulnerable function incorrectly classified as secure.

The results demonstrate that the ANN can capture nonlinear relationships between different source-code characteristics and security-related patterns.

---

## 13. Application

A **Streamlit-based web application** is developed for interactive vulnerability screening.

The application allows users to submit source code and receive:

* Vulnerability prediction
* Risk probability
* Security-pattern indicators
* Classification results

The system treats submitted code as text and does not execute it.

---

## 14. Security Safeguards

The application incorporates several security safeguards.

### Zero Dynamic Execution

User-submitted code is analyzed only as text. Functions such as `eval()` and `exec()` are not used to execute submitted code.

### Input Size Restriction

The application limits the size of submitted source code to prevent excessive resource consumption.

### Data Leakage Prevention

Feature preprocessing is fitted only on training data.

### Heuristic Verification

Security-sensitive patterns are presented separately from the ANN probability score to distinguish rule-based indicators from machine learning predictions.

---

## 15. Limitations

The project has several limitations:

### Syntactic and Semantic Limitations

The system primarily uses lexical, structural, and TF-IDF features. These features cannot completely capture complex program semantics, pointer relationships, or vulnerabilities involving multiple functions.

### Language Limitation

The dataset mainly consists of C/C++ systems software. Therefore, the model may not perform equally well on languages such as Python or JavaScript without additional training data.

### Machine Learning Limitation

The model provides vulnerability-risk predictions and should not be considered a replacement for:

* Static analysis
* Dynamic testing
* Fuzzing
* Manual security auditing
* Penetration testing

---

## 16. Conclusion

This project presents an **Artificial Neural Network-based approach for detecting security risks in AI-assisted and vibe-coded software**.

The system combines static source-code characteristics, security-sensitive patterns, and TF-IDF syntactic features to construct an 85-dimensional representation of source code. An ANN is then trained to classify functions as secure or vulnerable.

The proposed ANN achieved the highest recall and F1-score among the evaluated models, demonstrating its potential for identifying vulnerable code. The developed Streamlit application further provides an interactive method for screening source code and estimating security risk.

Overall, the project demonstrates how machine learning and neural networks can support automated software vulnerability detection and provide an additional security layer for rapidly generated AI-assisted software.
