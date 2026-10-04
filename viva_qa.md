# Comprehensive Viva Voce Questions & Answers

### Project: Artificial Neural Network-Based Detection and Classification of Security Risks in Vibe-Coded Software
**College ANN Mini-Project (20 Marks)**

---

### Q1: What is "vibe coding", and why does it present software security risks?
**Answer:**  
"Vibe coding" is a developer workflow where programmers generate code rapidly using natural language prompts and AI coding assistants (like GitHub Copilot, Cursor, ChatGPT, or Claude) without thoroughly reading or understanding every line.  
This introduces critical security hazards because:
1. Large language models are trained on public open-source repositories that historically contain insecure coding patterns.
2. AI-generated code frequently lacks input bounds checks, memory zeroing, and error handling.
3. Developers, lulled by syntactically clean code, deploy these modules without security audits or static analysis.

---

### Q2: What is the primary objective of your Artificial Neural Network (ANN) project?
**Answer:**  
The objective is to develop an automated screening system that extracts static lexical, structural, cyclomatic complexity, and security-sensitive features from source code and predicts whether a given function is likely to contain a security vulnerability ($0 = \text{Secure}$, $1 = \text{Vulnerable}$). It assists developers by flagging risky code snippets before deployment.

---

### Q3: What dataset did you use, and where does it originate?
**Answer:**  
We used the **Microsoft CodeXGLUE Defect Detection benchmark**, which is based on the **Devign dataset** (*Zhou et al., NeurIPS 2019*). It contains 27,318 real-world C/C++ functions extracted from major open-source projects (**QEMU** and **FFmpeg**) labeled with ground-truth vulnerability tags derived from security patch commits and CVE advisories.

---

### Q4: Why did you extract a subset of 6,000 samples instead of using the entire 27,318 functions?
**Answer:**  
To ensure fast, reproducible local training on a standard college machine without GPU cluster dependencies. We extracted a stratified subset (4,200 Train / 900 Val / 900 Test) using a fixed random seed (`42`). Crucially, **stratification strictly preserved the authentic class ratio** (~45.8% vulnerable, ~54.2% secure) across all splits, ensuring complete scientific integrity without data distortion.

---

### Q5: How did you extract features from raw source code?
**Answer:**  
We engineered an **85-dimensional feature space** consisting of two complementary parts:
1. **35 Domain-Specific Numerical Metrics:**
   - **Lexical/Structural:** Character count, line count, average line length, blank lines, comment ratio, maximum indentation depth, token count.
   - **Control Flow/Cyclomatic Complexity:** Counts of `if`, `else`, `for`, `while`, `switch`, `case`, `break`, `return`, `goto`, and total branching nodes.
   - **Operators & Memory Safety:** Pointer dereferences (`->`, `*`), array indexing (`[`), dynamic memory calls (`malloc`, `free`), unsafe buffer functions (`strcpy`, `sprintf`, `gets`), command executions (`system`, `popen`), format string specifiers (`%s`, `%n`), and null checks.
2. **50 TF-IDF Syntactic Code Tokens:** Capturing high-frequency programming constructs, keywords, and variable identifiers fit strictly on the training partition.

---

### Q6: How did you prevent data leakage during preprocessing?
**Answer:**  
We strictly isolated the training partition:
- The `TfidfVectorizer` was fitted **exclusively on the training code**. The validation and test sets were transformed using the trained vocabulary.
- The `StandardScaler` computed means ($\mu$) and standard deviations ($\sigma$) **only from the training feature matrix**. The validation and test matrices were standardized using those training parameters.
- At no point did test set statistics influence feature extraction or scaling.

---

### Q7: Explain the architecture of your Artificial Neural Network.
**Answer:**  
Our ANN is a regularized multi-layer perceptron (MLP) built in TensorFlow/Keras:
- **Input Layer:** 85 neurons (matching the feature dimension).
- **Hidden Layer 1:** 128 neurons, ReLU activation.
- **Batch Normalization 1 + Dropout (0.3):** Normalizes activations and prevents co-adaptation.
- **Hidden Layer 2:** 64 neurons, ReLU activation.
- **Batch Normalization 2 + Dropout (0.2).**
- **Hidden Layer 3:** 32 neurons, ReLU activation + Dropout (0.2).
- **Output Layer:** 1 neuron, **Sigmoid activation** ($\sigma(z) = \frac{1}{1 + e^{-z}}$), outputting a continuous risk probability between 0.0 and 1.0.

---

### Q8: What loss function and optimizer did you use, and why?
**Answer:**  
- **Loss Function:** **Binary Cross-Entropy (Log Loss)**:
  $$\mathcal{L} = - \frac{1}{N}\sum [y \log \hat{y} + (1-y)\log(1-\hat{y})]$$
  It is the theoretically optimal maximum likelihood loss function for binary classification.
- **Optimizer:** **Adam (Adaptive Moment Estimation)** with an initial learning rate of 0.001. Adam maintains adaptive learning rates for each parameter based on first ($m_t$) and second ($v_t$) moments of the gradients, converging faster than standard Stochastic Gradient Descent (SGD).

---

### Q9: How did EarlyStopping and ReduceLROnPlateau behave during training?
**Answer:**  
- `EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)` monitored validation loss. Training halted early at **epoch 17**, restoring the optimal weights from **epoch 7**. This prevented overfitting on code noise.
- `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4)` reduced the learning rate from $10^{-3}$ to $5 \times 10^{-4}$ at epoch 11, and down to $2.5 \times 10^{-4}$ at epoch 15 when validation loss plateaued.

---

### Q10: What baseline models did you benchmark against your ANN?
**Answer:**  
We benchmarked the ANN against four classic machine learning algorithms trained on the identical preprocessed dataset:
1. **Logistic Regression** (linear baseline with L2 regularization)
2. **Random Forest Classifier** (ensemble of 150 decision trees)
3. **Support Vector Machine (SVM)** (RBF kernel, $C=1.0$)
4. **Gradient Boosting Classifier** (sequential boosting ensemble)

---

### Q11: What were the actual experimental results on the holdout test set?
**Answer:**  
All models were evaluated on the exact same 900 test samples (487 Secure, 413 Vulnerable):
- **Logistic Regression:** Accuracy: 55.00%, Precision: 51.47%, Recall: 33.90%, F1: 0.4088, ROC-AUC: 0.5517
- **Random Forest:** Accuracy: 58.00%, Precision: 57.38%, Recall: 32.93%, F1: 0.4185, ROC-AUC: **0.6048**
- **Support Vector Machine:** Accuracy: **58.56%**, Precision: **58.55%**, Recall: 33.17%, F1: 0.4235, ROC-AUC: 0.5790
- **Gradient Boosting:** Accuracy: 56.56%, Precision: 54.20%, Recall: 34.38%, F1: 0.4207, ROC-AUC: 0.5792
- **Proposed ANN:** Accuracy: 56.44%, Precision: 53.35%, **Recall: 40.44%**, **F1-Score: 0.4601**, ROC-AUC: 0.5864

---

### Q12: Why did the ANN achieve the highest Recall and F1-Score, and why is Recall critical in cybersecurity?
**Answer:**  
- In software security, a **False Negative** means letting an exploitable vulnerability pass into production undetected, leading to potential breaches. A **False Positive** merely requires a developer to verify the code. Therefore, **Recall (sensitivity)** is far more critical than raw Accuracy.
- The baseline models suffered from low recall (~32–34%), as tree splits and hyperplanes defaulted toward the slightly larger secure class.
- The ANN achieved **40.44% Recall** and the **highest F1-Score (0.4601)** because its multi-layer dense transformations learned non-linear feature interactions between cyclomatic complexity, pointer dereferencing, and security-sensitive API calls.

---

### Q13: Are accuracy scores around 56%–59% realistic for vulnerability detection?
**Answer:**  
**Yes, absolutely.** Real-world software vulnerability detection on uncurated open-source benchmarks like Devign is notoriously hard because vulnerabilities often depend on complex multi-file inter-procedural dataflows. In the original Devign paper published at NeurIPS 2019 (*Zhou et al.*), baseline machine learning models scored between 55% and 60% accuracy. Fabricating 95% accuracy on this dataset would be scientifically fraudulent. Our results are authentic and reflect genuine scientific benchmarks.

---

### Q14: How does your application protect itself from malicious code submitted by users?
**Answer:**  
Our application enforces strict secure coding principles:
1. **Zero Dynamic Execution:** Submitted code is strictly treated as passive string text. We **never** call `eval()`, `exec()`, or pass input to `os.system()` or `subprocess`.
2. **Length Ceiling:** Inputs are strictly capped at 50,000 characters to prevent Denial-of-Service (DoS) and memory exhaustion.
3. **Safe Error Handling:** Null, empty, or whitespace inputs return graceful validation errors rather than unhandled exceptions.
4. **Automated Security Tests:** Pytest tests verify that code containing malicious payloads cannot create files or execute system commands.

---

### Q15: What is the difference between your ANN prediction and the "Security Pattern Indicators"?
**Answer:**  
- The **ANN prediction** is a learned probabilistic model outputting a continuous score $P(Y=1 \mid \mathbf{x})$ based on mathematical feature weights learned from 6,000 training functions.
- The **Security Pattern Indicators** are static, deterministic heuristic rules (e.g., detecting `strcpy`, `system`, or hardcoded API keys).  
We deliberately separate them in the UI so the user understands that heuristic pattern checks are static screening rules, not the ANN's internal weights.

---

### Q16: Why can't your ANN definitively prove that code is secure?
**Answer:**  
Machine learning models are probabilistic, not formal verification engines. An ANN evaluates statistical correlates of known vulnerabilities in its training distribution. It cannot account for novel zero-day attack patterns, business logic flaws, or cross-file architectural vulnerabilities. Hence, we include a mandatory disclaimer: *“This prediction is generated by a machine-learning model and should not be treated as a replacement for professional security auditing or static analysis.”*

---

### Q17: What are the main limitations of your approach?
**Answer:**  
1. **Isolated Function Scope:** The model analyzes standalone functions and cannot trace inter-procedural variable taint flow across multiple files.
2. **Language Specificity:** The training benchmark is C/C++ systems code. Modern vibe-coded software also includes JavaScript and Python, which require domain-specific vulnerability data.
3. **Lexical Representation:** While TF-IDF and static counts capture syntax, they do not parse the full semantic Abstract Syntax Tree or Control Flow Graph.

---

### Q18: What would be the next steps to enhance this project?
**Answer:**  
1. **Code Transformers:** Use pre-trained models like **CodeBERT** or **GraphCodeBERT** to generate contextual embeddings of source code.
2. **Graph Neural Networks (GNNs):** Convert source code into Joint AST and Control Flow Graphs (as in the original Devign paper).
3. **Multi-Class CWE Prediction:** Expand from binary classification to multi-class classification to identify the exact vulnerability type (e.g., CWE-120 buffer overflow vs CWE-78 command injection).
4. **Developer Tooling:** Package the model into an IDE extension (VS Code) or pre-commit git hook.

---

### Q19: Which Python libraries did you use and why?
**Answer:**  
- **TensorFlow / Keras:** Building, training, and regularizing the deep neural network.
- **Scikit-Learn:** Baseline classifiers (Logistic Regression, Random Forest, SVM, Gradient Boosting), `StandardScaler`, `TfidfVectorizer`, and evaluation metrics.
- **Pandas & NumPy:** Tabular data processing and fast array manipulations.
- **PyArrow:** Efficient reading and writing of compressed Parquet dataset files.
- **Matplotlib & Seaborn:** Generating publication-quality training curves, confusion matrices, and ROC curves.
- **Streamlit:** Fast, clean, reactive web UI for live demonstration.
- **Pytest:** Automated testing and verification of safety constraints.

---

### Q20: How would you summarize the conclusion of your mini-project in one minute?
**Answer:**  
"In this 20-mark mini project, we addressed the critical software security risk introduced by AI-assisted 'vibe coding'. We retrieved the genuine CodeXGLUE Devign vulnerability dataset of 27,318 C/C++ functions and trained a deep Artificial Neural Network using 85 extracted lexical, cyclomatic, and syntactic features. We demonstrated that our ANN achieved a superior **Recall of 40.44%** and **F1-Score of 0.4601**, outperforming Logistic Regression, Random Forest, SVM, and Gradient Boosting in detecting elusive security risks. Finally, we deployed a safe, interactive Streamlit application with zero code execution, complete test coverage, and transparent academic disclaimers."
