# Slide Presentation: GPU-Accelerated CKKS for Neural Networks

**Presenter:** Rafael Eduardo Contreras 
**Estimated Duration:** 45-50 Minutes (Expanded)

---

### **Part 1: Introduction & Motivation**
* **Slide 1: Title & Authorship**
    * Title: Python Library for CKKS Homomorphic Encryption Tensor Operations Accelerated by GPU.
    * Details: Universidad Central de Venezuela, Faculty of Sciences.
* **Slide 2: The Privacy Dilemma**
    * Explosion of Machine Learning (ML) in sensitive areas like medicine and finance.
    * The risk of data leaks when using third-party cloud services for inference.
* **Slide 3: Homomorphic Encryption (HE) as a Solution**
    * Definition: Computing directly on encrypted data.
    * The result, when decrypted, matches the output of the same operation on plaintext.

---

### **Part 2: Fundamentals of HE & Modern Schemes**
* **Slide 4: Origins and Evolution**
    * Proposed in 1978 as "Privacy Homomorphisms".
    * Evolution from Partially (RSA, Paillier) to Fully Homomorphic Encryption (FHE).
* **Slide 5: The "Noise" Problem**
    * Encryption involves a small "error" or noise for security.
    * Multiplication causes noise to grow much faster than addition.
    * If noise exceeds a threshold, decryption fails.
* **Slide 6: Modern FHE Schemes (Comparison)**
    * **BGV / BFV:** Exact arithmetic over integers (Zq). Modulus/Scale switching.
    * **TFHE:** Fast bootstrapping, gate-by-gate (AND, OR, MUX). Ideal for boolean logic.
    * **CKKS:** Approximate arithmetic for real/complex numbers.
* **Slide 7: Learning With Errors (LWE)**
    * Mathematical foundation: Difficulty of finding a secret vector in a noisy linear system.
    * Security based on lattice problems.
* **Slide 8: Ring-Learning With Errors (RLWE)**
    * Extension of LWE to polynomial rings ($Rq = Zq[x] / (x^n + 1)$).
    * Efficiency: Faster multiplication via NTT and compact ciphertexts.
    * Enables SIMD (parallel) operations.

---

### **Part 3: The CKKS Scheme in Depth**
* **Slide 9: Paradigm Shift: Approximate Arithmetic**
    * Unlike BGV/BFV, CKKS handles real and complex numbers natively.
    * Noise is treated like a rounding error in standard floating-point arithmetic.
* **Slide 10: Critical Parameters in CKKS**
    * Polynomial Degree ($N$): Affects security and number of slots.
    * Modulus ($Q$): Determines the "budget" for operations.
    * Scaling Factor ($\Delta$): Precision of the fixed-point representation.
* **Slide 11: Encoding & SIMD**
    * Canonical Embedding: Packing $N/2$ complex numbers into one polynomial.
    * Single Instruction, Multiple Data (SIMD) advantage.
* **Slide 12: Leveled CKKS**
    * Modulus Chain: $Q_L > Q_{L-1} > \dots > Q_0$.
    * Rescaling: Reducing the modulus to manage noise after multiplication.
* **Slide 13: Bootstrapped CKKS**
    * Evaluation of the decryption circuit as a homomorphic function.
    * "Refreshing" the ciphertext to reset noise and continue computing.

---

### **Part 4: Neural Networks Fundamentals**
* **Slide 14: Artificial Neurons & Layers**
    * Structure: Input, Weights, Bias, and Activation Function.
* **Slide 15: Superficial vs. Deep Neural Networks**
    * Shallow: Single hidden layer.
    * Deep (DNN): Multiple hidden layers for hierarchical feature learning.
* **Slide 16: Convolutional Neural Networks (CNN)**
    * Spatial invariance and parameter sharing.
    * Kernels, Stride, Padding, and Channels.
* **Slide 17: Training vs. Inference**
    * Training: High cost, backpropagation, requires labels (usually plaintext).
    * Inference: Focus of this work. Using pre-trained weights to predict on encrypted data.

---

### **Part 5: Neural Networks over CKKS**
* **Slide 18: Workflow for Private Inference**
    * Encryption of input $\rightarrow$ Remote evaluation $\rightarrow$ Decryption of result.
* **Slide 19: Linear Operations & Diagonalization**
    * Matrix-Vector multiplication in HE.
    * Optimizing rotations to minimize computational cost.
* **Slide 20: Convolutions on CKKS**
    * Implementation via **Im2Col** transformation.
    * Mapping spatial convolutions to optimized matrix multiplications.
* **Slide 21: Approximating Activation Functions**
    * The challenge: HE only supports polynomials (+, *).
    * Strategy: Replacing ReLU/Sigmoid with Taylor series or least-squares polynomials (e.g., $x^2$).

---

### **Part 6: Proposed GPU Library**
* **Slide 22: Limitations of Current Tools**
    * Most libraries (SEAL, HELib) rely exclusively on the CPU.
    * HE is computationally intensive; CPU bottlenecks make large models non-viable.
* **Slide 23: Technology Stack**
    * **FIDESlib:** Backend for CUDA/GPU-accelerated CKKS.
    * **OpenFHE-Python & Pybind11:** Bridging C++ performance with Python usability.
* **Slide 24: Objective & Proposed Library**
    * Creating a high-level Python interface for encrypted tensor operations on GPU.
    * Abstracting cryptographic complexity for ML researchers.
* **Slide 25: Implementation & Benchmarking Plan**
    * Unit tests for tensor operations.
    * Performance comparison: TenSEAL (CPU) vs. Concrete ML (GPU-TFHE) vs. Proposed (GPU-CKKS).
* **Slide 26: Summary & Future Work**
    * Enabling real-world private AI via hardware acceleration.
