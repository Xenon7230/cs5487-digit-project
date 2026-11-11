# A Comparative Study of Classical Machine Learning Methods for Handwritten Digit Classification

**CS5487 Machine Learning — Course Project Proposal**

Group members: ____________________  |  Date: ____________

## 1. Introduction

Handwritten digit recognition is a canonical multi-class problem and a natural testbed for the methods covered in this course. Deep networks dominate benchmarks, but classical classifiers remain valuable for understanding how regularization, margins, kernels and dimensionality reduction interact, and they are cheap and interpretable. We compare a fixed set of classical classifiers on the `digits4000` subset of MNIST under the instructor-provided, writer-separated protocol, asking **(RQ1)** how classical classifiers differ in accuracy and error patterns on held-out writers, and **(RQ2)** how PCA affects accuracy and computational cost.

## 2. Data and evaluation protocol

We use 4,000 28×28 grayscale digits (10 classes, 400 each), each stored as a 784-dimensional pixel vector in [0, 255], scaled deterministically to [0, 1]. The instructor's two predefined trials (2,000 train / 2,000 test each, with disjoint writers) are used exactly as given. Hyper-parameters are selected by 3-fold stratified cross-validation on the training set only, and each test set is evaluated once. PCA is fitted inside the CV pipeline to prevent leakage. We first reproduce the reference Euclidean 1-NN accuracy (91.35% / 91.85%) as a sanity check.

## 3. Planned methods

The five configurations cover five method families — instance-based, generative, linear discriminative, linear margin and kernel-based:

| Config | Method | Hyper-parameters |
|---|---|---|
| `knn_1` | Euclidean 1-NN baseline | fixed |
| `logreg` | Logistic regression, L2 | `C` ∈ {0.1, 1, 10} |
| `lda` | Shrinkage LDA | fixed (`lsqr`, `shrinkage='auto'`) |
| `svm_linear` | Linear SVM | `C` ∈ {0.1, 1, 10} |
| `svm_rbf` | RBF SVM | `C` ∈ {0.1, 1, 10} × `gamma` ∈ {0.001, 0.01, 0.1} |

## 4. Experiments

- **A (benchmark):** all five configurations on both trials; test accuracy, macro-F1, fit/predict time and selected parameters.
- **B (PCA):** logistic regression and RBF SVM with `n_components` ∈ {None, 50, 100, 200}; accuracy, cumulative explained variance and training time.
- **C (analysis):** RBF SVM, fixed in advance as the qualitative target — training-set CV heatmaps of the `C × gamma` grid, 10×10 confusion matrices, misclassified examples and frequent confusion pairs.

## 5. Expected outcomes and definition of success

We expect the 1-NN check to match the reference within 0.5 points; the RBF SVM to be strongest, plausibly in the low-to-mid 90s; shrinkage to make LDA usable, though its Gaussian assumption may trail discriminative methods; and PCA at 50–100 components to retain most accuracy (top components capturing roughly 80–95% of variance) while cutting training time several-fold. We do not assume PCA improves accuracy, and will report results regardless. Success means: baseline reproduced; the locked matrix completed without test-set tuning; leakage-free protocol verified; one-command reproducibility with a fixed seed.

## 6. Scope and limitations

Grids are intentionally small, so comparisons describe this configuration family. Without per-sample writer identities, inner CV is stratified rather than writer-grouped; only the final splits are writer-separated. Timing is environment-dependent. Deep learning, AutoML and external data are out of scope; third-party libraries will be cited in the final report.

**Work plan:** Weeks 10–11 pipeline and baseline; Week 12 Experiments A–B; Week 13 analysis and verification; Week 14 report and poster.
