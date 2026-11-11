# A Comparative Study of Classical Machine Learning Methods for Handwritten Digit Classification

**CS5487 Machine Learning — Course Project Proposal**

Group members: Shi Jixin, Li Tao, Zhang Zifan

### 1. Introduction

Handwritten digit recognition is a common multi-class classification problem in machine learning. Although the task appears relatively simple, variations in handwriting make some digits difficult to distinguish. This project aims to compare several classical machine learning methods on handwritten digit classification and investigate how dimensionality reduction affects classification performance and computational efficiency.

Specifically, we will explore two questions: (RQ1) How do different classical classifiers compare in terms of accuracy and classification errors? (RQ2) How does PCA affect classification performance and computational cost?

### 2. Dataset and Evaluation Protocol

We will use the provided `digits4000` dataset, containing 4,000 grayscale handwritten digit images (10 classes, 400 samples per class). Each image is represented as a 784-dimensional feature vector derived from its 28 × 28 pixel values.

The two predefined experimental trials will be used, each containing 2,000 training samples and 2,000 testing samples, with writers separated between the training and test sets. Hyperparameters will be selected through 3-fold stratified cross-validation using training data only. The test sets will be reserved for final evaluation, and PCA will be fitted within the training pipeline to avoid data leakage. Since individual writer identities are not provided for cross-validation, the internal validation will use stratified rather than writer-grouped splitting.

### 3. Planned Methods and Experiments

We will compare five classification methods: Euclidean 1-Nearest Neighbor (1-NN) as the baseline, Logistic Regression with L2 regularization, shrinkage Linear Discriminant Analysis (LDA), Linear SVM, and RBF-kernel SVM. Regularization parameters will be selected through cross-validation, with the RBF SVM also testing different kernel widths.

- **Experiment A – Classifier Comparison:** Evaluate all five classifiers on both experimental trials. We will compare their test accuracy and Macro-F1 scores and examine the selected hyperparameters.
- **Experiment B – PCA and Dimensionality Reduction:** Apply PCA to reduce the original 784-dimensional features to 50, 100, and 200 dimensions. Logistic Regression and RBF SVM will be evaluated on these representations and compared with their original-feature results. We will examine classification accuracy, explained variance, and training time.
- **Experiment C – Error Analysis:** Further analyze the RBF SVM predictions using confusion matrices and representative misclassified samples. We will pay particular attention to frequently confused digit pairs to understand the limitations of the classifiers.

### 4. Expected Outcomes and Success Criteria

We expect different classifiers to show different trade-offs in classification performance. Kernel-based methods may benefit from their ability to capture nonlinear decision boundaries, while PCA may reduce computational cost by using fewer features. However, the effect of dimensionality reduction on accuracy will be investigated experimentally rather than assumed in advance.

A successful project will provide a clear comparison of the selected classifiers, reproduce the provided 1-NN baseline, and identify the effects of PCA on classification performance. The conclusions will be supported by experimental results from both trials and further error analysis.