# Code Difficulty Classification for IOAI TST Kazakhstan — Hybrid BERT & XGBoost Pipeline

This repository contains a solution for the Code Difficulty Classification task from the third day of the IOAI TST (Team Selection Test) in Kazakhstan. The goal is to categorize programming code snippets into three difficulty levels: **Easy**, **Medium**, and **Hard**.

## Table of Contents
- [Problem Description](#problem-description)
- [Solution Approach](#solution-approach)
- [Feature Engineering & Extraction](#feature-engineering--extraction)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Dependencies](#dependencies)
- [Usage](#usage)

---

## Problem Description
The challenge is a multi-class text classification problem where the input is source code. Determining code difficulty requires capturing both the semantic meaning of the logic (via NLP) and structural complexity (via statistical features).

## Solution Approach
This solution employs a **hybrid ensemble strategy** that combines deep learning representations with classical machine learning features:
1.  **Semantic Layer:** A fine-tuned BERT-based model to understand the context and logic of the code.
2.  **Statistical Layer:** TF-IDF vectorization to capture the frequency of specific keywords and programming patterns.
3.  **Structural Layer:** Handcrafted features (code length, number of functions, loops, etc.) to measure complexity.
4.  **Ensemble Layer:** An XGBoost classifier that aggregates all the above features to make the final prediction.

## Feature Engineering & Extraction
To maximize performance, the pipeline extracts three types of data:
* **BERT Embeddings:** Fine-tuned on the competition dataset to extract high-level semantic features from the code snippets.
* **TF-IDF Features:** Applied to the raw code text to identify distinctive tokens associated with different difficulty levels.
* **Text-Based Features:** Manual extraction of metrics such as:
    * Line count and character count.
    * Frequency of complex keywords (e.g., `recursion`, `nested loops`).
    * Ratio of comments to code.

## Model Architecture
The final model is an **Ensemble Classifier (XGBoost)**. It takes the concatenated vector of BERT probabilities, TF-IDF scores, and handcrafted features as input. This approach allows the model to benefit from the strengths of both Transformers (contextual understanding) and Gradient Boosting (efficiency with tabular/structured data).

## Results
The model demonstrated strong performance by balancing the precision and recall across all three classes.

* **Final F1-Score (Evaluation Metric): 0.39765**

## Dependencies
The solution requires the following Python libraries:
* `torch` & `transformers` (Hugging Face)
* `datasets`
* `xgboost`
* `scikit-learn`
* `pandas`
* `numpy`

```bash
pip install torch transformers datasets xgboost scikit-learn pandas numpy
