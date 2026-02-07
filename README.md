# Airbnb Property Delisting Prediction (NLP)

The goal of this project is to use Natural Language Processing (NLP) models to predict whether a property listed on Airbnb will be unlisted in the next quarter. This analysis utilizes real Airbnb property descriptions, host biographies, and guest reviews to identify patterns associated with property churn.

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Data Preprocessing](#2-data-preprocessing)
3. [Feature Engineering](#3-feature-engineering)
4. [Installation & Setup](#4-installation--setup)
5. [Classification Models](#5-classification-models)
6. [Results & Evaluation](#6-results--evaluation)
7. [Conclusion](#7-conclusion)

---

## 1. Project Overview
This project addresses a binary classification problem:
* **Class 0:** Property remains listed.
* **Class 1:** Property becomes unlisted (churn).

The pipeline covers the entire lifecycle of an NLP project, from multilingual language detection to fine-tuning neural networks and evaluating Transformer-based embeddings.

---

## 2. Data Preprocessing

### 2.1 Language Detection & Cleaning
* **Hybrid Detection:** We employed `langdetect` and `langid`. A confidence threshold of 0.85 was used to decide between primary and secondary detection results.
* **Text Normalization:** Stripped HTML tags and non-alphabetic characters using Regular Expressions.
* **Stop-word Removal & Lemmatization:** Applied across 8 languages (English, French, Portuguese, Spanish, German, Italian, Dutch, and Russian) to standardize the text.

### 2.2 Data Integrity
To maintain data integrity, missing values in property and host descriptions were filled with an `"empty"` placeholder. This ensures that properties are not dropped simply because a specific text field was missing.



---

## 3. Feature Engineering
We compared five distinct methods to convert text into numerical vectors:

1. **Bag-of-Words (BoW):** Focuses on word frequency.
2. **TF-IDF:** Weights words based on their unique importance across the dataset.
3. **Word2Vec:** Captures semantic relationships (100-dimensional vectors).
4. **BERT (bert-base-multilingual-cased):** Generates deep contextual embeddings.
5. **LaBSE:** Specifically optimized for high-quality multilingual sentence alignment.

---

## 4. Installation & Setup

### 1. Clone the repository
```bash
git clone [https://github.com/your-username/Airbnb-predicting-property-delisting-NLP-](https://github.com/your-username/Airbnb-predicting-property-delisting-NLP-)
cd Airbnb-predicting-property-delisting-NLP-
```

### 2. Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```


## 5. Classification Models – Model Testing

To identify the most effective model and feature extraction method, we performed a comprehensive evaluation using **TF-IDF** and **LaBSE** (our top two performing extraction techniques). 

### 5.1 Training Methodology
* **Hyperparameter Tuning:** We employed `RandomizedSearchCV` to efficiently explore combinations of hyperparameters.
* **Cross-Validation:** 5-fold cross-validation was used consistently to ensure the models generalized well and to mitigate overfitting.
* **Model Selection:** We tested a variety of architectures, from linear models to deep learning and ensemble methods.

---

## 6. Results & Evaluation

### 6.1 Feature Extraction Comparison
Before finalizing our classifiers, we evaluated how different feature extraction methods performed using a baseline Logistic Regression model.

| Metric | BoW | TF-IDF | Word2Vec | BERT | LaBSE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Mean Accuracy** | 0.858 | **0.886** | 0.874 | 0.859 | 0.878 |
| **Mean Precision** | 0.732 | **0.769** | 0.739 | 0.742 | 0.755 |
| **Mean Recall** | 0.760 | 0.835 | **0.837** | 0.745 | 0.820 |
| **Mean F1 Score** | 0.745 | **0.800** | 0.785 | 0.743 | 0.786 |

> **Observation:** TF-IDF provided the best overall performance, followed closely by LaBSE.

### 6.2 Model Performance Summary (Weighted F1)
We compared the top models using both TF-IDF and LaBSE features.

| Feature Set | LogReg | KNN | MLP | XGBoost | **Average** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF** | **0.8038** | 0.7916 | 0.7950 | 0.7918 | **0.7956** |
| **LaBSE** | 0.7841 | 0.7965 | 0.7870 | 0.8013 | 0.7922 |

### 6.3 Performance Comparison (ROC AUC Curve)
To visualize the performance of all models simultaneously, we utilized the Receiver Operating Characteristic (ROC) curve. This comparison allows us to see how each model balances the True Positive Rate against the False Positive Rate across different thresholds.


![ROC AUC Curve for All Models Comparison](figures/AUC_models.png)
*Figure 1: Comparison of ROC Curves for Logistic Regression, KNN, MLP, XGBoost, and DistilBERT. The Area Under the Curve (AUC) confirms which models provide the most reliable separation between classes.*

### 6.4 Best Performing Model and Test Data Predictions
The **MLP Classifier** with **TF-IDF** features was selected as the final model due to its robust non-linear learning capabilities.

**Optimal Hyperparameters:**
* `hidden_layer_sizes`: (30)
* `activation`: 'tanh'
* `solver`: 'sgd'
* `learning_rate`: 'adaptive'
* `alpha`: 0.3594

Final predictions on the unseen test set resulted in **489** properties predicted as "Listed" (Class 0) and **206** as "Unlisted" (Class 1).
---

## 7. Conclusion

This project highlights the efficiency of traditional NLP techniques when applied to multilingual real-world data. While Transformer models like **DistilBERT** and **LaBSE** are highly sophisticated, **TF-IDF** combined with a **Multi-Layer Perceptron (MLP)** delivered the most reliable and robust results for predicting Airbnb property delistings.

**Key Takeaways:**
* **Simplicity vs. Power:** TF-IDF remains a strong contender for text classification due to its ability to capture term importance with low computational overhead.
* **Non-Linearity:** The MLP’s ability to learn complex patterns allowed it to edge out simpler models like Logistic Regression.
* **Multilingual Success:** Our preprocessing pipeline successfully handled 8 different languages, providing a clean foundation for feature extraction.

Final predictions on the unseen test set resulted in **489** properties predicted as "Listed" and **206** as "Unlisted."
