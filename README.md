# 🛡️ Isolation Forest for Anomaly Detection on Thyroid Dataset

# 📌 Project Overview

This project demonstrates **Anomaly Detection** using the **Isolation Forest** algorithm on a **Thyroid Dataset**.

Isolation Forest is an **unsupervised machine learning algorithm** that detects anomalies by randomly partitioning data. Observations that are isolated with fewer splits are considered anomalies.

The project also visualizes the detected anomalies using **Principal Component Analysis (PCA)**.

---

# 🎯 Objectives

- 🔍 Detect abnormal observations automatically
- 📊 Reduce dimensions using PCA
- 📈 Visualize anomalies in 2D space
- 🏥 Demonstrate anomaly detection for healthcare data
- ⚡ Learn one of the most popular outlier detection algorithms

---

# 🚀 Project Workflow

```
Load Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Feature Scaling
      │
      ▼
Isolation Forest Model
      │
      ▼
Predict Outliers
      │
      ▼
Principal Component Analysis
      │
      ▼
2D Visualization
      │
      ▼
Count Normal & Anomalous Records
```

---

# 📂 Dataset

Dataset Used

- 🏥 Thyroid Dataset

Target Column

- **Outlier_label**

Features

- Multiple numerical medical attributes used for anomaly detection.

---

# 🛠️ Technologies Used

- 🐍 Python
- 📊 Pandas
- 🔢 NumPy
- 📉 Matplotlib
- 🤖 Scikit-Learn
- 📐 StandardScaler
- 📊 PCA
- 🌲 Isolation Forest

---

# 📚 Libraries

```python
pandas
numpy
matplotlib
scikit-learn
```

Install dependencies

```bash
pip install pandas numpy matplotlib scikit-learn
```

---

# 📈 Project Steps

## ✅ Import Required Libraries

Import libraries for preprocessing, visualization, PCA, and anomaly detection.

---

## ✅ Load Dataset

Read the Thyroid dataset using Pandas.

---

## ✅ Separate Features and Target

- Feature matrix (X)
- Target labels (y)

---

## ✅ Standardize Features

StandardScaler normalizes the feature values before training the model.

---

## ✅ Train Isolation Forest

The model is initialized with:

```python
IsolationForest(
    n_estimators=200,
    contamination="auto",
    random_state=42
)
```

### Parameter Explanation

- 🌲 **n_estimators = 200**
  - Builds 200 isolation trees for stable anomaly detection.

- 📉 **contamination = "auto"**
  - Automatically estimates the proportion of anomalies.

- 🎲 **random_state = 42**
  - Ensures reproducible results.

---

## ✅ Predict Anomalies

The model classifies every observation as:

| Prediction | Meaning |
|------------|---------|
| **1** | Normal Observation |
| **-1** | Anomaly (Outlier) |

---

## ✅ Dimensionality Reduction

PCA converts the high-dimensional dataset into **2 principal components** for visualization.

---

## ✅ Data Visualization

Scatter plot showing:

- 🟢 Normal observations
- 🔴 Detected anomalies

---

## ✅ Count Outliers

The project calculates:

- Total Normal Records
- Total Outliers

---

# 📊 Output

The project produces:

- ✅ Scaled Dataset
- ✅ Isolation Forest Model
- ✅ Anomaly Predictions
- ✅ PCA Reduced Dataset
- ✅ Scatter Plot
- ✅ Number of Outliers
- ✅ Number of Normal Records

---

# 📈 Sample Visualization

```
                 PC2
                  ▲

        ● ● ● ●

     ● ● ●  ✖

  ● ● ● ● ●

      ✖

────────────────────────────► PC1

● = Normal Observation
✖ = Outlier
```

---

# 💡 Learning Outcomes

After completing this project, you will understand:

- ✅ Unsupervised Machine Learning
- ✅ Anomaly Detection
- ✅ Isolation Forest Algorithm
- ✅ Feature Scaling
- ✅ PCA for Visualization
- ✅ Outlier Detection Workflow
- ✅ Healthcare Data Analysis

---

# 🌍 Real-World Applications

🏦 Fraud Detection

💳 Credit Card Fraud

🏥 Disease Detection

🌐 Cybersecurity Intrusion Detection

🏭 Manufacturing Quality Inspection

🚗 IoT Sensor Monitoring

📈 Financial Risk Analysis

📡 Network Traffic Monitoring

---

# 📁 Project Structure

```
Isolation-Forest-Anomaly-Detection/
│
├── Isolation_Forest.py
├── thyroid_dataset.csv
├── README.md
└── requirements.txt
```

---

# ▶️ Run the Project

Clone the repository

Move into the project folder

```bash
cd Isolation-Forest-Anomaly-Detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the project

```bash
python Isolation_Forest.py
```

---

# 📌 Future Improvements

- 📈 Interactive anomaly visualization with Plotly
- 📊 Compare Isolation Forest with One-Class SVM
- 🌲 Compare with Local Outlier Factor (LOF)
- 📉 Precision, Recall, and F1-score evaluation
- 🤖 Hyperparameter tuning using GridSearchCV
- 🌐 Deploy using Streamlit

---
