# 🕵️‍♂️ Unsupervised Anomaly Detection for Financial Transaction Fraud

## 📌 Table of Contents

1. [Introduction](#1-introduction)
2. [Solution: Isolation Forest](#2-solution-isolation-forest)
3. [Dataset](#3-dataset)
4. [Methodology](#4-methodology)
5. [Key Findings](#5-key-findings)
6. [Limitations](#6-limitations)
7. [Future Work](#7-future-work)
8. [How to Run](#8-how-to-run)
9. [Technologies Used](#9-technologies-used)
10. [Author](#10-author)

---

## 1. Introduction

Financial institutions face increasing fraud risks, with traditional rule-based systems often lagging behind evolving fraudulent behavior. This project addresses:

- **Extreme class imbalance** (fraud < 0.2%)
- **Lack of labeled examples for new fraud**
- **High false positives**
- **Manual review burden**

## 2. Solution: Isolation Forest

We implement an **unsupervised anomaly detection** approach using the `Isolation Forest` algorithm. It works by isolating rare and different data points (anomalies) using random feature splits.

### ✅ Advantages:

- Detects novel fraud without labels
- Handles class imbalance effectively
- Reduces reliance on hardcoded rules
- Prioritizes cases for human review

## 3. Dataset

- 📊 **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 📀 284,807 transactions (only 492 frauds)
- 🔒 Features: V1–V28 (PCA-transformed), Time, Amount
- 🧪 `Class` label is used **only for evaluation**

## 4. Methodology

### 🔍 Data Preprocessing

- Dropped `Time`
- Scaled `Amount` using `StandardScaler`

### 🌲 Model Training

- Used `IsolationForest` from scikit-learn
- Key hyperparameter: `contamination` set to fraud rate (\~0.00172)

### 📈 Prediction

- Anomalies predicted as `-1`
- Used `decision_function` to score anomaly likelihood
- Created `is_anomaly` column for easier evaluation

## 5. Key Findings

| Contamination | Precision | Recall | F1-Score | Anomalies Predicted | Interpretation                    |
| ------------- | --------- | ------ | -------- | ------------------- | --------------------------------- |
| 0.0017        | 0.2825    | 0.2825 | 0.2825   | 492                 | Baseline                          |
| 0.0010        | 0.3684    | 0.2134 | 0.2703   | 285                 | High precision, lower recall      |
| 0.0020        | 0.2561    | 0.2967 | 0.2750   | 570                 | Slightly aggressive               |
| 0.0050        | 0.1608    | 0.4654 | 0.2390   | 1424                | High recall, many false positives |

### ♻️ Trade-off:

Choose contamination based on business priorities — precision vs. recall.

### 📊 Visualizations (saved in `results/`)

- `anomaly_score_distribution.png`
- `v1_amount_anomaly_scatter.png`
- `confusion_matrix.png`

## 6. Limitations

- No learning from labels (purely unsupervised)
- Hard to interpret PCA-transformed features
- No use of temporal patterns (e.g., transaction frequency over time)
- The `contamination` value is heuristic and sensitive

## 7. Future Work

- ✅ Semi-supervised learning (e.g., One-Class SVM with partial labels)
- ✅ Feature engineering from `Time`
- ✅ Model ensembles (e.g., combining LOF, Isolation Forest)
- ✅ Real-time API simulation for deployment
- ✅ Interactive web dashboard for visual insights

## 8. How to Run the Project

To run this project locally and explore the analysis, follow these steps:

### 📥 Clone the Repository:

```bash
git clone https://github.com/your-username/unsupervised-fraud-detection.git
cd unsupervised-fraud-detection
```

### 🛠️ Create and Activate Virtual Environment:

```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 📦 Install Dependencies:

```bash
pip install -r requirements.txt
```

(Generate `requirements.txt` if you haven't already by running `pip freeze > requirements.txt` in your activated environment.)

### 📁 Download Dataset:

1. Go to the Kaggle dataset page: **Credit Card Fraud Detection**
2. Download `creditcard.csv`
3. Place the `creditcard.csv` file in the **root directory** of your cloned repository (i.e., next to `README.md`)

### ▶️ Run the Main Analysis Script:

The core analysis, plotting, and evaluation are integrated into `analysis_and_iteration.py`:

```bash
python analysis_and_iteration.py
```

### 🧪 For Interactive Exploration:

Open the following Jupyter notebooks:

- `eda.ipynb`: For initial data exploration
- `data_preprocessing.ipynb`: For data cleaning and scaling
- `train_eval.ipynb`: For model training and initial evaluation
- `hyperparameter_tuning.ipynb`: For experimenting with the contamination parameter
- `analysis_and_iteration.py`: For running the complete flow and saving plots

The `analysis_and_iteration.py` script will print outputs to the console and save plots (anomaly score distribution, V1 vs. Amount scatter plot, confusion matrix) in the `results/` directory.

## 9. Technologies Used

- **Python 3.8+**
- **Pandas**, **NumPy** – Data manipulation
- **Scikit-learn** – Isolation Forest, metrics
- **Matplotlib**, **Seaborn** – Visualizations
- **Git & GitHub** – Version control and hosting

## 10. Author

**[Your Name]**\
📧 Email: [your-email@example.com](mailto\:your-email@example.com)\
🔗 LinkedIn: [linkedin.com/in/your-profile](https://linkedin.com/in/your-profile)

---

> ⭐️ Star this repo if you found it useful. Contributions and feedback are welcome!

