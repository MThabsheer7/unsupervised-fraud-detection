import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv(r".\data\predictions.csv")

print("\n--- Starting Phase 3: Analysis and Visualization ---")
# --- 3.1: Visualize the Distribution of Anomaly Scores ---
print("\nPlotting distribution of anomaly scores...")
plt.figure(figsize=(12, 6))
sns.histplot(df['anomaly_score'], bins=50, kde=True, color='purple')
plt.title('Distribution of Anomaly Scores from Isolation Forest', fontsize=16)
plt.xlabel('Anomaly Score (Lower values = More Anomalous)', fontsize=12)
plt.ylabel('Frequency / Density', fontsize=12)
# Add a vertical line at the threshold where samples become anomalies (-1)
# The threshold for predict() is derived from the contamination parameter.
# We can find the max score of predicted anomalies, which defines the boundary.
anomaly_threshold = df[df['is_anomaly'] == 1]['anomaly_score'].max()
plt.axvline(x=anomaly_threshold, color='red', linestyle='--', linewidth=2,
            label=f'Anomaly Threshold (max score of predicted anomalies): {anomaly_threshold:.4f}')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
#plt.show()
#plt.savefig('results/anomaly_score_distribution.png') # Save to results directory
print("Anomaly score distribution plot saved as 'results/anomaly_score_distribution.png'")

# --- 3.2: Scatter Plot of Key Features with Anomaly Status and Actual Class ---
# This plot helps visualize if predicted anomalies align with actual frauds in the feature space.
# We'll use V1 and Amount, as V1 often shows clear separation for fraud and Amount is crucial.
print("\nPlotting V1 vs. Amount colored by predicted anomaly and styled by actual class...")
plt.figure(figsize=(14, 8))

#plot legitimate transactions
sns.scatterplot(x=df['V1'][df['Class'] == 0], y=df['Amount'][df['Class'] == 0], color='blue', label='Actual Legitimate (Class 0)', alpha=0.1, s=10)
sns.scatterplot(x=df['V1'][df['Class'] == 1], y=df['Amount'][df['Class'] == 1],
                color='green', label='Actual Fraud (Class 1)', alpha=0.8, s=40, marker='X')
# Overlay predicted anomalies. We want to see how many of these align with the green 'X's.
sns.scatterplot(x=df['V1'][df['is_anomaly'] == 1], y=df['Amount'][df['is_anomaly'] == 1],
                edgecolor='red', facecolor='none', linewidth=1.5, s=100, marker='o',
                label='Predicted Anomaly', alpha=0.6)
plt.title('V1 vs. Scaled Amount: Predicted Anomalies vs. Actual Classes', fontsize=16)
plt.xlabel('V1 (Principal Component)', fontsize=12)
plt.ylabel('Scaled Amount', fontsize=12)
plt.legend(title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
#plt.show()
#plt.savefig('results/v1_amount_anomaly_scatter.png') # Save to results directory
#print("V1 vs. Amount scatter plot saved as 'results/v1_amount_anomaly_scatter.png'")

# --- 3.3: Formal Classification Report and Confusion Matrix ---
print("\n--- Generating Formal Classification Report and Confusion Matrix ---")
# Classification Report: Provides precision, recall, f1-score for each class
# Note: For unsupervised, 'is_anomaly' is our prediction, 'Class' is our true label.
# We care about the '1' class (Fraud/Anomaly).
print("\nClassification Report (Predicted Anomaly vs. Actual Fraud):")
# Ensure the target_names are correct: 0 is Legitimate, 1 is Fraud
y_true = pd.read_csv(r".\data\y_true.csv")
print(classification_report(y_true, df['is_anomaly'], target_names=['Legitimate', 'Fraud']))

# Confusion Matrix: Visualizes True Positives, False Positives, False Negatives, True Negatives
conf_matrix = confusion_matrix(y_true, df['is_anomaly'])
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', linewidths=.5,
            xticklabels=['Predicted Legitimate', 'Predicted Anomaly'],
            yticklabels=['Actual Legitimate', 'Actual Fraud'])
plt.title('Confusion Matrix for Anomaly Detection', fontsize=16)
plt.ylabel('Actual Class', fontsize=12)
plt.xlabel('Predicted Class', fontsize=12)
plt.tight_layout()
plt.show()
plt.savefig('results/confusion_matrix.png') # Save to results directory
print("Confusion matrix plot saved as 'results/confusion_matrix.png'")

print("\n--- Phase 3: Analysis and Visualization Complete ---")