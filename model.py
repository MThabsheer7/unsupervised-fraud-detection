import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from loguru import logger
import pickle
import os 

X = pd.read_csv(r".\data\preprocessed_data.csv")
y_true = pd.read_csv(r".\data\y_true.csv")

logger.info(f"Data loaded: X.shape: {X.shape}, y_true.shape: {y_true.shape}")

logger.info("\n--- Starting Isolation Forest Model Training ---")

# Calculate the actual fraud percentage from y_true
# This will be used to set the 'contamination' parameter.
# For a truly unsupervised scenario where labels are unknown, this value would be estimated or set based on domain knowledge/risk tolerance.
fraud_percentage = y_true.value_counts(normalize=True)[1]
logger.info(f"Calculated fraud percentage in the dataset: {fraud_percentage:.4f} ({fraud_percentage*100:.2f}%)")

# Initialize Isolation Forest model
# n_estimators: Number of trees in the forest. More trees generally lead to more stable results.
# contamination: The proportion of outliers in the dataset. This is crucial for defining the decision boundary.
# random_state: For reproducibility of results.
# n_jobs=-1: Use all available CPU cores for faster training.
iso_forest = IsolationForest(
    n_estimators=100,
    contamination=fraud_percentage,  # Setting based on known fraud rate for initial evaluation
    random_state=42,
    n_jobs=-1
)

# Fit the model to the data
# For unsupervised learning, we only fit on the features (X).
# The model learns the "normal" patterns and identifies deviations.
logger.info("Fitting Isolation Forest model to the feature data (X)...")
iso_forest.fit(X)
logger.success("Isolation Forest model training complete.")

# --- 2.3: Anomaly Prediction and Scoring ---
logger.info("\n--- Generating Anomaly Predictions and Scores ---")

# --- 2.3: Anomaly Prediction and Scoring ---
logger.info("\n--- Generating Anomaly Predictions and Scores ---")
predictions = iso_forest.predict(X)
logger.info(f"Shape of predictions array: {predictions.shape}")

# Get the anomaly scores: Lower score indicates higher anomaly likelihood
# The `decision_function` returns the raw anomaly score.
# Higher score = more normal; Lower score = more anomalous.
anomaly_scores = iso_forest.decision_function(X)
logger.info(f"Shape of anomaly scores array: {anomaly_scores.shape}")

# Add predictions and scores back to a DataFrame for easier analysis and comparison
# It's good practice to add these to a copy of the original df or create a results df.
df = pd.read_csv(r".\data\creditcard.csv")
df['anomaly_prediction'] = predictions
df['anomaly_score'] = anomaly_scores

# Create a boolean/integer column for easy identification of predicted anomalies (1 if anomaly, 0 if normal)
df['is_anomaly'] = df['anomaly_prediction'].apply(lambda x: 1 if x == -1 else 0)

logger.info("\nFirst 10 rows with added anomaly predictions and scores:")
logger.info(df[['Amount', 'Class', 'anomaly_score', 'anomaly_prediction', 'is_anomaly']].head(10))

logger.success("--- Anomaly Prediction and Scoring Complete ---")
df.to_csv(r".\data\results.csv", index=False)

# Assuming 'iso_forest' is your trained IsolationForest model instance
# from the previous step.

logger.info("\n--- Saving the Trained Model ---")

# Define the directory to save models
model_dir = 'models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir) # Create the directory if it doesn't exist
    logger.info(f"Created directory: {model_dir}")

# Define the file path for your model
model_filename = os.path.join(model_dir, 'isolation_forest_model.pkl')

try:
    # Save the model using pickle
    with open(model_filename, 'wb') as f: # 'wb' means write in binary mode
        pickle.dump(iso_forest, f)
    logger.info(f"Model saved successfully to: {model_filename}")
except Exception as e:
    logger.error(f"Error saving the model: {e}")

# --- How to Load the Model (for future use) ---
logger.info("\n--- Demonstrating How to Load the Saved Model ---")

loaded_iso_forest_model = None
try:
    # Load the model using pickle
    with open(model_filename, 'rb') as f: # 'rb' means read in binary mode
        loaded_iso_forest_model = pickle.load(f)
    logger.info(f"Model loaded successfully from: {model_filename}")

    # You can now use loaded_iso_forest_model for new predictions
    # Example: Check if the loaded model is the same as the original
    # (This is just a conceptual check; direct comparison of model objects isn't straightforward)
    # You'd typically check by making predictions and comparing results.
    if loaded_iso_forest_model is not None:
        logger.info(f"Type of loaded model: {type(loaded_iso_forest_model)}")
        # You could now use loaded_iso_forest_model.predict(new_data)
        # For instance, making predictions on the first 5 samples of X with the loaded model:
        # loaded_predictions = loaded_iso_forest_model.predict(X.head(5))
        # print(f"Predictions with loaded model on first 5 samples: {loaded_predictions}")
except FileNotFoundError:
    logger.error(f"Error: Model file not found at {model_filename}. Please ensure it was saved correctly.")
except Exception as e:
    logger.error(f"An error occurred while loading the model: {e}")

logger.success("--- Model Saving and Loading Demonstration Complete ---")