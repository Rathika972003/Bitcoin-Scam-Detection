import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load pre-trained model
with open("bitcoin_scam_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load dataset
features_df = pd.read_csv("elliptic_txs_features.csv", header=None)
labels_df = pd.read_csv("elliptic_txs_classes.csv")

# Define the selected features for prediction (ensure these match training)
selected_features = [54, 90, 56, 91, 143, 151, 155, 55]

# Set up Streamlit layout
st.set_page_config(page_title="Bitcoin Scam Detection", layout="centered")
st.title("🔍 Bitcoin Scam Detector")

# User input for transaction ID
tx_id = st.text_input("Enter Transaction ID:", "")

if st.button("Check Transaction", key="check_button"):  # Ensure unique key for button
    if tx_id.strip().isdigit():
        tx_id = int(tx_id)

        # Find the transaction in the dataset
        if tx_id in features_df[0].values:
            tx_index = features_df[features_df[0] == tx_id].index[0]
            transaction_data = features_df.iloc[tx_index, selected_features]

            # Ensure data is in correct format
            transaction_data = np.array(transaction_data, dtype=float).reshape(1, -1)  # Convert to float

            # Predict using the model
            prediction = model.predict(transaction_data)[0]

            # DEBUG: Show raw model output
            st.write("### Raw Model Output:", prediction)

            # Correct label mapping if needed
            if prediction == 1:
                st.success("⚠️ This transaction is classified as **SCAM**.")  # Changed from LEGIT
            else:
                st.error("✅ This transaction is classified as **LEGIT**.")  # Changed from SCAM
        else:
            st.warning("Transaction ID not found in dataset.")
    else:
        st.warning("Please enter a valid numeric Transaction ID.")

