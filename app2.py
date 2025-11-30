import re
import socket
import whois
import requests
import tldextract
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit as st
from urllib.parse import urlparse

# Load dataset (to get columns & lookup known URLs)
df = pd.read_csv("phishing.csv")

# Load model
clf = joblib.load("phishing_model.pkl")
explainer = shap.TreeExplainer(clf)

FEATURE_COLUMNS = [c for c in df.columns if c not in ["url", "status"]]

# --- Feature extraction function ---
def extract_features(url):
    """Extract a subset of features similar to dataset columns."""
    parsed = urlparse(url)
    ext = tldextract.extract(url)
    features = {}

    # Basic lexical features
    features["length_url"] = len(url)
    features["length_hostname"] = len(parsed.netloc)
    features["ip"] = 1 if re.match(r"^\d+\.\d+\.\d+\.\d+$", parsed.netloc) else 0
    features["nb_dots"] = url.count(".")
    features["nb_hyphens"] = url.count("-")
    features["nb_at"] = url.count("@")
    features["nb_qm"] = url.count("?")
    features["nb_and"] = url.count("&")
    features["nb_or"] = url.count("|")
    features["nb_eq"] = url.count("=")
    features["nb_slash"] = url.count("/")
    features["nb_www"] = url.lower().count("www")
    features["nb_com"] = url.lower().count(".com")

    # Protocol
    features["https_token"] = 1 if "https" in parsed.scheme else 0

    # WHOIS-based features
    try:
        w = whois.whois(ext.registered_domain)
        features["dns_record"] = 1 if w else 0
        if w.creation_date and w.expiration_date:
            age_days = (w.expiration_date - w.creation_date).days
            features["domain_registration_length"] = age_days
        else:
            features["domain_registration_length"] = -1
    except:
        features["dns_record"] = 0
        features["domain_registration_length"] = -1

    # External signals (simplified)
    try:
        r = requests.get(url, timeout=5)
        features["google_index"] = 1 if r.status_code == 200 else 0
    except:
        features["google_index"] = 0

    # Fill missing dataset columns with defaults
    for col in FEATURE_COLUMNS:
        if col not in features:
            features[col] = 0

    return features


# --- Streamlit UI ---
st.title("Phishing Webpage Detection")
st.write("Paste a URL. The model predicts if it's **Phishing** or **Legitimate** and explains the decision.")

user_url = st.text_input("Enter URL:")

if st.button("Check URL"):
    if not user_url.strip():
        st.warning("Please enter a URL.")
    else:
        # Check if URL exists in dataset
        row = df[df["url"] == user_url]

        if not row.empty:
            st.info("ℹ️ URL found in dataset → using dataset features.")
            X = row[FEATURE_COLUMNS]
            y_true = row["status"].values[0]
        else:
            st.info("ℹ️ New URL → extracting features dynamically.")
            feats = extract_features(user_url)
            X = pd.DataFrame([feats])[FEATURE_COLUMNS]
            y_true = "unknown"

        # Prediction
        pred = clf.predict(X)[0]
        proba = clf.predict_proba(X)[0][1]

        if pred == 1:
            st.error(f"⚠️ Predicted: **Phishing** (prob: {proba:.2f})")
        else:
            st.success(f"✅ Predicted: **Legitimate** (phishing prob: {proba:.2f})")

        if y_true != "unknown":
            st.info(f"Ground Truth (from dataset): {y_true}")

        # Show features
        # st.subheader("Extracted Features")
        # st.json(X.iloc[0].to_dict())
        
