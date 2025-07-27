# 🔍 Customer Churn Prediction Web App

A Streamlit web application to predict whether a customer will churn using a trained PyTorch model. This project uses:
- **PyTorch** for model inference
- **scikit-learn** for preprocessing
- **Streamlit** for the interactive web UI

---

## 📦 Features

- Accepts user input for relevant customer features
- Encodes and scales inputs using pre-fitted encoders and scaler
- Runs prediction using a trained ANN model
- Displays churn probability and decision
- Ready for local or cloud deployment

---

## 🧠 Model Details

- **Architecture**: 3-layer ANN (12 inputs → 64 → 32 → 1)
- **Activation**: ReLU (hidden), Sigmoid (output)
- **Loss**: Binary Cross Entropy
- **Optimizer**: Adam
- **Trained on**: Bank churn dataset (`Churn_Modelling.csv`)

---

## 🗂️ Folder Structure

