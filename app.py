import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pickle


class ChurnModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.fc1(x)
        return torch.sigmoid(x)

model = ChurnModel(12)
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()


with open("encoder_geo.pkl", 'rb') as f:
    geo_encoder = pickle.load(f)

with open("encoder_gender.pkl", 'rb') as f:
    gender_encoder = pickle.load(f)

with open("scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)


st.title("Customer Churn Prediction App")

st.markdown("### Enter Customer Details")


credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
gender = st.selectbox("Gender", ['Male', 'Female'])
age = st.slider("Age", 18, 100, 30)
tenure = st.slider("Tenure (Years)", 0, 10, 3)
balance = st.number_input("Account Balance", value=50000.0)
num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_credit_card = st.selectbox("Has Credit Card?", ['Yes', 'No'])
is_active = st.selectbox("Is Active Member?", ['Yes', 'No'])
estimated_salary = st.number_input("Estimated Salary", value=60000.0)
geography = st.selectbox("Geography", ['France', 'Spain', 'Germany'])

if st.button("Predict Churn"):
   
    # Encode categorical features
    gender_encoded = gender_encoder.transform([gender])[0]

    # Fix for one-hot encoded geography
    geo_encoded = geo_encoder.transform([[geography]])
    if hasattr(geo_encoded, "toarray"):
        geo_encoded = geo_encoded.toarray()
    geography_encoded = geo_encoded.flatten()



    has_credit_card = 1 if has_credit_card == 'Yes' else 0
    is_active = 1 if is_active == 'Yes' else 0

    input_data = np.array([[credit_score, gender_encoded, age, tenure, balance,
                            num_products, has_credit_card, is_active,
                            estimated_salary, *geography_encoded]])

  
    input_scaled = scaler.transform(input_data)

  
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

 
    with torch.no_grad():
        prob = model(input_tensor).item()

    st.markdown("### Prediction")
    if prob >= 0.5:
        st.error(f"⚠️ Customer is likely to churn. (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Customer is likely to stay. (Probability: {prob:.2f})")
