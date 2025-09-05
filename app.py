import gradio as gr
import numpy as np
import pickle
from sklearn.datasets import load_breast_cancer

# Load saved model
model = pickle.load(open("breast_cancer_model.pkl", "rb"))

# Load dataset to get feature names
data = load_breast_cancer()
feature_names = data.feature_names  # list of 30 feature names

def predict_cancer(*features):
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    return "Malignant" if prediction[0] == 0 else "Benign"

# Use actual feature names in Gradio input fields
inputs = [gr.Number(label=feature) for feature in feature_names]

iface = gr.Interface(
    fn=predict_cancer,
    inputs=inputs,
    outputs="text",
    title="Breast Cancer Detection",
    description="Enter the 30 features to predict whether the cancer is Malignant or Benign."
)

iface.launch()
