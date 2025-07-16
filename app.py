import gradio as gr
import numpy as np
import joblib
from pytorch_tabnet.tab_model import TabNetClassifier

# Load model and scaler
model = TabNetClassifier()
model.load_model("model.zip")
scaler = joblib.load("scaler.pkl")

# Prediction function
def predict(age, sex, cp, trestbps, chol, fbs, restecg,
            thalach, exang, oldpeak, slope):
    data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope]])
    scaled = scaler.transform(data)
    pred = model.predict(scaled)
    return "🫀 Heart Disease Detected" if pred[0] == 1 else "✅ Normal"

# Labels & Tooltips
sex_labels = {0: "Female", 1: "Male"}
cp_labels = {
    1: "Typical Angina",
    2: "Atypical Angina",
    3: "Non-anginal Pain",
    4: "Asymptomatic"
}
restecg_labels = {
    0: "Normal",
    1: "ST-T Abnormality",
    2: "Left Ventricular Hypertrophy"
}
slope_labels = {
    1: "Upsloping",
    2: "Flat",
    3: "Downsloping"
}

# Interface
with gr.Blocks(theme=gr.themes.Soft(), title="Heart Disease Predictor") as demo:
    gr.Markdown("# 🫀 Heart Disease Prediction App")
    gr.Markdown("Get a fast medical prediction based on vitals and test results using a deep learning model trained on heart disease data.")

    with gr.Row():
        with gr.Column():
            age = gr.Number(label="Age", info="Age of the patient in years", value=50)
            sex = gr.Radio(choices=list(sex_labels.keys()), label="Sex", value=1, info="0 = Female, 1 = Male")
            cp = gr.Radio(choices=list(cp_labels.keys()), label="Chest Pain Type", value=4, info=str(cp_labels))
            trestbps = gr.Number(label="Resting Blood Pressure (mm Hg)", value=130)
            chol = gr.Number(label="Serum Cholesterol (mg/dL)", value=250)
            fbs = gr.Radio(choices=[0, 1], label="Fasting Blood Sugar > 120 mg/dL?", value=0)

        with gr.Column():
            restecg = gr.Radio(choices=list(restecg_labels.keys()), label="Resting ECG Results", value=1, info=str(restecg_labels))
            thalach = gr.Number(label="Maximum Heart Rate Achieved", value=150)
            exang = gr.Radio(choices=[0, 1], label="Exercise Induced Angina?", value=0)
            oldpeak = gr.Number(label="ST Depression Induced by Exercise", value=1.5)
            slope = gr.Radio(choices=list(slope_labels.keys()), label="Slope of Peak Exercise ST Segment", value=2, info=str(slope_labels))

    submit_btn = gr.Button("🔍 Predict", variant="primary")

    output = gr.Textbox(label="Diagnosis")

    # On submit
    submit_btn.click(fn=predict,
                     inputs=[age, sex, cp, trestbps, chol, fbs, restecg,
                             thalach, exang, oldpeak, slope],
                     outputs=output)

    gr.Markdown("### 📊 Example Inputs")
    examples = gr.Examples(
        examples=[
            [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 3],
            [45, 0, 2, 120, 200, 0, 1, 160, 0, 1.0, 2],
            [58, 1, 4, 140, 230, 1, 2, 130, 1, 3.2, 1],
            [52, 0, 1, 130, 180, 0, 0, 150, 0, 0.0, 2],
            [69, 1, 4, 160, 289, 1, 2, 110, 1, 2.0, 3]
        ],
        inputs=[age, sex, cp, trestbps, chol, fbs, restecg,
                thalach, exang, oldpeak, slope]
    )

demo.launch()
