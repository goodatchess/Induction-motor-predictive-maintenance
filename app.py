from pathlib import Path
import joblib
import pandas as pd
import gradio as gr

ROOT = Path(__file__).resolve().parent
artifact = joblib.load(ROOT / "models" / "random_forest.joblib")
model = artifact["model"]
feature_columns = artifact["feature_columns"]


def predict(*values):
    names = [
        "current_1_rms", "current_1_variance", "current_1_peak_to_peak", "current_1_kurtosis",
        "current_2_rms", "current_2_variance", "current_2_peak_to_peak", "current_2_kurtosis",
        "vibration_rms", "vibration_variance", "vibration_peak_to_peak", "vibration_kurtosis",
    ]

    row = pd.DataFrame([dict(zip(names, values))]).reindex(columns=feature_columns)
    prediction = model.predict(row)[0]
    probabilities = model.predict_proba(row)[0]

    text = "\n".join(
        f"{label}: {prob:.1%}"
        for label, prob in sorted(
            zip(model.classes_, probabilities),
            key=lambda x: x[1],
            reverse=True
        )
    )
    return prediction, text


with gr.Blocks(title="Induction Motor Health Monitor") as demo:
    gr.Markdown("# Induction Motor Health Monitor")
    gr.Markdown(
        "Enter statistical features calculated from one synchronized "
        "motor-current/vibration window."
    )

    inputs = []
    labels = [
        "Current 1 RMS", "Current 1 Variance", "Current 1 Peak-to-Peak", "Current 1 Kurtosis",
        "Current 2 RMS", "Current 2 Variance", "Current 2 Peak-to-Peak", "Current 2 Kurtosis",
        "Vibration RMS", "Vibration Variance", "Vibration Peak-to-Peak", "Vibration Kurtosis",
    ]

    for i in range(0, len(labels), 4):
        with gr.Row():
            for label in labels[i:i+4]:
                box = gr.Number(label=label)
                inputs.append(box)

    button = gr.Button("Predict Bearing Health", variant="primary")
    result = gr.Textbox(label="Predicted Condition")
    probabilities = gr.Textbox(label="Class Probabilities")

    button.click(
        predict,
        inputs=inputs,
        outputs=[result, probabilities],
    )


if __name__ == "__main__":
    demo.launch()
