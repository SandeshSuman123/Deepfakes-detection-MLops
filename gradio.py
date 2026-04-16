import gradio as gr
import requests

def predict(image):
    # Save temp image
    image.save("temp.jpg")

    # Send to FastAPI
    with open("temp.jpg", "rb") as f:
        response = requests.post(
            "http://127.0.0.1:9001/predict",
            files={"file": f}
        )

    result = response.json()

    return f"{result['prediction']} ({result['confidence']:.2%})"


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Deepfake Detection",
    description="Upload image to detect fake or real"
)

demo.launch()