import gradio as gr
try:
    i = gr.Image(sources=["webcam"], webcam_options={"facing_mode": "user"})
    print("Normal dict passed.")
except Exception as e:
    print(f"Error 1: {e}")

try:
    i = gr.Image(sources=["webcam"], webcam_options={"facing_mode": {"exact": "environment"}})
    print("Exact dict passed.")
except Exception as e:
    print(f"Error 2: {e}")
