import gradio as gr
try:
    img = gr.Image(sources=["webcam"], streaming=True, webcam_options={"facing_mode": "environment"})
    print("facing_mode works")
except Exception as e:
    print(f"Error: {e}")
try:
    img2 = gr.Image(sources=["webcam"], streaming=True, webcam_options={"facingMode": "environment"})
    print("facingMode works")
except Exception as e:
    print(f"Error2: {e}")
