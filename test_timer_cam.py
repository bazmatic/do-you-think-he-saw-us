import gradio as gr

with gr.Blocks() as demo:
    with gr.Row():
        live_input = gr.Image(sources=["webcam"], streaming=True)
        out = gr.Textbox()
        btn = gr.Button("Start")
    timer = gr.Timer(active=False, value=1.0)
    btn.click(lambda: gr.Timer(active=True), outputs=timer)
    
    def log_frame(img):
        return f"Frame type: {type(img)}"
        
    timer.tick(log_frame, inputs=live_input, outputs=out)

demo.launch()
