import gradio as gr
def process_frame(img, state):
    with open("log.txt", "a") as f:
        f.write(f"frame received {type(img)}, state: {state}\n")
    return img, state+1

with gr.Blocks() as d:
    i = gr.Image(sources=["webcam"], streaming=True)
    o = gr.Image()
    s = gr.State(0)
    i.stream(process_frame, [i, s], [o, s])
d.launch()
