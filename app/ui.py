import gradio as gr
from src.inference import generate_question

def run_ui():
    def fn(topic, grade):
        return generate_question(topic, grade)
    demo = gr.Interface(fn=fn, inputs=[gr.Textbox(lines=2, placeholder='Enter topic'), gr.Slider(1,10,value=6)], outputs='text', title='Classroom Question Generator')
    demo.launch()
if __name__=='__main__':
    run_ui()
