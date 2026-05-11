import gradio as gr
import requests


def ask_rag_system(api_url, question, top_k):
    if not api_url or not question:
        return "Please provide all required information!"

    api_url = api_url.rstrip("/")
    endpoint = f"{api_url}/ask"

    payload = {"question": question, "top_k": int(top_k)}

    try:
        response = requests.post(endpoint, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()

        output_markdown = f"### Question: {data['question']}\n\n"
        for idx, item in enumerate(data['results']):
            output_markdown += f"#### Result #{idx + 1}\n"
            output_markdown += f"**Answer:** {item['answer']}\n\n"
            output_markdown += f"> **Context:** {item['context']}\n\n---\n"
        return output_markdown

    except requests.exceptions.RequestException as e:
        return f"**Connection Error:** `{e}`"


def build_ui():
    with gr.Blocks(title="RAG-QA System") as demo:
        gr.Markdown("# AI Question Answering Application (RAG System)")
        gr.Markdown("System combining **FAISS** (data retrieval) and **Llama 3** (answer generation).")

        with gr.Row():
            with gr.Column(scale=1):
                api_url_input = gr.Textbox(label="API URL (Ngrok)", placeholder="e.g., https://abcd-123.ngrok-free.app")
                question_input = gr.Textbox(label="Enter your question", placeholder="e.g., When did Beyonce start becoming popular?", lines=3)
                top_k_input = gr.Slider(minimum=1, maximum=5, value=3, step=1, label="Number of Results (Top K)")
                submit_btn = gr.Button("Submit Question", variant="primary")

            with gr.Column(scale=2):
                output_display = gr.Markdown("*Results will appear here...*")

        for event in [submit_btn.click, question_input.submit]:
            event(fn=ask_rag_system, inputs=[api_url_input, question_input, top_k_input], outputs=[output_display])

    return demo


def launch_ui(share=True):
    demo = build_ui()
    demo.launch(share=share)