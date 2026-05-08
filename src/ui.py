import gradio as gr
import requests


def ask_rag_system(api_url, question, top_k):
    if not api_url or not question:
        return "⚠️ Vui lòng nhập đầy đủ thông tin!"

    api_url = api_url.rstrip("/")
    endpoint = f"{api_url}/ask"

    payload = {"question": question, "top_k": int(top_k)}

    try:
        response = requests.post(endpoint, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()

        output_markdown = f"### 🎯 Câu hỏi: {data['question']}\n\n"
        for idx, item in enumerate(data['results']):
            output_markdown += f"#### 🔍 Kết quả thứ {idx + 1}\n"
            output_markdown += f"**🤖 Câu trả lời:** {item['answer']}\n\n"
            output_markdown += f"> **📄 Ngữ cảnh:** {item['context']}\n\n---\n"
        return output_markdown

    except requests.exceptions.RequestException as e:
        return f"❌ **Lỗi kết nối:** `{e}`"


def build_ui():
    with gr.Blocks(title="RAG-QA System") as demo:
        gr.Markdown("# 🧠 Ứng dụng Hỏi Đáp AI (RAG System)")
        gr.Markdown("Hệ thống kết hợp **FAISS** (truy xuất dữ liệu) và **Llama 3** (sinh câu trả lời).")

        with gr.Row():
            with gr.Column(scale=1):
                api_url_input = gr.Textbox(label="🌐 Đường dẫn API (Ngrok URL)", placeholder="VD: https://abcd-123.ngrok-free.app")
                question_input = gr.Textbox(label="❓ Nhập câu hỏi", placeholder="VD: When did Beyonce start becoming popular?", lines=3)
                top_k_input = gr.Slider(minimum=1, maximum=5, value=3, step=1, label="🔢 Số lượng kết quả (Top K)")
                submit_btn = gr.Button("🚀 Gửi câu hỏi", variant="primary")

            with gr.Column(scale=2):
                output_display = gr.Markdown("*Kết quả sẽ hiển thị ở đây...*")

        submit_btn.click(fn=ask_rag_system, inputs=[api_url_input, question_input, top_k_input], outputs=[output_display])
        question_input.submit(fn=ask_rag_system, inputs=[api_url_input, question_input, top_k_input], outputs=[output_display])

    return demo


def launch_ui(share=True):
    demo = build_ui()
    demo.launch(share=share)