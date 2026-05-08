from unsloth import FastLanguageModel
from transformers import TextStreamer

MODEL_NAME = 'MinhQuy24/llama3.2_3B_SQuAD_QA'
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True

model_QA, tokenizer_QA = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME, max_seq_length=MAX_SEQ_LENGTH, dtype=None, load_in_4bit=LOAD_IN_4BIT,
)
FastLanguageModel.for_inference(model_QA)
text_streamer = TextStreamer(tokenizer_QA, skip_prompt=True)


def format_prompt(context, question):
    return f"""<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are a helpful AI assistant providing accurate answers based on the given context.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Context: {context}
Question: {question}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Answer: """


def generate_answer(context, question):
    formatted_prompt = format_prompt(context, question)
    inputs = tokenizer_QA(formatted_prompt, return_tensors="pt").to("cuda")

    outputs = model_QA.generate(
        input_ids=inputs["input_ids"],
        max_new_tokens=64,
        use_cache=True,
        temperature=0.3,
        min_p=0.1,
        pad_token_id=tokenizer_QA.eos_token_id
    )

    full_text = tokenizer_QA.decode(outputs[0], skip_special_tokens=False)
    answer = full_text.split("Answer: ")[-1].replace('<|eot_id|>', '').strip()
    return answer