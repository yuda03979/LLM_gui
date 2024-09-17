from langchain_models import *



def create_paragraphs(bot_response, sentences_per_paragraph=2):
    sentences = sent_tokenize(bot_response)
    paragraphs = []
    current_paragraph = ""

    for i, sentence in enumerate(sentences, start=1):
        current_paragraph += " " + sentence
        if i % sentences_per_paragraph == 0:
            paragraphs.append(current_paragraph.strip())
            current_paragraph = ""

    if current_paragraph:
        paragraphs.append(current_paragraph.strip())

    formatted_paragraphs = "\n".join([f'<p style="text-align: right; direction: rtl;">{p}</p>' for p in paragraphs])
    return formatted_paragraphs


def chat(input_text, prompt, history, model_name, max_new_tokens, temperature, top_p, create_paragraphs_enabled):
    # Formatting user input as a right-aligned, RTL HTML div
    user_input = f'<div style="text-align: right; direction: rtl;">{input_text}</div>'

    # Constructing the conversation history for LangChain model
    messages = [
        SystemMessage(content=prompt),  # Initial system message with the prompt
    ]

    # Adding the previous conversation to the history
    for user_msg, bot_msg in history:
        messages.append(HumanMessage(content=user_msg))
        messages.append(AIMessage(content=bot_msg))

    messages.append(HumanMessage(content=user_input))
    generation_params = {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_new_tokens
    }

    response = generate_response(user_input, prompt, model_name, generation_params)
    if create_paragraphs_enabled:
        response = create_paragraphs(response)

    # Formatting the bot's response as a right-aligned, RTL HTML div
    bot_response = f'<div style="text-align: right; direction: rtl;">{response}</div>'

    # Updating the conversation history with the latest exchange
    history.append((user_input, bot_response))

    return history, history

with gr.Blocks() as demo:
    gr.Markdown("# Multi-Model Language Chatbot", elem_id="title")

    chatbot = gr.Chatbot(elem_id="chatbot")

    with gr.Row():
        message = gr.Textbox(placeholder="your message...", label="user", elem_id="message", scale=3)
        prompt = gr.Textbox(placeholder="your prompt...", label="prompt", elem_id="message", scale=3)
        submit = gr.Button("send", scale=1)

    with gr.Row():
        model_dropdown = gr.Dropdown(choices=models_names, value=current_model_name, label="choose model")
        create_paragraphs_checkbox = gr.Checkbox(label="create paragraphs", value=False)

    with gr.Accordion("settings", open=False):
        with gr.Row():
            with gr.Column():
                max_new_tokens = gr.Slider(minimum=1, maximum=2000, value=50, step=1, label="max new tokens")
                temperature = gr.Slider(minimum=0.1, maximum=1.0, value=0.7, step=0.1, label="temperature")
            with gr.Column():
                top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.9, step=0.1, label="Top P")
                top_k = gr.Slider(minimum=1, maximum=100, value=50, step=1, label="Top K")

    submit.click(chat, inputs=[message, prompt, chatbot, model_dropdown, max_new_tokens, temperature, top_p, create_paragraphs_checkbox], outputs=[chatbot, chatbot])

    demo.css = """
        #message, #message * {
            text-align: right !important;
            direction: rtl !important;
        }

        #chatbot, #chatbot * {
            text-align: right !important;
            direction: rtl !important;
        }

        #title, .label {
            text-align: right !important;
        }
    """

print("Starting the server. This may take a few minutes as all models are being loaded...")
demo.launch()