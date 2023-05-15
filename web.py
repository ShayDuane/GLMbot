import datetime
import gradio as gr
import mdtex2html

import requests
import json
import os
from config import *
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

chatglm_endpoint_chat = os.getenv("chatglm_endpoint_chat")
chatglm_endpoint_chat_stream = os.getenv("chatglm_endpoint_chat_stream")
emdbeddings_endpoint = os.getenv("emdbeddings_endpoint")
searchvectorsbase_endpoint = os.getenv("searchvectorsbase_endpoint")

"""Override Chatbot.postprocess"""


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text

def predict(input, chatbot, max_length, top_p, temperature, history, conversation_turn, stream, vector_base, index_name):
    chatbot.append((parse_text(input),""))
    prompt = input
    if vector_base:
        url = searchvectorsbase_endpoint
        headers_v = {'Content-Type': 'application/json'}
        data_v = {"prompt": input,"index_name": index_name}
        data_v = json.dumps(data_v)
        response = requests.post(url=url,headers=headers_v,data=data_v)
        if response.status_code == 200:
            result = json.loads(response.text)
            result = result["results"]
            context = "\n".join([doc for doc in result])
            prompt = prompt_template.replace("{question}", input).replace("{context}", context)
        else:
            prompt = input
    else:
        prompt = input
    headers = {'Content-Type': 'application/json'}
    history = [] if len(history) == conversation_turn else history
    data = {"prompt": prompt,"history": history,"max_length": max_length,"top_p": top_p,"temperature": temperature}
    if stream:
        res = requests.post(url=chatglm_endpoint_chat_stream,headers=headers,data=json.dumps(data),stream=True)
    else:
        res = requests.post(url=chatglm_endpoint_chat,headers=headers,data=json.dumps(data))
    for item in res.iter_lines():
        item = item.decode("utf-8")
        if item.strip() == '':
            continue 
        if stream:     
            item = item[6:]
        else:
            item = item
        try:
            item = json.loads(item)
        except:
            continue   
        response = item["response"]
        history = item["history"]
        chatbot[-1] = (parse_text(input),parse_text(response))       
        yield chatbot, history
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    log = "[" + time + "] " + '", prompt:"' + input + '", prompt_context:"' + prompt +'", response:"' + repr(response) + '"' + ", stream:" + str(stream) + '"' +"\n"
    print(log)


def get_vector_base(File, chatbot):
    if File is None:
        return
    url = emdbeddings_endpoint
    file = {'file': open(File.name, 'rb')}
    response = requests.post(url, files=file)
    if response.status_code == 200:
        result = json.loads(response.text)
        index_name = result['base_name']
        chatbot.append((parse_text("Loading vector knowledge base..."),
                    parse_text("Success! Please tick the Vector base checkbox and ask me questions")
                ))       
    else:
        chatbot.append((parse_text("Loading vector knowledge base..."),
                    parse_text("Failed! Please contact the administrator ShuaiqiDuan, may be a bug.")
                ))            
    return index_name, chatbot


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], []


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">ChatGLM</h1>""")

    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=13).style(
                    container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_length = gr.Slider(0, 4096, value=2048, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.7, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)
            conversation_turn = gr.Slider(5, 10, value=5, step=1, label="conversation_turn", interactive=True)
            stream = gr.Checkbox(label="stream_out", value=True, interactive=True)
            vector_base = gr.Checkbox(label="vector_base", value=False, interactive=True)
        with gr.Tab(label="vector_base"):
            File = gr.File(label="add_file",
                file_types=['.txt'],
                file_count="single",
                show_label=False)
            load_file_button = gr.Button("add file and load vector knowledge base based on the file")
    history = gr.State([])

    index_name = gr.State("")

    load_file_button.click(get_vector_base, [File, chatbot], [index_name, chatbot], show_progress=True)

    submitBtn.click(predict, [user_input, chatbot, max_length, top_p, temperature, history, conversation_turn, stream, vector_base, index_name], [chatbot, history],
                    show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])


    emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)

demo.queue().launch(share=False, inbrowser=True)