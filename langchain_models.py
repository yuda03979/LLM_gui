import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import nltk
from nltk.tokenize import sent_tokenize
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

nltk.download('punkt', quiet=True)


device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")
print(f"device: {device}")


cache_dir = "_cache"
current_model_name = "ChatGroq"
models_names = ["ChatGroq", "meta-llama/Llama-2-7b-hf"]
# model = AutoModelForCausalLM.from_pretrained(current_model_name, torch_dtype=torch.bfloat16, cache_dir=cache_dir).to(device)
# tokenizer = AutoTokenizer.from_pretrained(current_model_name, padding=True, cache_dir=cache_dir)


load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv('GROQ_API_KEY')


def instantiate_hf_model(model_name):
    # check if the model in the models_names

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, cache_dir=cache_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, cache_dir=cache_dir)
    return model, tokenizer


def generate_response(template, user_input, model_name, generation_params: dict):
    global model, tokenizer, current_model_name

    match model_name:
        case "meta-llama/Llama-2-7b-hf":


            if model_name != current_model_name:
                model, tokenizer = instantiate_hf_model(model_name)
                current_model_name = model_name

            params = {
                'temperature': generation_params['temperature'],
                'top_p': generation_params['top_p'],
                'max_length': generation_params['max_tokens'],
            }

            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                **params,
                device=device
            )

            llm = HuggingFacePipeline(pipeline=pipe)
            gc.collect()

            prompt_template = ChatPromptTemplate.from_template(template)
            prompt = prompt_template.invoke({"input": user_input})
            respawns = llm.invoke(prompt)


        case 'ChatGroq':
            llm = ChatGroq(model="llama3-8b-8192", **generation_params)
            current_model_name = model_name

            prompt_template = ChatPromptTemplate.from_template(template)
            prompt = prompt_template.invoke({"input": user_input})
            respawns = llm.invoke(prompt).content

        case _:
            raise ValueError(f"Unsupported model: {model_name}")

    return respawns