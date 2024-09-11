import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import torch
import nltk
from nltk.tokenize import sent_tokenize

from constants import *

# nltk.download('punkt')
#
# model_name = "yam-peleg/Hebrew-Mistral-7B"
# cache_dir = "hebrew_mistral_cache"
# os.makedirs(cache_dir, exist_ok=True)
#
# tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)


torch.backends.cudnn.benchmark = True


# quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
model = ''#AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, quantization_config=quantization_config)


models_dict = {
    models_names[i]: models[i] for i in range(len(models))
}

model_name = models_names[0]


def generate_response(model_name, input_text, max_new_tokens, min_length, no_repeat_ngram_size, num_beams, early_stopping, temperature, top_p, top_k):
   model = models_dict[model_name]
   # tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
   input_ids = ""#tokenizer(input_text, return_tensors="pt").to(model.device)
   # outputs = model.generate(
   #     **input_ids,
   #     max_new_tokens=max_new_tokens,
   #     min_length=min_length,
   #     no_repeat_ngram_size=no_repeat_ngram_size,
   #     num_beams=num_beams,
   #     early_stopping=early_stopping,
   #     temperature=temperature,
   #     top_p=top_p,
   #     top_k=top_k,
   #     pad_token_id=1,#tokenizer.eos_token_id,
   #     do_sample=True
   # )
   response = "this is a respawns"#tokenizer.decode(outputs[0], skip_special_tokens=True)
   return response




