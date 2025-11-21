from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.config import config

def load_tokenizer(model_name: str = None):
    return AutoTokenizer.from_pretrained(model_name or config.model_name)

def load_model(model_name: str = None):
    return AutoModelForSeq2SeqLM.from_pretrained(model_name or config.model_name)

def save_model_and_tokenizer(model, tokenizer, output_dir: str):
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
