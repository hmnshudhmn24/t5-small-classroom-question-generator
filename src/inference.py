from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.config import config

_cache = {}

def generate_question(topic: str, grade: int, model_dir: str = None, max_length: int = None):
    model_dir = model_dir or config.output_dir
    max_length = max_length or config.max_target_length
    key = model_dir
    if key not in _cache:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(config.device)
        _cache[key] = (tokenizer, model)
    tokenizer, model = _cache[key]
    prompt = f"topic: {topic} | grade: {grade}"
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True).to(config.device)
    outputs = model.generate(**inputs, max_length=max_length, num_beams=4)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
