from datasets import load_metric, load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.config import config

def evaluate(model_dir: str = None):
    model_dir = model_dir or config.output_dir
    ds = load_dataset('json', data_files='data/processed/dataset_clean.jsonl')['train']
    ds = ds.train_test_split(test_size=0.1, seed=config.seed)['test']

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(config.device)

    rouge = load_metric('rouge')
    predictions = []
    references = []

    for item in ds:
        input_text = item['input']
        inputs = tokenizer(input_text, return_tensors='pt', truncation=True).to(config.device)
        outs = model.generate(**inputs, max_length=config.max_target_length, num_beams=4)
        pred = tokenizer.decode(outs[0], skip_special_tokens=True)
        predictions.append(pred)
        references.append(item['target'])

    results = rouge.compute(predictions=predictions, references=references)
    print(results)

if __name__ == '__main__':
    evaluate()
