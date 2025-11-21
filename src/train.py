import os, random
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
)
from src.config import config

def set_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def preprocess_function(examples, tokenizer, max_input_length, max_target_length):
    inputs = examples['input']
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['target'], max_length=max_target_length, truncation=True)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

def main():
    set_seed(config.seed)
    data_path = Path('data/processed/dataset_clean.jsonl')
    if not data_path.exists():
        raise FileNotFoundError('Processed dataset not found. Run preprocessing first.')

    ds = load_dataset('json', data_files=str(data_path))['train']
    ds = ds.train_test_split(test_size=0.1, seed=config.seed)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)

    tokenized_train = ds['train'].map(lambda ex: preprocess_function(ex, tokenizer, config.max_input_length, config.max_target_length), batched=True)
    tokenized_eval = ds['test'].map(lambda ex: preprocess_function(ex, tokenizer, config.max_input_length, config.max_target_length), batched=True)

    args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        evaluation_strategy='epoch',
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        predict_with_generate=True,
        learning_rate=config.lr,
        num_train_epochs=config.epochs,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

if __name__ == '__main__':
    main()
