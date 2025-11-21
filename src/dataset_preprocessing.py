import argparse
import csv
import json
from pathlib import Path

def csv_to_jsonl(input_csv: str, output_jsonl: str):
    p_in = Path(input_csv)
    p_out = Path(output_jsonl)
    p_out.parent.mkdir(parents=True, exist_ok=True)
    with p_in.open(encoding='utf-8') as fin, p_out.open('w', encoding='utf-8') as fout:
        reader = csv.DictReader(fin)
        for row in reader:
            topic = (row.get('topic') or '').strip()
            grade = (row.get('grade') or '').strip()
            question = (row.get('generated_question') or '').strip()
            if not topic or not grade or not question:
                continue
            inp = f"topic: {topic} | grade: {grade}"
            fout.write(json.dumps({"input": inp, "target": question}, ensure_ascii=False) + "\n")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    csv_to_jsonl(args.input, args.output)
