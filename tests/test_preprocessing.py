from src import dataset_preprocessing
import json
def test_csv_to_jsonl(tmp_path):
    csv = tmp_path / 'd.csv'
    csv.write_text('topic,grade,generated_question\nHello,3,What is hello?\n')
    out = tmp_path / 'out.jsonl'
    dataset_preprocessing.csv_to_jsonl(str(csv), str(out))
    data = out.read_text().strip().splitlines()
    obj = json.loads(data[0])
    assert 'input' in obj and 'target' in obj
