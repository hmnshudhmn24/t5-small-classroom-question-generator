---
language:
  - en
tags:
  - question-generation
  - education
  - t5
  - nlp
license: apache-2.0
pipeline_tag: text-generation
model_name: t5-small-classroom-question-generator
base_model: t5-small
datasets:
  - custom
---

# 🎓 Classroom Question Generator (T5-Small)

An AI model that automatically generates **age-appropriate classroom questions** (Grades 1–10) from a simple topic.

### Example

**Input topic:** `Photosynthesis`  
**Grade:** `6`  
**Output question:** **"Why do plants need sunlight?"**

This project uses a fine-tuned **T5-small** Transformer and includes preprocessing, training, evaluation, inference, FastAPI API, and a Gradio UI.

---

# 🚀 Features

✓ Generates grade-appropriate questions for Grades 1–10  
✓ Designed for teachers, schools, and ed-tech platforms  
✓ End-to-end ML pipeline  
✓ Gradio UI + FastAPI server  
✓ Clean dataset format (`CSV → JSONL`)  
✓ Apache 2.0 license  
✓ HuggingFace model card included  

---

# 📁 Project Structure

```text
classroom-question-generator/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── src/
│   ├── config.py
│   ├── dataset_preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   └── model_utils.py
│
├── app/
│   ├── api.py
│   └── ui.py
│
├── model/
├── notebooks/
├── huggingface/
├── tests/
├── requirements.txt
├── LICENSE
└── README.md
```

---

# 📦 Installation

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

# 🔄 Dataset Preprocessing

```bash
python -m src.dataset_preprocessing   --input data/raw/dataset_raw.csv   --output data/processed/dataset_clean.jsonl
```

---

# 🏋️ Train Model

```bash
python -m src.train
```

---

# 🧪 Evaluate Model

```bash
python -m src.evaluate
```

---

# 🤖 Inference Example

```python
from src.inference import generate_question
print(generate_question("Photosynthesis", 6))
```

---

# 🌐 FastAPI Server

```bash
uvicorn app.api:app --reload --port 7860
```

---

# 🎨 Gradio UI

```bash
python app/ui.py
```

---

# 🎯 Prompt Format

```
topic: <topic> | grade: <grade>
```

---

# 📚 Example Outputs

| Topic          | Grade | Generated Question                              |
|----------------|-------|--------------------------------------------------|
| Photosynthesis | 6     | Why do plants need sunlight?                    |
| Gravity        | 7     | Why do objects fall toward the Earth?           |
| Water Cycle    | 4     | How does water move from the ground to the sky? |

---

# 📄 License

Apache License 2.0
