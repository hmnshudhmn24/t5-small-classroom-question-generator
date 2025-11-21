from fastapi import FastAPI
from pydantic import BaseModel
from src.inference import generate_question

app = FastAPI(title='Classroom Question Generator')

class Request(BaseModel):
    topic: str
    grade: int = 6

@app.post('/generate')
async def generate(req: Request):
    question = generate_question(req.topic, req.grade)
    return {'topic': req.topic, 'grade': req.grade, 'question': question}

@app.get('/')
async def root():
    return {'message':'Classroom Question Generator API. POST /generate with {topic, grade}.'}
