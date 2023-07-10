import json
import os
from os.path import join, dirname
from pprint import pprint

import torch
import openai
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from pydantic import BaseModel

from models import TransformerBody
from utils import read_dataset
from prompt import prompt_system


app = FastAPI()
origins = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

# ミドルウェアを追加
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="../templates")
load_dotenv()
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)
openai.api_key = os.environ.get("OPENAI_API_KEY")

class Paper(BaseModel):
    category: str
    abstract: str


def calc_sim(category: str, title: str):
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    try:
        dataset = read_dataset(f"../datasets/arxiv_{category.replace('.', '_')}_dataset.json")
    except FileNotFoundError:
        return None
    # Initialize the model
    transformer = TransformerBody(title, model_name, os.environ.get("DEVICE"))
    # Extract hidden states
    dataset = dataset.map(transformer.extract_similarity, batched=True, batch_size=30)

    # Extract the paper that has the top5 high similarity
    similarity = dataset["dataset"]["similarity"]
    top5_idx = torch.Tensor(similarity).argsort(descending=True)[:5].tolist()  # convert tensor to list
    top5_abstract = [dataset["dataset"][i]["abstract"] for i in top5_idx]
    top5_title = [dataset["dataset"][i]["title"] for i in top5_idx]

    print("Top 5 similar papers:")
    for i, title in enumerate(top5_title):
        if "\n" in title:
            title = title.replace("\n", " ")
        print(f"{i + 1}. {title} // similarity: {similarity[top5_idx[i]]:.3f}")
    return {
        "top5_abstract": [abstract for abstract in top5_abstract],
        "top5_title": [title for title in top5_title]
    }


def get_json(title: list, abstract: list):
    user_inputs = ""
    for i, (t, a) in enumerate(zip(title, abstract)):
        user_inputs += f"{i}.\ntitle : {t}\nabstract : {a}\n\n"
    prompt_user = user_inputs
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": prompt_user}
        ],
        temperature=1.0,
        max_tokens=2048,
    )
    data_str = response.choices[0].message.content.strip()

    return json.loads(data_str)


@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post('/search')
async def search(paper: Paper):
    similar_papers = calc_sim(paper.category, paper.abstract)
    if similar_papers is None:
       return JSONResponse(jsonable_encoder({"error": f"similar papers not found for category=\"{paper.category}\""}))

    titles = similar_papers["top5_title"]
    abstracts = similar_papers["top5_abstract"]

    gpt_results = get_json(titles, abstracts)
    pprint(gpt_results)

    data = {
        "data": gpt_results
    }

    return JSONResponse(jsonable_encoder(data))
