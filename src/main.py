import json

import torch
import openai
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel


from models import TransformerBody
from utils import read_dataset


app = FastAPI()
templates = Jinja2Templates(directory="../templates")

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
    transformer = TransformerBody(title, model_name, "mps")
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
        "top5_abstract": top5_abstract,
        "top5_title": top5_title
    }


def get_json(text):
    prompt_system = "Please operate as a Japanese speech modification program. " \
                    "Determine whether the user's speech contains discriminatory content. " \
                    "If the speech contains discriminatory content, output '1' at the beginning, " \
                    "and if it does not contain discriminatory content, output '0'. If you output '1', " \
                    "please display the modified sentence in a polite form as '1: Modified sentence'. " \
                    "If you output '0', a modified sentence is not necessary." \
                    "Output the 'Modified sentence' part in Japanese."
    prompt_user = text
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": prompt_user}
        ]
    )

    return response.choices[0].message.content.strip()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/search")
async def search(request: Request, paper: Paper):
    similar_papers = calc_sim(paper.category, paper.abstract)
    if similar_papers is None:
        return templates.TemplateResponse("error.html", {"request": request})

    titles = similar_papers["top5_title"]
    abstracts = similar_papers["top5_abstract"]

    gpt_results = {}

    data = {
        "data": gpt_results
    }

    return json.dumps(data)
