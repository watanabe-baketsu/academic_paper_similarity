import torch
import openai
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse
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
        return JSONResponse(jsonable_encoder({"error": f"similar papers not found for category=\"{paper.category}\" and abstract=\"{paper.abstract}\"!"}))

    titles = similar_papers["top5_title"]
    abstracts = similar_papers["top5_abstract"]

    gpt_results = [
        # ひとまずテストデータとして入れてます。
        {
            "論文のタイトル": "Detecting Phishing Sites Using ChatGPT",
            "概要": "大規模言語モデルChatGPTを使用してフィッシングサイトを検出する新しい方法を提案。ウェブクローラーを用いて情報を収集し、機械学習モデルを微調整することなくフィッシングサイトを検出する。GPT-4を使用した実験結果は精度98.3％、再現率98.4％を示す。",
            "長所": "ChatGPTの高い検出性能と、微調整なしでフィッシングサイトを検出できる点。",
            "短所": "GPT-4以前のモデルと比較した場合、誤検出（偽陰性）が増加する可能性がある。",
        }, 
    ]

    data = {
        "data": gpt_results
    }

    return JSONResponse(jsonable_encoder(data))
