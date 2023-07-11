import os
from os.path import join, dirname

from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv

from utils import read_dataset


load_dotenv()
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=os.environ.get("DEVICE"))


class EmbeddingSimilarity:
    def __init__(self, source_sentence):
        self.source_embedding = model.encode(source_sentence, convert_to_tensor=True).to("cpu")

    def extract_similarity(self, batch):
        sentence_embedding = batch["embedding"]
        similarity = util.cos_sim(self.source_embedding, sentence_embedding)
        return {"similarity": similarity[0]}


def extract_encoding(batch):
    sentence = batch["abstract"]
    sentence_embedding = model.encode(sentence, convert_to_tensor=True)
    return {"embedding": sentence_embedding}


def execute_encode():
    categories = [
        "cs.CR",
        "cs.RO",
        "cs.SD"
    ]
    dataset_dict = {}
    for category in categories:
        try:
            dataset = read_dataset(f"../datasets/arxiv_{category.replace('.', '_')}_dataset.json")
        except FileNotFoundError:
            return None
        # Extract encoding
        dataset = dataset.map(extract_encoding, batched=True, batch_size=30)
        dataset_dict[category] = dataset
    return dataset_dict
