from argparse import ArgumentParser

import torch

from models import TransformerBody
from utils import read_dataset


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--dataset_path", type=str, default="../datasets/arxiv_cs_CR_dataset.json")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--gpu_batch_size", type=int, default=30)
    parser.add_argument("--paper_title", type=str, default="Detecting Phishing Sites Using ChatGPT")
    parser.add_argument("--paper_abstract", type=str)

    args = parser.parse_args()

    test_paper = {
        "title": args.paper_title,
        "abstract": args.paper_abstract
    }

    # Load dataset
    dataset = read_dataset(args.dataset_path)
    # Initialize the model
    transformer = TransformerBody(test_paper["abstract"], args.model_name, args.device)
    # Extract hidden states
    dataset = dataset.map(transformer.extract_similarity, batched=True, batch_size=args.gpu_batch_size)

    # Extract the paper that has the top10 high similarity
    similarity = dataset["dataset"]["similarity"]
    top10_idx = torch.Tensor(similarity).argsort(descending=True)[:10].tolist()  # convert tensor to list
    top10_title = [dataset["dataset"][i]["title"] for i in top10_idx]

    print("Top 10 similar papers:")
    for i, title in enumerate(top10_title):
        if "\n" in title:
            title = title.replace("\n", " ")
        print(f"{i + 1}. {title} // similarity: {similarity[top10_idx[i]]:.3f}")

