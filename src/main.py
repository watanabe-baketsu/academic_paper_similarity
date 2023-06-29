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

    args = parser.parse_args()

    test_paper = {
        "title": "Detecting Phishing Sites Using ChatGPT",
        "abstract": "The rise of large language models (LLMs) has had a significant impact on various domains, "
                    "including natural language processing and artificial intelligence. While LLMs such as ChatGPT "
                    "have been extensively researched for tasks such as code generation and text synthesis, "
                    "their application in detecting malicious web content, particularly phishing sites, "
                    "has been largely unexplored. To combat the rising tide of automated cyber attacks facilitated by "
                    "LLMs, it is imperative to automate the detection of malicious web content, which requires "
                    "approaches that leverage the power of LLMs to analyze and classify phishing sites. In this "
                    "paper, we propose a novel method that utilizes ChatGPT to detect phishing sites. Our approach "
                    "involves leveraging a web crawler to gather information from websites and generate prompts based "
                    "on this collected data. This approach enables us to detect various phishing sites without the "
                    "need for fine-tuning machine learning models and identify social engineering techniques from the "
                    "context of entire websites and URLs. To evaluate the performance of our proposed method, "
                    "we conducted experiments using a dataset. The experimental results using GPT-4 demonstrated "
                    "promising performance, with a precision of 98.3% and a recall of 98.4%. Comparative analysis "
                    "between GPT-3.5 and GPT-4 revealed an enhancement in the latter's capability to reduce false "
                    "negatives. These findings not only highlight the potential of LLMs in efficiently identifying "
                    "phishing sites but also have significant implications for enhancing cybersecurity measures and "
                    "protecting users from the dangers of online fraudulent activities."
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

