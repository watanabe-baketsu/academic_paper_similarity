import json
import random

from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="../datasets/arxiv-metadata-oai-snapshot.json")
    parser.add_argument("--category", type=str, default="cs.CR")

    args = parser.parse_args()

    # JSON ファイルの読み込み
    with open(args.dataset_path, 'r') as f:
        data = [json.loads(line) for line in f]

    new_data = []
    for record in data:
        if " " in record['categories']:
            record_categories = record['categories'].split(" ")
            if args.category not in record_categories:
                continue
        if record['categories'] != args.category:
            continue
        new_record = {
            'title': record['title'],
            'abstract': record['abstract'],
            'categories': record['categories'],
        }
        new_data.append(new_record)
    print(f"Number of records: {len(new_data)}")

    # データのシャッフル
    random.shuffle(new_data)

    # データの合計サイズを取得
    total_size = len(new_data)

    new_json = {
        "dataset": new_data,
        "category": args.category,
        "total_size": total_size
    }

    with open(f"../datasets/arxiv_{args.category.replace('.', '_')}_dataset.json", 'w') as f:
        json.dump(new_json, f, indent=4)
