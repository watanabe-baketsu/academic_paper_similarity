# Academic Paper Similarity Inference

## Dataset Structure
The model is trained on the dataset from [here](https://www.kaggle.com/datasets/Cornell-University/arxiv).
Preprocessing is done on the dataset to remove the unnecessary columns and to remove the rows with missing values.

### Preprocessed Dataset
```json
{
  "dataset": [
    {
      "title": "Statistical Analysis Based Feature Selection Enhanced RF-PUF with >99.8%\n  Accuracy on Unmodified Commodity Transmitters for IoT Physical Security",
      "abstract": "<abstract content>",
      "categories": "cs.CR cs.IT cs.LG cs.MM cs.NI cs.SE"
    },
    {
      "title": "KYChain: User-Controlled KYC Data Sharing and Certification",
      "abstract": "<abstruct contents>",
        "categories": "cs.CR"
    }
  ],
  "total": 10000,
  "category": "cs.CR"
}
```

## Demo
```bash
cd src
uvicorn main:app --reload
```
