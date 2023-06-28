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
      "abstract": "<abstract content>"
    },
    {
      "title": "KYChain: User-Controlled KYC Data Sharing and Certification",
      "abstract": "<abstruct contents>"
    }
  ]
}
```

