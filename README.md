# paper_classification
For Academic Paper Category Classification

## Dataset Structure
The model is trained on the dataset from [here](https://www.kaggle.com/datasets/Cornell-University/arxiv).
Preprocessing is done on the dataset to remove the unnecessary columns and to remove the rows with missing values.
The ratio of training, validation and testing data is 8:1:1.

### Preprocessed Dataset
```json
{
    "training": [
        {
            "id": "2301.02213",
            "title": "<paper title>",
            "abstract": "<abstract strings>",
            "categories": "cs.LO math.LO",
            "label": [0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0]
          }
    ],
    "validation": [
        {
            "id": "2301.02214",
            "title": "<paper title>",
            "abstract": "<abstract strings>",
            "categories": "cs.LO",
            "label": [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
          }
    ],
    "testing": [
        {
            "id": "2301.02215",
            "title": "<paper title>",
            "abstract": "<abstract strings>",
            "categories": "cs.LO math.LO",
            "label": [0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0]
          }
    ]
}
```

