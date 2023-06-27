import numpy as np
import torch
from datasets import DatasetDict
from sklearn.metrics import classification_report, accuracy_score
from transformers import AutoModel, AutoTokenizer


class SimpleClassifiers:
    def __init__(self, dataset: DatasetDict):
        self.X_train = np.array(dataset["training"]["hidden_state"])
        self.y_train = np.array(dataset["training"]["label"])
        self.X_valid = np.array(dataset["validation"]["hidden_state"])
        self.y_valid = np.array(dataset["validation"]["label"])

    def random_forest(self):
        from sklearn.ensemble import RandomForestClassifier

        clf = RandomForestClassifier(random_state=0)
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_valid)
        print("#### Random Forest Report")
        print(classification_report(self.y_valid, y_pred, digits=6))

    def xgboost(self):
        from xgboost import XGBClassifier

        clf = XGBClassifier(random_state=0)
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_valid)
        print("#### XGBoost Report")
        print(classification_report(self.y_valid, y_pred, digits=6))

    def neural_network(self):
        from sklearn.neural_network import MLPClassifier

        clf = MLPClassifier(random_state=0)
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_valid)
        print("#### Neural Network Report")
        print(classification_report(self.y_valid, y_pred, digits=6))

    def evaluate_all(self):
        self.random_forest()
        self.xgboost()
        self.neural_network()


class TransformerBody:
    def __init__(self, model_name: str, device: str):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer_model = AutoModel.from_pretrained(model_name).to(self.device)

    def tokenize(self, data: DatasetDict) -> DatasetDict:
        # Tokenize the texts
        tokenized_inputs = self.tokenizer(data['visible_text'], padding="max_length", max_length=512, truncation=True,
                                          return_tensors="pt")
        return tokenized_inputs

    def extract_hidden_states(self, batch):
        inputs = {k: v.to(self.device) for k, v in batch.items() if k in self.tokenizer.model_input_names}
        with torch.no_grad():
            last_hidden_state = self.transformer_model(**inputs).last_hidden_state
        return {"hidden_state": last_hidden_state[:, 0].cpu().numpy()}