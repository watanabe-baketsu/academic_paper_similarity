from sentence_transformers import SentenceTransformer, util


class TransformerBody:
    def __init__(self, source_sentence: str, model_name: str, device: str):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.source_embedding = self.model.encode(source_sentence, convert_to_tensor=True)

    def extract_similarity(self, batch):
        sentence = batch["abstract"]
        sentence_embedding = self.model.encode(sentence, convert_to_tensor=True)
        similarity = util.cos_sim(self.source_embedding, sentence_embedding)
        return {"similarity": similarity[0]}
