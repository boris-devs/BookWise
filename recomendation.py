import numpy as np
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import PreTrainedTokenizerBase, TFPreTrainedModel


class HybridRecommender:
    def __init__(self, df: DataFrame):
        self.content_model = None
        self.collab_model = None
        self.book_ids = df.index.tolist()

    def build_content_based(self, combined_text: DataFrame):
        """Контентна фільтрація на основі TF-IDF"""
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, min_df=3)
        tfidf_matrix = vectorizer.fit_transform(combined_text)

        self.tfidf_matrix = tfidf_matrix
        self.vectorizer = vectorizer

    def recommend_similar_by_id_book(self, row_index: int, top_n: int = 5):
        """Рекомендація схожих книг за id"""
        idx = self.book_ids.index(row_index)
        sim_scores = list(enumerate(
            cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix)[0]
        ))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]

        return sim_scores

    def recommend_similar_by_desc(
            self,
            description_user: str,
            all_books_embeddings: DataFrame,
            tokenizer: PreTrainedTokenizerBase,
            model: TFPreTrainedModel,
            top_n: int = 20):
        """Рекомендація схожих книг за описом"""
        inputs = tokenizer(description_user, return_tensors="tf", truncation=True)
        outputs = model(**inputs, output_hidden_states=True)
        embeddings_user = outputs.hidden_states[-1][:, 0, :].numpy()

        similarities = cosine_similarity(embeddings_user, all_books_embeddings)[0]

        top_indices = np.argsort(similarities)[-top_n:][::-1]
        return [(idx, similarities[idx]) for idx in top_indices]
