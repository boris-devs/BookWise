import streamlit as st
import pickle

from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from data_collector import book_cl
from recomendation import HybridRecommender
from settings import MODEL_PATH
from utils import get_semantic_embeddings


@st.cache_resource
def get_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = TFAutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model


@st.cache_data
def load_all_data():
    df = book_cl.read_sample_data()
    embeddings = get_semantic_embeddings()
    return df, embeddings


class BookWiseApp:
    def __init__(self, dataset_df, semantic_embeddings, recommender: HybridRecommender):
        self.df = dataset_df
        self.semantic_embeddings = semantic_embeddings
        self.recommender = recommender

    def run(self):
        st.set_page_config(page_title="BookWise AI", page_icon="📚", layout="wide")
        st.title("📚 BookWise - Розумна система рекомендації книг")

        with st.status("Ініціалізація системи AI...", expanded=False) as status:
            st.write("Завантаження ваг BERT (768d)...")
            tokenizer, model = get_model_and_tokenizer(MODEL_PATH)

            st.write("Завантаження класифікатора жанрів...")
            with open(MODEL_PATH / "mlb.pkl", "rb") as f:
                mlb = pickle.load(f)

            status.update(label="Система готова!", state="complete", expanded=False)


        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Пошук за схожою книгою")
            name_book = st.selectbox("Почніть писати назву книги:", df["title"].values)
            row = df[df["title"] == name_book].iloc[0]

            id_book = int(row['id'])

            if st.button("Знайти схожі книги за книгою"):
                if id_book:
                    results = self.recommender.recommend_similar_by_id_book(id_book)
                    for idx, score in results:
                        st.info(f"📖 **{self.df.iloc[idx]['title']}**")
                else:
                    st.warning("Виберіть назву!")
        with col2:
            st.subheader("🔍 Пошук схожих книг за описом")
            description = st.text_area(
                "Опишіть, що ви хочете почитати:",
                placeholder="Наприклад: A story about a boy who discovers he is a wizard...",
                height=150
            )

            if st.button("Знайти книги за описом"):
                if description:
                    with st.spinner('BERT аналізує ваш запит...'):
                        recommendations = self.recommender.recommend_similar_by_desc(
                            description,
                            self.semantic_embeddings,
                            tokenizer,
                            model
                        )

                    st.markdown("### ✨ Найкращі збіги:")
                    for idx, score in recommendations:
                        book_title = self.df.iloc[idx]['title']
                        book_author = self.df.iloc[idx]['author']

                        with st.container():
                            st.success(f"**{book_title}** — {book_author}")
                            st.caption(f"Точність збігу: {score * 100:.1f}%")
                            st.divider()
                else:
                    st.error("Опис не може бути порожнім!")


if __name__ == "__main__":
    try:
        df, embeddings = load_all_data()
        hybrid_recommend = HybridRecommender(df)
        hybrid_recommend.build_content_based(book_cl.combined_text())
        app = BookWiseApp(
            dataset_df=df,
            semantic_embeddings=embeddings,
            recommender=hybrid_recommend
        )

        app.run()
    except Exception as e:
        st.error(f"Помилка при запуску додатка: {e}")
        st.stop()
