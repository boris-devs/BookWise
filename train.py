from classifier import GenreClassifier
from settings import MODEL_PATH
from data_collector import book_cl
from utils import save_semantic_embeddings


def run_training_model():
    if any(MODEL_PATH.iterdir()):
        print(f"Model already exists in {MODEL_PATH}! To re-train, delete this folder or its content.")
        return

    print("Starting training...")
    common_genres = book_cl.most_common_genres()
    genres_classifier = GenreClassifier(num_genres=len(common_genres))

    genres_classifier.train(
        book_cl.all_descriptions(),
        common_genres,
        book_cl.all_genres()
    )
    save_semantic_embeddings(book_cl.all_descriptions(), genres_classifier.tokenizer, genres_classifier.model)
    print("Training completed successfully.")


if __name__ == "__main__":
    run_training_model()
