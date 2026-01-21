import numpy as np
import pandas as pd
from pandas import Series
from transformers import PreTrainedTokenizerBase, TFPreTrainedModel

from settings import MODEL_PATH


def save_genre_logits(
        descriptions_list: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        model: TFPreTrainedModel
) -> np.ndarray:
    embeddings = []
    print(f"Starting creating embeddings for  {len(descriptions_list)} books...")
    for text in descriptions_list:
        inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=512)
        outputs = model(**inputs)
        logits = outputs.logits.numpy()[0]
        embeddings.append(logits)
    embeddings = np.array(embeddings)
    book_embeddings_path = MODEL_PATH / "book_logits_base.npy"
    np.save(book_embeddings_path, embeddings)
    return embeddings


def get_genres_logits(path: str = MODEL_PATH / "book_logits_base.npy") -> np.ndarray:
    return np.load(path)


def save_semantic_embeddings(df: Series, tokenizer: PreTrainedTokenizerBase, model: TFPreTrainedModel):
    descriptions = df.fillna("").tolist()
    batch_size = 16
    all_embeddings = []

    print(f"Starting save {len(descriptions)} descriptions to embeddings...")
    for i in range(0, len(descriptions), batch_size):
        batch_texts = descriptions[i:i + batch_size]

        inputs = tokenizer(batch_texts, return_tensors="tf", padding=True, truncation=True, max_length=512)

        outputs = model(inputs, output_hidden_states=True)
        embeddings = outputs.hidden_states[-1][:, 0, :].numpy()

        all_embeddings.append(embeddings)

    final_embeddings = np.vstack(all_embeddings)
    book_embeddings_path = MODEL_PATH / "books_embeddings_768.npy"
    np.save(book_embeddings_path, final_embeddings)


def get_semantic_embeddings(path: str = MODEL_PATH / "books_embeddings_768.npy") -> np.ndarray:
    return np.load(path)
