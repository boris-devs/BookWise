import pickle
import tensorflow as tf
from pandas import DataFrame
from transformers import (AutoTokenizer, TFAutoModelForSequenceClassification, PreTrainedTokenizerBase,
                          TFPreTrainedModel)
from sklearn.preprocessing import MultiLabelBinarizer

from settings import MODEL_PATH


class GenreClassifier:
    def __init__(self, num_genres=100):
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = TFAutoModelForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=num_genres
        )
        self.mlb = None

    def train(self, descriptions: DataFrame, common_genres: list, all_genres: DataFrame):
        inputs = self.tokenizer(descriptions.tolist(),
                                padding="max_length",
                                truncation=True,
                                max_length=128,
                                return_tensors='tf')

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.BinaryAccuracy()]
        )

        train_data = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        }

        self.mlb = MultiLabelBinarizer(classes=common_genres)
        genre_matrix = self.mlb.fit_transform(all_genres)

        history = self.model.fit(
            train_data,
            genre_matrix,
            epochs=3,
            batch_size=8,
            validation_split=0.2
        )

        self.model.save_pretrained(MODEL_PATH)
        self.tokenizer.save_pretrained(MODEL_PATH)

        with open(MODEL_PATH / "mlb.pkl", "wb") as f:
            pickle.dump(self.mlb, f)
        print(f"Saved in: {MODEL_PATH}")

        return history

    def load_model(self) -> tuple[
        PreTrainedTokenizerBase,
        TFPreTrainedModel,
        MultiLabelBinarizer
    ]:
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

        with open(MODEL_PATH / "mlb.pkl", "rb") as f:
            self.mlb = pickle.load(f)

        return self.tokenizer, self.model, self.mlb


def genres_predict(desc, tokenizer: PreTrainedTokenizerBase, model: TFPreTrainedModel,
                   mlb: MultiLabelBinarizer, threshold=0.5):
    inputs = tokenizer(desc, return_tensors="tf")
    outputs = model(**inputs)

    probs = tf.nn.sigmoid(outputs.logits).numpy()[0]
    predictions = []
    for genre_name, prob in zip(mlb.classes_, probs):
        if prob > threshold:
            predictions.append({"genre": genre_name, "percent": f"{prob * 100:.1f}"})

    predictions.sort(key=lambda k: k["percent"], reverse=True)
    return predictions
