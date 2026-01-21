import os
import ast

import pandas as pd
from collections import Counter

from pandas import DataFrame


class BookDataCollector:

    def __init__(self):
        self.dataset_path = os.path.join(os.getcwd(), 'datasets', 'goodreads_ds.csv')
        self.df = pd.read_csv(self.dataset_path)

    def read_sample_data(self) -> DataFrame:
        return self.df

    def most_common_genres(self) -> list:
        self.df["genres_list"] = self.df["genres"].apply(lambda x: ast.literal_eval(x) if x else [])
        all_genres = [genre for sublist in self.df['genres_list'] for genre in sublist]
        genre_counts = Counter(all_genres).most_common(100)
        return [genre for genre, count in genre_counts]

    def all_genres(self) -> DataFrame:
        genres = self.df['genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        return genres

    def all_descriptions(self) -> DataFrame:
        return self.df['description']

    def combined_text(self) -> DataFrame:
        author = self.df['author'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x).fillna('')
        combined_text = (self.df['id'].astype(str)
                         + self.df['title'].fillna('')
                         + ' '
                         + self.df['description'].fillna('')
                         + ' '
                         + author.str.join(' '))
        self.df['combined_text'] = combined_text.str.replace(r'[^a-zA-Z\s]', ' ', regex=True)

        return self.df['combined_text']


book_cl = BookDataCollector()
