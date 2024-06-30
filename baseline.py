import numpy as np
from sentence_transformers import util, SentenceTransformer
import pandas as pd


class find_school():

    def __init__(self, model: str ='sentence-transformers/LaBSE') -> None:
        self.model = SentenceTransformer(model)
        
    def encode(self, sentence: str|list) -> np.ndarray:
        return self.model.encode(sentence)

    def find_similar(self, query:np.ndarray, corpus: np.ndarray) -> np.ndarray:
        return util.semantic_search(query, corpus, top_k=2)
    
    @staticmethod
    def clean_input(self, sentence:str) -> str:
        cleaned_sentence = sentence
        return cleaned_sentence



corpus = [find_school.clean_input(x) for x in pd.read_csv('reference_schools.csv', index_col=0).name.values]
model = find_school()
corpus = model.encode(corpus)
query = model.clean_input(input())
result = model.find_similar(query, corpus)