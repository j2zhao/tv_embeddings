import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def default_answer_query(question_embedding, embeddings:pd.DataFrame, top_values:int, embeddings_name: str, id_name:str):
    """
    Given a question and embeddings -> return top 
    """
    cosine_similarities = cosine_similarity([question_embedding], embeddings[embeddings_name])[0]
    top_indices = cosine_similarities.argsort()[-top_values:][::-1]
    top_ids = []
    for index in top_indices:
        top_ids.append(embeddings.at[index, id_name])
    return [top_ids]

def check_answer(real_answer, top_ids, n):
    """
    given real_answer and top_values, return true, false
    """
    if isinstance(real_answer, list):
        for ans in real_answer:
            if ans in top_ids[:n]:
                return True
        return False
    else:
        return real_answer in top_ids[:n]