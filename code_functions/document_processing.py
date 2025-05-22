from datasets import load_dataset
import pandas as pd


def retrieve_dataset_generate(path,
                            dataset_name, 
                            split="full",
                            columns=[],
                            joined_columns={}):
    corpus_data = load_dataset(path, dataset_name, split=split)
    corpus_data = corpus_data.to_pandas()
    if len(columns) > 0:
        corpus_data = corpus_data[columns]
    for new_col in joined_columns:
        corpus_data[new_col] = corpus_data[joined_columns[new_col][0]]
        for col in joined_columns[new_col][1:]:
            corpus_data[new_col] += '\n' + corpus_data[col]
        corpus_data.drop(columns=joined_columns[new_col], inplace=True)
    return corpus_data

def retrieve_dataset_match(id_df: pd.DataFrame, 
                           id_column, 
                           path, 
                           dataset_name, 
                           split="full", 
                           columns=[]):
    corpus_data = load_dataset(path, dataset_name, split=split)
    corpus_data = corpus_data.to_pandas()
    if len(columns) > 0:
        corpus_data = corpus_data[columns]
    corpus_data = pd.merge(id_df, corpus_data, on=id_column, how='left')
    corpus_data.drop(columns=[id_column], inplace=True)
    # if is_s2orc:
    #     corpus_data["externalids"] = 
    return corpus_data

# def add_external_paper_link(org_df:pd.DataFrame, 
#                             column_id:str, 
#                             path,
#                             dataset_name,
#                             columns=[],
#                             joined_columns={}):
    
#     corpus_external_data = retrieve_dataset_generate("princeton-nlp/LitSearch", "corpus_s2orc")

    

# def retrieve_queries_dataset():
#     query_data = load_dataset("princeton-nlp/LitSearch", "query", split="full")