import pandas as pd
from gritlm import GritLM
import numpy as np
model_path = "GritLM/GritLM-7B"
raw_instruction = "Given a research query, retrieve the title and abstract of the relevant research paper"
raw_instruction = "Given a research query, retrieve the title and abstract of the relevant research paper"
raw_instruction = "Given a research query, retrieve the passage from the relevant research paper"

def generate_embeddings(texts:pd.DataFrame, raw_instruction:str):
    print(len(texts))
    model = GritLM(model_path, torch_dtype="auto", device_map="auto", mode="embedding") 
    print(model.device)
    ndarr = model.encode(texts, batch_size=256, 
                 instruction=raw_instruction, 
                 show_progress_bar=True).astype(np.float16)
    print(ndarr.shape)
    for i, embedding in enumerate(ndarr):
        ndarr[i] = np.zeros_like(embedding)
    return pd.Series(ndarr)