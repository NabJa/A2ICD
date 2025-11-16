import os

import pandas as pd
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from tqdm import trange

load_dotenv()


def load_vector_store() -> Chroma:
    """Load the vector store containing ICD-10 codes."""

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vector_store = Chroma(
        chroma_cloud_api_key=os.getenv("CHROMA_CLOUD_API_KEY"),
        tenant=os.getenv("CHROMA_CLOUD_TENANT_ID"),
        collection_name="icd",
        database="icd",
        embedding_function=embeddings,
    )

    return vector_store


def create_vector_store() -> None:
    icd_df = pd.read_csv(
        "https://raw.githubusercontent.com/Bobrovskiy/ICD-10-CSV/refs/heads/master/2020/diagnosis.csv"
    )

    icd_df["text"] = icd_df["CodeWithSeparator"] + ": " + icd_df["LongDescription"]

    texts = icd_df["text"].to_list()

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vector_store = Chroma(
        chroma_cloud_api_key=os.getenv("CHROMA_CLOUD_API_KEY"),
        tenant=os.getenv("CHROMA_CLOUD_TENANT_ID"),
        collection_name="icd",
        database="icd",
        embedding_function=embeddings,
    )

    for i in trange(0, len(texts), 300):
        batch_texts = texts[i : i + 300]
        _ = vector_store.add_texts(batch_texts)


if __name__ == "__main__":
    create_vector_store()
