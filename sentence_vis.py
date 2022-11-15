import json
from collections import Counter
from collections import defaultdict
from typing import List

import hdbscan
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import umap
from matplotlib import pyplot as plt
from transformers import AutoModel
from transformers import AutoTokenizer

DATA_PATH = "data/short_tweet.txt"
JSON_DATA_PATH = "data/Appliances.json"

TOKENIZER = "sentence-transformers/all-MiniLM-L6-v2"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def import_json_data(JSON_DATA_PATH: str, max_cnt: int = 500) -> List:
    """
    Read data from json file.
    """
    reviews = defaultdict(list)
    cnt = 0
    try:
        with open(JSON_DATA_PATH, "r") as f:
            for line in f:
                record = json.loads(line)
                if record.get("reviewText") and record.get("overall"):
                    reviews["reviewText"].append(record["reviewText"])
                    reviews["overall"].append(record["overall"])
                cnt += 1
                if cnt >= max_cnt:
                    break
            return reviews
    except:
        raise Exception("The given JSON_DATA_PATH is not valid")


def import_txt_data(DATA_PATH: str) -> List:
    """
    Import the data.
    Output the raw dataset as a list.
    """
    try:
        with open(DATA_PATH) as f:
            lines = [line.rstrip() for line in f]
            return lines
    except:
        raise Exception("The given DATA_PATH is not valid")


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def sentence_transform(tweets_list: List, tokenizer: str, model: str) -> List:
    """
    Convert the tweets into the embeddings.
    Output the numerical values of the embeddings.
    """

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    model = AutoModel.from_pretrained(model)

    # Tokenize sentences
    encoded_input = tokenizer(
        tweets_list, padding=True, truncation=True, return_tensors="pt"
    )

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    return sentence_embeddings


# class varianceReductionImplementer(self):
def reduce_dimension(embeddings):
    """
    Perform dimension reduction.
    """
    embedder = umap.UMAP(n_neighbors=15, n_components=2, metric="cosine")
    return embedder.fit_transform(embeddings)


def clustering(embeddings):
    """
    Clustering the embeddings.
    """
    labels = hdbscan.HDBSCAN(
        min_samples=None,
        min_cluster_size=10,
    ).fit_predict(embeddings)
    return labels


def main():
    # Import dataset
    texts = import_json_data(JSON_DATA_PATH=JSON_DATA_PATH)
    # Feed the texts into the BERT model
    embeddings = sentence_transform(texts["reviewText"], TOKENIZER, MODEL)
    # Variance Reduction
    embeddings_red = reduce_dimension(embeddings)
    # Clustering
    labels = clustering(embeddings_red)
    print(Counter(labels))
    # print(labels)
    fig, ax = plt.subplots()
    scatter = ax.scatter(
        embeddings_red[:, 0],
        embeddings_red[:, 1],
        c=texts["overall"],
        label=texts["overall"],
    )
    # plt.legend()
    legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
    ax.add_artist(legend1)

    plt.show()
    plt.savefig("embeddings_vis.png")


if __name__ == "__main__":
    main()
