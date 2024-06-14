import concurrent.futures
from typing import Callable, List, Literal, Optional, Union

import numpy as np
import weave
from litellm import embedding
from scipy.spatial.distance import chebyshev, cityblock, hamming, jaccard, minkowski
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from weave_example_demo.llm_types.models.embedding_model import (
    SentenceTransformersModel,
)


class VectorStore(weave.Object):

    embedding_model: Optional[Union[str, SentenceTransformersModel]] = (
        "text-embedding-3-small"
    )
    embedding_function: Optional[Callable] = None
    articles: Union[List[str], List[dict], str] = []
    article_embeddings: Optional[List[dict]] = None
    key: Optional[str] = None
    multiprocess_embeddings: Optional[bool] = False
    ranking_method: Literal[
        "cosine",
        "euclidean",
        "manhattan",
        "chebyshev",
        "minkowski",
        "hamming",
        "jaccard",
    ] = "cosine"
    embeddings_matrix: Optional[np.ndarray] = None
    name: str = "VectorStore"

    def __init__(
        self,
        articles: Union[List[str], List[dict], str],
        embedding_model: Optional[
            Union[str, SentenceTransformersModel]
        ] = "text-embedding-3-small",
        key: Optional[str] = None,
        multiprocess_embeddings: Optional[bool] = False,
        ranking_method: str = "cosine",
    ):
        super().__init__(
            embedding_model=embedding_model,
            articles=articles,
            key=key,
            multiprocess_embeddings=multiprocess_embeddings,
        )
        self.embedding_model = embedding_model
        self.init_embedding_model(embedding_model)
        self.key = key
        self.multiprocess_embeddings = multiprocess_embeddings
        if isinstance(articles, str) and articles.startswith("weave://"):
            articles_ref = weave.ref(articles)
            articles_data = articles_ref.get()
            # TODO: Remove the [:250] once we have a better way to handle large datasets
            self.articles = [dict(row) for row in articles_data.rows][:100]
        else:
            self.articles = articles
        self.article_embeddings = self.docs_to_embeddings(self.articles)

        # Initialize ranking methods
        self.ranking_method = ranking_method

        # Precompute embeddings for cosine similarity and Euclidean distance
        self.embeddings_matrix = np.array(
            [doc_emb["embedding"] for doc_emb in self.article_embeddings]
        )

    @weave.op()
    def set_embedding_model(
        self,
        embedding_model: Optional[
            Union[str, SentenceTransformersModel]
        ] = "text-embedding-3-small",
    ):
        self.embedding_model = embedding_model
        self.weave_init_embedding_model(embedding_model)

    @weave.op()
    def weave_init_embedding_model(
        self,
        embedding_model: Optional[
            Union[str, SentenceTransformersModel]
        ] = "text-embedding-3-small",
    ):
        name = None
        model_type = getattr(embedding_model, "model_type", None)
        if model_type == "sentence_transformers":
            self.embedding_function = embedding_model.litellm_compatible_embedding
            name = "sentence_transformers"
        else:
            self.embedding_function = embedding
            name = "litellm"

        return {"embedding_function": name}

    def init_embedding_model(
        self,
        embedding_model: Optional[
            Union[str, SentenceTransformersModel]
        ] = "text-embedding-3-small",
    ):
        name = None
        if isinstance(embedding_model, SentenceTransformersModel):
            self.embedding_function = embedding_model.litellm_compatible_embedding
            name = "sentence_transformers"
        else:
            self.embedding_function = embedding
            name = "litellm"

        return {"embedding_function": name}

    def docs_to_embeddings(self, docs: Union[List[dict], List[str]]) -> List[dict]:

        def embed_doc(doc):
            text_to_embed = doc[self.key] if self.key else doc
            # TODO: Add batches of data to make it faster
            response = self.embedding_function(
                input=text_to_embed, model=self.embedding_model
            ).data[0]["embedding"]
            return {"embedding": response, "original_doc": doc}

        document_embeddings = []
        if self.multiprocess_embeddings:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(embed_doc, doc) for doc in docs]
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="Embedding documents",
                ):
                    document_embeddings.append(future.result())
        else:
            for doc in tqdm(docs, desc="Embedding documents"):
                document_embeddings.append(embed_doc(doc))

        return document_embeddings

    @weave.op()
    def get_distance_function(self, ranking_method: str) -> Callable:
        distance_functions = {
            "cosine": self._get_cosine_distances,
            "euclidean": self._get_euclidean_distances,
            "manhattan": self._get_manhattan_distances,
            "chebyshev": self._get_chebyshev_distances,
            "minkowski": self._get_minkowski_distances,
            "hamming": self._get_hamming_distances,
            "jaccard": self._get_jaccard_distances,
        }
        if ranking_method not in distance_functions:
            raise ValueError(f"Unknown ranking method: {ranking_method}")
        return distance_functions[ranking_method]

    @weave.op()
    def get_most_relevant_documents(
        self,
        query: str,
        n: int = 1,
        ranking_method: Optional[
            Literal[
                "cosine",
                "euclidean",
                "manhattan",
                "chebyshev",
                "minkowski",
                "hamming",
                "jaccard",
            ]
        ] = None,
    ) -> list:
        if ranking_method:
            _ranking_method = ranking_method
        else:
            _ranking_method = self.ranking_method

        # BUG: Serlializing the internal function of an underlying weave object
        # is causing a strange BoxedStr error. This is a workaround for now
        self.weave_init_embedding_model(self.embedding_model)
        distance_function = self.get_distance_function(_ranking_method)
        return distance_function(query, n)

    @weave.op()
    def _get_cosine_distances(self, query: str, n: int) -> list:
        query_embedding = self.embedding_function(
            input=query, model=self.embedding_model
        ).data[0]["embedding"]
        cosine_similarities = cosine_similarity(
            [query_embedding], self.embeddings_matrix
        ).flatten()
        top_n_indices = np.argsort(cosine_similarities)[-n:][::-1]
        return [
            {
                "document": self.article_embeddings[i]["original_doc"],
                "score": cosine_similarities[i],
            }
            for i in top_n_indices
        ]

    @weave.op()
    def _get_euclidean_distances(self, query: str, n: int) -> list:
        query_embedding = self.embedding_function(
            input=query, model=self.embedding_model
        ).data[0]["embedding"]
        euclidean_distances = np.linalg.norm(
            self.embeddings_matrix - query_embedding, axis=1
        )
        top_n_indices = np.argsort(euclidean_distances)[:n]
        return [
            {
                "document": self.article_embeddings[i]["original_doc"],
                "score": euclidean_distances[i],
            }
            for i in top_n_indices
        ]

    @weave.op()
    def _get_manhattan_distances(self, query: str, n: int) -> list:
        query_embedding = self.embedding_function(
            input=query, model=self.embedding_model
        ).data[0]["embedding"]
        distances = np.array(
            [cityblock(query_embedding, emb) for emb in self.embeddings_matrix]
        )
        top_n_indices = np.argsort(distances)[:n]
        return [
            {
                "document": self.article_embeddings[i]["original_doc"],
                "score": distances[i],
            }
            for i in top_n_indices
        ]

    @weave.op()
    def _get_chebyshev_distances(self, query: str, n: int) -> list:
        query_embedding = self.embedding_function(
            input=query, model=self.embedding_model
        ).data[0]["embedding"]
        distances = np.array(
            [chebyshev(query_embedding, emb) for emb in self.embeddings_matrix]
        )
        top_n_indices = np.argsort(distances)[:n]
        return [
            {
                "document": self.article_embeddings[i]["original_doc"],
                "score": distances[i],
            }
            for i in top_n_indices
        ]

    @weave.op()
    def _get_minkowski_distances(self, query: str, n: int, p: int = 3) -> list:
        query_embedding = self.embedding_function(
            input=query, model=self.embedding_model
        ).data[0]["embedding"]
        distances = np.array(
            [minkowski(query_embedding, emb, p)
             for emb in self.embeddings_matrix]
        )
        top_n_indices = np.argsort(distances)[:n]
        return [
            {
                "document": self.article_embeddings[i]["original_doc"],
                "score": distances[i],
            }
            for i in top_n_indices
        ]

    @weave.op()
    def _get_hamming_distances(self, query: str, n: int) -> list:
        query_embedding = self.embedding_function(
            input=query, model=self.embedding_model
        ).data[0]["embedding"]
        distances = np.array(
            [hamming(query_embedding, emb) for emb in self.embeddings_matrix]
        )
        top_n_indices = np.argsort(distances)[:n]
        return [
            {
                "document": self.article_embeddings[i]["original_doc"],
                "score": distances[i],
            }
            for i in top_n_indices
        ]

    @weave.op()
    def _get_jaccard_distances(self, query: str, n: int) -> list:
        query_embedding = self.embedding_function(
            input=query, model=self.embedding_model
        ).data[0]["embedding"]
        distances = np.array(
            [jaccard(query_embedding, emb) for emb in self.embeddings_matrix]
        )
        top_n_indices = np.argsort(distances)[:n]
        return [
            {
                "document": self.article_embeddings[i]["original_doc"],
                "score": distances[i],
            }
            for i in top_n_indices
        ]
