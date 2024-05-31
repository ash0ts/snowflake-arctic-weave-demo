from typing import List, Optional

import numpy as np
import weave
from litellm import embedding
from pydantic import Field
from weave import Object


class VectorStore(Object):
    embedding_model: str = Field(default="text-embedding-3-small")
    articles: List[str]
    article_embeddings: List[List[float]] = Field(default_factory=list)

    def __init__(
        self,
        # Make this an refable index
        embedding_model: Optional[str] = "text-embedding-3-small",
        articles: List[str] = [],
    ):
        super().__init__(embedding_model=embedding_model, articles=articles)
        self.embedding_model = embedding_model
        self.articles = articles
        self.article_embeddings = self.docs_to_embeddings(self.articles)

    def docs_to_embeddings(self, docs: List[str]) -> List[List[float]]:
        document_embeddings = []
        for doc in docs:
            response = embedding(input=doc, model=self.embedding_model).data[0][
                "embedding"
            ]
            document_embeddings.append(response)
        return document_embeddings

    @weave.op()
    def get_most_relevant_document(self, query: str) -> str:
        query_embedding = embedding(input=query, model=self.embedding_model).data[0][
            "embedding"
        ]
        similarities = [
            np.dot(query_embedding, doc_emb)
            / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb))
            for doc_emb in self.article_embeddings
        ]
        # Get the index of the most similar document
        most_relevant_doc_index = np.argmax(similarities)
        return self.articles[most_relevant_doc_index]
