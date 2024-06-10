from sentence_transformers import SentenceTransformer
from typing import List, Dict, Union, Optional
import weave
from dataclasses import dataclass


class SentenceTransformersModel(weave.Model):
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    model: Optional[SentenceTransformer] = None
    model_type: str = "sentence_transformers"  # TODO: Replace with naem
    name: str = "SentenceTransformersModel"

    def __init__(
        self,
        model_name: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        super().__init__(
            model_name=model_name
        )
        self.model_name = model_name
        self.model_type = "sentence_transformers"
        # Initialize the Sentence Transformer model

        self.set_model(model_name)

    # The first weave op may cuase an incomplete object to be serialized so
    # we seperate the two set models for now until we have a better weave way
    def set_model(self, model_name: str) -> str:
        self.model = SentenceTransformer(model_name)
        return self.model_type

    @weave.op()
    def weave_set_model(self, model_name: str) -> str:
        self.model = SentenceTransformer(model_name)
        return self.model_type

    @weave.op()
    def predict(self, texts: List[str]) -> Dict[str, List[float]]:
        """
        Embed incoming text and return the embeddings.

        Args:
            texts (List[str]): List of texts to be embedded.

        Returns:
            Dict[str, List[float]]: Dictionary containing the embeddings.
        """
        self.weave_set_model(self.model_name)
        embeddings = self.model.encode(texts)  # Compute embeddings
        # Convert to list for JSON serialization
        embedding_data = [{"embedding": embedding.tolist()}
                          for embedding in embeddings]
        return embedding_data

    @weave.op()
    def litellm_compatible_embedding(self, input: Union[List[str], str], model: Optional[str] = None) -> List[float]:

        @dataclass
        class LitellmCompatibleEmbedding:
            data: List[Dict[str, List[float]]]

        compatible_embedding = LitellmCompatibleEmbedding(
            data=self.predict([input] if isinstance(input, str) else input))
        return compatible_embedding
