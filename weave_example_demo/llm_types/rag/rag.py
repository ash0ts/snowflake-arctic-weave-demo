import asyncio
from itertools import product
from typing import Optional, Union

import weave
from dotenv import load_dotenv
from litellm import completion
from weave import Model

from weave_example_demo.llm_types.prompts import (
    PromptTemplate,
    rag_human_prompts,
    rag_system_prompts,
)
from weave_example_demo.llm_types.rag.vector_store import VectorStore
from weave_example_demo.scorers.llm_guard_scorer import LLMGuardScorer
from weave_example_demo.scorers.tonic_validate_scorer import TonicValidateScorer

# Load environment variables from a .env file
load_dotenv()


# class RAGModel(BaseModel):
class RAGModel(Model):
    model_name: str = "gpt-3.5-turbo-1106"
    prompt_template: PromptTemplate
    temperature: float = 0.0
    vector_store: Optional[Union[VectorStore, str]] = None
    name: str = "RAGModel"

    def __init__(
        self,
        vector_store: Union[VectorStore, str],
        system_prompt: Optional[str] = None,
        human_prompt: Optional[str] = None,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
    ):
        super().__init__(
            model_name=model_name,
            prompt_template=PromptTemplate(
                system_prompt=system_prompt, human_prompt=human_prompt
            ),
            temperature=temperature,
            # vector_store=vector_store,
        )
        self.model_name = model_name
        self.prompt_template = PromptTemplate(
            system_prompt=system_prompt, human_prompt=human_prompt
        )
        self.temperature = temperature
        self.set_vector_store(vector_store)

    @weave.op()
    def set_vector_store(self, vector_store: Union[VectorStore, str]):
        if isinstance(vector_store, str) and vector_store.startswith("weave://"):
            vector_store_ref = weave.ref(vector_store)
            self.vector_store = vector_store_ref.get()
        else:
            self.vector_store = vector_store

    @weave.op()
    def predict(self, question: str, n_documents: int = 2) -> dict:
        self.set_vector_store(self.vector_store)
        _context = self.vector_store.get_most_relevant_documents(
            query=question, n=n_documents
        )

        context_documents = self.pre_process_context(_context)
        context = "\n\n".join(
            [f"Context {i+1}:\n{doc}" for i,
                doc in enumerate(context_documents)]
        )

        human_prompt_args = {
            "question": question,
            "context": context,
        }
        messages = self.prompt_template.format_prompt(
            human_prompt_args=human_prompt_args
        )
        completion_args = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
        }
        if "gpt" not in self.model_name:
            completion_args["system"] = messages.pop(0)["content"]
        response = completion(**completion_args)
        answer = response.choices[0].message.content

        return self.post_process_result(answer, _context)

    def post_process_result(self, answer: str, context: list) -> dict:
        """
        Post-process the answer and context.
        Subclasses can override this method to perform custom post-processing on the answer and context.
        """
        return {"answer": answer, "context": context}

    # TODO: Change the default pre-processing to not be BioASQ
    def pre_process_context(self, context: list) -> list:
        """
        Pre-process the context.
        Subclasses can override this method to perform custom pre-processing on the context used in the LLM.
        """
        return [doc["document"]["passage"] for doc in context]
