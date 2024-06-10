import argparse
import os
from dotenv import load_dotenv
import weave
from weave_example_demo.llm_types.rag.vector_store import VectorStore
from weave_example_demo.llm_types.models.embedding_model import SentenceTransformersModel


def get_relevant_documents(vector_store, query, n, ranking_method=None):
    if ranking_method:
        return vector_store.get_most_relevant_documents(query=query, n=n, ranking_method=ranking_method)
    return vector_store.get_most_relevant_documents(query=query, n=n)


@weave.op()
def create_bioasq_vector_store(args):

    dataset_ref = f"weave:///a-sh0ts/{args['project_name']}/object/{args['dataset_name']}:{args['alias']}"

    embedding_model = SentenceTransformersModel(
        model_name=args['embedding_model_name'])
    vector_store = VectorStore(articles=dataset_ref, key=args['key'],
                               embedding_model=embedding_model, multiprocess_embeddings=args['multiprocess_embeddings'])

    documents = get_relevant_documents(
        vector_store, args['query'], args['num_docs'], args['ranking_method'])
    for doc in documents:
        print(doc)

    weave.publish(vector_store, name=args['store_name'])

    return {"embedding_model": embedding_model, "vector_store": vector_store}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publish BioASQ Vector Store")
    parser.add_argument("--project_name", type=str, default="bioasq-rag-data",
                        help="Name of the Weave project")
    parser.add_argument("--query", type=str, default="What is MSQ?",
                        help="Query to search for relevant documents")
    parser.add_argument("--num_docs", type=int, default=3,
                        help="Number of relevant documents to retrieve")
    parser.add_argument("--ranking_method", type=str, default=None,
                        help="Ranking method to use for retrieving documents")
    parser.add_argument("--store_name", type=str, default="BioASQVectorStore",
                        help="Name to publish the vector store under")
    parser.add_argument("--alias", type=str, default="latest",
                        help="Alias for the dataset reference")
    parser.add_argument("--dataset_name", type=str, default="TextCorpusFiltered",
                        help="Name of the dataset to publish the vector store under")
    parser.add_argument("--multiprocess_embeddings", type=bool, default=True,
                        help="Whether to use multiprocess embeddings")
    parser.add_argument("--key", type=str, default="passage",
                        help="Key to use for the vector store")
    parser.add_argument("--embedding_model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Name of the embedding model to use")

    args = parser.parse_args()
    weave.init(args.project_name)
    args_dict = vars(args)
    create_bioasq_vector_store(args_dict)
