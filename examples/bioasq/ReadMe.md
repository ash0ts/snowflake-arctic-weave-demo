# BioASQ RAG Demo

BioASQ RAG Demo is a Retrieval-Augmented Generation (RAG) system designed to answer biomedical questions using the BioASQ dataset. The system retrieves relevant passages, scores their relevance, summarizes the most pertinent information, and synthesizes a final answer.

## Features

- Retrieval of relevant biomedical passages
- Relevance scoring of retrieved passages
- Summarization of key information
- Synthesis of final answers

## Badges

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10-blue.svg)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/bioasq-rag-demo.git
    cd bioasq-rag-demo
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the data publishing script:
    ```bash
    python -m examples.bioasq.bioasq_publish_data
    ```

4. Create and publish the vector store:
    ```bash
    python -m examples.bioasq.bioasq_vector_store
    ```

## Usage

### Advanced RAG Model

1. Open `examples/bioasq/bioasq_rag_advanced.ipynb`.
2. Run the notebook cells to:
   - Load the published dataset and vector store.
   - Define the question-to-query, article relevance, summarization, and synthesis models.
   - Create the `BioASQAdvancedRAGModel` class.
   - Instantiate the RAG model and evaluate it on a subset of the BioASQ dataset.

### Simple RAG Model and Agent

1. Open `examples/bioasq/bioasq_rag_and_agent_simple.ipynb`.
2. Run the notebook cells to:
   - Load the published dataset and vector store.
   - Define the RAG model and agent.
   - Evaluate the RAG model on a subset of the BioASQ dataset.
   - Interact with the RAG agent by asking biomedical questions.

## Support

For support, please open an issue on the GitHub repository