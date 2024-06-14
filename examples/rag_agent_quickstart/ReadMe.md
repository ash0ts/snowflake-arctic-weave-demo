# RAG Agent Quickstart

## What is it?

This project demonstrates the use of Retrieval-Augmented Generation (RAG) models and agent-based models for question answering and information retrieval tasks. It includes examples of how to set up and use these models with various prompts and evaluation metrics.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/rag_agent_quickstart.git
    cd rag_agent_quickstart
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Running the RAG Model

To run the RAG model example, execute the following command:
```sh
python -m examples.rag_agent_quickstart.rag
```

### Running the Agent Model

To run the agent model example, execute the following command:
```sh
python -m examples.rag_agent_quickstart.agent
```

## Examples

### RAG Model

The RAG model is initialized with a set of articles and a list of questions. It uses different combinations of system and human prompts to generate answers and evaluates them using various scorers.

### Agent Model

The agent model integrates tools like search and calculation with the RAG model to answer more complex queries. It demonstrates both single-turn and multi-turn interactions.
