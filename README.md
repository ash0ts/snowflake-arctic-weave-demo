# Snowflake Arctic Weave Demo

## Overview
This project is a Streamlit application that integrates Weights & Biases Weave, Snowflake Arctic, and Replicate to demonstrate LLMOps tracking and efficient language model operations. The application allows users to interact with a language model, manage chat history, and evaluate model responses using various scoring mechanisms.

## Features
- **[Weights & Biases Weave](https://wandb.me/weave)**: Tracks and evaluates model performance.
- **Streamlit Interface**: User-friendly interface for interacting with the language model.
- **Snowflake Arctic**: Efficient language model operations.
- **Replicate**: Model hosting and deployment.
- **Scoring Mechanisms**: Evaluates model responses using Ragas, LLMGuard, TonicValidate, and DeepEval.

## Setup

### Installation
1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Set up environment variables:
    - Create a `.env` file in the root directory and add the following:
        ```env
        REPLICATE_API_TOKEN=<your_replicate_api_token>
        # Or any API Key for an LLM provider compatible with litellm
        ```

### Iterating and Evaluating LLM Types

- You can adjust the prompts in `src/llm_types/prompts.py`
- You can adjust the scorers and metrics used in `src/llm_types/<type>/<type>.py`
- You can adjust the models you try in `src/llm_types/<type>/<type>.py`

When done making changes, you can run the following commands to evaluate the model:

```sh
python -m src.llm_types.rag.rag # For Evaluating a RAG model
python -m src.llm_types.agent.agent # For Running the ReAct Agent (Evaluation WIP)
```

### Running the Application

Ensure to set the `.streamlit/secrets.toml` file with the appropriate API Key and reference to a Weave Model. The file should look like this:

```toml
REPLICATE_API_TOKEN = "<your_replicate_api_token>"
ANY_OTHER_NECCESARY_LLM_API_KEY_FOR_LITELLM = "<your_llm_api_key>"
WEAVE_MODEL_REF = "<your_weave_model_ref>"
```

Start the Streamlit application to interact with a selected model after running one of the above commands. 
```sh
streamlit run src/app/streamlit_app.py
```

## Project Structure
- `src/app/streamlit_app.py`: Main Streamlit application.
- `src/llm_types/rag/rag.py`: RAG model implementation.
- `src/llm_types/rag/vector_store.py`: Vector store for document embeddings.
- `src/llm_types/agent/agent.py`: ReAct Agent model for tool integration.
- `src/llm_types/agent/tools.py`: Tools for the agent model.
- `src/llm_types/prompts.py`: Prompt templates for llm types.
- `src/scorers`: Scoring mechanisms for evaluating model responses.
- `.streamlit/secrets.toml`: Secrets for Streamlit application.

## License
This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Contributing
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## Contact
For any questions or issues, please open an issue on the repository or contact the maintainers.