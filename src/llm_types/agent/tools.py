from typing import Any, Callable, Dict

from pydantic import BaseModel, Field

from src.llm_types.rag.rag import RAGModel

# Implement based on weave ref


class RAGTool(BaseModel):
    rag_model: RAGModel

    def __init__(self, rag_model: RAGModel):
        super().__init__(rag_model=rag_model)
        self.rag_model = rag_model

    def __call__(self, prompt: str) -> str:
        return self.rag_model.predict(prompt)

    @property
    def __name__(self):
        return "RAGTool"


class ToolRegistry(BaseModel):
    tools: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    def add_tool(
        self, key: str, function: Callable, description: str, parameters: Dict[str, Any]
    ):
        self.tools[key] = {
            "function": function,
            "dict": {
                "type": "function",
                "function": {
                    "name": key,
                    "description": description,
                    "parameters": parameters,
                },
            },
        }


# Define the tool functions


def search_tool(query: str) -> str:
    return (
        f"Searching for {query}... Results found: [Example result 1, Example result 2]"
    )


def calculate_tool(expression: str) -> str:
    try:
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return str(e)


# Create the tool registry instance
tool_registry = ToolRegistry()

# Add tools to the registry
tool_registry.add_tool(
    key="search_tool",
    function=search_tool,
    description="Search for a query",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"},
        },
        "required": ["query"],
    },
)

tool_registry.add_tool(
    key="calculate_tool",
    function=calculate_tool,
    description="Calculate the result of an expression",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The expression to calculate",
            },
        },
        "required": ["expression"],
    },
)