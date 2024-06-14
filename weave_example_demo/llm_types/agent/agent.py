import json
from typing import Callable, List

import weave
from dotenv import load_dotenv
from litellm import completion
from weave import Model

from weave_example_demo.llm_types.agent.tools import (
    ToolRegistry,
)
from weave_example_demo.llm_types.prompts import (
    PromptTemplate,
)

# Load environment variables from a .env file
load_dotenv()


class LLMAgentModel(Model):

    model_name: str
    tools: List[Callable]
    agent_tool_registry: ToolRegistry
    prompt_template: PromptTemplate
    temperature: float
    max_tokens: int

    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        human_prompt: str,
        allowed_tools: List[str],
        agent_tool_registry: ToolRegistry,
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ):
        super().__init__(
            model_name=model_name,
            tools=[
                agent_tool_registry.tools[tool]["function"] for tool in allowed_tools
            ],
            agent_tool_registry=agent_tool_registry,
            prompt_template=PromptTemplate(
                system_prompt=system_prompt, human_prompt=human_prompt
            ),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.model_name = model_name
        self.tools = [
            self.agent_tool_registry.tools[tool]["function"] for tool in allowed_tools
        ]
        self.agent_tool_registry = agent_tool_registry
        self.prompt_template = PromptTemplate(
            system_prompt=system_prompt, human_prompt=human_prompt
        )
        self.temperature = temperature
        self.max_tokens = max_tokens

    @weave.op()
    def call_tool(self, tool_name: str, *args, **kwargs):
        if tool_name in self.agent_tool_registry.tools:
            tool = self.agent_tool_registry.tools[tool_name]["function"]
            return tool(*args, **kwargs)
        raise ValueError(f"Tool <{tool_name}> not found")

    @weave.op()
    def run_chat(
        self, messages: List[str], tools: List[dict], tool_choice: str = "auto"
    ) -> dict:
        response = completion(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            tools=tools,
            tool_choice=tool_choice,
        )
        response_message = response.choices[0].message
        has_tool_calls = hasattr(response_message, "tool_calls")
        print(response_message)
        print(has_tool_calls)

        if has_tool_calls:
            tool_calls = response_message.tool_calls
            messages.append(response_message)
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_kwargs = json.loads(tool_call.function.arguments)
                function_response = self.call_tool(
                    function_name, **function_kwargs)
                # TODO: Add tool parsers to registry
                if function_name == "RAGTool":
                    function_response = function_response["answer"]
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    }
                )
        else:
            messages.append(response_message)

        response_message_content = str(response_message.content)
        return {
            "messages": messages,
            "response_message_content": response_message_content,
        }

    @weave.op()
    def react_prompting(self, messages: List[str], tools: List[dict]) -> str:
        while True:
            response = self.run_chat(messages, tools)
            messages = response["messages"]
            response_message_content = response["response_message_content"]
            if response_message_content.startswith("FINISH"):
                break

        return {
            "messages": messages,
            "response_message_content": response_message_content,
        }

    @weave.op()
    def predict(self, question: str, multithought: bool = True) -> str:
        tool_names = ", ".join(
            [
                tool.__name__ if hasattr(tool, "__name__") else str(tool)
                for tool in self.tools
            ]
        )

        system_prompt_args = {"tools": tool_names}
        human_prompt_args = {"question": question}

        prompt = self.prompt_template.format_prompt(
            system_prompt_args=system_prompt_args, human_prompt_args=human_prompt_args
        )

        tools = [
            self.agent_tool_registry.tools[tool]["dict"]
            for tool in self.agent_tool_registry.tools
        ]
        if multithought:
            response = self.react_prompting(prompt, tools)[
                "messages"][-1]["content"]
        else:
            response = self.run_chat(prompt, tools)["messages"][-1]["content"]
        return response
