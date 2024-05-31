import json
from typing import Callable, List

import weave
from dotenv import load_dotenv
from litellm import completion
from weave import Model

from src.llm_types.agent.tools import RAGTool, tool_registry
from src.llm_types.prompts import (
    PromptTemplate,
    agent_human_prompt_template,
    agent_system_prompt_template,
    rag_human_prompts,
    rag_system_prompts,
)
from src.llm_types.rag.rag import RAGModel
from src.llm_types.rag.vector_store import VectorStore

# Load environment variables from a .env file
load_dotenv()


class LLMAgentModel(Model):

    model_name: str
    tools: List[Callable]
    prompt_template: PromptTemplate
    temperature: float
    max_tokens: int

    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        human_prompt: str,
        tools: List[str],
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ):
        super().__init__(
            model_name=model_name,
            tools=[tool_registry.tools[tool]["function"] for tool in tools],
            prompt_template=PromptTemplate(
                system_prompt=system_prompt, human_prompt=human_prompt
            ),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.model_name = model_name
        self.tools = [tool_registry.tools[tool]["function"] for tool in tools]
        self.prompt_template = PromptTemplate(
            system_prompt=system_prompt, human_prompt=human_prompt
        )
        self.temperature = temperature
        self.max_tokens = max_tokens

    @weave.op()
    def call_tool(self, tool_name: str, *args, **kwargs):
        if tool_name in tool_registry.tools:
            tool = tool_registry.tools[tool_name]["function"]
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
    def predict(self, question: str, multiturn: bool = True) -> str:
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

        tools = [tool_registry.tools[tool]["dict"]
                 for tool in tool_registry.tools]
        if multiturn:
            response = self.react_prompting(prompt, tools)[
                "messages"][-1]["content"]
        else:
            response = self.run_chat(prompt, tools)["messages"][-1]["content"]
        return response


def main():
    weave.init("agentic-rag-example")

    # For Agentic RAG
    articles = [
        "Novo Nordisk and Eli Lilly rival soars 32 percent after promising weight loss drug results Shares of Denmarks Zealand Pharma shot 32 percent higher in morning trade, after results showed success in its liver disease treatment survodutide, which is also on trial as a drug to treat obesity. The trial “tells us that the 6mg dose is safe, which is the top dose used in the ongoing [Phase 3] obesity trial too,” one analyst said in a note. The results come amid feverish investor interest in drugs that can be used for weight loss.",
        "Berkshire shares jump after big profit gain as Buffetts conglomerate nears $1 trillion valuation Berkshire Hathaway shares rose on Monday after Warren Buffetts conglomerate posted strong earnings for the fourth quarter over the weekend. Berkshires Class A and B shares jumped more than 1.5%, each. Class A shares are higher by more than 17% this year, while Class B have gained more than 18%. Berkshire was last valued at $930.1 billion, up from $905.5 billion where it closed on Friday, according to FactSet. Berkshire on Saturday posted fourth-quarter operating earnings of $8.481 billion, about 28 percent higher than the $6.625 billion from the year-ago period, driven by big gains in its insurance business. Operating earnings refers to profits from businesses across insurance, railroads and utilities. Meanwhile, Berkshires cash levels also swelled to record levels. The conglomerate held $167.6 billion in cash in the fourth quarter, surpassing the $157.2 billion record the conglomerate held in the prior quarter.",
        "Highmark Health says its combining tech from Google and Epic to give doctors easier access to information Highmark Health announced it is integrating technology from Google Cloud and the health-care software company Epic Systems. The integration aims to make it easier for both payers and providers to access key information they need, even if its stored across multiple points and formats, the company said. Highmark is the parent company of a health plan with 7 million members, a provider network of 14 hospitals and other entities",
        "Rivian and Lucid shares plunge after weak EV earnings reports Shares of electric vehicle makers Rivian and Lucid fell Thursday after the companies reported stagnant production in their fourth-quarter earnings after the bell Wednesday. Rivian shares sank about 25 percent, and Lucids stock dropped around 17 percent. Rivian forecast it will make 57,000 vehicles in 2024, slightly less than the 57,232 vehicles it produced in 2023. Lucid said it expects to make 9,000 vehicles in 2024, more than the 8,428 vehicles it made in 2023.",
        "Mauritius blocks Norwegian cruise ship over fears of a potential cholera outbreak Local authorities on Sunday denied permission for the Norwegian Dawn ship, which has 2,184 passengers and 1,026 crew on board, to access the Mauritius capital of Port Louis, citing “potential health risks.” The Mauritius Ports Authority said Sunday that samples were taken from at least 15 passengers on board the cruise ship. A spokesperson for the U.S.-headquartered Norwegian Cruise Line Holdings said Sunday that 'a small number of guests experienced mild symptoms of a stomach-related illness' during Norwegian Dawns South Africa voyage.",
        "Intuitive Machines lands on the moon in historic first for a U.S. company Intuitive Machines Nova-C cargo lander, named Odysseus after the mythological Greek hero, is the first U.S. spacecraft to soft land on the lunar surface since 1972. Intuitive Machines is the first company to pull off a moon landing — government agencies have carried out all previously successful missions. The companys stock surged in extended trading Thursday, after falling 11 percent in regular trading.",
        "Lunar landing photos: Intuitive Machines Odysseus sends back first images from the moon Intuitive Machines cargo moon lander Odysseus returned its first images from the surface. Company executives believe the lander caught its landing gear sideways in the moons surface while touching down and tipped over. Despite resting on its side, the companys historic IM-1 mission is still operating on the moon.",
    ]
    vector_store = VectorStore(articles=articles)
    rag_model = RAGModel(
        model_name="gpt-3.5-turbo",
        system_prompt=rag_system_prompts[-1],
        human_prompt=rag_human_prompts[-1],
        vector_store=vector_store,
    )
    tool_registry.add_tool(
        key="RAGTool",
        function=RAGTool(rag_model=rag_model),
        description="Generate a response using RAG model",
        parameters={
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The prompt for the RAG model",
                },
            },
            "required": ["prompt"],
        },
    )

    tools = [
        "search_tool",
        "calculate_tool",
        #  "RAGTool" #BUG: Slice issue
    ]
    agent = LLMAgentModel(
        model_name="gpt-3.5-turbo",
        tools=tools,
        human_prompt=agent_human_prompt_template,
        system_prompt=agent_system_prompt_template,
    )

    # Example usage - single turn analysis
    # TODO: add support for parallel multitool
    question = "Calculate the square root of 4"
    print(agent.predict(question, multiturn=False))

    # Example usage - multi turn analysis via REACT
    question = "Search my RAG to analyze Rivian and Lucid. Should I put or call?"
    print(agent.predict(question, multiturn=True))


if __name__ == "__main__":
    main()
