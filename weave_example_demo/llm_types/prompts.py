from dataclasses import dataclass
from typing import Dict, Optional

import weave

rag_system_prompts = [
    "You are an expert in finance and answer questions related to finance, financial services, and financial markets. When responding based on provided information, be sure to cite the source.",  # Good
    "You know a bit about finance. Try to answer the questions.",  # Medium
    "Answer finance questions.",  # Bad
]

rag_human_prompts = [
    """
    Use the following information to answer the subsequent question. If the answer cannot be found, write "I don't know."
    Context:
    \"\"\"
    {context}
    \"\"\"
    Question: {question}
    """,  # Good
    """
    Here is some info. Answer the question if you can.
    Context:
    \"\"\"
    {context}
    \"\"\"
    Question: {question}
    """,  # Medium
    """
    Info:
    \"\"\"
    {context}
    \"\"\"
    Q: {question}
    """,  # Bad
]

agent_system_prompt_template = (
    "You are an intelligent assistant with access to the following tools: {tools}. "
    "Use the tools to answer the questions. Follow the format below:\n"
    "Question: the input question you must answer\n"
    "Thought: you should always think about what to do\n"
    "Action: the action to take, should be one of [{tools}]\n"
    "Action Input: the input to the action\n"
    "Observation: the result of the action\n"
    "... (this Thought/Action/Action Input/Observation can repeat N times)\n"
    "Thought: I now know the final answer\n"
    "FINISH: the final answer to the original input question\n"
    "You must start the final answer with FINISH"
)

agent_human_prompt_template = "Question: {question}\n" "Response:"


@dataclass
class PromptTemplate:
    system_prompt: str
    human_prompt: str

    @weave.op()
    def format_prompt(
        self,
        system_prompt_args: Optional[Dict[str, str]] = {},
        human_prompt_args: Optional[Dict[str, str]] = {},
    ):
        "A formatting function for OpenAI models"
        system_prompt_formatted = self.system_prompt.format(
            **system_prompt_args)
        human_prompt_formatted = self.human_prompt.format(**human_prompt_args)
        messages = [
            {"role": "system", "content": system_prompt_formatted},
            {"role": "user", "content": human_prompt_formatted},
        ]
        return messages
