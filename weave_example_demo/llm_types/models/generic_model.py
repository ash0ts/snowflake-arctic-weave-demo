from typing import Optional

import weave
from litellm import completion

from weave_example_demo.llm_types.prompts import PromptTemplate


class GenericLLMModel(weave.Model):
    model_name: str = "gpt-3.5-turbo-1106"
    prompt_template: PromptTemplate
    temperature: float = 0.0
    name: str = "GenericLLMModel"

    def __init__(
        self,
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
        )
        self.model_name = model_name
        self.prompt_template = PromptTemplate(
            system_prompt=system_prompt, human_prompt=human_prompt
        )
        self.temperature = temperature

    @weave.op()
    def predict(
        self,
        human_prompt_args: Optional[dict] = {},
        system_prompt_args: Optional[dict] = {},
    ) -> dict:
        messages = self.prompt_template.format_prompt(
            human_prompt_args=human_prompt_args, system_prompt_args=system_prompt_args
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
        return {"answer": answer}
