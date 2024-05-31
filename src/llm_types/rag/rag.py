import asyncio
from itertools import product
from typing import Optional

import weave
from dotenv import load_dotenv
from litellm import completion
from weave import Model

from src.llm_types.prompts import PromptTemplate, rag_human_prompts, rag_system_prompts
from src.llm_types.rag.vector_store import VectorStore
from src.scorers.llm_guard_scorer import LLMGuardScorer
from src.scorers.tonic_validate_scorer import TonicValidateScorer

# Load environment variables from a .env file
load_dotenv()


# class RAGModel(BaseModel):
class RAGModel(Model):
    model_name: str = "gpt-3.5-turbo-1106"
    prompt_template: PromptTemplate
    temperature: float = 0.0
    vector_store: VectorStore

    def __init__(
        self,
        vector_store: VectorStore,
        system_prompt: Optional[str],
        human_prompt: Optional[str],
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
    ):
        super().__init__(
            model_name=model_name,
            prompt_template=PromptTemplate(
                system_prompt=system_prompt, human_prompt=human_prompt
            ),
            temperature=temperature,
            vector_store=vector_store,
        )
        self.model_name = model_name
        self.prompt_template = PromptTemplate(
            system_prompt=system_prompt, human_prompt=human_prompt
        )
        self.temperature = temperature
        self.vector_store = vector_store

    @weave.op()
    def predict(
        self, question: str
    ) -> (
        dict
    ):  # note: `question` will be used later to select data from our evaluation rows

        context = self.vector_store.get_most_relevant_document(question)
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
            # "response_format": {"type": "text"},
        }
        if "gpt" not in self.model_name:
            completion_args["system"] = messages.pop(0)["content"]
        if "claude" in self.model_name:
            del completion_args["response_format"]
        response = completion(**completion_args)
        answer = response.choices[0].message.content
        return {"answer": answer, "context": context}


def main():
    weave.init("rag-qa-evaluation-qs-experiments")
    questions = [
        {
            "question": "What significant result was reported about Zealand Pharma's obesity trial?",
            "ground_truth": "The significant result reported about Zealand Pharma's obesity trial is that the 6mg dose of the drug survodutide was found to be safe, which is the top dose used in the ongoing Phase 3 obesity trial.",
        },
        {
            "question": "How much did Berkshire Hathaway's cash levels increase in the fourth quarter?",
            "ground_truth": "Berkshire Hathaway's cash levels increased to $167.6 billion in the fourth quarter, surpassing the previous record of $157.2 billion.",
        },
        {
            "question": "What is the goal of Highmark Health's integration of Google Cloud and Epic Systems technology?",
            "ground_truth": "The goal of Highmark Health's integration of Google Cloud and Epic Systems technology is to make it easier for both payers and providers to access key information they need, even if it's stored across multiple points and formats.",
        },
        {
            "question": "What were Rivian and Lucid's vehicle production forecasts for 2024?",
            "ground_truth": "Rivian forecasted it will make 57,000 vehicles in 2024, slightly less than the 57,232 vehicles it produced in 2023. Lucid expects to make 9,000 vehicles in 2024, more than the 8,428 vehicles it made in 2023.",
        },
        {
            "question": "Why was the Norwegian Dawn cruise ship denied access to Mauritius?",
            "ground_truth": "The Norwegian Dawn cruise ship was denied access to Mauritius due to fears of a potential cholera outbreak, as samples were taken from at least 15 passengers on board who experienced mild symptoms of a stomach-related illness.",
        },
        {
            "question": "Which company achieved the first U.S. moon landing since 1972?",
            "ground_truth": "Intuitive Machines achieved the first U.S. moon landing since 1972 with its Nova-C cargo lander named Odysseus.",
        },
        {
            "question": "What issue did Intuitive Machines' lunar lander encounter upon landing on the moon?",
            "ground_truth": "Intuitive Machines' lunar lander Odysseus encountered an issue where it tipped over upon landing on the moon, likely due to its landing gear catching sideways in the moon's surface.",
        },
    ]
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
    prompt_combinations = list(product(rag_system_prompts, rag_human_prompts))
    for model_name in [
        "gpt-3.5-turbo",
        # "claude-3-haiku-20240307",
        # "gpt-4-turbo",
        # "claude-3-opus-20240229",
        # "replicate/snowflake/snowflake-arctic-instruct"
    ]:
        for system_prompt, human_prompt in prompt_combinations:
            model = RAGModel(
                model_name=model_name,
                system_prompt=system_prompt,
                human_prompt=human_prompt,
                vector_store=vector_store,
            )
            model.predict(
                "What significant result was reported about Zealand Pharma's obesity trial?"
            )
            scorers = [
                # RagasScorer(metrics=["faithfulness",
                #             "context_recall", "context_precision"]),
                TonicValidateScorer(
                    metrics=[
                        "AnswerSimilarityMetric",
                        "AugmentationPrecisionMetric",
                        "AnswerConsistencyMetric",
                    ]
                ),
                LLMGuardScorer(
                    metrics=["NoRefusal", "Relevance", "Sensitive"]),
                # DeepEvalScorer(metrics=["ToxicityMetric",
                #                "BiasMetric", "HallucinationMetric"]),
            ]
            evaluation = weave.Evaluation(dataset=questions, scorers=scorers)
            asyncio.run(evaluation.evaluate(model))


if __name__ == "__main__":
    main()
