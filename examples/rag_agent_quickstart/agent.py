import weave
from weave_example_demo.llm_types.rag.vector_store import VectorStore
from weave_example_demo.llm_types.rag.rag import RAGModel
import asyncio
from weave_example_demo.llm_types.prompts import agent_human_prompt_template, agent_system_prompt_template, rag_human_prompts, rag_system_prompts
from weave_example_demo.llm_types.agent.agent import LLMAgentModel
from weave_example_demo.llm_types.agent.tools import RAGTool, ToolRegistry, search_tool, calculate_tool, search_tool_kwargs, calculate_tool_kwargs
from itertools import product
from examples.rag_agent_quickstart.rag import ExampleRAGModel


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
    rag_model = ExampleRAGModel(
        model_name="gpt-3.5-turbo",
        system_prompt=rag_system_prompts[-1],
        human_prompt=rag_human_prompts[-1],
        vector_store=vector_store,
    )
    tool_registry = ToolRegistry()
    tool_registry.add_tool(**search_tool_kwargs)
    tool_registry.add_tool(**calculate_tool_kwargs)
    tool_registry.add_tool(
        key="RAGTool",
        function=RAGTool(rag_model=rag_model),
        description="Generate a response using the RAG model",
        parameters={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question for the RAG model",
                },
                "n_documents": {
                    "type": "integer",
                    "description": "The number of documents to retrieve",
                    "default": 2,
                },
            },
            "required": ["question", "n_documents"],
        },
    )

    allowed_tools = [
        "search_tool",
        "calculate_tool",
        "RAGTool"
    ]
    agent = LLMAgentModel(
        model_name="gpt-3.5-turbo",
        allowed_tools=allowed_tools,
        agent_tool_registry=tool_registry,
        human_prompt=agent_human_prompt_template,
        system_prompt=agent_system_prompt_template,
    )

    # Example usage - single turn analysis
    # TODO: add support for parallel multitool
    question = "Calculate the square root of 4"
    print(agent.predict(question, multithought=False))

    # Example usage - multi turn analysis via REACT
    question = "Search my RAG to analyze Rivian and Lucid. Should I put or call?"
    print(agent.predict(question, multithought=True))


if __name__ == "__main__":
    main()
