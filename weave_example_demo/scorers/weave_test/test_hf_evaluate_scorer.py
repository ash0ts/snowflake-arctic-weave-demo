import weave
from weave import Evaluation
from weave_example_demo.scorers.hf_evaluate_scorer import HFEvaluateScorer
import asyncio

# Define example predictions and references
examples = [
    {"question": "the cat sat on the mat",
        "references": [["the cat ate the mat"]]},
    {"question": "the dog ran in the park",
        "references": [["the dog played in the park"]]}
]

# Initialize the HFEvaluateScorer with desired metrics
scorer = HFEvaluateScorer(metrics=["google_bleu"])

# Create an Evaluation with the examples and scorer
evaluation = Evaluation(

    dataset=examples,
    scorers=[scorer]
)


@ weave.op()
def function_to_evaluate(question: str):
    # here's where you would add your LLM call and return the output
    return {'predictions': ["the cat sat on the mat"]}


# Initialize Weave tracking
weave.init("test-hf-evaluate")

# Run the evaluation
asyncio.run(evaluation.evaluate(function_to_evaluate))
