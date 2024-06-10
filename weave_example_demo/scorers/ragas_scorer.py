from typing import List, Optional

import weave
from dotenv import load_dotenv

from weave_example_demo.scorers.third_party_metrics_scorer import ThirdPartyMetricsScorer

load_dotenv()


class RagasScorer(ThirdPartyMetricsScorer):
    metrics_module_name: str = "ragas.metrics"
    answer_key: Optional[str] = "answer"
    context_key: Optional[str] = "context"

    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        run_all_metrics: bool = False,
        answer_key: str = "answer",
        context_key: str = "context",
    ):
        super().__init__(metrics=metrics, run_all_metrics=run_all_metrics)
        self.answer_key = answer_key
        self.context_key = context_key

    # @weave.op()
    def score(self, question: str, ground_truth: str, model_output: dict) -> dict:
        from datasets import Dataset
        from ragas import evaluate

        metric_modules = self.get_metric_modules()
        qa_dataset = Dataset.from_dict(
            {
                "question": [question],
                "ground_truth": [ground_truth],
                "answer": [model_output[self.answer_key]],
                "contexts": [
                    (
                        [model_output[self.context_key]]
                        if isinstance(model_output[self.context_key], str)
                        else model_output[self.context_key]
                    )
                ],
            }
        )
        result = evaluate(qa_dataset, metrics=metric_modules,
                          raise_exceptions=False)
        return result.to_pandas().to_dict(orient="records")


def ragas_score(question, ground_truth, model_output):
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness

    metric_modules = [
        faithfulness,
        # answer_relevancy,
        # answer_correctness,
        # context_recall,
        # context_precision
    ]
    qa_dataset = Dataset.from_dict(
        {
            "question": [question],
            "ground_truth": [ground_truth],
            "answer": [model_output["answer"]],
            "contexts": [[model_output["context"]]],
        }
    )
    result = evaluate(qa_dataset, metrics=metric_modules,
                      raise_exceptions=False)
    return result.to_pandas().to_dict(orient="records")


def main():
    weave.init("test-ragas")
    # val1 = ragas_score(
    #     "What is the capital of France?",
    #     "Paris",
    #     {"answer": "Paris", "context": ["Paris is the capital of France."]}
    # )
    # print(val1)
    scorer = RagasScorer(metrics=["answer_correctness"])
    val2 = scorer.score(
        "What is the capital of France?",
        "Paris",
        {"answer": "Paris", "context": "Paris is the capital of France."},
    )
    print(val2)


if __name__ == "__main__":
    main()
