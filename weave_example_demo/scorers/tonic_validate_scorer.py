from typing import List, Optional

import weave
from dotenv import load_dotenv

from weave_example_demo.scorers.third_party_metrics_scorer import (
    ThirdPartyMetricsScorer,
)

load_dotenv()


class TonicValidateScorer(ThirdPartyMetricsScorer):
    metrics_module_name: str = "tonic_validate.metrics"
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

    @weave.op()
    def score(self, question: str, ground_truth: str, model_output: dict) -> dict:
        from tonic_validate import Benchmark, ValidateScorer

        metric_modules = [metric() for metric in self.get_metric_modules()]

        def get_llm_response(question):
            return {
                "llm_answer": model_output[self.answer_key],
                "llm_context_list": (
                    [model_output[self.context_key]]
                    if isinstance(model_output[self.context_key], str)
                    else model_output[self.context_key]
                ),
            }

        benchmark = Benchmark(questions=[question], answers=[ground_truth])
        scorer = ValidateScorer(metrics=metric_modules)
        run = scorer.score(benchmark, get_llm_response)
        return run.run_data[0].scores


@weave.op()
def tonic_validate_scorer(question, ground_truth, model_output):
    from tonic_validate import Benchmark, ValidateScorer

    def get_llm_response(question):
        return {
            "llm_answer": model_output["answer"],
            "llm_context_list": (
                [model_output["context"]]
                if isinstance(model_output["context"], str)
                else model_output["context"]
            ),
        }

    benchmark = Benchmark(questions=[question], answers=[ground_truth])
    scorer = ValidateScorer()
    run = scorer.score(benchmark, get_llm_response)
    return run.run_data[0].scores


def main():
    weave.init("test-tonic-validate")
    # val1 = tonic_validate_scorer(
    #     "What is the capital of France?",
    #     "Paris",
    #     {"answer": "Paris", "context": ["Paris is the capital of France."]}
    # )
    # print(val1)
    scorer = TonicValidateScorer(metrics=["AnswerSimilarityMetric"])
    val2 = scorer.score(
        "What is the capital of France?",
        "Paris",
        {"answer": "Paris", "context": ["Paris is the capital of France."]},
    )
    print(val2)


if __name__ == "__main__":
    main()
