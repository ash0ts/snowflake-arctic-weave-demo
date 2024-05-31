from typing import List, Optional

import weave
from dotenv import load_dotenv

from src.scorers.third_party_metrics_scorer import ThirdPartyMetricsScorer

load_dotenv()


def unpack_test_result(test_result):
    results = {}
    results["success"] = test_result.success
    for metric in test_result.metrics:
        results[metric.__name__] = {
            "score": metric.score,
            "reason": metric.reason,
            "success": metric.success,
            "cost": metric.evaluation_cost,
            "verdicts": metric.verdicts,
            "statements": getattr(metric, "statements", []),
        }
    return results


class DeepEvalScorer(ThirdPartyMetricsScorer):
    metrics_module_name: str = "deepeval.metrics"
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
        from deepeval import evaluate
        from deepeval.test_case import LLMTestCase

        metric_modules = [metric() for metric in self.get_metric_modules()]

        # TODO: figure out difference between context and retrieval context
        test_case = LLMTestCase(
            input=question,
            expected_output=ground_truth,
            # Replace this with the actual output from your LLM application
            context=(
                [model_output[self.context_key]]
                if isinstance(model_output[self.context_key], str)
                else model_output[self.context_key]
            ),
            actual_output=model_output[self.answer_key],
            retrieval_context=(
                [model_output[self.context_key]]
                if isinstance(model_output[self.context_key], str)
                else model_output[self.context_key]
            ),
        )
        test_results = evaluate([test_case], metric_modules)
        return unpack_test_result(test_results[0])


def main():
    weave.init("test-deepeval")
    scorer = DeepEvalScorer(
        metrics=[
            "AnswerRelevancyMetric",
            "ToxicityMetric",
            "HallucinationMetric",
            "ContextualPrecisionMetric",
        ]
    )
    val2 = scorer.score(
        "What is the capital of France?",
        "Paris",
        {"answer": "Paris", "context": ["Paris is the capital of France."]},
    )
    print(val2)


if __name__ == "__main__":
    main()
