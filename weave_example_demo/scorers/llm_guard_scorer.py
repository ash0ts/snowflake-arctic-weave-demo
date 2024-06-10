from typing import List, Optional

import weave
from dotenv import load_dotenv

from weave_example_demo.scorers.third_party_metrics_scorer import ThirdPartyMetricsScorer

load_dotenv()


class LLMGuardScorer(ThirdPartyMetricsScorer):
    metrics_module_name: str = "llm_guard.output_scanners"
    answer_key: Optional[str] = "answer"

    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        run_all_metrics: bool = False,
        answer_key: str = "answer",
        context_key: str = "context",
    ):
        super().__init__(metrics=metrics, run_all_metrics=run_all_metrics)
        self.answer_key = answer_key

    @weave.op()
    def score(self, question: str, ground_truth: str, model_output: dict) -> dict:
        from llm_guard import scan_output

        output_scanners = [metric() for metric in self.get_metric_modules()]
        # TODO; Think about sanitization especially w.r.t input_scanners
        sanitized_response_text, results_valid, results_score = scan_output(
            output_scanners, question, model_output[self.answer_key]
        )
        return {
            "sanitized_response_text": sanitized_response_text,
            "results_valid": results_valid,
            "results_score": results_score,
        }


def main():
    weave.init("test-llm-guard")
    scorer = LLMGuardScorer(metrics=["Bias"])
    val2 = scorer.score("What is the capital of France?",
                        "Paris", {"answer": "Paris"})
    print(val2)


if __name__ == "__main__":
    main()
