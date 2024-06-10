from typing import List, Optional

import weave
from weave import Scorer
from dotenv import load_dotenv
import evaluate
from typing import Optional

load_dotenv()


class HFEvaluateScorer(Scorer):

    metrics: List[str]

    def __init__(self, metrics: List[str]):
        super().__init__(metrics=metrics)
        self.metrics = metrics

    @weave.op()
    def score(self, model_output: Optional[dict] = {}, **kwargs: dict) -> dict:
        results = {}
        for metric_name in self.metrics:
            metric = evaluate.load(metric_name)
            features = metric._info().features

            # Find the feature dictionary that is a subset of model_output or kwargs
            matching_feature = next((feature for feature in features if set(
                feature.keys()) <= set(model_output.keys()) or set(feature.keys()) <= set(kwargs.keys())), None)

            # Merge model_output and kwargs, giving priority to kwargs
            merged_kwargs = {**model_output, **kwargs}

            if matching_feature is None:
                raise ValueError(
                    f"Invalid arguments for metric {metric_name}. Expected one of: {features} but got {merged_kwargs}")

            # Filter merged_kwargs to only include keys present in the matching feature dictionary
            sanitized_kwargs = {
                k: v for k, v in merged_kwargs.items() if k in matching_feature}
            result = metric.compute(**sanitized_kwargs)
            results[metric_name] = result

        return results


def main():
    weave.init("test-hf-evaluate")
    scorer = HFEvaluateScorer(metrics=["google_bleu"])
    val2 = scorer.score(predictions=["the cat sat on the mat"], references=[
                        ["the cat ate the mat"]], moody=True)
    print(val2)


if __name__ == "__main__":
    main()
