import importlib
import inspect
from typing import List, Optional

from pydantic import Field
from weave.flow.scorer import Scorer


class ThirdPartyMetricsScorer(Scorer):
    metrics: List[str] = Field(default_factory=list)
    # TODO: Allow for custom filtering logic for run_all_metrics packages as this may cause unintended packages to run like in RAGAS
    run_all_metrics: bool = False
    all_metrics: List[str] = Field(default_factory=list)
    metrics_module_name: str

    def __init__(
        self, metrics: Optional[List[str]] = None, run_all_metrics: bool = False
    ):
        super().__init__(metrics=metrics or [], run_all_metrics=run_all_metrics)
        self.validate_metrics(self.metrics_module_name, self.metrics)
        if self.run_all_metrics:
            self.metrics = self.all_metrics

    def validate_metrics(self, metrics_module_name: str, metrics: List[str]):
        try:
            metrics_module = importlib.import_module(metrics_module_name)
            all_metrics = getattr(metrics_module, "__all__", [])
            if len(all_metrics) == 0:
                # BUG: This grabs all the folder names in the metrics_module and the defined classes in the __init__.py file
                # We want this to only grab the defined classes in the __init__.py file
                all_metrics = [name for name,
                               obj in inspect.getmembers(metrics_module)]
            self.all_metrics = all_metrics
        except ImportError:
            raise ImportError(
                f"The '{metrics_module_name}' module is not installed or cannot be found."
            )

        if not all(metric in all_metrics for metric in metrics):
            invalid_metrics = [
                metric for metric in metrics if metric not in all_metrics
            ]
            raise ValueError(
                f"Metrics {invalid_metrics} are not valid. Must be one of {all_metrics}."
            )

    # TODO: Figure out the best way to have arguments to instantiated metrics for certain packages
    def get_metric_modules(self):
        metric_modules = []
        for metric in self.metrics:
            module = importlib.import_module(self.metrics_module_name)
            metric_module = getattr(module, metric)
            metric_modules.append(metric_module)
        return metric_modules
