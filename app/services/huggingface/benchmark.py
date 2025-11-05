"""
Darwin Neuropsychiatric Benchmark Suite

Comprehensive benchmark for psychiatric AI models:
- Depression diagnosis (PHQ-9 correlation, AUC)
- Bipolar episode prediction (lead time, sensitivity)
- ADHD classification (theta/beta ratio, specificity)
- Treatment response (8-week outcome prediction)
- Multimodal fusion (ablation studies)

Standardized metrics for comparing models across:
- Modalities (EEG-only vs multimodal)
- Architectures (traditional ML vs deep learning)
- Training data (size, quality, diversity)
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, classification_report
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)


class BenchmarkTask(str, Enum):
    """Available benchmark tasks"""
    DEPRESSION_DIAGNOSIS = "depression_diagnosis"
    BIPOLAR_PREDICTION = "bipolar_prediction"
    ADHD_CLASSIFICATION = "adhd_classification"
    TREATMENT_RESPONSE = "treatment_response"
    MULTIMODAL_FUSION = "multimodal_fusion"


@dataclass
class BenchmarkMetrics:
    """Standard metrics for psychiatric tasks"""
    # Classification
    accuracy: float
    precision: float
    recall: float  # Sensitivity
    specificity: float
    f1_score: float
    auc_roc: float
    
    # Confusion matrix
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    
    # Task-specific
    task_specific: Dict[str, float] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Complete benchmark result"""
    task: BenchmarkTask
    model_name: str
    dataset: str
    metrics: BenchmarkMetrics
    metadata: Dict[str, Any] = field(default_factory=dict)


class DarwinNeuroBenchmark:
    """
    Darwin Neuropsychiatric Benchmark Suite.
    
    Standardized evaluation for psychiatric AI models.
    """
    
    def __init__(self):
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn required")
        
        logger.info("Darwin Neuro Benchmark initialized")
    
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> BenchmarkMetrics:
        """
        Compute standard metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (for AUC)
        
        Returns:
            BenchmarkMetrics
        """
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # Specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # AUC
        if y_prob is not None and len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_prob)
        else:
            auc = 0.0
        
        return BenchmarkMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            specificity=specificity,
            f1_score=f1,
            auc_roc=auc,
            true_positives=int(tp),
            true_negatives=int(tn),
            false_positives=int(fp),
            false_negatives=int(fn)
        )
    
    def evaluate_depression_diagnosis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        phq9_scores: Optional[np.ndarray] = None
    ) -> BenchmarkResult:
        """
        Evaluate depression diagnosis task.
        
        Target metrics:
        - AUC > 0.85
        - Sensitivity > 0.80 (detect true cases)
        - Specificity > 0.80 (avoid false alarms)
        - PHQ-9 correlation > 0.70 (if scores available)
        
        Args:
            y_true: True labels (0=healthy, 1=depression)
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            phq9_scores: Optional PHQ-9 scores for correlation
        
        Returns:
            BenchmarkResult
        """
        metrics = self.compute_metrics(y_true, y_pred, y_prob)
        
        # Task-specific: PHQ-9 correlation
        if phq9_scores is not None:
            from scipy.stats import pearsonr
            phq9_correlation, _ = pearsonr(y_prob, phq9_scores)
            metrics.task_specific['phq9_correlation'] = phq9_correlation
        
        result = BenchmarkResult(
            task=BenchmarkTask.DEPRESSION_DIAGNOSIS,
            model_name="unknown",
            dataset="unknown",
            metrics=metrics,
            metadata={
                'target_auc': 0.85,
                'target_sensitivity': 0.80,
                'target_specificity': 0.80,
                'passes_threshold': (
                    metrics.auc_roc >= 0.85 and
                    metrics.recall >= 0.80 and
                    metrics.specificity >= 0.80
                )
            }
        )
        
        return result
    
    def evaluate_bipolar_prediction(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        days_before_episode: Optional[np.ndarray] = None
    ) -> BenchmarkResult:
        """
        Evaluate bipolar episode prediction.
        
        Target metrics:
        - Sensitivity > 0.75 (catch episodes early)
        - Lead time: 3-7 days before episode
        - False alarm rate < 0.30
        
        Args:
            y_true: True labels (0=stable, 1=episode)
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            days_before_episode: Days between prediction and actual episode
        
        Returns:
            BenchmarkResult
        """
        metrics = self.compute_metrics(y_true, y_pred, y_prob)
        
        # Task-specific: lead time
        if days_before_episode is not None:
            # Only for true positives
            tp_mask = (y_true == 1) & (y_pred == 1)
            if tp_mask.sum() > 0:
                mean_lead_time = days_before_episode[tp_mask].mean()
                metrics.task_specific['mean_lead_time_days'] = mean_lead_time
        
        # False alarm rate
        false_alarm_rate = metrics.false_positives / (metrics.false_positives + metrics.true_negatives)
        metrics.task_specific['false_alarm_rate'] = false_alarm_rate
        
        result = BenchmarkResult(
            task=BenchmarkTask.BIPOLAR_PREDICTION,
            model_name="unknown",
            dataset="unknown",
            metrics=metrics,
            metadata={
                'target_sensitivity': 0.75,
                'target_lead_time': (3, 7),  # days
                'target_false_alarm_rate': 0.30,
                'passes_threshold': (
                    metrics.recall >= 0.75 and
                    false_alarm_rate <= 0.30
                )
            }
        )
        
        return result
    
    def evaluate_adhd_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        theta_beta_ratios: Optional[np.ndarray] = None
    ) -> BenchmarkResult:
        """
        Evaluate ADHD classification.
        
        Target metrics:
        - Specificity > 0.80 (avoid misdiagnosis)
        - Sensitivity > 0.75
        - Theta/beta ratio correlation
        
        Args:
            y_true: True labels (0=no ADHD, 1=ADHD)
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            theta_beta_ratios: EEG theta/beta ratios
        
        Returns:
            BenchmarkResult
        """
        metrics = self.compute_metrics(y_true, y_pred, y_prob)
        
        # Task-specific: theta/beta correlation
        if theta_beta_ratios is not None:
            from scipy.stats import pearsonr
            tb_correlation, _ = pearsonr(y_prob, theta_beta_ratios)
            metrics.task_specific['theta_beta_correlation'] = tb_correlation
        
        result = BenchmarkResult(
            task=BenchmarkTask.ADHD_CLASSIFICATION,
            model_name="unknown",
            dataset="unknown",
            metrics=metrics,
            metadata={
                'target_specificity': 0.80,
                'target_sensitivity': 0.75,
                'passes_threshold': (
                    metrics.specificity >= 0.80 and
                    metrics.recall >= 0.75
                )
            }
        )
        
        return result
    
    def evaluate_treatment_response(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        baseline_severity: Optional[np.ndarray] = None
    ) -> BenchmarkResult:
        """
        Evaluate treatment response prediction.
        
        Target metrics:
        - AUC > 0.75
        - Balanced accuracy (important for treatment decisions)
        
        Args:
            y_true: True labels (0=non-responder, 1=responder)
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            baseline_severity: Baseline symptom severity
        
        Returns:
            BenchmarkResult
        """
        metrics = self.compute_metrics(y_true, y_pred, y_prob)
        
        # Balanced accuracy
        balanced_acc = (metrics.recall + metrics.specificity) / 2
        metrics.task_specific['balanced_accuracy'] = balanced_acc
        
        # Stratify by severity if available
        if baseline_severity is not None:
            # Split by median severity
            median = np.median(baseline_severity)
            mild_mask = baseline_severity < median
            severe_mask = baseline_severity >= median
            
            if mild_mask.sum() > 0:
                mild_acc = accuracy_score(y_true[mild_mask], y_pred[mild_mask])
                metrics.task_specific['accuracy_mild'] = mild_acc
            
            if severe_mask.sum() > 0:
                severe_acc = accuracy_score(y_true[severe_mask], y_pred[severe_mask])
                metrics.task_specific['accuracy_severe'] = severe_acc
        
        result = BenchmarkResult(
            task=BenchmarkTask.TREATMENT_RESPONSE,
            model_name="unknown",
            dataset="unknown",
            metrics=metrics,
            metadata={
                'target_auc': 0.75,
                'target_balanced_accuracy': 0.75,
                'passes_threshold': (
                    metrics.auc_roc >= 0.75 and
                    balanced_acc >= 0.75
                )
            }
        )
        
        return result
    
    def evaluate_multimodal_fusion(
        self,
        y_true: np.ndarray,
        predictions: Dict[str, Tuple[np.ndarray, np.ndarray]]  # modality -> (y_pred, y_prob)
    ) -> Dict[str, BenchmarkResult]:
        """
        Evaluate multimodal fusion via ablation study.
        
        Tests:
        - EEG only
        - Digital phenotype only
        - Clinical only
        - EEG + Digital
        - EEG + Clinical
        - Digital + Clinical
        - All modalities (full fusion)
        
        Args:
            y_true: Ground truth labels
            predictions: Dict mapping modality combination to predictions
        
        Returns:
            Dict of benchmark results per configuration
        """
        results = {}
        
        for config_name, (y_pred, y_prob) in predictions.items():
            metrics = self.compute_metrics(y_true, y_pred, y_prob)
            
            result = BenchmarkResult(
                task=BenchmarkTask.MULTIMODAL_FUSION,
                model_name=f"fusion_{config_name}",
                dataset="unknown",
                metrics=metrics,
                metadata={'modality_config': config_name}
            )
            
            results[config_name] = result
        
        # Compute complementarity scores
        if 'full' in results and 'eeg_only' in results:
            full_auc = results['full'].metrics.auc_roc
            eeg_auc = results['eeg_only'].metrics.auc_roc
            complementarity = full_auc - eeg_auc
            
            results['full'].metadata['complementarity_score'] = complementarity
        
        return results
    
    def generate_report(
        self,
        results: List[BenchmarkResult]
    ) -> str:
        """
        Generate benchmark report.
        
        Args:
            results: List of benchmark results
        
        Returns:
            Markdown report
        """
        report = "# Darwin Neuropsychiatric Benchmark Report\n\n"
        
        # Group by task
        by_task = {}
        for result in results:
            task = result.task
            if task not in by_task:
                by_task[task] = []
            by_task[task].append(result)
        
        # Report per task
        for task, task_results in by_task.items():
            report += f"## {task.value.replace('_', ' ').title()}\n\n"
            
            report += "| Model | Dataset | Accuracy | Precision | Recall | Specificity | F1 | AUC |\n"
            report += "|-------|---------|----------|-----------|--------|-------------|----|----|\\n"
            
            for result in task_results:
                m = result.metrics
                report += f"| {result.model_name} | {result.dataset} | "
                report += f"{m.accuracy:.3f} | {m.precision:.3f} | {m.recall:.3f} | "
                report += f"{m.specificity:.3f} | {m.f1_score:.3f} | {m.auc_roc:.3f} |\n"
            
            report += "\n"
            
            # Task-specific metrics
            if task_results and task_results[0].metrics.task_specific:
                report += "**Task-Specific Metrics:**\n\n"
                for key, value in task_results[0].metrics.task_specific.items():
                    report += f"- {key}: {value:.3f}\n"
                report += "\n"
        
        return report


# Factory function
def get_benchmark() -> DarwinNeuroBenchmark:
    """Factory function"""
    return DarwinNeuroBenchmark()


# Example usage
if __name__ == "__main__":
    benchmark = get_benchmark()
    
    # Simulate depression diagnosis evaluation
    np.random.seed(42)
    n_samples = 500
    
    y_true = np.random.randint(0, 2, n_samples)
    y_prob = np.random.rand(n_samples)
    y_pred = (y_prob > 0.5).astype(int)
    
    result = benchmark.evaluate_depression_diagnosis(y_true, y_pred, y_prob)
    
    print("=" * 60)
    print("Depression Diagnosis Benchmark")
    print("=" * 60)
    print(f"\nAccuracy: {result.metrics.accuracy:.3f}")
    print(f"Sensitivity: {result.metrics.recall:.3f}")
    print(f"Specificity: {result.metrics.specificity:.3f}")
    print(f"AUC: {result.metrics.auc_roc:.3f}")
    print(f"\nPasses threshold: {result.metadata['passes_threshold']}")

