# Bias Mitigated Hiring Pipeline

A fair and transparent machine-learning pipeline for candidate screening that actively detects and mitigates demographic bias throughout every stage of the hiring process.

---

## Motivation

Traditional resume-screening models trained on historical hiring data can inadvertently learn and perpetuate demographic biases (e.g., gender, race, age). This project provides a drop-in pipeline that:

1. **Removes** sensitive attributes and their proxies from features before training.  
2. **Trains** a classifier with a fairness-aware reweighing strategy.  
3. **Evaluates** the resulting model against multiple fairness metrics so that bias is measurable and reportable.

---

## Project Structure

```
Bias-Mitigated-Hiring-Pipeline/
├── data_utils.py          # Dataset loading and preprocessing helpers
├── fairness_metrics.py    # Fairness evaluation (demographic parity, equalized odds, …)
├── pipeline.py            # End-to-end bias-mitigated hiring pipeline class
├── train.py               # Training & evaluation entry point
├── requirements.txt       # Python dependencies
└── README.md
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the pipeline on the built-in synthetic dataset

```bash
python train.py
```

### 3. Use your own dataset

```python
from pipeline import BiasMitigatedHiringPipeline

pipeline = BiasMitigatedHiringPipeline(
    sensitive_attributes=["gender", "race"],
    target_column="hired",
    fairness_threshold=0.1,
)
pipeline.fit(X_train, y_train)
results = pipeline.evaluate(X_test, y_test)
print(results)
```

---

## Fairness Metrics Computed

| Metric | Description |
|---|---|
| **Demographic Parity Difference** | Difference in positive prediction rates between groups |
| **Equalized Odds Difference** | Max difference in TPR and FPR across groups |
| **Disparate Impact Ratio** | Ratio of positive prediction rates (≥ 0.8 is fair by the 80 % rule) |
| **Accuracy** | Overall prediction accuracy |

---

## How Bias Mitigation Works

### Pre-processing – Reweighing
Each training sample is assigned a weight that corrects for the under- or over-representation of combinations of sensitive-attribute values and labels. This ensures the model does not learn from a skewed distribution.

### In-processing – Fair Feature Selection
Sensitive attributes and columns that are highly correlated with them (correlation > configurable threshold) are dropped from the feature set so that the model cannot use proxies for protected characteristics.

### Post-processing – Threshold Calibration *(optional)*
Per-group decision thresholds can be independently tuned so that false-positive and false-negative rates are equalized across demographic groups.

---

## Configuration

`BiasMitigatedHiringPipeline` accepts the following keyword arguments:

| Parameter | Default | Description |
|---|---|---|
| `sensitive_attributes` | `[]` | Column names of protected characteristics |
| `target_column` | `"hired"` | Name of the binary outcome column |
| `correlation_threshold` | `0.5` | Drop features correlated above this level with a sensitive attribute |
| `fairness_threshold` | `0.1` | Warn when a fairness metric (parity/odds) exceeds this value |
| `model_type` | `"logistic"` | Base classifier: `"logistic"` or `"random_forest"` |
| `random_state` | `42` | Seed for reproducibility |

---

## License

MIT
