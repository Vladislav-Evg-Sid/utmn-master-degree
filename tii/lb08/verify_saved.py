"""Load this lab's trusted artifacts in a fresh Python process."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
pipeline = joblib.load(ROOT / '08_feature_pipeline.joblib')
model = joblib.load(ROOT / '08_demo_model.joblib')
X = pd.read_csv(ROOT / '08_input_val.csv')
expected = np.load(ROOT / '08_validation_reference.npy')
actual = pipeline.transform(X)
schema = pd.read_csv(ROOT / '08_feature_schema.csv').feature_name.tolist()
assert schema == pipeline.named_steps['prep'].get_feature_names_out().tolist()
assert len(schema) == len(set(schema)) == actual.shape[1]
assert actual.shape == expected.shape and np.isfinite(actual).all()
np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
predictions = pd.read_csv(ROOT / '08_validation_predictions.csv')
labels = pd.read_csv(ROOT / '08_y_val.csv')
assert predictions.order_id.tolist() == labels.order_id.tolist()
np.testing.assert_array_equal(predictions.late, labels.late)
np.testing.assert_allclose(model[:-1].transform(X), actual, atol=1e-12, rtol=1e-12)
np.testing.assert_allclose(model.predict_proba(X)[:, 1], predictions.probability, atol=1e-12, rtol=1e-12)
np.testing.assert_array_equal(model.predict(X), predictions.prediction)
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
metrics = pd.read_csv(ROOT / '08_validation_metrics.csv').iloc[0]
for name, value in {
    'balanced_accuracy': balanced_accuracy_score(labels.late, predictions.prediction),
    'f1': f1_score(labels.late, predictions.prediction, zero_division=0),
    'roc_auc': roc_auc_score(labels.late, predictions.probability),
}.items():
    np.testing.assert_allclose(metrics[name], value, atol=1e-12, rtol=1e-12)
result = {'fresh_process': True, 'rows': len(X), 'columns': actual.shape[1],
          'max_abs_diff': float(np.max(np.abs(actual - expected))),
          'predictions_equal': True, 'schema_equal': True,
          'row_ids_equal': True, 'metrics_recomputed': True,
          'model_preprocessing_equal': True}
(ROOT / '08_reload_check.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
print(json.dumps(result))
