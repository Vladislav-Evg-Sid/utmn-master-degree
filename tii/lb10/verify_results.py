"""Audit saved results without running the model on test again."""
from pathlib import Path
import ast,json,hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,balanced_accuracy_score,log_loss
ROOT=Path(__file__).resolve().parent
n=json.loads((ROOT/'10_student.ipynb').read_text(encoding='utf-8'))
test_calls=0
for c in n['cells']:
 if c['cell_type']!='code':continue
 assert c['execution_count'] is not None
 assert not any(o['output_type']=='error' for o in c['outputs'])
 for node in ast.walk(ast.parse(''.join(c['source']))):
  if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='predict_proba':
   if any(isinstance(a,ast.Name) and a.id=='Xtest' for a in node.args):test_calls+=1
assert test_calls==1
decision=json.loads((ROOT/'10_frozen_decision.json').read_text(encoding='utf-8'))
assert not decision['test_used_for_selection']
grid=pd.read_csv(ROOT/'10_validation_threshold_grid.csv')
best=grid.sort_values(['Стоимость_руб','Recall','distance_to_half','порог'],ascending=[True,False,True,True]).iloc[0]
np.testing.assert_allclose(best['порог'],decision['selected_threshold'])
expected=sorted(set([round(t,2) for t in np.arange(.05,1,.05)]+[200/2200]))
np.testing.assert_allclose(grid['порог'],expected)
def check_table(filename, predictions, probability='probability'):
 for _,r in pd.read_csv(ROOT/filename).iterrows():
  prob=predictions['dummy_probability'] if str(r['модель']).startswith('Dummy') else predictions[probability]
  pred=(prob>=r['порог']).astype(int)
  tn,fp,fn,tp=confusion_matrix(predictions.late,pred,labels=[0,1]).ravel()
  np.testing.assert_array_equal([tn,fp,fn,tp],r[['TN','FP','FN','TP']].to_numpy(dtype=int))
  assert r['Стоимость_руб']==200*fp+2000*fn
  for name,fnc in [('Accuracy',accuracy_score),('Precision',precision_score),('Recall',recall_score),('F1',f1_score),('BalancedAccuracy',balanced_accuracy_score)]:
   kwargs={'zero_division':0} if name in ['Precision','Recall','F1'] else {}
   np.testing.assert_allclose(r[name],fnc(predictions.late,pred,**kwargs),atol=1e-12)
  np.testing.assert_allclose(r.LogLoss,log_loss(predictions.late,prob),atol=1e-12)
val=pd.read_csv(ROOT/'10_validation_predictions.csv')
for name in ['10_validation_models.csv','10_validation_fixed_thresholds.csv','10_validation_threshold_grid.csv']:check_table(name,val)
test=pd.read_csv(ROOT/'10_test_predictions_once.csv')
check_table('10_final_test_once.csv',test)
splits=pd.read_csv(ROOT/'split_ids_06.csv').set_index('order_id')
assert val.order_id.is_unique and test.order_id.is_unique
assert splits.loc[val.order_id,'split'].eq('val').all()
assert splits.loc[test.order_id,'split'].eq('test').all()
assert hashlib.sha256((ROOT/'orders_dirty.csv').read_bytes()).hexdigest()==decision['source_sha256']
print('PASS: all saved metrics and costs, threshold/tie rule, IDs, source hash and single test call. No new test inference.')
