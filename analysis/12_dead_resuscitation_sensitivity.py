# 12 dead resuscitation sensitivity
# See companion notebook and docs/PROVENANCE.md.
# Repository configuration; no data are loaded by this cell.
import os
from pathlib import Path
if "EMS_RRI_REPOSITORY_ROOT" in os.environ:
    REPOSITORY_ROOT = Path(os.environ["EMS_RRI_REPOSITORY_ROOT"]).expanduser().resolve()
elif "__file__" in globals():
    REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
else:
    REPOSITORY_ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
                            if (p / "reference/scoring_rules.json").exists()), None)
    if REPOSITORY_ROOT is None:
        raise RuntimeError("Start Jupyter inside the repository, or set EMS_RRI_REPOSITORY_ROOT.")
ANALYSIS_ROOT = Path(os.environ.get("EMS_RRI_ANALYSIS_ROOT", str(REPOSITORY_ROOT / "work"))).expanduser().resolve()
ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
print("Analysis data/output root:", ANALYSIS_ROOT)


# Standalone Colab cell: 2024 -> 2025 sensitivity analysis.
# Reclassify eDisposition.19 = 4219007 (Dead Without Resuscitation Efforts)
# as high acuity. Keep the original 2024 downstream split membership for
# previously eligible records; assign the newly eligible cases separately.

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, brier_score_loss)
from xgboost import XGBClassifier

ROOT = Path(str(ANALYSIS_ROOT))
FILE24 = ROOT / 'datasets/nemsis_2024.parquet'
FILE25 = ROOT / 'datasets/nemsis_2025.parquet'
OUT = ROOT / 'revision_analysis/outputs/12_forward_dead_resuscitation_sensitivity'
assert FILE24.exists() and FILE25.exists(), f'Missing input: {FILE24} or {FILE25}'
OUT.mkdir(parents=True, exist_ok=True)

df24, df25 = pd.read_parquet(FILE24), pd.read_parquet(FILE25)
HIGH_ORIGINAL = {'4219001', '4219003'}
DEAD = '4219007'
LOW = '4219005'
def coded(df):
    return df['eDisposition_19'].astype('string').str.strip()

def eligible(df, include_dead):
    c = coded(df)
    high = HIGH_ORIGINAL | ({DEAD} if include_dead else set())
    out = df.loc[c.isin(high | {LOW})].copy()
    out['_high_acuity'] = coded(out).isin(high).astype('int8')
    return out

original24 = eligible(df24, False)
dead24 = df24.loc[coded(df24).eq(DEAD)].copy()
dead24['_high_acuity'] = np.int8(1)
forward25 = eligible(df25, True)
print('Cohorts:', {'2024_original': len(original24), '2024_added_dead': len(dead24),
                   '2024_revised': len(original24) + len(dead24),
                   '2025_revised': len(forward25),
                   '2025_dead': int(coded(forward25).eq(DEAD).sum())})
assert (len(original24), len(dead24), len(forward25)) == (115877, 885, 312114), \
    'Cohort counts differ from the earlier notebooks; inspect the input files.'

# Recreate the ORIGINAL forward experiment's stratified 2024 split exactly.
base_train, base_temp = train_test_split(original24, test_size=.30,
    random_state=42, shuffle=True, stratify=original24['_high_acuity'])
base_val, base_test = train_test_split(base_temp, test_size=.50,
    random_state=42, shuffle=True, stratify=base_temp['_high_acuity'])
# New cases are all positive, so split them without stratification.
dead_train, dead_temp = train_test_split(dead24, test_size=.30,
    random_state=42, shuffle=True)
dead_val, dead_test = train_test_split(dead_temp, test_size=.50,
    random_state=42, shuffle=True)
train24 = pd.concat([base_train, dead_train], ignore_index=True)
val24 = pd.concat([base_val, dead_val], ignore_index=True)
test24 = pd.concat([base_test, dead_test], ignore_index=True)
assert len(train24) + len(val24) + len(test24) == 116762
print('2024 partitions:', len(train24), len(val24), len(test24),
      '| 2025 forward:', len(forward25))

# Same 18 baseline fields and one EMS-RRI field as the forward main analysis.
numeric = ['computed_EMSChuteTimeMin', 'computed_EMSDispatchCenterTimeSec',
           'computed_EMSSceneResponseTimeMin', 'computed_EMSSceneToPatientTimeMin',
           'computed_EMSSystemResponseTimeMin', 'computed_ageinyear']
baseline = numeric + ['computed_NasemsoRegion', 'computed_USCensusDivision',
    'computed_USCensusRegion', 'computed_Urbanicity', 'eArrest_05',
    'eSituation_01', 'eSituation_02', 'eSituation_07', 'eSituation_08',
    'eSituation_13', 'eSituation_18', 'eSituation_20']
enhanced = baseline + ['ems_rri_basic']
assert len(baseline) == 18 and len(enhanced) == 19
for year, df in [('2024', df24), ('2025', df25)]:
    missing = sorted(set(enhanced + ['eDisposition_19']) - set(df.columns))
    assert not missing, f'{year} is missing columns: {missing}'
assert 'eDisposition_19' not in enhanced

def make_X(df, predictors, nums):
    X = df[predictors].copy()
    for name in nums:
        X[name] = pd.to_numeric(X[name], errors='coerce')
    return X

def make_model(nums, cats):
    numeric_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
        ('scaler', RobustScaler())])
    category_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))])
    pre = ColumnTransformer([('numeric', numeric_pipe, nums),
                             ('categorical', category_pipe, cats)])
    # Matches the executed primary forward downstream notebook (Analysis 08).
    clf = XGBClassifier(n_estimators=500, learning_rate=.05, max_depth=6,
        min_child_weight=5, subsample=.8, colsample_bytree=.8,
        tree_method='hist', objective='binary:logistic', eval_metric='logloss',
        random_state=42, n_jobs=-1)
    return Pipeline([('preprocessor', pre), ('classifier', clf)])

ytrain = train24['_high_acuity'].to_numpy()
yval = val24['_high_acuity'].to_numpy()
models, thresholds, probabilities = {}, {}, {}
partitions = {'2024 internal test': test24, '2025 forward': forward25}
for name, fields, nums in [('Baseline', baseline, numeric),
                           ('Baseline + EMS-RRI', enhanced, numeric + ['ems_rri_basic'])]:
    cats = [col for col in fields if col not in nums]
    model = make_model(nums, cats)
    print('Fitting', name, 'using 2024 training records...')
    model.fit(make_X(train24, fields, nums), ytrain)
    val_prob = model.predict_proba(make_X(val24, fields, nums))[:, 1]
    candidate = np.round(np.arange(.01, 1.00, .01), 2)
    f1_values = [f1_score(yval, val_prob >= t, zero_division=0) for t in candidate]
    threshold = float(candidate[int(np.argmax(f1_values))])
    models[name], thresholds[name] = model, threshold
    print(name, '2024-validation F1 threshold:', threshold)
    for cohort, data in partitions.items():
        probabilities[(cohort, name)] = model.predict_proba(
            make_X(data, fields, nums))[:, 1]

def measures(y, prob, threshold):
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'n': len(y), 'prevalence': y.mean(),
        'precision': precision_score(y, pred, zero_division=0),
        'recall': recall_score(y, pred, zero_division=0),
        'f1': f1_score(y, pred, zero_division=0),
        'roc_auc': roc_auc_score(y, prob),
        'pr_auc': average_precision_score(y, prob),
        'brier': brier_score_loss(y, prob),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}

rows, subgroup_rows, paired_rows = [], [], []
for cohort, data in partitions.items():
    y = data['_high_acuity'].to_numpy()
    dead_mask = coded(data).eq(DEAD).to_numpy()
    assert dead_mask.sum() > 0 and y[dead_mask].min() == 1
    print('\n', cohort, '| total:', len(data), '| dead subgroup:', dead_mask.sum())
    for name in models:
        prob = probabilities[(cohort, name)]
        m = measures(y, prob, thresholds[name])
        rows.append({'dataset': cohort, 'model': name,
                     'threshold': thresholds[name], **m})
        dead_pred = (prob[dead_mask] >= thresholds[name]).astype(int)
        subgroup_rows.append({'dataset': cohort, 'model': name,
            'dead_n': int(dead_mask.sum()), 'detected_n': int(dead_pred.sum()),
            'dead_recall': float(dead_pred.mean())})

    pb = probabilities[(cohort, 'Baseline')]
    pe = probabilities[(cohort, 'Baseline + EMS-RRI')]
    metrics = ('f1', 'recall', 'roc_auc', 'pr_auc', 'brier')
    rng = np.random.default_rng(42 if cohort == '2024 internal test' else 43)
    deltas = {metric: [] for metric in metrics}
    for b in range(500):
        idx = rng.integers(0, len(y), size=len(y))  # paired records
        yy = y[idx]
        if len(np.unique(yy)) != 2:
            continue
        mb = measures(yy, pb[idx], thresholds['Baseline'])
        me = measures(yy, pe[idx], thresholds['Baseline + EMS-RRI'])
        for metric in metrics:
            deltas[metric].append(me[metric] - mb[metric])
        if (b + 1) % 100 == 0:
            print(cohort, 'paired bootstrap', b + 1, '/ 500')
    point_b = measures(y, pb, thresholds['Baseline'])
    point_e = measures(y, pe, thresholds['Baseline + EMS-RRI'])
    for metric in metrics:
        lo, hi = np.percentile(deltas[metric], [2.5, 97.5])
        paired_rows.append({'dataset': cohort, 'metric': metric,
            'delta_enhanced_minus_baseline': point_e[metric] - point_b[metric],
            'lower_95': lo, 'upper_95': hi})
    # Both subgroup labels are positive: bootstrap paired recall difference.
    db = (pb[dead_mask] >= thresholds['Baseline']).astype(int)
    de = (pe[dead_mask] >= thresholds['Baseline + EMS-RRI']).astype(int)
    sub_rng = np.random.default_rng(44 if cohort == '2024 internal test' else 45)
    sub_delta = [float((de[ii] - db[ii]).mean())
        for ii in (sub_rng.integers(0, len(db), size=len(db)) for _ in range(500))]
    lo, hi = np.percentile(sub_delta, [2.5, 97.5])
    subgroup_rows.append({'dataset': cohort, 'model': 'Paired recall difference',
        'dead_n': int(len(db)), 'detected_n': np.nan,
        'dead_recall': float(de.mean() - db.mean()),
        'lower_95': lo, 'upper_95': hi})

overall = pd.DataFrame(rows)
paired = pd.DataFrame(paired_rows)
dead_subgroup = pd.DataFrame(subgroup_rows)
for name, table in [('overall_metrics', overall), ('paired_differences', paired),
                    ('dead_subgroup_recall', dead_subgroup)]:
    table.to_csv(OUT / f'{name}.csv', index=False)
    print('\n' + name.upper() + '\n' + table.round(6).to_string(index=False))
config = {'design': '2024 development -> 2025 forward',
    '2024_original_split_preserved': True, 'new_2024_dead_cases_added': len(dead24),
    '2025_full_revised_cohort': len(forward25), 'thresholds': thresholds,
    'predictors_baseline': baseline, 'predictors_enhanced': enhanced,
    'paired_bootstrap_repetitions': 500, 'class_imbalance_setting':
    'Matches Analysis 08: no scale_pos_weight parameter was set.'}
(OUT / 'configuration.json').write_text(json.dumps(config, indent=2))
print('\nSaved results to:', OUT)
