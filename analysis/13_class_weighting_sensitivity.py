# 13 class weighting sensitivity
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


# One standalone Colab cell: compare class-weighted and original downstream models.
# The weighted ratio is calculated from each 2024 TRAINING partition only.
# This is a robustness comparison, not a replacement for reported results.

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

root = Path(str(ANALYSIS_ROOT))
inputs = [root / 'datasets/nemsis_2024.parquet',
          root / 'datasets/nemsis_2025.parquet']
assert all(path.exists() for path in inputs), f'Check input paths: {inputs}'
out = root / 'revision_analysis/outputs/13_downstream_class_weight_comparison'
out.mkdir(parents=True, exist_ok=True)
raw24, raw25 = (pd.read_parquet(path) for path in inputs)

def code(df):
    return df['eDisposition_19'].astype('string').str.strip()

def acuity_records(df, include_dead):
    high = {'4219001', '4219003'} | ({'4219007'} if include_dead else set())
    keep = code(df).isin(high | {'4219005'})
    selected = df.loc[keep].copy()
    selected['_high_acuity'] = code(selected).isin(high).astype('int8')
    return selected

original24 = acuity_records(raw24, False)
original25 = acuity_records(raw25, False)
dead24 = raw24.loc[code(raw24).eq('4219007')].copy()
dead24['_high_acuity'] = np.int8(1)
revised25 = acuity_records(raw25, True)
assert (len(original24), len(original25), len(dead24), len(revised25)) == (
    115877, 309790, 885, 312114), 'Unexpected record counts: verify the yearly files.'

# Reproduce the original 2024 forward-analysis partitions for eligible records.
base_train, base_temp = train_test_split(original24, test_size=.30,
    random_state=42, shuffle=True, stratify=original24['_high_acuity'])
base_val, base_test = train_test_split(base_temp, test_size=.50,
    random_state=42, shuffle=True, stratify=base_temp['_high_acuity'])
# For the separate sensitivity analysis, add the previously excluded records.
dead_train, dead_temp = train_test_split(dead24, test_size=.30,
    random_state=42, shuffle=True)
dead_val, dead_test = train_test_split(dead_temp, test_size=.50,
    random_state=42, shuffle=True)
splits = {
    'primary': (base_train, base_val, base_test, original25),
    'dead_included': (
        pd.concat([base_train, dead_train], ignore_index=True),
        pd.concat([base_val, dead_val], ignore_index=True),
        pd.concat([base_test, dead_test], ignore_index=True),
        revised25),
}

numeric = ['computed_EMSChuteTimeMin', 'computed_EMSDispatchCenterTimeSec',
    'computed_EMSSceneResponseTimeMin', 'computed_EMSSceneToPatientTimeMin',
    'computed_EMSSystemResponseTimeMin', 'computed_ageinyear']
baseline = numeric + ['computed_NasemsoRegion', 'computed_USCensusDivision',
    'computed_USCensusRegion', 'computed_Urbanicity', 'eArrest_05',
    'eSituation_01', 'eSituation_02', 'eSituation_07', 'eSituation_08',
    'eSituation_13', 'eSituation_18', 'eSituation_20']
features = {'baseline': (baseline, numeric),
            'ems_rri': (baseline + ['ems_rri_basic'], numeric + ['ems_rri_basic'])}
assert len(baseline) == 18 and len(features['ems_rri'][0]) == 19
for year, df in [('2024', raw24), ('2025', raw25)]:
    missing = set(features['ems_rri'][0] + ['eDisposition_19']) - set(df.columns)
    assert not missing, f'{year} missing fields: {sorted(missing)}'

def predictor_data(data, columns, numeric_columns):
    x = data[columns].copy()
    for field in numeric_columns:
        x[field] = pd.to_numeric(x[field], errors='coerce')
    return x

def build_model(numeric_columns, categorical_columns, positive_weight):
    prep = ColumnTransformer([
        ('numeric', Pipeline([
            ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
            ('scaler', RobustScaler())]), numeric_columns),
        ('categorical', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))]), categorical_columns)])
    parameters = dict(n_estimators=500, learning_rate=.05, max_depth=6,
        min_child_weight=5, subsample=.8, colsample_bytree=.8,
        tree_method='hist', objective='binary:logistic', eval_metric='logloss',
        random_state=42, n_jobs=-1)
    if positive_weight is not None:
        parameters['scale_pos_weight'] = positive_weight
    return Pipeline([('preprocessor', prep),
                     ('classifier', XGBClassifier(**parameters))])

def evaluate(y, probability, threshold):
    pred = (probability >= threshold).astype('int8')
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(n=len(y), prevalence=float(y.mean()),
        precision=precision_score(y, pred, zero_division=0),
        recall=recall_score(y, pred, zero_division=0),
        f1=f1_score(y, pred, zero_division=0),
        roc_auc=roc_auc_score(y, probability),
        pr_auc=average_precision_score(y, probability),
        brier=brier_score_loss(y, probability),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))

records, predictions, selected_thresholds, ratios = [], {}, {}, {}
for analysis, (train, val, test, forward) in splits.items():
    y_train = train['_high_acuity'].to_numpy()
    y_val = val['_high_acuity'].to_numpy()
    ratio = float((y_train == 0).sum() / (y_train == 1).sum())
    ratios[analysis] = ratio
    print('\n', analysis, 'sizes:', len(train), len(val), len(test), len(forward),
          '| 2024 training negative/positive ratio:', round(ratio, 5))
    for weighting, weight in [('original', None), ('class_weighted', ratio)]:
        for name, (columns, numeric_columns) in features.items():
            categorical_columns = [f for f in columns if f not in numeric_columns]
            model = build_model(numeric_columns, categorical_columns, weight)
            print('Fitting', analysis, weighting, name, '...')
            model.fit(predictor_data(train, columns, numeric_columns), y_train)
            val_probability = model.predict_proba(
                predictor_data(val, columns, numeric_columns))[:, 1]
            candidates = np.round(np.arange(.01, 1.00, .01), 2)
            scores = [f1_score(y_val, val_probability >= t, zero_division=0)
                      for t in candidates]
            threshold = float(candidates[int(np.argmax(scores))])
            selected_thresholds[(analysis, weighting, name)] = threshold
            print('  2024 validation-selected F1 threshold:', threshold)
            for dataset, data in [('2024 internal test', test), ('2025 forward', forward)]:
                probability = model.predict_proba(
                    predictor_data(data, columns, numeric_columns))[:, 1]
                predictions[(analysis, weighting, name, dataset)] = probability
                metrics = evaluate(data['_high_acuity'].to_numpy(),
                                   probability, threshold)
                records.append(dict(analysis=analysis, weighting=weighting,
                    model=name, dataset=dataset, threshold=threshold, **metrics))

results = pd.DataFrame(records)
results.to_csv(out / 'all_model_metrics.csv', index=False)
print('\nALL MODEL METRICS\n' + results.round(6).to_string(index=False))

# Check that the original unweighted runs were recreated before interpreting
# any comparison. Small deviations can occur across software versions.
reference_f1 = {('primary', 'baseline'): .805654,
                ('primary', 'ems_rri'): .805868,
                ('dead_included', 'baseline'): .808656,
                ('dead_included', 'ems_rri'): .810256}
for (analysis, name), expected in reference_f1.items():
    observed = results.loc[(results.analysis == analysis) &
        (results.weighting == 'original') & (results.model == name) &
        (results.dataset == '2025 forward'), 'f1'].iloc[0]
    status = 'MATCH' if abs(observed - expected) < .0001 else 'CHECK'
    print('Original-result check:', status, analysis, name,
          'current F1', round(observed, 6), 'earlier F1', expected)

# Paired uncertainty: the same 2025 record indices are sampled for each pair.
# Reports changes in the class-weighted approach and the incremental EMS-RRI
# effect under class weighting. Each model retains its own validation threshold.
comparison_rows = []
pairs = [('baseline: weighted minus original',
          ('class_weighted', 'baseline'), ('original', 'baseline')),
         ('ems_rri: weighted minus original',
          ('class_weighted', 'ems_rri'), ('original', 'ems_rri')),
         ('weighted: ems_rri minus baseline',
          ('class_weighted', 'ems_rri'), ('class_weighted', 'baseline'))]
metric_names = ['f1', 'recall', 'precision', 'roc_auc', 'pr_auc', 'brier']
for analysis, (_, _, _, forward) in splits.items():
    y = forward['_high_acuity'].to_numpy()
    for pair_number, (label, left, right) in enumerate(pairs):
        lp = predictions[(analysis, *left, '2025 forward')]
        rp = predictions[(analysis, *right, '2025 forward')]
        lt = selected_thresholds[(analysis, *left)]
        rt = selected_thresholds[(analysis, *right)]
        left_point, right_point = evaluate(y, lp, lt), evaluate(y, rp, rt)
        changes = {metric: [] for metric in metric_names}
        rng = np.random.default_rng(20260926 + pair_number +
                                    (10 if analysis == 'dead_included' else 0))
        print('Bootstrapping', analysis, label, '(500 paired repetitions)...')
        for repetition in range(500):
            index = rng.integers(0, len(y), len(y))
            yy = y[index]
            if np.unique(yy).size != 2:
                continue
            l = evaluate(yy, lp[index], lt)
            r = evaluate(yy, rp[index], rt)
            for metric in metric_names:
                changes[metric].append(l[metric] - r[metric])
            if (repetition + 1) % 100 == 0:
                print(' ', repetition + 1, '/ 500')
        for metric in metric_names:
            lo, hi = np.percentile(changes[metric], [2.5, 97.5])
            comparison_rows.append(dict(analysis=analysis, dataset='2025 forward',
                comparison=label, measure=metric,
                difference=left_point[metric] - right_point[metric],
                lower_95=lo, upper_95=hi))

comparisons = pd.DataFrame(comparison_rows)
comparisons.to_csv(out / 'paired_2025_comparisons.csv', index=False)
print('\nPAIRED 2025 COMPARISONS\n' +
      comparisons.round(6).to_string(index=False))

# The original selective gate uses the PRIMARY outcome and the baseline model.
# Report how retraining with weights changes accepted and abstained error rates.
gate_rows = []
primary_y = original25['_high_acuity'].to_numpy()
score = pd.to_numeric(original25['ems_rri_basic'], errors='coerce').to_numpy()
for weighting in ['original', 'class_weighted']:
    probability = predictions[('primary', weighting, 'baseline', '2025 forward')]
    threshold = selected_thresholds[('primary', weighting, 'baseline')]
    errors = (probability >= threshold) != primary_y
    for cutoff in [81.8182, 72.7273]:
        accepted = score > cutoff
        abstained = ~accepted
        accepted_error = float(errors[accepted].mean())
        abstained_error = float(errors[abstained].mean())
        gate_rows.append(dict(weighting=weighting, cutoff=cutoff,
            accepted_n=int(accepted.sum()), abstained_n=int(abstained.sum()),
            abstained_pct=100 * float(abstained.mean()),
            accepted_error_pct=100 * accepted_error,
            abstained_error_pct=100 * abstained_error,
            abstained_over_accepted_error_ratio=abstained_error / accepted_error))
gating = pd.DataFrame(gate_rows)
gating.to_csv(out / 'selective_gate_point_estimates.csv', index=False)
print('\nSELECTIVE GATE POINT ESTIMATES\n' + gating.round(6).to_string(index=False))

# Single-class subgroup: compare detection among reclassified records.
dead_rows = []
mask = code(revised25).eq('4219007').to_numpy()
assert mask.sum() == 2324
for weighting in ['original', 'class_weighted']:
    for name in features:
        probability = predictions[('dead_included', weighting, name, '2025 forward')]
        threshold = selected_thresholds[('dead_included', weighting, name)]
        detected = int((probability[mask] >= threshold).sum())
        dead_rows.append(dict(weighting=weighting, model=name,
            dead_n=int(mask.sum()), detected_n=detected,
            dead_recall=detected / int(mask.sum())))
dead_results = pd.DataFrame(dead_rows)
dead_results.to_csv(out / 'dead_subgroup_detection.csv', index=False)
print('\nDEAD SUBGROUP DETECTION\n' + dead_results.round(6).to_string(index=False))

settings = dict(design='2024 development, 2025 forward evaluation',
    experimental_status='class weighting comparison; no manuscript model selected',
    training_negative_to_positive_ratio=ratios,
    thresholds={'|'.join(key): value for key, value in selected_thresholds.items()},
    weighted_parameter='scale_pos_weight = 2024 training negatives / positives',
    unweighted_parameter='scale_pos_weight not specified, matching original code',
    paired_bootstrap_repetitions=500,
    notes='All models use the same imputation, RobustScaler, one-hot encoding, and XGBoost settings.')
(out / 'settings.json').write_text(json.dumps(settings, indent=2))
print('\nSaved comparison files to:', out)
