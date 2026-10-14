"""Check private core analysis inputs; no model fitting or record output."""
import ast
import os
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('EMS_RRI_ANALYSIS_ROOT', str(ROOT/'work'))).expanduser()/'datasets'
tree = ast.parse((ROOT/'analysis/01_forward_temporal_validation.py').read_text())
features = []
for node in tree.body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {'numeric_features','categorical_features'} for t in node.targets):
        features.extend(ast.literal_eval(node.value))
assert len(features) == 32
required = set(features) | {'ems_rri_basic', 'ems_rri_domain_weighted',
    'primary_impression_eSituation_11', 'primary_symptom_eSituation_09'}
for year, expected in [(2024,200000),(2025,500000)]:
    path = DATA/f'nemsis_{year}.parquet'
    meta = pq.ParquetFile(path)
    missing = sorted(required-set(meta.schema_arrow.names))
    if missing: raise ValueError(f'{year}: missing columns: {missing}')
    if meta.metadata.num_rows != expected: raise ValueError(f'{year}: expected {expected} rows, found {meta.metadata.num_rows}')
    frame = pd.read_parquet(path, columns=['ems_rri_basic','ems_rri_domain_weighted'])
    if frame.isna().any().any(): raise ValueError(f'{year}: missing score values')
    if not ((frame >= 0) & (frame <= 100)).all().all(): raise ValueError(f'{year}: scores outside 0–100')
    print(f'{year}: core columns, rows, score range and completeness passed')
    if year == 2024:
        train, _ = train_test_split(frame, test_size=.30, random_state=42, shuffle=True)
        print('Training-only 20th percentiles:', train.quantile(.20).to_dict())
print('Core checks complete. This does not verify input provenance, every audit column, or model results.')
