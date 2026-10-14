"""Synthetic checks of the supplied scorer, without executing its data-loading entrypoint."""
import ast
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def load_scoring_definitions():
    tree = ast.parse((ROOT / 'preprocessing/05_calculate_ems_rri.py').read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) or
             (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'RULES' for t in n.targets))]
    namespace = {'np': np, 'pd': pd, 'Path': Path}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'scoring_definitions', 'exec'), namespace)
    return namespace

class ScoringTests(unittest.TestCase):
    def test_documentation_applicability_and_weighting(self):
        ns = load_scoring_definitions()
        rows = []
        for _ in range(7):
            row = {}
            for rule in ns['RULES']:
                c = rule['raw_column']
                row[c] = 'observed'
                row[c + '_documented'] = 1
                row[c + '_not_applicable'] = 0
            row['eArrest_01'] = '3001001'
            row['eScene_07'] = '9923001'
            rows.append(row)
        # Default 11 applicable; missing two gives the paper's 9/11 example.
        rows[1]['eDispatch_01_documented'] = 0
        rows[1]['eDispatch_02_documented'] = 0
        # A non-applicable default field is excluded even if marked documented.
        rows[2]['eDispatch_01_not_applicable'] = 1
        # Conditional triggers activate nine arrest fields and one scene field.
        rows[3]['eArrest_01'] = '3001003'
        rows[4]['eScene_07'] = '9923003'
        # Missing trigger: only its default field is counted as undocumented.
        rows[5]['eArrest_01'] = None
        rows[5]['eArrest_01_documented'] = 0
        # No applicable fields produces an undefined score.
        for rule in ns['RULES']:
            rows[6][rule['raw_column'] + '_not_applicable'] = 1
        frame = pd.DataFrame(rows)
        captured = {}
        def capture(df, *args, **kwargs): captured['scores'] = df.copy()
        with tempfile.TemporaryDirectory() as tmp:
            ns['OUTPUT_DIR'] = Path(tmp)
            with patch.object(pd, 'read_parquet', return_value=frame), patch.object(pd.DataFrame, 'to_parquet', capture), contextlib.redirect_stdout(io.StringIO()), np.errstate(invalid='ignore', divide='ignore'):
                ns['process_year'](2024, Path(tmp)/'input.parquet', Path(tmp)/'output.parquet')
        scores = captured['scores']
        self.assertEqual(scores.ems_rri_applicable_item_count.tolist(), [11,11,10,20,12,11,0])
        self.assertEqual(scores.ems_rri_basic.iloc[1], 81.8182)
        self.assertEqual(scores.ems_rri_basic.iloc[2], 100)
        self.assertEqual(scores.ems_rri_basic.iloc[5], 90.9091)
        # Default weights sum to 1+1+1+0.1+0.8 = 3.9; missing dispatch removes 1.
        self.assertEqual(scores.ems_rri_domain_weighted.iloc[1], round(100*2.9/3.9,4))
        self.assertTrue(pd.isna(scores.ems_rri_basic.iloc[6]))
        self.assertTrue(pd.isna(scores.ems_rri_domain_weighted.iloc[6]))

if __name__ == '__main__':
    unittest.main()
