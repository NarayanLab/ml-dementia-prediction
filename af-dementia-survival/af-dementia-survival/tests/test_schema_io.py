
import pandas as pd
from afds.io import load_csv, load_config, infer_features
from afds.schema import validate_schema

def test_schema_and_features():
    df = load_csv('data/example/example_cohort.csv')
    cfg = load_config('configs/default.yaml')
    validate_schema(df, cfg['columns']['id'], cfg['columns']['time'], cfg['columns']['event'])
    feats = infer_features(df, cfg['columns']['id'], cfg['columns']['time'], cfg['columns']['event'], cfg['features']['include'], cfg['features']['exclude'])
    assert set(feats) == set(cfg['features']['include'])
