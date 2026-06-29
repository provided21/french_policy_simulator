# -*- coding: utf-8 -*-
"""Compute Spearman rank correlation: LLM vs IFOP by CSP and by Age."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

from src.retriever import create_database_adapter

# ── IFOP Benchmarks ──────────────────────────
# From IFOP/CGT 2025-04, n=2,023
# Source: docs/survey_data_catalog.md

# CSP mapping: French occupation -> IFOP CSP categories
CSP_IFOP = {
    # IFOP CSP categories and their support% for "back to 62"
    'Ouvriers': 73,                        # Workers
    'Employes': 77,                        # Employees
    'Professions intermediaires': 65,      # Intermediate professions
    'Cadres et prof intellectuelles sup': 54,  # Executives
}

# Age group benchmarks from IFOP (approximate from cross-tabs)
AGE_IFOP = {
    '18-24': 78,
    '25-34': 72,
    '35-49': 65,
    '50-64': 58,
    '65+': 41,
}

# ── Load LLM Results ────────────────────────
db = create_database_adapter()
queries = db.list_queries()
db.close()

print(f'Database has {len(queries)} queries:')
for _, q in queries.iterrows():
    print(f'  {q["query_id"]}  |  {q.get("question","")[:60]}  |  {q.get("total_responses",0)} responses')

# Load the Run 2 result (best prompt)
# Find the query with ~10000 responses that uses the retirement question
target_id = None
for _, q in queries.iterrows():
    if q.get('total_responses', 0) >= 5000:
        target_id = q['query_id']
        print(f'\nUsing query: {target_id}')
        break

if target_id is None:
    print('ERROR: No query with enough responses found.')
    sys.exit(1)

db = create_database_adapter()
df = db.load(target_id)
question = db.get_question(target_id)
db.close()

print(f'\nLoaded {len(df)} responses')
print(f'Question: {question}')
print(f'Valid stances: {df["stance"].value_counts().to_dict()}')
print(f'Mean support_score: {df["support_score"].mean():.3f}')

# ── PCS -> CSP mapping ──────────────────────
# Our data has PCS categories. Map them to IFOP CSP.
PCS_TO_CSP = {
    'Ouvriers': 'Ouvriers',
    'Employes': 'Employes',
    'Professions intermediaires': 'Professions intermediaires',
    'Cadres et professions intellectuelles superieures': 'Cadres et prof intellectuelles sup',
    'Agriculteurs exploitants': None,  # no IFOP benchmark
    'Artisans, commercants, chefs d\'entreprise': None,
    'Retraites': None,
    'Autres sans activite professionnelle': None,
}

# Map using PCS field if available, otherwise try occupation
if 'pcs' in df.columns:
    df['csp'] = df['pcs'].map(PCS_TO_CSP)
elif 'occupation' in df.columns:
    # Try to infer from occupation
    def occupation_to_csp(occ):
        if pd.isna(occ): return None
        occ = str(occ).strip()
        # Check against known PCS categories
        for pcs, csp in PCS_TO_CSP.items():
            if pcs.lower() in occ.lower():
                return csp
        return None
    df['csp'] = df['occupation'].apply(occupation_to_csp)

valid = df[df['csp'].notna()].copy()
print(f'\n--- Spearman by CSP (n={len(valid)}) ---')
csp_llm = valid.groupby('csp')['support_score'].mean() * 100  # convert to 0-100 scale
print('LLM support % by CSP:')
print(csp_llm.sort_values(ascending=False).to_string())

# Compute Spearman
common = [c for c in CSP_IFOP if c in csp_llm.index]
if len(common) >= 3:
    ifop_vals = [CSP_IFOP[c] for c in common]
    llm_vals = [csp_llm[c] for c in common]
    r, p = spearmanr(ifop_vals, llm_vals)
    print(f'\nIFOP:  {dict(zip(common, ifop_vals))}')
    print(f'LLM:   {dict(zip(common, [round(v,1) for v in llm_vals]))}')
    print(f'\nSpearman r = {r:.3f}  (p = {p:.3f})')
    print(f'  -> {"Significant" if p < 0.05 else "Not significant (n too small)"}')
    print(f'  -> Rank order {"preserved" if r > 0.5 else "NOT preserved"}')
else:
    print(f'Only {len(common)} matching CSP categories, need >= 3')

# ── By Age ──────────────────────────────────
print(f'\n--- Spearman by Age ---')
if 'age' in df.columns:
    df_valid = df[df['age'].notna()].copy()
    df_valid['age_group'] = pd.cut(df_valid['age'].astype(float),
        bins=[18, 25, 35, 50, 65, 120],
        labels=['18-24', '25-34', '35-49', '50-64', '65+'])
    age_llm = df_valid.groupby('age_group')['support_score'].mean() * 100
    print('LLM support % by age:')
    print(age_llm.to_string())

    common_age = [a for a in AGE_IFOP if a in age_llm.index]
    if len(common_age) >= 3:
        ifop_vals_a = [AGE_IFOP[a] for a in common_age]
        llm_vals_a = [age_llm[a] for a in common_age]
        r_a, p_a = spearmanr(ifop_vals_a, llm_vals_a)
        print(f'\nSpearman r = {r_a:.3f}  (p = {p_a:.3f})')
        print(f'  -> {"Significant" if p_a < 0.05 else "Not significant"}')
        print(f'  -> Rank order {"preserved" if r_a > 0.5 else "NOT preserved"}')
else:
    print('No age column found')

print('\nDone.')
