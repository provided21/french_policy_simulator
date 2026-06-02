import sqlite3, pandas as pd, numpy as np, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

conn = sqlite3.connect('data/results.db')
df = pd.read_sql_query("""
    SELECT support_score, stance, age, sex, occupation, education_level, departement
    FROM responses WHERE query_id = '20260511_022424' AND stance IS NOT NULL
""", conn)
conn.close()

print('=== support_score distribution ===')
print(df['support_score'].describe())
print(f'std={df["support_score"].std():.4f}')

print('\n=== by stance ===')
print(df.groupby('stance')['support_score'].describe())

print('\n=== by occupation ===')
occ = df.groupby('occupation')['support_score'].agg(['mean','std','count']).sort_values('mean')
print(occ.to_string())

print('\n=== by education ===')
edu = df.groupby('education_level')['support_score'].agg(['mean','std','count']).sort_values('mean')
print(edu.to_string())

print('\n=== by age bin ===')
df['age_bin'] = pd.cut(df['age'], bins=[18,25,35,45,55,65,100], labels=['18-25','25-35','35-45','45-55','55-65','65+'])
print(df.groupby('age_bin')['support_score'].agg(['mean','std','count']).to_string())

print('\n=== by sex ===')
print(df.groupby('sex')['support_score'].agg(['mean','std','count']).to_string())

# Compare variance between groups vs within groups
print('\n=== ANOVA-like: between-group vs within-group variance ===')
grand_mean = df['support_score'].mean()
for col in ['occupation','education_level','age_bin','sex']:
    groups = df.groupby(col)['support_score']
    between_var = sum(len(g) * (g.mean() - grand_mean)**2 for _, g in groups) / len(df)
    within_var = sum((len(g)-1) * g.var() for _, g in groups) / (len(df) - df[col].nunique())
    total_var = df['support_score'].var()
    print(f'{col}: between_var={between_var:.6f}, within_var={within_var:.6f}, ratio={between_var/total_var:.4f}')
