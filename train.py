import pandas as pd
import numpy as np
import json
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.decomposition import PCA

np.random.seed(42)

TRAIT_PREFIX = {'O': 'OPN', 'C': 'CSN', 'E': 'EXT', 'A': 'AGR', 'N': 'EST'}
TRAIT_NAME = {'O': 'Openness', 'C': 'Conscientiousness', 'E': 'Extraversion', 'A': 'Agreeableness', 'N': 'Neuroticism'}

ITEM_TEXT = {
 'EXT1':"I am the life of the party.", 'EXT2':"I don't talk a lot.", 'EXT3':"I feel comfortable around people.",
 'EXT4':"I keep in the background.", 'EXT5':"I start conversations.", 'EXT6':"I have little to say.",
 'EXT7':"I talk to a lot of different people at parties.", 'EXT8':"I don't like to draw attention to myself.",
 'EXT9':"I don't mind being the center of attention.", 'EXT10':"I am quiet around strangers.",
 'EST1':"I get stressed out easily.", 'EST2':"I am relaxed most of the time.", 'EST3':"I worry about things.",
 'EST4':"I seldom feel blue.", 'EST5':"I am easily disturbed.", 'EST6':"I get upset easily.",
 'EST7':"I change my mood a lot.", 'EST8':"I have frequent mood swings.", 'EST9':"I get irritated easily.",
 'EST10':"I often feel blue.",
 'AGR1':"I feel little concern for others.", 'AGR2':"I am interested in people.", 'AGR3':"I insult people.",
 'AGR4':"I sympathize with others' feelings.", 'AGR5':"I am not interested in other people's problems.",
 'AGR6':"I have a soft heart.", 'AGR7':"I am not really interested in others.", 'AGR8':"I take time out for others.",
 'AGR9':"I feel others' emotions.", 'AGR10':"I make people feel at ease.",
 'CSN1':"I am always prepared.", 'CSN2':"I leave my belongings around.", 'CSN3':"I pay attention to details.",
 'CSN4':"I make a mess of things.", 'CSN5':"I get chores done right away.", 'CSN6':"I often forget to put things back in their proper place.",
 'CSN7':"I like order.", 'CSN8':"I shirk my duties.", 'CSN9':"I follow a schedule.", 'CSN10':"I am exacting in my work.",
 'OPN1':"I have a rich vocabulary.", 'OPN2':"I have difficulty understanding abstract ideas.", 'OPN3':"I have a vivid imagination.",
 'OPN4':"I am not interested in abstract ideas.", 'OPN5':"I have excellent ideas.", 'OPN6':"I do not have a good imagination.",
 'OPN7':"I am quick to understand things.", 'OPN8':"I use difficult words.", 'OPN9':"I spend time reflecting on things.",
 'OPN10':"I am full of ideas.",
}

# ---------------------------------------------------------------------------
# Load + clean
# ---------------------------------------------------------------------------
item_cols = [f'{p}{i}' for p in ['EXT','EST','AGR','CSN','OPN'] for i in range(1,11)]
df = pd.read_csv('bigfive_raw.csv', usecols=item_cols)
df = df.replace(0, np.nan).dropna()
df = df[(df[item_cols] >= 1).all(axis=1) & (df[item_cols] <= 5).all(axis=1)]
print('clean rows', len(df))

# ---------------------------------------------------------------------------
# Data-driven reverse-key detection (via correlation-matrix eigenvector sign)
# ---------------------------------------------------------------------------
detected_reverse = {}
for trait_key, prefix in TRAIT_PREFIX.items():
    cols = [f'{prefix}{i}' for i in range(1, 11)]
    X = df[cols].values.astype(float)
    Xc = X - X.mean(axis=0)
    corr = np.corrcoef(Xc, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(corr)
    top = eigvecs[:, np.argmax(eigvals)]
    if np.sum(top > 0) < np.sum(top < 0):
        top = -top
    reverse_idx = {i+1 for i, loading in enumerate(top) if loading < 0}
    detected_reverse[prefix] = reverse_idx

print('detected reverse keys:', {k: sorted(v) for k,v in detected_reverse.items()})

# ---------------------------------------------------------------------------
# Recode + compute validated full 10-item trait scores (0-100)
# ---------------------------------------------------------------------------
recoded = pd.DataFrame(index=df.index)
for trait_key, prefix in TRAIT_PREFIX.items():
    for i in range(1, 11):
        col = f'{prefix}{i}'
        recoded[col] = 6 - df[col] if i in detected_reverse[prefix] else df[col]

full_scores = pd.DataFrame(index=df.index)
for trait_key, prefix in TRAIT_PREFIX.items():
    cols = [f'{prefix}{i}' for i in range(1, 11)]
    mean_1to5 = recoded[cols].mean(axis=1)
    full_scores[trait_key] = (mean_1to5 - 1) / 4 * 100

# ---------------------------------------------------------------------------
# Pick best 4-item short form per trait (highest item-rest correlation)
# ---------------------------------------------------------------------------
chosen_items = {}
for trait_key, prefix in TRAIT_PREFIX.items():
    cols = [f'{prefix}{i}' for i in range(1, 11)]
    corrs = {}
    for c in cols:
        rest = [x for x in cols if x != c]
        rest_mean = recoded[rest].mean(axis=1)
        corrs[c] = np.corrcoef(recoded[c], rest_mean)[0, 1]
    ranked = sorted(corrs.items(), key=lambda kv: -kv[1])
    top4 = [c for c, _ in ranked[:4]]
    chosen_items[trait_key] = top4

print('\nchosen 4-item short forms:')
for k, v in chosen_items.items():
    flags = ['rev' if int(c[len(TRAIT_PREFIX[k]):]) in detected_reverse[TRAIT_PREFIX[k]] else 'str' for c in v]
    print(k, list(zip(v, flags)))

# ---------------------------------------------------------------------------
# Train/test split, then per trait: Ridge regression (RAW 1-5 answers for the
# 4 chosen items -> full 10-item trait score) + a RandomForest for comparison
# ---------------------------------------------------------------------------
idx = df.index
train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)

models_export = {}
metrics_report = {}

for trait_key, prefix in TRAIT_PREFIX.items():
    feats = chosen_items[trait_key]
    X = df.loc[idx, feats].values.astype(float)   # RAW answers, exactly what a user would type in
    y = full_scores.loc[idx, trait_key].values

    X_train, X_test = X[idx.isin(train_idx)], X[idx.isin(test_idx)]
    y_train, y_test = y[idx.isin(train_idx)], y[idx.isin(test_idx)]

    ridge = Ridge(alpha=5.0)
    ridge.fit(X_train, y_train)
    pred = ridge.predict(X_test)
    r2 = r2_score(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5

    rf = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    r2_rf = r2_score(y_test, pred_rf)
    rmse_rf = mean_squared_error(y_test, pred_rf) ** 0.5

    weights = list(ridge.coef_) + [ridge.intercept_]
    models_export[trait_key] = {
        'items': feats,
        'reverse': [c in feats and (int(c[len(prefix):]) in detected_reverse[prefix]) for c in feats],
        'weights': [round(float(w), 5) for w in weights],
    }
    metrics_report[trait_key] = {
        'ridge_r2': round(float(r2), 4), 'ridge_rmse': round(float(rmse), 2),
        'rf_r2': round(float(r2_rf), 4), 'rf_rmse': round(float(rmse_rf), 2),
        'rf_importances': [round(float(x), 3) for x in rf.feature_importances_],
    }
    print(f"{trait_key} ({TRAIT_NAME[trait_key]}): ridge R2={r2:.3f} RMSE={rmse:.2f}  |  RF R2={r2_rf:.3f} RMSE={rmse_rf:.2f}")

# ---------------------------------------------------------------------------
# PCA on the 5 full trait scores across the whole cleaned population
# ---------------------------------------------------------------------------
pop_matrix = full_scores.values
pca = PCA(n_components=2)
pca.fit(pop_matrix)
proj_all = pca.transform(pop_matrix)

sample_n = 900
sample_idx = np.random.choice(len(pop_matrix), size=sample_n, replace=False)
pop_sample_scores = pop_matrix[sample_idx].round(1).tolist()
pop_sample_proj = proj_all[sample_idx].round(2).tolist()

pca_export = {
    'mean': [round(float(x),3) for x in pca.mean_],
    'components': [[round(float(x),4) for x in row] for row in pca.components_],
    'explained_variance_ratio': [round(float(x),4) for x in pca.explained_variance_ratio_],
}

print('\nPCA explained variance ratio:', pca_export['explained_variance_ratio'])
print('PCA components (rows=PC1,PC2; cols=O,C,E,A,N):')
print(pca.components_)

# ---------------------------------------------------------------------------
# Export everything the frontend needs
# ---------------------------------------------------------------------------
export = {
    'meta': {
        'n_total': int(len(df)),
        'n_train': int(len(train_idx)),
        'n_test': int(len(test_idx)),
        'source': 'IPIP-50 "Big-Five Factor Markers", Open Psychometrics online test (2016-2018), via Kaggle dataset tunguz/big-five-personality-test',
    },
    'item_text': {c: ITEM_TEXT[c] for trait in chosen_items.values() for c in trait},
    'models': models_export,
    'metrics': metrics_report,
    'pca': pca_export,
    'population_sample_scores': pop_sample_scores,   # Nx5 -> O,C,E,A,N (0-100)
    'population_sample_proj': pop_sample_proj,        # Nx2 -> PC1,PC2
    'population_mean_scores': [round(float(full_scores[k].mean()),2) for k in ['O','C','E','A','N']],
}

with open('model_export.json', 'w') as f:
    json.dump(export, f)

print('\nExported model_export.json, size:', __import__('os').path.getsize('model_export.json'), 'bytes')
