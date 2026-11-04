"""Stage 2: XGBoost (5 folds). Saves OOF + test probs."""
import pandas as pd, numpy as np, re, time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import xgboost as xgb
from clean import clean

TYPO_MAP = {
    'anlst': 'analyst', 'analist': 'analyst',
    'enginer': 'engineer', 'eng.': 'engineer', 'eng': 'engineer',
    'manger': 'manager', 'mgr': 'manager', 'scientst': 'scientist',
    'exectuive': 'executive', 'spec.': 'specialist', 'spec': 'specialist',
    'rep': 'representative', 'sr': 'senior', 'jr': 'junior',
    'assoc': 'associate', 'adm': 'admin', 'dev': 'developer',
    'ops': 'operations', 'assit': 'assistant', 'sucess': 'success',
    'strateigst': 'strategist', 'managment': 'management',
    'recruter': 'recruiter', 'consulant': 'consultant',
    'accoutant': 'accountant', 'engneer': 'engineer',
    'developr': 'developer', 'systmes': 'systems', 'systme': 'system',
    'administor': 'administrator', 'coordinater': 'coordinator',
    'desginer': 'designer', 'architecte': 'architect', 'lawer': 'lawyer',
}

def norm_title(jt):
    if pd.isna(jt):
        return 'unknown'
    t = re.sub(r'\s+', ' ', str(jt).lower().strip()).replace('.', ' ')
    return ' '.join(TYPO_MAP.get(w, w) for w in t.split())

t0 = time.time()
tr = pd.read_csv('../data/train.csv')
te = pd.read_csv('../data/test.csv')

df_tr, fx = clean(tr)
df_te, _ = clean(te, fx_rates=fx)
df_tr['title_clean'] = df_tr['job_title'].apply(norm_title)
df_te['title_clean'] = df_te['job_title'].apply(norm_title)

tstats = df_tr.groupby('title_clean').agg(
    title_med_sal=('sal_usd', 'median'), title_n=('sal_usd', 'size')).reset_index()
tstats['title_log_med_sal'] = np.log1p(tstats['title_med_sal'])
df_tr = df_tr.merge(tstats, on='title_clean', how='left')
df_te = df_te.merge(tstats, on='title_clean', how='left')
for d in (df_tr, df_te):
    d['sal_title_ratio'] = d['sal_usd'] / (d['title_med_sal'] + 1)
    d['title_freq'] = np.log1p(d['title_n'])
    d['title_rare'] = (d['title_n'] < 20).astype(int)

CATS = ['currency_clean', 'gender_clean', 'edu_clean', 'dept_clean', 'remote_clean',
        'perf_rating_clean', 'country_clean', 'source_dataset', 'notes_clean', 'title_clean']
NUMS = ['log_sal_usd', 'sal_usd', 'per_month', 'fx_rate', 'age_clean', 'yoe_clean',
        'tenure_clean', 'perf_score_norm', 'import_batch', 'hire_year', 'hire_month',
        'jt_c_level', 'jt_vp', 'jt_director', 'jt_head', 'jt_manager', 'jt_lead',
        'jt_senior', 'jt_junior', 'jt_intern', 'jt_staff', 'jt_principal',
        'sal_title_ratio', 'title_freq', 'title_log_med_sal', 'title_rare']
FEATS = CATS + NUMS

y_cat = df_tr['pay_band'].astype('category')
labels = y_cat.cat.codes.values
Xtr, Xte = df_tr[FEATS].copy(), df_te[FEATS].copy()
for c in CATS:
    Xtr[c] = Xtr[c].astype('category')
    Xte[c] = Xte[c].astype('category')
# label-encode for XGB
for c in CATS:
    Xtr[c] = Xtr[c].cat.codes
    Xte[c] = Xte[c].cat.codes

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_xgb = np.zeros((len(Xtr), 4))
test_xgb = np.zeros((len(Xte), 4))

for fi, (trn, val) in enumerate(skf.split(Xtr, labels)):
    warr = (len(trn) / (4 * np.bincount(labels[trn])))[labels[trn]]
    dwtr = xgb.DMatrix(Xtr.iloc[trn], label=labels[trn], weight=warr)
    dwval = xgb.DMatrix(Xtr.iloc[val], label=labels[val])
    m = xgb.train({
        'objective': 'multi:softprob', 'num_class': 4, 'eval_metric': 'mlogloss',
        'learning_rate': 0.03, 'max_depth': 7, 'min_child_weight': 10,
        'subsample': 0.8, 'colsample_bytree': 0.75, 'lambda': 2.0,
        'tree_method': 'hist', 'seed': 42 + fi},
        dwtr,
        num_boost_round=5000,
        evals=[(dwval, 'val')],
        early_stopping_rounds=200, verbose_eval=False)
    oof_xgb[val] = m.predict(dwval, iteration_range=(0, m.best_iteration))
    test_xgb += m.predict(xgb.DMatrix(Xte), iteration_range=(0, m.best_iteration)) / 5
    sc = f1_score(labels[val], oof_xgb[val].argmax(1), average='macro')
    print(f'fold {fi}: macroF1={sc:.5f} best_iter={m.best_iteration}', flush=True)
    np.savez('stage_xgb.npz', oof_xgb=oof_xgb, test_xgb=test_xgb)
print('time:', round(time.time() - t0, 1), 's')
