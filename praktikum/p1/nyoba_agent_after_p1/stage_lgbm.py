"""Stage 1: LightGBM ensemble (4 seeds x 5 folds). Saves OOF + test probs."""
import pandas as pd, numpy as np, re, time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
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

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
SEEDS = [42, 43, 44, 45]
oof_lgb = np.zeros((len(Xtr), 4))
test_lgb = np.zeros((len(Xte), 4))

for fi, (trn, val) in enumerate(skf.split(Xtr, labels)):
    for seed in SEEDS:
        dtr = lgb.Dataset(Xtr.iloc[trn], label=labels[trn])
        dval = lgb.Dataset(Xtr.iloc[val], label=labels[val], reference=dtr)
        w = len(trn) / (4 * np.bincount(labels[trn]))
        dtr.set_weight(w[labels[trn]])
        m = lgb.train({
            'objective': 'multiclass', 'num_class': 4, 'metric': 'multi_logloss',
            'learning_rate': 0.03, 'num_leaves': 127, 'min_data_in_leaf': 15,
            'feature_fraction': 0.75, 'bagging_fraction': 0.8, 'bagging_freq': 5,
            'lambda_l2': 2.0, 'max_bin': 511, 'verbose': -1, 'seed': seed},
            dtr, num_boost_round=5000, valid_sets=[dval],
            callbacks=[lgb.early_stopping(200, verbose=False)])
        oof_lgb[val] += m.predict(Xtr.iloc[val], num_iteration=m.best_iteration) / len(SEEDS)
        test_lgb += m.predict(Xte, num_iteration=m.best_iteration) / (len(SEEDS) * 5)
    sc = f1_score(labels[val], oof_lgb[val].argmax(1), average='macro')
    print(f'fold {fi}: macroF1={sc:.5f}', flush=True)
    np.savez('stage_lgb.npz', oof_lgb=oof_lgb, test_lgb=test_lgb)

print('LGB ensemble OOF macro-F1:', round(f1_score(labels, oof_lgb.argmax(1), average='macro'), 5))
print('time:', round(time.time() - t0, 1), 's')
