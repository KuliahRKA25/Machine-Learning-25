"""Generator for the final solution notebook (self-contained)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.cells = []


def code(s):
    nb.cells.append(nbf.v4.new_code_cell(s))


def md(s):
    nb.cells.append(nbf.v4.new_markdown_cell(s))


md("""\
# Praktikum 1 ML RKA 2026 — pay_band classification (F1-macro)

Dirty-HR-data competition: 60,430 train / 29,070 test rows, 20 features,
target `pay_band` (4 ordered classes, Band 4+ only ~2%).
Metric: **F1-macro**.

## Key findings from EDA
- `salary_raw` has ~10 formats: `k`-suffix (`46.4k`), EU decimal
  (`20.855,27`), US (`48,455.37`), symbols (`£`,`$`,`€`),
  and **monthly** vs annual (`per month`/`monthly`/`/month` values are ~12× smaller).
- `currency` is dirty (`-`, `unknown`, `?`, NaN) but **country → currency is
  100% deterministic**, so currency is always recoverable.
- FX rates implied by the data (median salary per currency / USD median) are
  realistic and make pay-band medians line up across currencies.
- `pay_band` is driven by a **latent seniority level** that jointly determines
  salary, age, years_experience and job_title. Salary-only model = 0.672
  macro-F1; full feature set = 0.863.
- Test set has extra dirt: age as text (`b. 1989`, `54 yrs`, `30 y.o.`,
  `born 2001`) and yoe as text (`168 months`, `27 yrs`) — both recoverable.
- `performance_review_score` is bimodal: 1–5 scale and 20–100 scale (÷20).
- Train/test missingness differs: train drops age/yoe (32%/36%), test drops
  hire_date (76%) but keeps age/yoe (~92% recoverable).

## Pipeline
1. Parse salary → annual USD salary (FX-normalized).
2. Normalize all categorical typos/abbreviations.
3. Recover text-formatted age/yoe.
4. Job-title normalization + title salary stats + salary/title-median ratio.
5. LightGBM (4 seeds × 5 folds) + XGBoost, probability blend, mild class-bias
   tuning for macro-F1.

## Results
- LightGBM ensemble OOF macro-F1: **0.8637**; XGBoost OOF: **0.8601**
- Blend (0.95/0.05) + class-bias OOF: **0.8648**
- OOF on rows with age & yoe present (test-like coverage): **0.8937**
- Kaggle: **public 0.87128 / private 0.86735**
  (previous best private was 0.85919; private is the ranking metric)
""")

code("""\
import pandas as pd, numpy as np, re
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

pd.set_option('display.width', 200)
""")

md("""\
## 1. Load data & sanity checks
""")

code("""\
tr = pd.read_csv('train.csv')
te = pd.read_csv('test.csv')
sub = pd.read_csv('sample_submission.csv')

print(f"train: {tr.shape}, test: {te.shape}")
print(f"\\nTarget distribution:\\n{tr['pay_band'].value_counts().to_string()}")
print(f"\\nNull rates (train vs test):")
na = pd.DataFrame({'train': tr.isna().mean(), 'test': te.isna().mean()})
print(na[na.max(axis=1) > 0].round(3).to_string())
""")

md("""\
## 2. Cleaning functions

### 2.1 Salary parsing
Handles `k`-suffix, EU/US/decimal-comma numbers, currency symbols,
and monthly-vs-annual detection.
""")

code("""\
def parse_salary(s):
    \"\"\"-> (value, currency_hint, per_month_flag)\"\"\"
    if pd.isna(s):
        return (np.nan, None, False)
    s = str(s).strip()
    per_month = bool(re.search(r'(per month|monthly|/month)', s, re.I))
    cur = None
    m = re.search(r'(USD|EUR|GBP|PHP|KES|ZAR|SGD|CAD|BRL|INR|AUD|NGN)', s)
    if m:
        cur = m.group(1)
    if s.startswith('£'):
        cur = cur or 'GBP'
    if s.startswith('$'):
        cur = cur or 'USD'
    if s.startswith('€'):
        cur = cur or 'EUR'
    km = re.search(r'([\\d.,]+)\\s*k\\b', s, re.I)
    s2 = s.replace('£', '').replace('$', '').replace('€', '')
    s2 = re.sub(r'(?i)(per month|monthly|/month|USD|EUR|GBP|PHP|KES|ZAR|SGD|CAD|BRL|INR|AUD|NGN)', '', s2)
    s2 = s2.strip().strip(',').strip()
    if km:
        v = km.group(1)
        if re.match(r'^[\\d.,]+$', v):
            return (float(v.replace(',', '')) * 1000, cur, per_month)
    if not s2:
        return (np.nan, cur, per_month)
    if re.match(r'^\\d{1,3}(\\.\\d{3})+,\\d+$', s2):        # EU: 20.855,27
        return (float(s2.replace('.', '').replace(',', '.')), cur, per_month)
    if re.match(r'^\\d{1,3}(,\\d{3})+(\\.\\d+)?$', s2):      # US: 48,455.37
        return (float(s2.replace(',', '')), cur, per_month)
    if re.match(r'^\\d{1,3},\\d{2}$', s2):                   # decimal comma: 384,54
        return (float(s2.replace(',', '.')), cur, per_month)
    if re.match(r'^\\d+(\\.\\d+)?$', s2):                    # plain
        return (float(s2), cur, per_month)
    return (np.nan, cur, per_month)
""")

md("""\
### 2.2 Category normalization maps
Country→currency is 100% deterministic (verified on train), which also
repairs the dirty `currency` column. Gender/education/department/remote/
performance have case + abbreviation + typo variants.
""")

code("""\
COUNTRY_TO_CUR = {
    'US': 'USD', 'USA': 'USD', 'United States': 'USD',
    'UK': 'GBP', 'United Kingdom': 'GBP',
    'Kenya': 'KES', 'KE': 'KES',
    'Germany': 'EUR', 'DE': 'EUR', 'Deutschland': 'EUR',
    'Philippines': 'PHP', 'PH': 'PHP',
    'South Africa': 'ZAR', 'S. Africa': 'ZAR', 'ZA': 'ZAR',
    'Singapore': 'SGD', 'SG': 'SGD',
    'Brazil': 'BRL', 'BR': 'BRL', 'Brasil': 'BRL',
    'Canada': 'CAD', 'CA': 'CAD',
    'India': 'INR', 'IN': 'INR', 'Bharat': 'INR',
    'Australia': 'AUD', 'AU': 'AUD', 'Aus': 'AUD',
    'Nigeria': 'NGN', 'NG': 'NGN',
    'France': 'EUR', 'FR': 'EUR',
    'Netherlands': 'EUR', 'NL': 'EUR', 'Holland': 'EUR',
}
GENDER_MAP = {
    'male': 'male', 'Male': 'male', 'MALE': 'male', 'Man': 'male', 'M': 'male',
    'female': 'female', 'Female': 'female', 'FEMALE': 'female', 'Woman': 'female', 'F': 'female',
    'NB': 'nonbinary', 'Non-binary': 'nonbinary', 'non-binary': 'nonbinary',
    'Other': 'other', 'Prefer not to say': 'prefer_not_to_say',
}
EDU_MAP = {
    "bachelor's degree": 'bachelors', "Bachelor's": 'bachelors', 'Bachelors': 'bachelors', 'BSc': 'bachelors',
    'Associate Degree': 'associates', 'High School': 'high_school',
    "Master's": 'masters', 'MSc': 'masters', 'PhD': 'phd', 'PhD.': 'phd',
}
REMOTE_MAP = {'On-site': 'onsite', 'Onsite': 'onsite', 'On site': 'onsite',
              'remote': 'remote', 'Remote': 'remote', 'REMOTE': 'remote',
              'Hybrid': 'hybrid', 'hybrid': 'hybrid'}
PERF_MAP = {'Meets Expectations': 'meets', 'meets': 'meets',
            'Exceeds Expectations': 'exceeds', 'exceeds': 'exceeds',
            'Below Expectations': 'below'}

DEPT_MAP = {}
for canon in ['Customer Support', 'Product', 'Operations', 'HR', 'Finance',
              'Marketing', 'IT', 'Design', 'Sales', 'Data Science', 'Legal']:
    DEPT_MAP[canon] = canon.lower().replace(' ', '_')
    DEPT_MAP[canon + ' Department'] = canon.lower().replace(' ', '_')
    DEPT_MAP[canon + ' Dept'] = canon.lower().replace(' ', '_')
    DEPT_MAP[canon + ' Dept.'] = canon.lower().replace(' ', '_')
DEPT_MAP['Product team'] = 'product'

COUNTRY_CANON = {
    'US': 'US', 'USA': 'US', 'United States': 'US',
    'UK': 'UK', 'United Kingdom': 'UK', 'Kenya': 'KE', 'KE': 'KE',
    'Germany': 'DE', 'DE': 'DE', 'Deutschland': 'DE',
    'Philippines': 'PH', 'PH': 'PH', 'South Africa': 'ZA', 'S. Africa': 'ZA', 'ZA': 'ZA',
    'Singapore': 'SG', 'SG': 'SG', 'Brazil': 'BR', 'BR': 'BR', 'Brasil': 'BR',
    'Canada': 'CA', 'CA': 'CA', 'India': 'IN', 'IN': 'IN', 'Bharat': 'IN',
    'Australia': 'AU', 'AU': 'AU', 'Aus': 'AU', 'Nigeria': 'NG', 'NG': 'NG',
    'France': 'FR', 'FR': 'FR', 'Netherlands': 'NL', 'NL': 'NL', 'Holland': 'NL',
}
""")

md("""\
### 2.3 Main cleaning pass
- currency: column → salary hint → country map (always resolves)
- annualize monthly salaries (×12), FX-normalize to USD with rates derived
  from train medians (label-free)
- recover text-formatted age (`b. YYYY`, `N yrs`, `N y.o.`, `N years old`)
  and yoe (`N months` ÷12, `N yrs`)
- clip sentinel dirt (age 18–75, yoe/tenure 0–50), divide 20–100-scale
  performance scores by 20
""")

code("""\
JOB_SENIORITY = [
    ('c_level', r'\\b(cfo|coo|cmo|cto|ceo|cpo|chro|ciso|cio|cko)\\b'),
    ('vp', r'\\b(vp|vice president)\\b'),
    ('director', r'\\bdirector\\b'), ('head', r'\\bhead of\\b'),
    ('manager', r'\\bmanager\\b'), ('lead', r'\\blead\\b'),
    ('senior', r'\\bsenior\\b'), ('junior', r'\\bjunior\\b'),
    ('intern', r'\\bintern\\b'), ('staff', r'\\bstaff\\b'),
    ('principal', r'\\bprincipal\\b'),
]

def job_features(jt):
    if pd.isna(jt):
        return {f'jt_{n}': 0 for n, _ in JOB_SENIORITY}
    low = str(jt).lower()
    return {f'jt_{n}': int(bool(re.search(p, low))) for n, p in JOB_SENIORITY}

def clean(df, fx_rates=None):
    df = df.copy()
    parsed = df['salary_raw'].apply(parse_salary)
    df['sal'] = [p[0] for p in parsed]
    df['cur_hint'] = [p[1] for p in parsed]
    df['per_month'] = np.array([p[2] for p in parsed], dtype=int)

    dirty_cur = df['currency'].isin(['-', 'unknown', '?']) | df['currency'].isna()
    df['currency_clean'] = df['currency'].where(~dirty_cur, df['cur_hint'])
    still = df['currency_clean'].isna() | df['currency_clean'].isin(['-', 'unknown', '?'])
    df.loc[still, 'currency_clean'] = df.loc[still, 'country'].map(COUNTRY_TO_CUR)

    df['sal_annual'] = df['sal'] * np.where(df['per_month'] == 1, 12.0, 1.0)
    if fx_rates is None:
        med = df.groupby('currency_clean')['sal_annual'].median()
        fx_rates = (med['USD'] / med).to_dict()
    df['fx_rate'] = df['currency_clean'].map(fx_rates)
    df['sal_usd'] = df['sal_annual'] * df['fx_rate']
    df['log_sal_usd'] = np.log1p(df['sal_usd'])

    df['gender_clean'] = df['gender'].map(GENDER_MAP).fillna('unknown')
    df['edu_clean'] = df['education_level'].map(EDU_MAP).fillna('unknown')
    df['dept_clean'] = df['department'].map(DEPT_MAP)
    df['dept_clean'] = df['dept_clean'].fillna(
        df['department'].str.lower().str.replace(r'\\s*(dept\\.?|department|team)$', '', regex=True))
    df['remote_clean'] = df['remote_status'].map(REMOTE_MAP).fillna('unknown')
    df['perf_rating_clean'] = df['performance_rating'].map(PERF_MAP).fillna('unknown')
    df['country_clean'] = df['country'].map(COUNTRY_CANON).fillna(df['country'])

    prs = pd.to_numeric(df['performance_review_score'], errors='coerce')
    df['perf_score_norm'] = np.where(prs > 5, prs / 20.0, prs)

    REF_YEAR = 2026
    def _parse_age(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, (int, float)):
            return float(v)
        v = str(v).strip()
        m = re.match(r'^(?:b\\.|born)\\s*(\\d{4})$', v)
        if m:
            return float(REF_YEAR - int(m.group(1)))
        m = re.match(r'^(\\d+)\\s*(?:yrs|years\\s*old|y\\.o\\.|years)$', v)
        if m:
            return float(m.group(1))
        return np.nan
    def _parse_yoe(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, (int, float)):
            return float(v)
        v = str(v).strip()
        m = re.match(r'^(\\d+)\\s*months$', v)
        if m:
            return float(m.group(1)) / 12.0
        m = re.match(r'^(\\d+)\\s*(?:yrs|years)$', v)
        if m:
            return float(m.group(1))
        return np.nan
    age = pd.to_numeric(df['age'], errors='coerce').fillna(df['age'].apply(_parse_age))
    df['age_clean'] = age.where((age >= 18) & (age <= 75))
    yoe = pd.to_numeric(df['years_experience'], errors='coerce').fillna(df['years_experience'].apply(_parse_yoe))
    df['yoe_clean'] = yoe.where((yoe >= 0) & (yoe <= 50))
    ten = pd.to_numeric(df['tenure_years'], errors='coerce')
    df['tenure_clean'] = ten.where((ten >= 0) & (ten <= 50))
    df['import_batch'] = pd.to_numeric(df['import_batch'], errors='coerce')

    jf = df['job_title'].apply(job_features)
    for k in jf[0]:
        df[k] = [r[k] for r in jf]

    df['hire_dt'] = pd.to_datetime(df['hire_date'], errors='coerce', format='mixed')
    mask = df['hire_dt'].isna() & df['hire_date'].notna()
    if mask.any():
        df.loc[mask, 'hire_dt'] = pd.to_datetime(df.loc[mask, 'hire_date'], errors='coerce', format='%Y%m%d')
    df['hire_year'] = df['hire_dt'].dt.year
    df['hire_month'] = df['hire_dt'].dt.month
    df['notes_clean'] = df['notes'].fillna('none')
    return df, fx_rates
""")

md("""\
## 3. Feature engineering
Job titles carry typos/abbreviations (`anlst`/`analist`/`analyst`,
`enginer`/`eng.`/`engineer`, `manger`/`mgr`/`manager`). Normalizing them
into a categorical feature is the single biggest win (0.824 → 0.863 OOF).
Also: title median salary, title frequency, and salary-vs-title-median ratio.
""")

code("""\
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
    t = re.sub(r'\\s+', ' ', str(jt).lower().strip()).replace('.', ' ')
    return ' '.join(TYPO_MAP.get(w, w) for w in t.split())

df_tr, fx = clean(tr)
df_te, _ = clean(te, fx_rates=fx)
df_tr['title_clean'] = df_tr['job_title'].apply(norm_title)
df_te['title_clean'] = df_te['job_title'].apply(norm_title)

# title salary stats from TRAIN only (salary is a feature, not the target)
tstats = df_tr.groupby('title_clean').agg(
    title_med_sal=('sal_usd', 'median'), title_n=('sal_usd', 'size')).reset_index()
tstats['title_log_med_sal'] = np.log1p(tstats['title_med_sal'])
df_tr = df_tr.merge(tstats, on='title_clean', how='left')
df_te = df_te.merge(tstats, on='title_clean', how='left')
for d in (df_tr, df_te):
    d['sal_title_ratio'] = d['sal_usd'] / (d['title_med_sal'] + 1)
    d['title_freq'] = np.log1p(d['title_n'])
    d['title_rare'] = (d['title_n'] < 20).astype(int)

print(f"recovered test age coverage: {(1 - df_te['age_clean'].isna().mean()) * 100:.1f}%")
print(f"recovered test yoe coverage: {(1 - df_te['yoe_clean'].isna().mean()) * 100:.1f}%")
print(f"unresolved currency in test: {df_te['currency_clean'].isna().sum()}")
print(f"implied FX rates: { {k: round(v, 4) for k, v in sorted(fx.items())} }")
""")

md("""\
## 4. Cross-validated ensemble
LightGBM (4 seeds × 5 stratified folds, balanced class weights) blended with
XGBoost. OOF macro-F1 is the model-selection metric.
""")

code("""\
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
bands = y_cat.cat.categories
Xtr, Xte = df_tr[FEATS].copy(), df_te[FEATS].copy()
for c in CATS:
    Xtr[c] = Xtr[c].astype('category')
    Xte[c] = Xte[c].astype('category')

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
SEEDS = [42, 43, 44, 45]
oof_lgb = np.zeros((len(Xtr), 4))
oof_xgb = np.zeros((len(Xtr), 4))
test_lgb = np.zeros((len(Xte), 4))
test_xgb = np.zeros((len(Xte), 4))

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

    Xtr_f, Xte_f = Xtr.copy(), Xte.copy()
    for c in CATS:
        Xtr_f[c] = Xtr_f[c].cat.codes
        Xte_f[c] = Xte_f[c].cat.codes
    warr = np.full(len(trn), len(trn) / (4 * np.bincount(labels[trn])))[labels[trn]]
    mx = xgb.train({
        'objective': 'multi:softprob', 'num_class': 4, 'eval_metric': 'mlogloss',
        'learning_rate': 0.03, 'max_depth': 7, 'min_child_weight': 10,
        'subsample': 0.8, 'colsample_bytree': 0.75, 'lambda': 2.0,
        'tree_method': 'hist', 'seed': 42 + fi},
        xgb.DMatrix(Xtr_f.iloc[trn], label=labels[trn]),
        num_boost_round=5000,
        evals=[(xgb.DMatrix(Xtr_f.iloc[val], label=labels[val]), 'val')],
        early_stopping_rounds=200, verbose_eval=False, sample_weight=warr)
    oof_xgb[val] = mx.predict(xgb.DMatrix(Xtr_f.iloc[val]), iteration_range=(0, mx.best_iteration))
    test_xgb += mx.predict(xgb.DMatrix(Xte_f), iteration_range=(0, mx.best_iteration)) / 5
    print(f'fold {fi} done', flush=True)

def mf(p):
    return f1_score(labels, p.argmax(1), average='macro')
print(f'LGB ensemble OOF macro-F1: {mf(oof_lgb):.5f}')
print(f'XGB OOF macro-F1:          {mf(oof_xgb):.5f}')
""")

md("""\
## 5. Blend + class-bias tuning
Small per-class probability multipliers, tuned on OOF with restricted deltas
(0.95–1.15) to avoid overfitting the OOF sample.
""")

code("""\
best = (None, -1)
for a in np.arange(0.6, 1.01, 0.05):
    s = mf(a * oof_lgb + (1 - a) * oof_xgb)
    print(f'blend lgb={a:.2f}: {s:.5f}')
    if s > best[1]:
        best = (a, s)
alpha = best[0]
oof = alpha * oof_lgb + (1 - alpha) * oof_xgb
test_p = alpha * test_lgb + (1 - alpha) * test_xgb

bias = np.ones(4)
def _mf_b(b):
    return f1_score(labels, (oof * b).argmax(1), average='macro')
improved = True
while improved:
    improved = False
    for c in range(4):
        for d in (0.95, 1.0, 1.05, 1.1, 1.15):
            cand = bias.copy()
            cand[c] *= d
            if _mf_b(cand) > _mf_b(bias) + 1e-6:
                bias = cand
                improved = True
print(f'bias: {np.round(bias, 4)}, OOF macro-F1: {_mf_b(bias):.5f}')

pred = (test_p * bias).argmax(1)
sub['pay_band'] = [bands[p] for p in pred]
sub.to_csv('submission.csv', index=False)
print(sub['pay_band'].value_counts().to_string())
""")

with open('praktikum1_solution.ipynb', 'w') as f:
    nbf.write(nb, f)
print('Created praktikum1_solution.ipynb with', len(nb.cells), 'cells')
