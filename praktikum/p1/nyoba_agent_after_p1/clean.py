"""Data cleaning for praktikum-1 dirty HR data.

Key facts established by EDA:
- salary_raw: many formats (k-suffix, EU '20.855,27', US '48,455.37', symbols, /month)
- 'per month|monthly|/month' rows are ~12x smaller -> annualize x12
- country -> currency is 100% deterministic (fallback when currency col dirty)
- FX to USD derived from train medians (label-free)
- performance_review_score: bimodal 1-5 and 20-100 scales (divide latter by 20)
- age/yoe/tenure have sentinel dirt (-1, 0, 999, >100)
- pay_band driven mainly by normalized salary + job seniority
"""
import pandas as pd
import numpy as np
import re

CUR_LIST = ['USD', 'EUR', 'GBP', 'PHP', 'KES', 'ZAR', 'SGD', 'CAD', 'BRL', 'INR', 'AUD', 'NGN']

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
    "Master's": 'masters', 'MSc': 'masters',
    'PhD': 'phd', 'PhD.': 'phd',
}

DEPT_CANON = {
    'Customer Support': 'customer_support', 'Customer Support Department': 'customer_support',
    'Product': 'product', 'Product Department': 'product', 'Product Dept': 'product', 'Product Dept.': 'product', 'Product team': 'product',
    'Operations': 'operations', 'Operations Department': 'operations', 'Operations Dept.': 'operations', 'Operations Dept': 'operations',
    'HR': 'hr',
    'Finance': 'finance', 'Finance Dept.': 'finance', 'Finance Dept': 'finance',
    'Marketing': 'marketing', 'Marketing Dept': 'marketing', 'Marketing Dept.': 'marketing',
    'IT': 'it', 'IT Dept.': 'it', 'IT Dept': 'it',
    'Design': 'design', 'Design Dept': 'design',
    'Sales': 'sales', 'Sales Dept': 'sales', 'Sales Dept.': 'sales',
    'Data Science': 'data_science', 'Data Science Dept.': 'data_science', 'Data Science Dept': 'data_science',
    'Legal': 'legal', 'Legal Dept': 'legal',
}

REMOTE_MAP = {
    'On-site': 'onsite', 'Onsite': 'onsite', 'On site': 'onsite',
    'remote': 'remote', 'Remote': 'remote', 'REMOTE': 'remote',
    'Hybrid': 'hybrid', 'hybrid': 'hybrid',
}

PERF_MAP = {
    'Meets Expectations': 'meets', 'meets': 'meets',
    'Exceeds Expectations': 'exceeds', 'exceeds': 'exceeds',
    'Below Expectations': 'below',
}

COUNTRY_CANON = {
    'US': 'US', 'USA': 'US', 'United States': 'US',
    'UK': 'UK', 'United Kingdom': 'UK',
    'Kenya': 'KE', 'KE': 'KE',
    'Germany': 'DE', 'DE': 'DE', 'Deutschland': 'DE',
    'Philippines': 'PH', 'PH': 'PH',
    'South Africa': 'ZA', 'S. Africa': 'ZA', 'ZA': 'ZA',
    'Singapore': 'SG', 'SG': 'SG',
    'Brazil': 'BR', 'BR': 'BR', 'Brasil': 'BR',
    'Canada': 'CA', 'CA': 'CA',
    'India': 'IN', 'IN': 'IN', 'Bharat': 'IN',
    'Australia': 'AU', 'AU': 'AU', 'Aus': 'AU',
    'Nigeria': 'NG', 'NG': 'NG',
    'France': 'FR', 'FR': 'FR',
    'Netherlands': 'NL', 'NL': 'NL', 'Holland': 'NL',
}


def parse_salary(s):
    """-> (value, currency_hint, per_month_flag)."""
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
    km = re.search(r'([\d.,]+)\s*k\b', s, re.I)
    s2 = s.replace('£', '').replace('$', '').replace('€', '')
    s2 = re.sub(r'(?i)(per month|monthly|/month|USD|EUR|GBP|PHP|KES|ZAR|SGD|CAD|BRL|INR|AUD|NGN)', '', s2)
    s2 = s2.strip().strip(',').strip()
    if km:
        v = km.group(1)
        if re.match(r'^[\d.,]+$', v):
            return (float(v.replace(',', '')) * 1000, cur, per_month)
    if not s2:
        return (np.nan, cur, per_month)
    # European: 20.855,27
    if re.match(r'^\d{1,3}(\.\d{3})+,\d+$', s2):
        return (float(s2.replace('.', '').replace(',', '.')), cur, per_month)
    # US: 48,455.37
    if re.match(r'^\d{1,3}(,\d{3})+(\.\d+)?$', s2):
        return (float(s2.replace(',', '')), cur, per_month)
    # decimal comma: 384,54
    if re.match(r'^\d{1,3},\d{2}$', s2):
        return (float(s2.replace(',', '.')), cur, per_month)
    # plain
    if re.match(r'^\d+(\.\d+)?$', s2):
        return (float(s2), cur, per_month)
    return (np.nan, cur, per_month)


JOB_SENIORITY_PATTERNS = [
    ('c_level', r'\b(cfo|coo|cmo|cto|ceo|cpo|chro|ciso|cio|cko)\b'),
    ('vp', r'\b(vp|vice president)\b'),
    ('director', r'\bdirector\b'),
    ('head', r'\bhead of\b'),
    ('manager', r'\bmanager\b'),
    ('lead', r'\blead\b'),
    ('senior', r'\bsenior\b'),
    ('junior', r'\bjunior\b'),
    ('intern', r'\bintern\b'),
    ('staff', r'\bstaff\b'),
    ('principal', r'\bprincipal\b'),
]


def job_features(jt):
    """-> dict of seniority flags + normalized title."""
    if pd.isna(jt):
        return {f'jt_{n}': 0 for n, _ in JOB_SENIORITY_PATTERNS}
    low = str(jt).lower()
    out = {}
    for name, pat in JOB_SENIORITY_PATTERNS:
        out[f'jt_{name}'] = 1 if re.search(pat, low) else 0
    return out


def clean(df, fx_rates=None):
    df = df.copy()
    parsed = df['salary_raw'].apply(parse_salary)
    df['sal'] = [p[0] for p in parsed]
    df['cur_hint'] = [p[1] for p in parsed]
    df['per_month'] = np.array([p[2] for p in parsed], dtype=int)
    # currency: col if valid, else hint, else country map
    dirty_cur = df['currency'].isin(['-', 'unknown', '?']) | df['currency'].isna()
    df['currency_clean'] = df['currency'].where(~dirty_cur, df['cur_hint'])
    still_missing = df['currency_clean'].isna() | df['currency_clean'].isin(['-', 'unknown', '?'])
    df.loc[still_missing, 'currency_clean'] = df.loc[still_missing, 'country'].map(COUNTRY_TO_CUR)

    # annualize monthly salaries
    df['sal_annual'] = df['sal'] * np.where(df['per_month'] == 1, 12.0, 1.0)

    # FX normalize to USD
    if fx_rates is None:
        med = df.groupby('currency_clean')['sal_annual'].median()
        fx_rates = (med['USD'] / med).to_dict()
    df['fx_rate'] = df['currency_clean'].map(fx_rates)
    df['sal_usd'] = df['sal_annual'] * df['fx_rate']
    df['log_sal_usd'] = np.log1p(df['sal_usd'])

    # categories
    df['gender_clean'] = df['gender'].map(GENDER_MAP).fillna('unknown')
    df['edu_clean'] = df['education_level'].map(EDU_MAP).fillna('unknown')
    df['dept_clean'] = df['department'].map(DEPT_CANON).fillna(df['department'].str.lower().str.replace(r'\s*(dept\.?|department|team)$', '', regex=True))
    df['remote_clean'] = df['remote_status'].map(REMOTE_MAP).fillna('unknown')
    df['perf_rating_clean'] = df['performance_rating'].map(PERF_MAP).fillna('unknown')
    df['country_clean'] = df['country'].map(COUNTRY_CANON).fillna(df['country'])

    # performance score: 20-100 scale -> /20 (coerce non-numeric to NaN)
    prs = pd.to_numeric(df['performance_review_score'], errors='coerce')
    df['perf_score_norm'] = np.where(prs > 5, prs / 20.0, prs)

    # dirty numerics -> NaN (coerce non-numeric first; parse text formats)
    REF_YEAR = 2026
    def _parse_age(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, (int, float)):
            return float(v)
        v = str(v).strip()
        m = re.match(r'^(?:b\.|born)\s*(\d{4})$', v)
        if m:
            return float(REF_YEAR - int(m.group(1)))
        m = re.match(r'^(\d+)\s*(?:yrs|years\s*old|y\.o\.|years)$', v)
        if m:
            return float(m.group(1))
        return np.nan
    def _parse_yoe(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, (int, float)):
            return float(v)
        v = str(v).strip()
        m = re.match(r'^(\d+)\s*months$', v)
        if m:
            return float(m.group(1)) / 12.0
        m = re.match(r'^(\d+)\s*(?:yrs|years)$', v)
        if m:
            return float(m.group(1))
        return np.nan
    age = pd.to_numeric(df['age'], errors='coerce')
    age = age.fillna(df['age'].apply(_parse_age))
    df['age_clean'] = age.where((age >= 18) & (age <= 75))
    yoe = pd.to_numeric(df['years_experience'], errors='coerce')
    yoe = yoe.fillna(df['years_experience'].apply(_parse_yoe))
    df['yoe_clean'] = yoe.where((yoe >= 0) & (yoe <= 50))
    ten = pd.to_numeric(df['tenure_years'], errors='coerce')
    df['tenure_clean'] = ten.where((ten >= 0) & (ten <= 50))
    df['import_batch'] = pd.to_numeric(df['import_batch'], errors='coerce')

    # job seniority
    jf = df['job_title'].apply(job_features)
    for k in list(jf[0].keys()) if len(jf) else []:
        df[k] = [r[k] for r in jf]

    # hire date parsing
    df['hire_dt'] = pd.to_datetime(df['hire_date'], errors='coerce', format='mixed')
    # try extra formats pandas may miss
    mask = df['hire_dt'].isna() & df['hire_date'].notna()
    if mask.any():
        alt = pd.to_datetime(df.loc[mask, 'hire_date'], errors='coerce', format='%Y%m%d')
        df.loc[mask, 'hire_dt'] = alt
    df['hire_year'] = df['hire_dt'].dt.year
    df['hire_month'] = df['hire_dt'].dt.month

    # notes
    df['notes_clean'] = df['notes'].fillna('none')
    return df, fx_rates
