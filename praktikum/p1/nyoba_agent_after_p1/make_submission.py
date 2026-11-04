"""Stage 3: blend, bias-tune, predict, write submission."""
import numpy as np, pandas as pd
from sklearn.metrics import f1_score

zl = np.load('stage_lgb.npz')
zx = np.load('stage_xgb.npz')
oof_lgb, test_lgb = zl['oof_lgb'], zl['test_lgb']
oof_xgb, test_xgb = zx['oof_xgb'], zx['test_xgb']

tr = pd.read_csv('../data/train.csv')
labels = pd.Categorical(tr['pay_band']).codes
bands = pd.Categorical(tr['pay_band']).categories


def mf(p):
    return f1_score(labels, p.argmax(1), average='macro')

print('LGB OOF:', round(mf(oof_lgb), 5))
print('XGB OOF:', round(mf(oof_xgb), 5))

best = (None, -1)
for a in np.arange(0.6, 1.01, 0.05):
    s = mf(a * oof_lgb + (1 - a) * oof_xgb)
    print(f'blend lgb={a:.2f}: {s:.5f}')
    if s > best[1]:
        best = (a, s)
alpha = best[0]
oof = alpha * oof_lgb + (1 - alpha) * oof_xgb
test_p = alpha * test_lgb + (1 - alpha) * test_xgb
print('best alpha:', round(alpha, 2), 'OOF:', round(best[1], 5))

# mild bias tuning (restricted deltas to avoid OOF overfit)
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
print('bias:', np.round(bias, 4))
print('OOF with bias:', round(_mf_b(bias), 5))

pred = (test_p * bias).argmax(1)
sub = pd.read_csv('../data/sample_submission.csv')
sub['pay_band'] = [bands[p] for p in pred]
sub.to_csv('submission.csv', index=False)
print(sub['pay_band'].value_counts().to_string())
print('wrote submission.csv')
