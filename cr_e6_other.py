import json, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def diagnostic(X, y, seed=42):
    idx = np.arange(len(X))
    tr, te = train_test_split(idx, test_size=.2, random_state=seed, stratify=y)
    fi, va = train_test_split(tr, test_size=.25, random_state=seed, stratify=y[tr])
    acc = lambda m, c: accuracy_score(y[te], m.fit(X[tr][:, c], y[tr]).predict(X[te][:, c])) * 100
    allc = list(range(X.shape[1]))
    f = acc(RandomForestClassifier(100, random_state=seed, n_jobs=1), allc)
    d4 = acc(DecisionTreeClassifier(max_depth=4, random_state=seed), allc)
    val = {i: accuracy_score(y[va], DecisionTreeClassifier(max_depth=4, random_state=seed)
                             .fit(X[fi][:, [i]], y[fi]).predict(X[va][:, [i]])) for i in allc}
    b = max(val, key=val.get)
    one = acc(DecisionTreeClassifier(max_depth=4, random_state=seed), [b])
    return dict(forest=f, d4=d4, one=one, margin=f - one, best=int(b))


out = {}
m = pd.read_excel('PMU_data.xlsx')
m = m.drop(columns=[c for c in m.columns if str(c).startswith('Unnamed')])
ycol = [c for c in m.columns if 'abel' in str(c)][0]
Xm = m.drop(columns=[ycol])
out['Mendeley'] = diagnostic(Xm.values.astype(float), m[ycol].astype(str).values)
out['Mendeley']['best'] = str(Xm.columns[out['Mendeley']['best']])
out['Mendeley']['rows'] = len(m)

kg = pd.read_csv('classData.csv')
lab = kg[['G', 'C', 'B', 'A']].astype(str).agg(''.join, axis=1)
names = {'0000': 'NF', '1001': 'LG', '0110': 'LL', '1011': 'LLG', '0111': 'LLL', '1111': 'LLLG'}
lab = lab.map(names).fillna(lab).values
K = ['Ia', 'Ib', 'Ic', 'Va', 'Vb', 'Vc']
Xk = kg[K].values.astype(float)
out['Kaggle 6'] = diagnostic(Xk, lab)
lab5 = np.where(np.isin(lab, ['LLL', 'LLLG']), 'LLL/G', lab)
out['Kaggle 5'] = diagnostic(Xk, lab5)
for k in ('Kaggle 6', 'Kaggle 5'):
    out[k]['best'] = K[out[k]['best']]
print(pd.Series(lab).value_counts().to_dict())
for k, v in out.items():
    print(f"{k:10} forest {v['forest']:6.2f}  d4 {v['d4']:6.2f}  1f {v['one']:6.2f}  margin {v['margin']:6.2f}  best {v['best']}")
json.dump(out, open('results/cr_e6.json', 'w'), indent=1)
