"""[25]-rule MLP baseline on ROC3: tanh hidden, sigmoid out, layers scaled by components.
ROC3 = 1 APUF evaluated twice -> l=3-ish; run (16,32,16) and (32,64,32), N=100k, raw C."""
import time, numpy as np
from pypuf.simulation import ArbiterPUF
from sklearn.neural_network import MLPClassifier
src = open('experiments/encoding-induced-false-security-pufs-roc3.py').read()
ns = {}
exec(src.split('rows = []')[0], ns)
for inst in [0, 1]:
    puf = ArbiterPUF(n=64, seed=400+inst, noisiness=0.0)
    Ctr, ytr, _ = ns['gen'](puf, 100000, np.random.default_rng(2000+inst))
    Cte, yte, _ = ns['gen'](puf, 10000, np.random.default_rng(3000+inst))
    for hl in [(16, 32, 16), (32, 64, 32)]:
        m = MLPClassifier(hidden_layer_sizes=hl, activation='tanh', max_iter=100, random_state=0)
        m.fit(Ctr.astype(np.float32), ytr)
        print('inst', inst, hl, round(float(m.score(Cte.astype(np.float32), yte)), 4), flush=True)
print('done', round(time.time() - time.time(), 1) if False else 'done')
