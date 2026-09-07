import csv
import sys
from collections import defaultdict
import numpy as np

for path in sys.argv[1:]:
    agg = defaultdict(list)
    for r in csv.DictReader(open(path)):
        agg[(r["method"], r.get("split", ""), r.get("sigma", ""))].append(
            (float(r["acc"]), float(r["ari"]))
        )
    print("==", path)
    for kk in sorted(agg):
        a = np.array(agg[kk])
        se = a[:, 1].std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
        print(
            f"   {kk[0]:16s} {kk[1]:6s} sig={kk[2]:8s} "
            f"ACC={a[:, 0].mean():.4f} ARI={a[:, 1].mean():.4f} +-{se:.4f} n={len(a)}"
        )
