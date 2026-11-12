"""
Thesis-quality EGT figures (PNG, 300 dpi) + CSV tables.

Usage:  python scripts/egt_figures.py [--r 0.82 --phi 0.8 --x0 0.6]
Outputs: docs/figures/egt_*.png, docs/figures/egt_equilibria.csv
"""
import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app.services import egt  # noqa: E402

FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 10})


def main(r, phi, x0):
    scen = {k: dict(v) for k, v in egt.SCENARIOS.items()}
    scen["cv_assisted"]["params"] = {"phi": phi, "r": r}
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharex=True, sharey=True)
    for ax, (k, sc) in zip(axes.flat, scen.items()):
        p = egt.EGTParams(**sc["params"])
        t, tr = egt.simulate(p, x0, 0.5, 0.5)
        for i, (lab, ls) in enumerate([("x (worker safe)", "-"), ("y (supervisor strict)", "--"),
                                       ("z (management invests)", ":")]):
            ax.plot(t, tr[:, i], ls, color="black", label=lab)
        ax.set_title(sc["label"], fontsize=9)
        ax.set_ylim(-0.02, 1.02)
    axes[1, 0].set_xlabel("time")
    axes[1, 1].set_xlabel("time")
    axes[0, 0].set_ylabel("proportion")
    axes[1, 0].set_ylabel("proportion")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "egt_scenarios.png", dpi=300)

    # initial-condition robustness (3-D phase portrait)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(projection="3d")
    p = egt.EGTParams(phi=phi, r=r)
    rng = np.random.default_rng(1)
    for _ in range(14):
        s0 = rng.uniform(0.1, 0.9, 3)
        _, tr = egt.simulate(p, *s0)
        ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color="black", lw=0.7)
        ax.scatter(*s0, color="gray", s=8)
    ax.set_xlabel("x worker")
    ax.set_ylabel("y supervisor")
    ax.set_zlabel("z management")
    fig.tight_layout()
    fig.savefig(FIG / "egt_phase_portrait.png", dpi=300)

    # sensitivity of x to recall r and penalty Fw
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, (par, vals, base) in zip(axes, [("r", np.linspace(0, 1, 21), egt.EGTParams(phi=phi)),
                                            ("Fw", np.linspace(0, 12, 25), egt.EGTParams())]):
        res = egt.sensitivity(base, par, vals, x0, 0.5, 0.5)
        for key, ls in (("x", "-"), ("y", "--"), ("z", ":")):
            ax.plot(vals, [q[key] for q in res], ls, color="black", label=key)
        ax.set_xlabel(par)
        ax.set_ylabel("state at t=50")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "egt_sensitivity.png", dpi=300)

    rows = []
    for k, sc in scen.items():
        for e in egt.equilibrium_stability(egt.EGTParams(**sc["params"])):
            rows.append({"scenario": k, **e})
    pd.DataFrame(rows).to_csv(FIG / "egt_equilibria.csv", index=False)
    pd.DataFrame([asdict(egt.EGTParams())]).T.rename(columns={0: "value"}).to_csv(FIG / "egt_parameters.csv")
    print("figures written to", FIG)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=float, default=0.8, help="detector recall from validation")
    ap.add_argument("--phi", type=float, default=0.8, help="camera coverage")
    ap.add_argument("--x0", type=float, default=0.5, help="1 - unsafe behaviour rate")
    a = ap.parse_args()
    main(a.r, a.phi, a.x0)
