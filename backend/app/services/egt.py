"""
Tripartite Evolutionary Game Theory (EGT) model - Phase 1 behavioural modelling.

Players and strategies
    Worker (W)      : Safe behaviour (prob. x)       | Unsafe behaviour (1-x)
    Supervisor (S)  : Strict supervision (prob. y)   | Lax supervision (1-y)
    Management (M)  : Invest in safety (prob. z)     | Not invest (1-z)
                      (training, incentives, vision-based monitoring system)

Payoffs are first defined for every *pure* strategy profile (w, s, m) in
{0,1}^3 (see `pure_payoffs`).  Expected payoffs for mixed populations and the
replicator dynamics are then derived mechanically from that table, so the
simulation is exactly the mathematical model (no hand-drawn curves):

    dx/dt = x(1-x) [E(U_W | Safe)     - E(U_W | Unsafe)]
    dy/dt = y(1-y) [E(U_S | Strict)   - E(U_S | Lax)]
    dz/dt = z(1-z) [E(U_M | Invest)   - E(U_M | Not invest)]

Link with Phase 2 (computer vision):
    When management invests, the vision system raises the probability that an
    unsafe act is detected and corrected:
        alpha_I = 1 - (1 - alpha0)(1 - phi * r)     (strict supervision)
        beta_I  = 1 - (1 - beta0) (1 - phi * r)     (lax supervision)
    r   = recall of the detector measured in the validation step
    phi = camera coverage of the work area (0-1)
    x0  = 1 - Unsafe Behaviour Rate observed by the vision system

Stability of the 8 pure-strategy equilibria is assessed with the Jacobian
(Friedman, 1991 method): at a vertex the Jacobian is diagonal, eigenvalues
    lambda_i = (1 - 2 s_i) * F_i(vertex);  ESS <=> all lambda_i < 0.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, asdict, field

import numpy as np


@dataclass
class EGTParams:
    # ---- worker
    Cw: float = 2.0     # extra cost of safe behaviour (time, effort, discomfort)
    Bw: float = 2.0     # private benefit of unsafe behaviour (time saved, output bonus)
    Rw: float = 1.0     # incentive paid by management for safe behaviour (if investing)
    Fw: float = 4.0     # penalty to worker when an unsafe act is detected
    Lw: float = 10.0    # worker's loss if an accident occurs
    # ---- supervisor
    Cs: float = 2.0     # cost of strict supervision
    Rs: float = 1.0     # reward from management for strict supervision (if investing)
    Fs: float = 3.0     # penalty from management audit on lax supervisor when worker is unsafe
    Ls: float = 8.0     # supervisor's accountability loss if an accident occurs
    # ---- management
    Cm: float = 3.0     # cost of safety investment (training, incentive fund, CV system)
    G: float = 4.0      # reputation / regulatory / productivity benefit of investing
    Lm: float = 30.0    # management loss if an accident occurs
    Fg: float = 10.0    # government penalty after an accident without safety investment
    eta: float = 0.5    # fraction of accident probability removed by investment (engineering controls)
    # ---- accident & detection
    pi: float = 0.2     # probability that an uncorrected unsafe act leads to an accident
    alpha0: float = 0.6 # detection probability under strict human supervision
    beta0: float = 0.1  # detection probability under lax supervision
    phi: float = 0.0    # camera coverage of the work area (0..1)  [Phase-2 link]
    r: float = 0.0      # recall of the CV detector (0..1)          [Phase-2 link]

    def detection(self, strict: int, invest: int) -> float:
        a0 = self.alpha0 if strict else self.beta0
        if invest:
            return 1.0 - (1.0 - a0) * (1.0 - self.phi * self.r)
        return a0


PARAM_DOC = {
    "Cw": "Worker: extra cost of safe behaviour", "Bw": "Worker: benefit of unsafe behaviour",
    "Rw": "Worker: incentive for safe behaviour", "Fw": "Worker: penalty if caught",
    "Lw": "Worker: accident loss", "Cs": "Supervisor: cost of strict supervision",
    "Rs": "Supervisor: reward for strict supervision", "Fs": "Supervisor: audit penalty for lax supervision",
    "Ls": "Supervisor: accountability loss", "Cm": "Management: safety investment cost",
    "G": "Management: benefit of investing", "Lm": "Management: accident loss",
    "Fg": "Management: government penalty", "eta": "Accident reduction by investment",
    "pi": "Accident probability of uncorrected unsafe act", "alpha0": "Detection prob. (strict)",
    "beta0": "Detection prob. (lax)", "phi": "CV camera coverage", "r": "CV detector recall",
}


# ------------------------------------------------------------------ payoffs
def pure_payoffs(p: EGTParams, w: int, s: int, m: int) -> tuple[float, float, float]:
    """Payoffs (U_W, U_S, U_M) for pure profile w,s,m in {0,1} (1 = Safe/Strict/Invest)."""
    D = p.detection(s, m)
    A = 0.0 if w else p.pi * (1.0 - D) * (1.0 - p.eta * m)      # accident probability

    if w:
        uw = -p.Cw + m * p.Rw
    else:
        uw = p.Bw - D * p.Fw - A * p.Lw

    if s:
        us = -p.Cs + m * p.Rs - A * p.Ls
    else:
        us = -A * p.Ls - m * (1 - w) * p.Fs

    if m:
        um = -p.Cm + p.G - w * p.Rw - s * p.Rs + (1 - s) * (1 - w) * p.Fs - A * p.Lm
    else:
        um = -A * p.Lm - A * p.Fg
    return uw, us, um


def payoff_table(p: EGTParams) -> list[dict]:
    rows = []
    for w, s, m in itertools.product((1, 0), repeat=3):
        uw, us, um = pure_payoffs(p, w, s, m)
        rows.append({"worker": "Safe" if w else "Unsafe", "supervisor": "Strict" if s else "Lax",
                     "management": "Invest" if m else "Not invest",
                     "U_W": round(uw, 3), "U_S": round(us, 3), "U_M": round(um, 3)})
    return rows


def payoff_tensor(p: EGTParams) -> np.ndarray:
    """T[w, s, m, player] for pure profiles (index 1 = Safe/Strict/Invest)."""
    T = np.zeros((2, 2, 2, 3))
    for w, s, m in itertools.product((0, 1), repeat=3):
        T[w, s, m] = pure_payoffs(p, w, s, m)
    return T


def _gaps_from_tensor(T: np.ndarray, x, y, z):
    """Expected payoff difference (strategy 1 - strategy 0) for each player."""
    py, pz, px = np.array([1 - y, y]), np.array([1 - z, z]), np.array([1 - x, x])
    Fw = np.einsum("j,k,jk->", py, pz, T[1, :, :, 0] - T[0, :, :, 0])
    Fs = np.einsum("i,k,ik->", px, pz, T[:, 1, :, 1] - T[:, 0, :, 1])
    Fm = np.einsum("i,j,ij->", px, py, T[:, :, 1, 2] - T[:, :, 0, 2])
    return float(Fw), float(Fs), float(Fm)


def fitness_gaps(p: EGTParams, x, y, z):
    """F_W, F_S, F_M = expected payoff of strategy 1 minus strategy 0."""
    return _gaps_from_tensor(payoff_tensor(p), x, y, z)


def replicator(p: EGTParams, state, T=None):
    x, y, z = state
    Fx, Fy, Fz = _gaps_from_tensor(payoff_tensor(p) if T is None else T, x, y, z)
    return np.array([x * (1 - x) * Fx, y * (1 - y) * Fy, z * (1 - z) * Fz])


# ------------------------------------------------------------------ simulation
def simulate(p: EGTParams, x0=0.5, y0=0.5, z0=0.5, t_max=50.0, dt=0.05):
    """Classical 4th-order Runge-Kutta integration of the replicator system."""
    n = int(t_max / dt) + 1
    traj = np.zeros((n, 3))
    traj[0] = [x0, y0, z0]
    T = payoff_tensor(p)
    for k in range(1, n):
        s = traj[k - 1]
        k1 = replicator(p, s, T)
        k2 = replicator(p, s + dt / 2 * k1, T)
        k3 = replicator(p, s + dt / 2 * k2, T)
        k4 = replicator(p, s + dt * k3, T)
        traj[k] = np.clip(s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), 0.0, 1.0)
    t = np.linspace(0, t_max, n)
    return t, traj


def equilibrium_stability(p: EGTParams) -> list[dict]:
    out = []
    for v in itertools.product((0, 1), repeat=3):
        F = fitness_gaps(p, *v)
        eig = [(1 - 2 * v[i]) * F[i] for i in range(3)]
        if all(e < 0 for e in eig):
            kind = "ESS (stable)"
        elif any(e > 0 for e in eig):
            kind = "Unstable" if all(e > 0 for e in eig) else "Saddle point"
        else:
            kind = "Undetermined (zero eigenvalue)"
        out.append({"point": f"E({v[0]},{v[1]},{v[2]})",
                    "meaning": f"{'Safe' if v[0] else 'Unsafe'} / {'Strict' if v[1] else 'Lax'} / "
                               f"{'Invest' if v[2] else 'Not invest'}",
                    "eigenvalues": [round(e, 3) for e in eig], "stability": kind})
    return out


def sensitivity(base: EGTParams, param: str, values, x0=0.5, y0=0.5, z0=0.5, t_max=50.0):
    res = []
    for v in values:
        p = EGTParams(**{**asdict(base), param: float(v)})
        _, tr = simulate(p, x0, y0, z0, t_max=t_max, dt=0.1)
        res.append({"value": float(v), "x": float(tr[-1, 0]), "y": float(tr[-1, 1]), "z": float(tr[-1, 2])})
    return res


SCENARIOS = {
    "baseline": {"label": "Baseline - human supervision only", "params": {}},
    "cv_assisted": {"label": "CV-assisted supervision (phi=0.8, r from validation)",
                    "params": {"phi": 0.8, "r": 0.8}},
    "high_penalty": {"label": "Stronger worker penalty", "params": {"Fw": 8.0}},
    "cv_low_recall": {"label": "CV with weak detector (phi=0.8, r=0.4)", "params": {"phi": 0.8, "r": 0.4}},
}


def run(params: dict | None = None, x0=0.5, y0=0.5, z0=0.5, t_max=50.0,
        sens_param: str | None = "Fw", sens_range=(0, 12, 13)) -> dict:
    p = EGTParams(**{k: float(v) for k, v in (params or {}).items() if k in EGTParams.__dataclass_fields__})
    t, tr = simulate(p, x0, y0, z0, t_max)
    step = max(1, len(t) // 300)          # down-sample for the web chart
    out = {
        "params": asdict(p), "param_doc": PARAM_DOC,
        "initial": {"x0": x0, "y0": y0, "z0": z0},
        "t": [round(v, 3) for v in t[::step]],
        "x": [round(v, 4) for v in tr[::step, 0]],
        "y": [round(v, 4) for v in tr[::step, 1]],
        "z": [round(v, 4) for v in tr[::step, 2]],
        "final": {"x": float(tr[-1, 0]), "y": float(tr[-1, 1]), "z": float(tr[-1, 2])},
        # mean of the second half of the run - robust summary when the system oscillates
        "time_average": {k: float(tr[len(tr) // 2:, i].mean()) for i, k in enumerate("xyz")},
        "equilibria": equilibrium_stability(p),
        "payoff_table": payoff_table(p),
        "detection_probabilities": {
            "strict_no_invest": p.detection(1, 0), "lax_no_invest": p.detection(0, 0),
            "strict_invest": round(p.detection(1, 1), 4), "lax_invest": round(p.detection(0, 1), 4)},
    }
    if sens_param:
        lo, hi, n = sens_range
        out["sensitivity"] = {"param": sens_param,
                              "results": sensitivity(p, sens_param, np.linspace(lo, hi, int(n)), x0, y0, z0, t_max)}
    return out


def calibrate_from_system(unsafe_behaviour_rate: float | None, recall: float | None, coverage: float = 0.8):
    """Phase-2 -> Phase-1 bridge: initial worker state and CV detection parameters."""
    x0 = 0.5 if unsafe_behaviour_rate is None else float(np.clip(1 - unsafe_behaviour_rate, 0.01, 0.99))
    return {"x0": round(x0, 3), "phi": coverage, "r": round(recall if recall is not None else 0.0, 3)}


__all__ = ["EGTParams", "run", "simulate", "equilibrium_stability", "payoff_table",
           "calibrate_from_system", "SCENARIOS", "field"]
