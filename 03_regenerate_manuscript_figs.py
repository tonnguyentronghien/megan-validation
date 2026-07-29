# -*- coding: utf-8 -*-
"""Regenerate manuscript Fig. 2 and Fig. 3 (same code paths as the original
notebooks, fixed seeds). Run from the repository root:
    python scripts/03_regenerate_manuscript_figs.py
Outputs: figures/fig2_threshold_sensitivity.png, figures/fig3_stability_index.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

w = np.array([0.121, 0.184, 0.067, 0.148, 0.120, 0.093, 0.022, 0.142, 0.103])
Z = np.array([
    [0.00, 3.00, 3.00, 4.00, 3.00, 1.00, 3.00, 3.00, 3.67],
    [3.00, 0.00, 3.00, 4.00, 3.00, 2.00, 2.67, 4.00, 3.00],
    [2.33, 1.00, 0.00, 1.33, 2.33, 3.67, 1.00, 2.33, 2.67],
    [1.00, 2.67, 1.67, 0.00, 4.00, 3.00, 1.00, 2.00, 2.00],
    [3.00, 4.00, 3.00, 3.00, 0.00, 3.00, 1.33, 4.00, 2.33],
    [3.00, 2.67, 3.00, 1.33, 1.33, 0.00, 2.67, 2.67, 2.67],
    [2.33, 3.00, 1.00, 2.33, 0.00, 0.00, 0.00, 3.00, 1.00],
    [2.33, 2.67, 2.33, 1.67, 1.00, 1.00, 1.33, 0.00, 3.33],
    [2.00, 2.67, 3.00, 3.00, 2.00, 2.00, 1.00, 3.33, 0.00]])
n = 9
I = np.eye(n)

# ===========================================================================
# Fig. 2 — MEAN+SD threshold distributions under perturbation
# (identical to the notebook cell that produced fig2_new.png)
# ===========================================================================


def msd_threshold(Zm):
    X = Zm / Zm.sum(axis=1).max()
    T = X @ np.linalg.inv(I - X)
    off = T[~np.eye(n, dtype=bool)]
    return off.mean() + off.std()


def simulate(model, sigma, iters=2000):
    rng = np.random.default_rng(42)
    out = []
    for _ in range(iters):
        if model == "DI":
            Zp = np.maximum(Z * (1 + rng.normal(0, sigma, (n, n))), 0)
            np.fill_diagonal(Zp, 0)
            out.append(msd_threshold(Zp))
        else:
            wp = np.maximum(w * (1 + rng.normal(0, sigma, n)), 1e-6)
            wp /= wp.sum()
            out.append(msd_threshold(wp[:, None] * Z))
    return np.array(out)


fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), dpi=300)
for ax, sigma in zip(axes, [0.10, 0.15]):
    a_di = simulate("DI", sigma)
    a_bdm = simulate("BDIM", sigma)
    ax.hist(a_di, bins=40, alpha=0.65, color="#c44e52",
            label=f"DI ($\\mu$={a_di.mean():.3f}, "
                  f"CV={a_di.std()/a_di.mean()*100:.1f}%)")
    ax.hist(a_bdm, bins=40, alpha=0.65, color="#4c72b0",
            label=f"BDI-M ($\\mu$={a_bdm.mean():.3f}, "
                  f"CV={a_bdm.std()/a_bdm.mean()*100:.1f}%)")
    ax.axvline(a_di.mean(), color="#c44e52", ls="--", lw=1)
    ax.axvline(a_bdm.mean(), color="#4c72b0", ls="--", lw=1)
    ax.set_title(f"{int(sigma*100)}% Gaussian perturbation", fontsize=10)
    ax.set_xlabel(r"MEAN+SD threshold ($\alpha$)", fontsize=9)
    ax.set_ylabel("Frequency", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("figures/fig2_threshold_sensitivity.png", bbox_inches="tight")
plt.close()
print("saved figures/fig2_threshold_sensitivity.png")

# ===========================================================================
# Fig. 3 — Stability Index comparison, BDI-M vs BDI-F vs DI
# (identical protocol to the notebook cell: fixed MEAN+SD baseline threshold,
#  normalize-first formulation, 3,000 iterations per noise level)
# ===========================================================================
np.random.seed(42)
w9 = w.copy()
w8 = np.delete(w9, 6); w8 = w8 / w8.sum()
A8 = np.delete(np.delete(Z, 6, 0), 6, 1)
D9 = Z / np.max(Z.sum(axis=1))
D8 = A8 / np.max(A8.sum(axis=1))


def get_msd(T):
    off = T[~np.eye(T.shape[0], dtype=bool)]
    return off.mean() + off.std()


def run_si_sim(D, w_base, is_bdi, sigma, iterations=3000):
    m = D.shape[0]
    Im = np.eye(m)
    M_base = (np.diag(w_base) @ D) if is_bdi else D
    T_base = M_base @ np.linalg.inv(Im - M_base)
    alpha_base = get_msd(T_base)
    R_base = (T_base >= alpha_base).astype(int)
    changes = 0
    for _ in range(iterations):
        noise = np.random.normal(0, sigma, m)
        if is_bdi:
            wp = np.maximum(w_base * (1 + noise), 1e-6)
            wp /= wp.sum()
            Mk = np.diag(wp) @ D
        else:
            Mk = np.maximum(D * (1 + noise.reshape(-1, 1)), 0)
        if np.max(np.abs(np.linalg.eigvals(Mk))) >= 1:
            continue
        Tk = Mk @ np.linalg.inv(Im - Mk)
        if not np.array_equal((Tk >= alpha_base).astype(int), R_base):
            changes += 1
    return 1 - changes / iterations


noise_levels = [0.05, 0.10, 0.15, 0.20, 0.30]
rows = []
for s in noise_levels:
    rows.append([s * 100,
                 run_si_sim(D9, w9, True, s),
                 run_si_sim(D8, w8, True, s),
                 run_si_sim(D9, w9, False, s)])
    print(f"sigma={s:.2f}  BDI-M={rows[-1][1]:.4f}  "
          f"BDI-F={rows[-1][2]:.4f}  DI={rows[-1][3]:.4f}")

rows = np.array(rows)
plt.figure(figsize=(10, 6), dpi=300)
plt.plot(rows[:, 0], rows[:, 1], "o-", label="BDI-M (Proposed 9x9)",
         color="#4c72b0")
plt.plot(rows[:, 0], rows[:, 2], "s-", label="BDI-F (Reduced 8x8)",
         color="#55a868")
plt.plot(rows[:, 0], rows[:, 3], "x--", label="DI (Baseline)",
         color="#c44e52")
plt.title("Stability Index Comparison: BDI-M vs. BDI-F vs. DI",
          fontweight="bold")
plt.xlabel("Perturbation Level (%)")
plt.ylabel("Stability Index (SI)")
plt.legend(frameon=False)
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("figures/fig3_stability_index.png", bbox_inches="tight")
plt.close()
print("saved figures/fig3_stability_index.png")
