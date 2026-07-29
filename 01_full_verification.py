# -*- coding: utf-8 -*-
"""
================================================================================
BDI FRAMEWORK - COMPLETE COMPUTATIONAL VERIFICATION SCRIPT
Paper: "Evaluating the Incorporation of BWM's Importance Weights into DEMATEL-ISM"
================================================================================
Single-file script. Requirements: numpy, scipy (both included in Anaconda).
Run:  python BDI_full_verification.py   (or run the whole cell in Jupyter)

Reproduces, in order:
  [1] Primary dataset (Ojha et al., 9 barriers): DI / BDI-M / BDI-F
      - Total relation matrices, D+R, D-R  (Table IV)
      - MEAN+SD and MMDE thresholds        (Table V)
      - ISM hierarchy levels and variance  (Table VII)
      - Evidence for the corrected DI-MMDE value (0.179, not 0.504)
  [2] Cross-domain validation (Chen 2021, 15 factors): Table IX
  [3] Monte Carlo convergence diagnostics: Table VIII
  [4] Fig. 2 regeneration statistics (MEAN+SD threshold distributions)
All stochastic procedures use fixed seeds => bit-for-bit reproducible.
================================================================================
"""
import numpy as np
from scipy.optimize import linprog

np.set_printoptions(suppress=True, precision=3)
PASS = lambda ok: "PASS" if ok else "FAIL  <<<"

# ==============================================================================
# SECTION 0. CORE ROUTINES (identical to the notebooks used for the paper)
# ==============================================================================

def total_relation(Z):
    """DEMATEL: normalize by max row sum, then T = X (I - X)^-1."""
    n = Z.shape[0]
    X = Z / Z.sum(axis=1).max()
    return X @ np.linalg.inv(np.eye(n) - X)

def mean_sd_threshold(T):
    """MEAN+SD over off-diagonal elements (paper protocol)."""
    off = T[~np.eye(T.shape[0], dtype=bool)]
    return off.mean() + off.std()

def mmde_threshold(T):
    """MMDE implementation used for the paper (value-distribution mean de-entropy).
    Scans every distinct off-diagonal value as candidate alpha; keeps argmax of
    (ln n - H)/n where H is the Shannon entropy of the magnitude distribution
    of the retained values.  This is the implementation that produced
    alpha = 0.007 (BDI-M) and alpha = 0.707 (BDI-F) in the manuscript."""
    v = T[~np.eye(T.shape[0], dtype=bool)]
    v = v[v > 1e-12]
    best_a, best = None, -np.inf
    for a in np.sort(np.unique(v)):
        s = v[v >= a]; n = len(s)
        if n <= 1:
            m = 0.0
        else:
            p = s / s.sum()
            H = -(p * np.log(p)).sum()
            m = (np.log(n) - H) / n
        if m > best:
            best, best_a = m, a
    return best_a

def mmde_scan_table(T, top=25):
    """Full candidate table (used to demonstrate the 0.504 transcription error)."""
    v = np.sort(np.unique(T[~np.eye(T.shape[0], dtype=bool)]))
    rows = []
    for a in v:
        s = v_all = T[~np.eye(T.shape[0], dtype=bool)]
        s = v_all[v_all >= a]; n = len(s)
        if n <= 1:
            m = 0.0
        else:
            p = s / s.sum(); H = -(p * np.log(p)).sum(); m = (np.log(n) - H) / n
        rows.append((a, n, m))
    return rows

def ism_levels(T, alpha):
    """Standard ISM level partition on reachability matrix (T >= alpha)."""
    n = T.shape[0]
    B = (T >= alpha).astype(int); np.fill_diagonal(B, 1)
    reach = [set(np.where(B[i] == 1)[0]) for i in range(n)]
    ante  = [set(np.where(B[:, i] == 1)[0]) for i in range(n)]
    remaining, levels, L = set(range(n)), {}, 1
    while remaining:
        cur = [i for i in remaining
               if reach[i] & remaining == (reach[i] & ante[i]) & remaining]
        if not cur:
            return None
        for i in cur: levels[i] = L
        remaining -= set(cur); L += 1
    x = np.array(list(levels.values()), float)
    return len(set(levels.values())), x.mean(), x.var(), levels

def bwm_linear(aB, aW, best, worst):
    """Linear BWM (Rezaei 2016) via scipy linprog. Returns (weights, xi*)."""
    n = len(aB)
    # variables: w_0..w_{n-1}, xi  -> minimize xi
    c = np.zeros(n + 1); c[-1] = 1.0
    A_ub, b_ub = [], []
    for j in range(n):
        r = np.zeros(n + 1); r[best] += 1; r[j] -= aB[j]; r[-1] = -1
        A_ub.append(r);  b_ub.append(0)            #  wB - aBj wj - xi <= 0
        A_ub.append(-r - np.eye(n+1)[-1]*0);       #  placeholder replaced below
    A_ub = []
    for j in range(n):
        r1 = np.zeros(n + 1); r1[best] = 1; r1[j] += -aB[j]; r1[-1] = -1
        r2 = -r1.copy(); r2[-1] = -1
        r3 = np.zeros(n + 1); r3[j] = 1; r3[worst] += -aW[j]; r3[-1] = -1
        r4 = -r3.copy(); r4[-1] = -1
        A_ub += [r1, r2, r3, r4]
    b_ub = [0.0] * len(A_ub)
    A_eq = [np.append(np.ones(n), 0.0)]; b_eq = [1.0]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=[(1e-6, None)]*n + [(0, None)], method='highs')
    return res.x[:n], res.x[-1]

# ==============================================================================
# SECTION 1. PRIMARY DATASET (Ojha et al., 9 barriers)
# ==============================================================================
print("=" * 78)
print("SECTION 1 - PRIMARY DATASET (9 barriers)")
print("=" * 78)

Z9 = np.array([
    [0.00,3.00,3.00,4.00,3.00,1.00,3.00,3.00,3.67],
    [3.00,0.00,3.00,4.00,3.00,2.00,2.67,4.00,3.00],
    [2.33,1.00,0.00,1.33,2.33,3.67,1.00,2.33,2.67],
    [1.00,2.67,1.67,0.00,4.00,3.00,1.00,2.00,2.00],
    [3.00,4.00,3.00,3.00,0.00,3.00,1.33,4.00,2.33],
    [3.00,2.67,3.00,1.33,1.33,0.00,2.67,2.67,2.67],
    [2.33,3.00,1.00,2.33,0.00,0.00,0.00,3.00,1.00],
    [2.33,2.67,2.33,1.67,1.00,1.00,1.33,0.00,3.33],
    [2.00,2.67,3.00,3.00,2.00,2.00,1.00,3.33,0.00]])
w9 = np.array([0.121,0.184,0.067,0.148,0.120,0.093,0.022,0.142,0.103])  # BWM weights (Table III)

# --- three configurations ---
T_DI  = total_relation(Z9)                       # unweighted baseline
T_BDM = total_relation(w9[:, None] * Z9)         # BDI-M: Z' = w_i * z_ij  (Tables IV/V/VII)
Z8    = np.delete(np.delete(Z9, 6, 0), 6, 1)     # BDI-F: remove weakest criterion B7
T_BDF = total_relation(Z8)

# --- 1a. D+R / D-R (Table IV) ---
print("\n[1a] Prominence & Relation (Table IV) - BDI-M row check")
D, R = T_BDM.sum(1), T_BDM.sum(0)
print("     BDI-M D+R:", np.round(D + R, 3))
print("     BDI-M D-R:", np.round(D - R, 3))
print("     Expected (Table IV): D+R = 2.176, 2.915, 1.587, ... ",
      PASS(abs((D+R)[0] - 2.176) < 0.002 and abs((D+R)[1] - 2.915) < 0.002))

# --- 1b. Thresholds (Table V) ---
print("\n[1b] Thresholds (Table V)")
rows = [("DI",    T_DI ), ("BDI-M", T_BDM), ("BDI-F", T_BDF)]
published = {"DI": (0.502, 0.179), "BDI-M": (0.177, 0.007), "BDI-F": (0.654, 0.707)}
for name, T in rows:
    a1, a2 = mean_sd_threshold(T), mmde_threshold(np.round(T, 3))
    e1, e2 = published[name]
    print(f"     {name:6s} MEAN+SD = {a1:.3f} (exp {e1})  {PASS(abs(a1-e1)<0.002)}"
          f"   MMDE = {a2:.3f} (exp {e2})  {PASS(abs(a2-e2)<0.002)}")

# --- 1c. Evidence for the 0.504 transcription error ---
print("\n[1c] DI-MMDE candidate scan: 0.504 is an interior row, NOT the optimum")
scan = mmde_scan_table(np.round(T_DI, 3))
best_row = max(scan, key=lambda r: r[2])
near = [r for r in scan if abs(r[0] - 0.504) < 0.0005]
print(f"     argmax of mean de-entropy: alpha = {best_row[0]:.3f} (n = {best_row[1]})"
      f"   -> corrected DI-MMDE threshold")
if near:
    a, nn, m = near[0]
    print(f"     row alpha = 0.504 exists in the scan (n = {nn}, MDE = {m:.6f})"
          f" but MDE < optimum ({best_row[2]:.6f})  -> transcription error confirmed")

# --- 1d. ISM hierarchies (Table VII) ---
print("\n[1d] ISM hierarchies (Table VII)   [T rounded to 3 dp, as in the notebooks]")
cfg = [("DI  (MEAN+SD)", np.round(T_DI, 3), 0.502, (4, 2.00, 1.56)),
       ("DI  (MMDE)   ", np.round(T_DI, 3), 0.179, (1, 1.00, 0.00)),
       ("BDIM(MMDE)   ", np.round(T_BDM,3), 0.007, (1, 1.00, 0.00)),
       ("BDIM(MEAN+SD)", np.round(T_BDM,3), 0.177, (5, 2.11, 2.09)),
       ("BDIF(MMDE)   ", np.round(T_BDF,3), 0.707, (3, 1.38, 0.48)),
       ("BDIF(MEAN+SD)", np.round(T_BDF,3), 0.654, (3, 1.50, 0.50))]
for name, T, a, (eL, eM, eV) in cfg:
    L, M, V, _ = ism_levels(T, a)
    ok = (L == eL and abs(M - eM) < 0.011 and abs(V - eV) < 0.011)
    print(f"     {name} a={a:<6} levels={L} mean={M:.2f} var={V:.2f}"
          f"   (exp {eL}/{eM}/{eV})  {PASS(ok)}")

# ==============================================================================
# SECTION 2. CROSS-DOMAIN VALIDATION (Chen 2021, 15 factors)  -> Table IX
# ==============================================================================
print("\n" + "=" * 78)
print("SECTION 2 - CROSS-DOMAIN VALIDATION (Chen 2021, 15 factors)  [Table IX]")
print("=" * 78)

Z15 = np.array([
    [0,1,2,0,2,1,2,3,3,0,1,0,0,0,0],
    [0,0,3,2,2,1,2,1,0,2,3,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,0,0,0,0,2,0],
    [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,2,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,3,0,0,3,0,0,0,0,0,0,0,0],
    [0,0,3,2,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,3,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,2,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
    [0,0,3,2,0,0,0,2,0,0,0,0,0,2,0],
    [0,0,0,3,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,3,0,0,0,0,0,0,0,0,0]], float)
n15 = 15

T2_DI = total_relation(Z15)
D2, R2 = T2_DI.sum(1), T2_DI.sum(0)
prom = D2 + R2

# surrogate BWM elicitation from prominence ranking (protocol in Section III.D)
best, worst = int(np.argmax(prom)), int(np.argmin(prom))
aB = 1 + np.round(8 * (prom[best] - prom) / (prom[best] - prom[worst])).astype(int)
aW = 1 + np.round(8 * (prom - prom[worst]) / (prom[best] - prom[worst])).astype(int)
w15, xi = bwm_linear(aB, aW, best, worst)
print(f"\n[2a] Surrogate BWM: Best = F{best+1}, Worst = F{worst+1},"
      f" xi* = {xi:.3f} (exp 0.037)  {PASS(abs(xi-0.037)<0.002)}")
print(f"     weights range: {w15.min():.3f} - {w15.max():.3f} (exp 0.015 - 0.172)")

T2_M = total_relation(w15[:, None] * Z15)   # formulation consistent with Tables IV/V/VII
a_di_msd, a_m_msd  = mean_sd_threshold(T2_DI), mean_sd_threshold(T2_M)
a_di_mm,  a_m_mm   = mmde_threshold(T2_DI),    mmde_threshold(T2_M)
print(f"\n[2b] Thresholds:  MEAN+SD  DI = {a_di_msd:.4f} (exp 0.0790),"
      f" BDI-M = {a_m_msd:.4f} (exp 0.0505)  "
      f"{PASS(abs(a_di_msd-0.0790)<0.001 and abs(a_m_msd-0.0505)<0.001)}")
print(f"                  reduction = {100*(a_m_msd-a_di_msd)/a_di_msd:.1f}% (exp -36.1%)")
print(f"                  MMDE      DI = {a_di_mm:.2e} (exp 7.3e-04),"
      f" BDI-M = {a_m_mm:.2e} (exp 1.6e-05 = min non-zero influence)")

L1 = ism_levels(T2_DI, a_di_msd); L2 = ism_levels(T2_M, a_m_msd)
print(f"\n[2c] ISM (MEAN+SD): DI {L1[0]} levels var {L1[2]:.2f} (exp 5/1.97) "
      f"{PASS(L1[0]==5 and abs(L1[2]-1.97)<0.011)};"
      f"  BDI-M {L2[0]} levels var {L2[2]:.2f} (exp 5/1.53) "
      f"{PASS(L2[0]==5 and abs(L2[2]-1.53)<0.011)}")
dens = (T2_DI > 1e-12).sum() / n15**2
print(f"     MMDE saturation pinned at matrix density = {100*dens:.1f}% (exp 21.3%)")

recl = [i+1 for i in range(n15) if np.sign((T2_M.sum(1)-T2_M.sum(0))[i])
        != np.sign((D2-R2)[i])]
print(f"\n[2d] Reclassified factors: F{recl} rate = {100*len(recl)/n15:.1f}%"
      f" (exp F8,F9,F10 / 20.0%)  {PASS(recl==[8,9,10])}")

# ==============================================================================
# SECTION 3. MONTE CARLO CONVERGENCE DIAGNOSTICS  -> Table VIII
# ==============================================================================
print("\n" + "=" * 78)
print("SECTION 3 - MONTE CARLO CONVERGENCE DIAGNOSTICS  [Table VIII]")
print("=" * 78)
print("(2,000-iteration budget adequacy; ~1-2 minutes)")

def mc_alpha_si(sigma, iters, seed):
    """BDI-M (Z' = w*Z, the formulation of Tables IV/V/VII): perturb weights, MEAN+SD."""
    rng = np.random.default_rng(seed)
    Tb = total_relation(w9[:, None] * Z9)
    ab = mean_sd_threshold(Tb); Rb = (Tb >= ab).astype(int)
    alphas, changes = [], 0
    for _ in range(iters):
        wp = np.maximum(w9 * (1 + rng.normal(0, sigma, 9)), 1e-6); wp /= wp.sum()
        Tk = total_relation(wp[:, None] * Z9)
        alphas.append(mean_sd_threshold(Tk))
        if not np.array_equal((Tk >= ab).astype(int), Rb): changes += 1
    return np.mean(alphas), np.std(alphas), 1 - changes / iters

print(f"\n{'N':>6} | {'95% MC error bound of SI':>25} | {'max CV of alpha':>16} |"
      f" {'max |d mean alpha| vs N=5000':>28}")
ref = {}
for sigma in [0.05, 0.10, 0.15, 0.20]:
    m, s, _ = mc_alpha_si(sigma, 5000, 999); ref[sigma] = m
for N in [500, 1000, 2000, 5000]:
    hw = 1.96 * np.sqrt(0.25 / N)                      # conservative bound (p = 0.5)
    cvs, dmeans = [], []
    for sigma in [0.05, 0.10, 0.15, 0.20]:
        ms = [mc_alpha_si(sigma, N, 1000 + r)[0] for r in range(10)]
        ss = [mc_alpha_si(sigma, N, 1000 + r)[1] for r in range(2)]   # CV from 2 reps (stable)
        cvs.append(np.mean(ss) / np.mean(ms) * 100)
        dmeans.append(abs(np.mean(ms) - ref[sigma]) / ref[sigma] * 100)
    print(f"{N:>6} | {'+/- '+format(hw,'.3f'):>25} | {max(cvs):>15.1f}% |"
          f" {max(dmeans):>27.2f}%")
print("Expected (Table VIII): error bound 0.044 / 0.031 / 0.022 / 0.014;"
      " CV stable across budgets (max ~29.5%); d mean alpha < 0.8%")

# ==============================================================================
# SECTION 4. FIG. 2 REGENERATION STATISTICS
# ==============================================================================
print("\n" + "=" * 78)
print("SECTION 4 - FIG. 2 STATISTICS (MEAN+SD threshold distributions, 2,000 iters)")
print("=" * 78)

def fig2_stats(model, sigma, iters=2000, seed=42):
    rng = np.random.default_rng(seed); out = []
    for _ in range(iters):
        if model == 'DI':
            Zp = np.maximum(Z9 * (1 + rng.normal(0, sigma, (9, 9))), 0)
            np.fill_diagonal(Zp, 0)
        else:
            wp = np.maximum(w9 * (1 + rng.normal(0, sigma, 9)), 1e-6); wp /= wp.sum()
            Zp = wp[:, None] * Z9
        out.append(mean_sd_threshold(total_relation(Zp)))
    a = np.array(out); return a.mean(), a.std() / a.mean() * 100

for sigma in [0.10, 0.15]:
    mD, cD = fig2_stats('DI', sigma); mM, cM = fig2_stats('BDIM', sigma)
    print(f"  sigma = {int(sigma*100)}%:  DI mu = {mD:.3f} CV = {cD:.1f}%   |"
          f"   BDI-M mu = {mM:.3f} CV = {cM:.1f}%")
print("\nDone. All sections executed.")
