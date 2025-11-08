# app.py
# SmartSurvey Synthetic Data Lab (Bangladesh Context)
# Generalized PLS-SEM synthetic dataset generator (SmartPLS-ready)
# Author scaffold: ChatGPT for Professor Mahbub
# ---------------------------------------------------------------

import io
import json
import numpy as np
import pandas as pd
import streamlit as st
from itertools import combinations
from scipy.stats import zscore, pearsonr
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor

st.set_page_config(page_title="SmartSurvey Synthetic Data Lab", page_icon="🧪", layout="wide")


# ----------------------------- Utility Functions -----------------------------

def set_seed(seed=2025):
    np.random.seed(seed)

def cronbach_alpha(df):
    if df.shape[1] < 2:
        return np.nan
    corr = df.corr().values
    if np.isnan(corr).any():
        return np.nan
    k = df.shape[1]
    avg_corr = (np.sum(corr) - k) / (k * (k - 1))
    return (k * avg_corr) / (1 + (k - 1) * avg_corr)

def composite_reliability(loadings, error_vars):
    s = np.sum(loadings)
    return (s*s) / ((s*s) + np.sum(error_vars))

def ave(loadings):
    return float(np.mean(np.array(loadings)**2))

def htmt(blocks: dict, data: pd.DataFrame):
    # Heterotrait–Monotrait ratio
    def _avg_hetero(c1_cols, c2_cols):
        vals = []
        for i in c1_cols:
            for j in c2_cols:
                r = data[i].corr(data[j])
                if pd.notna(r):
                    vals.append(abs(r))
        return np.mean(vals) if vals else np.nan

    def _avg_mono(cols):
        vals = []
        for i, j in combinations(cols, 2):
            r = data[i].corr(data[j])
            if pd.notna(r):
                vals.append(abs(r))
        return np.mean(vals) if vals else np.nan

    cons = list(blocks.keys())
    mat = pd.DataFrame(index=cons, columns=cons, dtype=float)
    for a, b in combinations(cons, 2):
        het = _avg_hetero(blocks[a], blocks[b])
        mono_a = _avg_mono(blocks[a])
        mono_b = _avg_mono(blocks[b])
        denom = np.sqrt(mono_a * mono_b) if (mono_a and mono_b) else np.nan
        mat.loc[a, b] = het/denom if (denom and denom > 0) else np.nan
        mat.loc[b, a] = mat.loc[a, b]
    np.fill_diagonal(mat.values, 1.0)
    return mat

def vif_dataframe(X: pd.DataFrame):
    if X.shape[1] == 0:
        return pd.DataFrame(columns=["Variable", "VIF"])
    X_ = X.copy()
    X_ = X_.assign(const=1.0)
    cols = [c for c in X_.columns if c != "const"]
    out = []
    for i, col in enumerate(cols):
        out.append({"Variable": col, "VIF": variance_inflation_factor(X_.values, i)})
    return pd.DataFrame(out)

def likertize_array(x, scale=5, noise=0.45, rng=None):
    rng = rng or np.random.default_rng(2025)
    vals = x + rng.normal(0, noise, size=len(x))
    if scale == 5:
        vals = np.clip(np.round(vals), 1, 5)
    else:
        # map approximately to 1–7
        vals = np.clip(np.round((vals - vals.min()) / (vals.max()-vals.min()+1e-9) * 6 + 1), 1, 7)
    return vals.astype(int)

def random_correlation_matrix(k, base=0.25, jitter=0.10, rng=None):
    """
    Construct a positive semi-definite correlation-like matrix with moderate correlations,
    biased to behavioral survey patterns typical in Bangladesh context.
    """
    rng = rng or np.random.default_rng(2025)
    A = rng.normal(0, 1, (k, k))
    S = np.dot(A, A.T)
    D = np.diag(1.0 / np.sqrt(np.diag(S)))
    C = D @ S @ D
    # blend with a target average correlation
    target = np.full((k, k), base)
    np.fill_diagonal(target, 1.0)
    C = 0.6 * C + 0.4 * target
    # final tweak
    C = C + rng.normal(0, jitter, C.shape)
    C = (C + C.T) / 2
    np.fill_diagonal(C, 1.0)
    # ensure PSD by projecting to nearest PSD via eigen clipping
    vals, vecs = np.linalg.eigh(C)
    vals[vals < 1e-6] = 1e-6
    C_psd = (vecs @ np.diag(vals) @ vecs.T)
    # rescale to corr
    d = np.sqrt(np.diag(C_psd))
    C_psd = C_psd / np.outer(d, d)
    np.fill_diagonal(C_psd, 1.0)
    return C_psd

def generate_latents_general(n, constructs, roles, paths, r2_targets, moderation_terms, dist_cfg, rng=None):
    """
    constructs: list of construct names
    roles: dict {construct: role in {"IV","DV","Mediator","Moderator","Control"}}
    paths: list of tuples (src, dst, beta) for linear relations (no interactions)
    r2_targets: dict {endogenous: target_R2}
    moderation_terms: list of tuples (iv, moderator, dst, beta_int)
    dist_cfg: dict for distribution choices
    Returns dict of latent arrays (standardized), incl. automatically created interaction latents.
    """
    rng = rng or np.random.default_rng(2025)
    k = len(constructs)

    # Step 1: exogenous seeds ~ correlated normals
    exogs = [c for c in constructs if all(dst != c for (_, dst, _) in paths) and
             all(dst != c for (_, _, dst, _) in moderation_terms)]
    # include pure Moderators and Controls as exogenous
    exogs += [c for c in constructs if roles.get(c) in ["Moderator", "Control"] and c not in exogs]
    exogs = sorted(list(dict.fromkeys(exogs)))  # unique, stable order

    corr_exog = random_correlation_matrix(len(exogs),
                                          base=dist_cfg.get("avg_corr", 0.25),
                                          jitter=dist_cfg.get("corr_jitter", 0.08),
                                          rng=rng)
    Z = rng.multivariate_normal(mean=np.zeros(len(exogs)), cov=corr_exog, size=n)
    latents = {exogs[i]: Z[:, i] for i in range(len(exogs))}

    # Step 2: recursively build endogenous constructs
    # do a simple topological loop: iterate until all have values
    remaining = [c for c in constructs if c not in latents]
    guard = 0
    while remaining and guard < 10*len(constructs):
        guard += 1
        assigned = []
        for c in remaining:
            preds = [(src, b) for (src, dst, b) in paths if dst == c]
            mods  = [(iv, m, b) for (iv, m, dst2, b) in moderation_terms if dst2 == c]
            ready = all((p in latents) for (p, _) in preds) and all((iv in latents and m in latents) for (iv, m, _) in mods)
            if not ready:
                continue
            # linear predictor
            pred_val = 0.0
            if preds:
                pred_val += sum(b * latents[src] for (src, b) in preds)
            if mods:
                for (iv, m, bint) in mods:
                    pred_val += bint * (latents[iv] * latents[m])
                latents.update({f"{iv}x{m}": latents[iv] * latents[m] for (iv, m, _) in mods})
            # noise calibrated to hit target R²
            var_signal = np.var(pred_val)
            r2 = r2_targets.get(c, 0.45)
            var_e = (var_signal * (1 - r2)) / max(r2, 1e-6) if var_signal > 1e-12 else 1.0
            e = rng.normal(0, np.sqrt(max(var_e, 1e-6)), size=n)
            latents[c] = pred_val + e
            assigned.append(c)
        remaining = [c for c in remaining if c not in assigned]

    # Step 3: optional distributional shaping (Bangladesh survey tendencies)
    if dist_cfg.get("skew_controls", False):
        # Weakly skew some controls if present
        if "AGE" in latents: latents["AGE"] = zscore(np.maximum(latents["AGE"], -1) + 0.3)
        if "INCOME" in latents: latents["INCOME"] = zscore(latents["INCOME"] + rng.gamma(1.5, 0.3, n) - 0.4)

    # Standardize all latents
    for k in list(latents.keys()):
        latents[k] = zscore(latents[k])

    # Return
    return latents


def generate_items(latents, measurement_spec, likert_scale=5, load_min=0.72, load_max=0.90, rng=None):
    rng = rng or np.random.default_rng(2025)
    items = {}
    load_record = {}
    for cons, spec in measurement_spec.items():
        k = spec.get("k", 4)
        lmin = spec.get("load_min", load_min)
        lmax = spec.get("load_max", load_max)
        lam = rng.uniform(lmin, lmax, size=k)
        load_record[cons] = lam
        lv = latents.get(cons, zscore(rng.normal(size=len(next(iter(latents.values()))))))
        for j in range(k):
            noise_sd = np.sqrt(max(1 - lam[j]**2, 1e-6))
            x = lam[j] * lv + rng.normal(0, noise_sd, size=len(lv))
            if likert_scale in [5, 7]:
                x = likertize_array(x, scale=likert_scale, noise=0.0, rng=rng)
            items[f"{cons}{j+1}"] = x
    df = pd.DataFrame(items)
    return df, load_record


def structural_preview(latents, paths, moderation_terms):
    lv = pd.DataFrame({k: zscore(v) for k, v in latents.items()})
    results = []
    # collect destinations
    dsts = set([dst for (_, dst, _) in paths] + [dst for (_, _, dst, _) in moderation_terms])
    for dst in sorted(dsts):
        preds = [src for (src, dst2, _) in paths if dst2 == dst]
        betas = [b for (_, dst2, b) in paths if dst2 == dst]
        mods  = [(iv, m, b) for (iv, m, dst2, b) in moderation_terms if dst2 == dst]
        Xcols = preds.copy()
        for (iv, m, b) in mods:
            coln = f"{iv}x{m}"
            if coln not in lv.columns:
                lv[coln] = lv[iv] * lv[m]
            Xcols += [iv, m, coln]  # include main effects + interaction
        Xcols = list(dict.fromkeys(Xcols))
        if dst not in lv.columns or len(Xcols) == 0:
            continue
        X = lv[Xcols].values
        y = lv[dst].values
        lm = LinearRegression().fit(X, y)
        # rough t via corr for single predictor; here we just report OLS betas
        for name, coef in zip(Xcols, lm.coef_):
            rel = f"{name} → {dst}" if "x" not in name else f"{name} → {dst} (interaction)"
            results.append({"Relation": rel, "β (OLS preview)": float(coef)})
    return pd.DataFrame(results)


# ----------------------------- Sidebar: Configuration -----------------------------

st.sidebar.title("Configuration")

seed = st.sidebar.number_input("Random seed", min_value=1, max_value=10_000_000, value=2025, step=1)
set_seed(seed)

n = st.sidebar.number_input("Sample size (n)", min_value=100, max_value=20000, value=850, step=50)

with st.sidebar.expander("Measurement settings"):
    likert_scale = st.radio("Likert scale", [5, 7], index=0, horizontal=True)
    default_items = st.slider("Default items per construct", 3, 7, 4)
    load_min = st.slider("Loading min (λ)", 0.60, 0.95, 0.72, 0.01)
    load_max = st.slider("Loading max (λ)", 0.60, 0.98, 0.90, 0.01)

with st.sidebar.expander("Correlation & distribution"):
    avg_corr = st.slider("Average exogenous correlation", 0.05, 0.50, 0.25, 0.01)
    corr_jitter = st.slider("Correlation jitter", 0.00, 0.20, 0.08, 0.01)
    skew_controls = st.checkbox("Weak skew on controls (AGE/INCOME if present)", value=False)

with st.sidebar.expander("Significance profile (quick)"):
    profile = st.selectbox("Preset", ["All significant (default)", "One weak path", "Custom"], index=0)
    weak_scale = st.slider("Weak path β (for 'One weak path')", 0.00, 0.40, 0.10, 0.01)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: For a deliberately non-significant path, reduce β or set lower R² for that endogenous construct.")


# ----------------------------- Main: Define Model -----------------------------

st.title("🧪 SmartSurvey Synthetic Data Lab (Bangladesh Context)")
st.write("Define constructs, paths, and statistical targets. Generate SmartPLS-ready data with diagnostics.")

st.subheader("1) Define constructs")
st.caption("Add constructs with roles (IV/DV/Mediator/Moderator/Control). The app will create items and LVs accordingly.")

# Construct table builder
if "construct_rows" not in st.session_state:
    st.session_state.construct_rows = [
        {"name": "PE", "role": "IV", "items": default_items},
        {"name": "PV", "role": "IV", "items": default_items},
        {"name": "EC", "role": "IV", "items": default_items},
        {"name": "AIA","role": "IV", "items": default_items},
        {"name": "TR", "role": "IV", "items": default_items},
        {"name": "ATT","role": "Mediator", "items": default_items},
        {"name": "BI", "role": "DV", "items": default_items},
        {"name": "PCIS","role": "Moderator", "items": default_items},
    ]

roles_options = ["IV", "DV", "Mediator", "Moderator", "Control"]

cols_hdr = st.columns([3,2,2,1])
cols_hdr[0].markdown("**Construct**")
cols_hdr[1].markdown("**Role**")
cols_hdr[2].markdown("**Items**")
cols_hdr[3].markdown("")

to_remove = []
for i, row in enumerate(st.session_state.construct_rows):
    c1, c2, c3, c4 = st.columns([3,2,2,1])
    row["name"] = c1.text_input("name", value=row["name"], key=f"name_{i}", label_visibility="collapsed")
    row["role"] = c2.selectbox("role", roles_options, index=roles_options.index(row["role"]), key=f"role_{i}", label_visibility="collapsed")
    row["items"] = c3.number_input("items", min_value=1, max_value=10, value=row["items"], step=1, key=f"items_{i}", label_visibility="collapsed")
    if c4.button("🗑", key=f"del_{i}"):
        to_remove.append(i)
if to_remove:
    st.session_state.construct_rows = [r for j, r in enumerate(st.session_state.construct_rows) if j not in to_remove]

if st.button("➕ Add construct"):
    st.session_state.construct_rows.append({"name": f"C{len(st.session_state.construct_rows)+1}", "role": "IV", "items": default_items})

constructs = [r["name"] for r in st.session_state.construct_rows]
roles = {r["name"]: r["role"] for r in st.session_state.construct_rows}
measurement_spec = {r["name"]: {"k": r["items"], "load_min": load_min, "load_max": load_max} for r in st.session_state.construct_rows}

st.markdown("---")
st.subheader("2) Define paths")
st.caption("Add structural relations. For moderation, add an interaction in the next subsection.")

# Path editor
if "paths" not in st.session_state:
    st.session_state.paths = [
        ("PE","ATT",0.26),
        ("PV","ATT",0.20),
        ("EC","ATT",0.24),
        ("AIA","ATT",0.18),
        ("TR","ATT",0.20),
        ("ATT","BI",0.60),
        ("PCIS","BI",0.12),
    ]

def path_row_ui(idx, src, dst, b):
    c1, c2, c3, c4 = st.columns([3,3,3,1])
    src_new = c1.selectbox("src", constructs, index=constructs.index(src) if src in constructs else 0,
                           key=f"p_src_{idx}", label_visibility="collapsed")
    dst_new = c2.selectbox("dst", constructs, index=constructs.index(dst) if dst in constructs else 0,
                           key=f"p_dst_{idx}", label_visibility="collapsed")
    beta = c3.slider("β", 0.00, 0.90, float(b), 0.01, key=f"p_beta_{idx}", label_visibility="collapsed")
    delete = c4.button("🗑", key=f"p_del_{idx}")
    return (src_new, dst_new, beta, delete)

to_del = []
for i, (s, d, b) in enumerate(st.session_state.paths):
    s2, d2, b2, delbtn = path_row_ui(i, s, d, b)
    st.session_state.paths[i] = (s2, d2, b2)
    if delbtn: to_del.append(i)
if to_del:
    st.session_state.paths = [p for j, p in enumerate(st.session_state.paths) if j not in to_del]

if st.button("➕ Add path"):
    st.session_state.paths.append((constructs[0], constructs[-1], 0.30))

st.markdown("---")
st.subheader("3) Moderation (optional)")
st.caption("Define interaction terms (IV × Moderator → DV). The app will create product indicators at the latent level.")

if "moderations" not in st.session_state:
    st.session_state.moderations = [("ATT","PCIS","BI",0.12)]

def mod_row_ui(idx, iv, mod, dst, b):
    c1, c2, c3, c4, c5 = st.columns([3,3,3,3,1])
    iv2 = c1.selectbox("IV", constructs, index=constructs.index(iv) if iv in constructs else 0,
                       key=f"m_iv_{idx}", label_visibility="collapsed")
    m2 = c2.selectbox("Moderator", constructs, index=constructs.index(mod) if mod in constructs else 0,
                      key=f"m_mod_{idx}", label_visibility="collapsed")
    d2 = c3.selectbox("DV", constructs, index=constructs.index(dst) if dst in constructs else 0,
                      key=f"m_dst_{idx}", label_visibility="collapsed")
    b2 = c4.slider("β (interaction)", 0.00, 0.90, float(b), 0.01, key=f"m_beta_{idx}", label_visibility="collapsed")
    delete = c5.button("🗑", key=f"m_del_{idx}")
    return (iv2, m2, d2, b2, delete)

to_del_m = []
for i, (iv, m, d, b) in enumerate(st.session_state.moderations):
    iv2, m2, d2, b2, delm = mod_row_ui(i, iv, m, d, b)
    st.session_state.moderations[i] = (iv2, m2, d2, b2)
    if delm: to_del_m.append(i)
if to_del_m:
    st.session_state.moderations = [mm for j, mm in enumerate(st.session_state.moderations) if j not in to_del_m]

if st.button("➕ Add moderation"):
    st.session_state.moderations.append((constructs[0], constructs[1], constructs[-1], 0.20))

st.markdown("---")
st.subheader("4) Target R² for endogenous constructs")
st.caption("Set R² targets; noise is calibrated to meet these approximately.")

# Collect destinations/endogenous constructs
endogenous = sorted(set([d for (_, d, _) in st.session_state.paths] + [d for (_, _, d, _) in st.session_state.moderations]))
if "r2_map" not in st.session_state:
    st.session_state.r2_map = {c: 0.55 for c in endogenous}

cols_r2 = st.columns(min(4, max(1, len(endogenous))))
for i, c in enumerate(endogenous):
    with cols_r2[i % len(cols_r2)]:
        st.session_state.r2_map[c] = st.slider(f"R²: {c}", 0.05, 0.90, float(st.session_state.r2_map[c]), 0.01)

# Apply preset profile if chosen
if profile == "One weak path" and st.session_state.paths:
    # weaken the last path
    s, d, _ = st.session_state.paths[-1]
    st.session_state.paths[-1] = (s, d, weak_scale)

# ----------------------------- Generate -----------------------------

st.markdown("---")
if st.button("🚀 Generate dataset", type="primary"):
    with st.spinner("Simulating latent variables and items..."):
        rng = np.random.default_rng(seed)
        dist_cfg = {"avg_corr": avg_corr, "corr_jitter": corr_jitter, "skew_controls": skew_controls}

        latents = generate_latents_general(
            n=n,
            constructs=constructs,
            roles=roles,
            paths=st.session_state.paths,
            r2_targets=st.session_state.r2_map,
            moderation_terms=st.session_state.moderations,
            dist_cfg=dist_cfg,
            rng=rng
        )

        items_df, load_record = generate_items(
            latents=latents,
            measurement_spec=measurement_spec,
            likert_scale=likert_scale,
            load_min=load_min,
            load_max=load_max,
            rng=rng
        )

    st.success(f"Dataset generated: {items_df.shape[0]} rows × {items_df.shape[1]} indicators.")

    # -------- Measurement diagnostics --------
    st.subheader("Measurement diagnostics")
    diag_rows = []
    blocks = {}
    for cons, spec in measurement_spec.items():
        cols = [f"{cons}{i+1}" for i in range(spec["k"])]
        blocks[cons] = cols
        dfc = items_df[cols]
        alpha = cronbach_alpha(dfc)
        loads = load_record[cons]
        err_vars = [1 - l**2 for l in loads]
        CR = composite_reliability(loads, err_vars)
        AVE = ave(loads)
        diag_rows.append({
            "Construct": cons,
            "Items": spec["k"],
            "α": round(alpha, 3) if pd.notna(alpha) else np.nan,
            "CR": round(CR, 3),
            "AVE": round(AVE, 3),
            "min λ": round(float(np.min(loads)), 3),
            "max λ": round(float(np.max(loads)), 3)
        })
    diag_df = pd.DataFrame(diag_rows)
    st.dataframe(diag_df, use_container_width=True)
    st.caption("Guideline: α ≥ 0.70, CR ≥ 0.70, AVE ≥ 0.50; indicator loadings ideally ≥ 0.70.")

    st.markdown("**HTMT (Heterotrait–Monotrait)**")
    htmt_mat = htmt(blocks, items_df)
    st.dataframe(htmt_mat, use_container_width=True)
    st.caption("HTMT < 0.85 (strict) or < 0.90 (lenient).")

    st.markdown("**Fornell–Larcker (latent correlations)**")
    lv_scores = pd.DataFrame({k: zscore(v) for k, v in latents.items()})
    st.dataframe(lv_scores[constructs].corr(), use_container_width=True)

    st.markdown("**√AVE per construct**")
    sqrt_ave = pd.DataFrame({"√AVE": [np.sqrt(ave(load_record[c])) for c in constructs]}, index=constructs)
    st.dataframe(sqrt_ave, use_container_width=True)

    # -------- Structural preview and VIF --------
    st.subheader("Structural preview (OLS coefficients)")
    sp = structural_preview(latents, st.session_state.paths, st.session_state.moderations)
    st.dataframe(sp, use_container_width=True)
    st.caption("SmartPLS will estimate exact path significance via bootstrapping; this is indicative preview.")

    st.markdown("**Collinearity (VIF) among predictors per endogenous DV**")
    for dst in endogenous:
        # gather predictors
        preds = [src for (src, d, _) in st.session_state.paths if d == dst]
        mods = [(iv, m) for (iv, m, d, _) in st.session_state.moderations if d == dst]
        Xcols = preds.copy()
        for (iv, m) in mods:
            coln = f"{iv}x{m}"
            if coln not in lv_scores.columns:
                lv_scores[coln] = lv_scores[iv] * lv_scores[m]
            Xcols += [iv, m, coln]
        Xcols = list(dict.fromkeys(Xcols))
        st.markdown(f"**Predictors → {dst}**")
        if Xcols:
            st.dataframe(vif_dataframe(lv_scores[Xcols]), use_container_width=True)
        else:
            st.info("No predictors defined.")

    # -------- Downloads --------
    st.subheader("Download data")

    # items only
    buf_csv = io.BytesIO()
    items_df.to_csv(buf_csv, index=False)
    st.download_button("Download Items (CSV)", buf_csv.getvalue(), file_name="synthetic_items.csv", mime="text/csv")

    buf_xlsx = io.BytesIO()
    with pd.ExcelWriter(buf_xlsx, engine="openpyxl") as w:
        items_df.to_excel(w, index=False, sheet_name="Items")
    st.download_button("Download Items (XLSX)", buf_xlsx.getvalue(),
                       file_name="synthetic_items.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # items + latents
    full_df = items_df.copy()
    for c in lv_scores.columns:
        full_df[f"LV_{c}"] = lv_scores[c]

    buf2 = io.BytesIO()
    full_df.to_csv(buf2, index=False)
    st.download_button("Download Items + Latents (CSV)", buf2.getvalue(),
                       file_name="synthetic_items_latents.csv", mime="text/csv")

    buf2x = io.BytesIO()
    with pd.ExcelWriter(buf2x, engine="openpyxl") as w:
        full_df.to_excel(w, index=False, sheet_name="Data")
    st.download_button("Download Items + Latents (XLSX)", buf2x.getvalue(),
                       file_name="synthetic_items_latents.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ----------------------------- Guidelines -----------------------------

st.markdown("---")
with st.expander("Guidelines & SmartPLS workflow", expanded=False):
    st.markdown("""
**Measurement (reflective)**  
- Indicator loadings (λ): ≥ 0.70 preferred (0.60–0.70 acceptable with strong CR/AVE).  
- Reliability: α ≥ 0.70, CR ≥ 0.70.  
- Convergent validity: AVE ≥ 0.50.  
- Discriminant validity: HTMT < 0.85/0.90; √AVE > inter-construct correlations.

**Structural**  
- R² benchmarks: ~0.25 weak, ~0.50 moderate, ~0.75 substantial.  
- Collinearity: VIF < 3.3 preferred (≤ 5 acceptable by some).  
- For non-significant paths (teaching cases), reduce β or increase DV noise via lower R².

**SmartPLS steps**  
1. Import the “Items” CSV.  
2. Create constructs and assign indicators (Mode A / reflective).  
3. Draw paths and, for moderation, add an interaction (product indicator or two-stage).  
4. Run PLS Algorithm → check loadings, α, CR, AVE, HTMT, FL, VIF.  
5. Bootstrapping (e.g., 5,000 subsamples) → get t/p for paths and outer loadings.  
6. Iterate in this app if you need different significance patterns.
""")
