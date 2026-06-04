"""
Interactive HMM + Monte Carlo CKD Progression App

Run locally:
    pip install streamlit pandas numpy matplotlib scikit-learn
    streamlit run ckd_hmm_monte_carlo_app.py

In Google Colab:
    !pip install streamlit pyngrok
    !streamlit run ckd_hmm_monte_carlo_app.py &>/content/logs.txt &
    # then expose with pyngrok if you use ngrok
"""

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

STATE_NAMES = [
    "State 0: Stable / normal-mild",
    "State 1: Moderate dysfunction",
    "State 2: Severe dysfunction",
    "State 3: Kidney failure range",
]

DEFAULT_TRANSITION = np.array([
    [0.956, 0.039, 0.004, 0.001],
    [0.103, 0.812, 0.081, 0.004],
    [0.006, 0.118, 0.797, 0.079],
    [0.008, 0.008, 0.180, 0.805],
])


def normalize_rows(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=float)
    mat = np.clip(mat, 0, None)
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return mat / row_sums


def estimate_transition_matrix(df: pd.DataFrame, state_col: str, subject_col: str, time_col: str | None, n_states: int = 4) -> np.ndarray:
    counts = np.ones((n_states, n_states))  # Laplace smoothing prevents zero probabilities
    work = df.copy()
    if time_col and time_col in work.columns:
        work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
        work = work.sort_values([subject_col, time_col])
    else:
        work = work.sort_values([subject_col])

    for _, g in work.groupby(subject_col):
        states = pd.to_numeric(g[state_col], errors="coerce").dropna().astype(int).to_numpy()
        states = states[(states >= 0) & (states < n_states)]
        if len(states) < 2:
            continue
        for a, b in zip(states[:-1], states[1:]):
            counts[a, b] += 1
    return normalize_rows(counts)


def simulate_paths(transition: np.ndarray, start_state: int, horizon: int, n_sims: int, rng: np.random.Generator) -> np.ndarray:
    n_states = transition.shape[0]
    paths = np.zeros((n_sims, horizon + 1), dtype=int)
    paths[:, 0] = start_state
    for t in range(1, horizon + 1):
        prev = paths[:, t - 1]
        for s in range(n_states):
            idx = np.where(prev == s)[0]
            if len(idx) > 0:
                paths[idx, t] = rng.choice(n_states, size=len(idx), p=transition[s])
    return paths


def summarize_paths(paths: np.ndarray) -> pd.DataFrame:
    n_sims, n_steps = paths.shape
    n_states = int(paths.max()) + 1
    rows = []
    for t in range(n_steps):
        row = {"time_step": t}
        for s in range(n_states):
            row[f"P_state_{s}"] = np.mean(paths[:, t] == s)
        row["P_above_stable"] = np.mean(paths[:, t] > 0)
        row["P_failure_range"] = np.mean(paths[:, t] == 3) if n_states > 3 else np.nan
        row["expected_state"] = np.mean(paths[:, t])
        rows.append(row)
    return pd.DataFrame(rows)


def estimate_current_state_from_labs(creatinine: float, egfr: float | None) -> int:
    """Simple clinically motivated heuristic for choosing a starting state."""
    if egfr is not None and not np.isnan(egfr):
        if egfr >= 60:
            return 0
        if egfr >= 30:
            return 1
        if egfr >= 15:
            return 2
        return 3
    if creatinine < 1.4:
        return 0
    if creatinine < 2.0:
        return 1
    if creatinine < 4.0:
        return 2
    return 3


st.set_page_config(page_title="CKD HMM Monte Carlo App", layout="wide")
st.title("CKD Progression Monte Carlo Simulator")
st.caption("Uses an HMM transition matrix to simulate possible future kidney-function states.")

st.markdown(
    """
This app treats the patient's kidney-function state as **hidden** and uses a transition matrix to simulate future paths.
You can either upload a dataset with predicted HMM states, or use the default transition matrix from the MIMIC demo model.
"""
)

with st.sidebar:
    st.header("1. Data input")
    uploaded_file = st.file_uploader("Upload HMM prediction CSV", type=["csv"])
    st.write("Expected columns if uploading data: `subject_id`, `charttime`, `predicted_hmm_state`.")

    st.header("2. Simulation settings")
    horizon = st.slider("Number of future time steps", 1, 30, 10)
    n_sims = st.slider("Monte Carlo simulations", 100, 20000, 5000, step=100)
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)

# Load or estimate transition matrix
transition = DEFAULT_TRANSITION.copy()
df = None
source_msg = "Using default transition matrix from the previous MIMIC demo HMM."

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded dataset preview")
    st.dataframe(df.head(20), use_container_width=True)

    needed_cols = {"subject_id", "predicted_hmm_state"}
    if needed_cols.issubset(df.columns):
        time_col = "charttime" if "charttime" in df.columns else None
        transition = estimate_transition_matrix(
            df=df,
            state_col="predicted_hmm_state",
            subject_col="subject_id",
            time_col=time_col,
            n_states=4,
        )
        source_msg = "Estimated transition matrix from uploaded patient trajectories."
    else:
        st.warning("Uploaded file does not include `subject_id` and `predicted_hmm_state`, so the app is using the default matrix.")

st.subheader("Transition Matrix")
st.write(source_msg)
transition_df = pd.DataFrame(transition, index=STATE_NAMES, columns=STATE_NAMES)
st.dataframe(transition_df.round(3), use_container_width=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Choose starting state")
    input_mode = st.radio("Input mode", ["Pick state manually", "Estimate from labs"], horizontal=True)

    if input_mode == "Pick state manually":
        start_state = st.selectbox(
            "Current kidney-function state",
            options=list(range(4)),
            format_func=lambda i: STATE_NAMES[i],
        )
    else:
        creatinine = st.number_input("Creatinine (mg/dL)", min_value=0.1, max_value=15.0, value=1.2, step=0.1)
        egfr_input = st.number_input("eGFR, optional; enter 0 if unknown", min_value=0.0, max_value=150.0, value=75.0, step=1.0)
        egfr = np.nan if egfr_input == 0 else egfr_input
        start_state = estimate_current_state_from_labs(creatinine, egfr)
        st.info(f"Estimated starting state: **{STATE_NAMES[start_state]}**")

with col2:
    st.subheader("Optional: select patient from uploaded data")
    if df is not None and {"subject_id", "predicted_hmm_state"}.issubset(df.columns):
        patient_ids = sorted(df["subject_id"].dropna().unique().tolist())
        selected_patient = st.selectbox("Patient", patient_ids)
        g = df[df["subject_id"] == selected_patient].copy()
        if "charttime" in g.columns:
            g["charttime"] = pd.to_datetime(g["charttime"], errors="coerce")
            g = g.sort_values("charttime")
        current = int(g["predicted_hmm_state"].dropna().iloc[-1])
        st.write(f"Last observed HMM state: **{STATE_NAMES[current]}**")
        use_patient_state = st.checkbox("Use selected patient's last state as simulation start", value=False)
        if use_patient_state:
            start_state = current
        show_cols = [c for c in ["subject_id", "charttime", "creatinine_mg_dl", "estimated_egfr_2021", "predicted_hmm_state", "predicted_hmm_state_name"] if c in g.columns]
        st.dataframe(g[show_cols].tail(10), use_container_width=True)
    else:
        st.write("Upload a prediction dataset to select a real patient trajectory.")

rng = np.random.default_rng(seed)
paths = simulate_paths(transition, start_state, horizon, n_sims, rng)
summary = summarize_paths(paths)

st.subheader("Monte Carlo Results")

metric_cols = st.columns(4)
final = summary.iloc[-1]
metric_cols[0].metric("P(above stable)", f"{final['P_above_stable']:.1%}")
metric_cols[1].metric("P(kidney failure range)", f"{final['P_failure_range']:.1%}")
metric_cols[2].metric("Expected state", f"{final['expected_state']:.2f}")
metric_cols[3].metric("Simulations", f"{n_sims:,}")

plot_df = summary.set_index("time_step")[["P_state_0", "P_state_1", "P_state_2", "P_state_3"]]
st.line_chart(plot_df)

st.write("State probability table")
st.dataframe(summary.round(4), use_container_width=True)

# Histogram of final states
st.subheader("Distribution of Final States")
final_states = pd.Series(paths[:, -1]).value_counts(normalize=True).sort_index()
final_states_df = pd.DataFrame({
    "state": [STATE_NAMES[i] for i in final_states.index],
    "probability": final_states.values,
})
st.bar_chart(final_states_df.set_index("state"))

st.subheader("Download results")
csv = summary.to_csv(index=False).encode("utf-8")
st.download_button("Download Monte Carlo summary CSV", csv, "monte_carlo_ckd_summary.csv", "text/csv")

paths_small = pd.DataFrame(paths[: min(1000, len(paths))])
paths_small.columns = [f"t_{i}" for i in range(paths_small.shape[1])]
paths_csv = paths_small.to_csv(index=False).encode("utf-8")
st.download_button("Download sample simulated paths CSV", paths_csv, "monte_carlo_ckd_paths_sample.csv", "text/csv")

st.markdown(
    """
### Interpretation note
These are **simulation-based probabilities**, not guaranteed clinical diagnoses. The app estimates future risk under the transition matrix you provide. If the transition matrix comes from MIMIC-IV demo data, the result should be described as hospital-based kidney-function trajectory modeling, not definitive long-term outpatient CKD progression.
"""
)
