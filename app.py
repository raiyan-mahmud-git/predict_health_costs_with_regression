"""Local Streamlit interface for the MEPS annual expenditure model."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from health_costs.meps_data import MEPS_FEATURES
from health_costs.meps_model import (
    MEPS_MODEL_DIR,
    load_meps_artifacts,
    predict_meps_expenditure,
    train_meps_project,
)

st.set_page_config(
    page_title="CareCost | Health Expenditure Estimator",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    /* Keep the app palette tied to Streamlit's selected theme. Hard-coded
       light surfaces made headings, tabs, and captions disappear in dark mode. */
    .stApp { background: var(--background-color); color: var(--text-color); }
    html, body, [class*="css"] { font-family: ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif; }
    h1, h2, h3 { font-family: ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif; letter-spacing: -0.035em; color: var(--text-color); }
    .block-container { max-width: 1080px; padding-top: 2.3rem; padding-bottom: 4rem; }
    .hero { background: linear-gradient(120deg, #102f3a 0%, #155b59 100%); color: white; padding: 2.2rem 2.5rem; border-radius: 24px; margin-bottom: 1.4rem; box-shadow: 0 18px 45px rgba(21, 66, 72, .16); }
    .hero h1 { color: white; font-size: 2.65rem; margin: .3rem 0 .55rem; }
    .hero p { color: #d4e8e4; max-width: 670px; margin: 0; font-size: 1.06rem; }
    .eyebrow { text-transform: uppercase; letter-spacing: .16em; font-size: .74rem; font-weight: 700; color: #8ee0c9; }
    .panel { background: var(--secondary-background-color); color: var(--text-color); border: 1px solid color-mix(in srgb, var(--text-color) 14%, transparent); padding: 1.35rem 1.5rem; border-radius: 18px; box-shadow: 0 8px 28px rgba(16,47,58,.045); }
    .result { background: color-mix(in srgb, #147d6e 13%, var(--secondary-background-color)); color: var(--text-color); border: 1px solid color-mix(in srgb, #147d6e 30%, transparent); padding: 1.35rem 1.6rem; border-radius: 18px; }
    .fineprint { color: var(--text-color); opacity: .72; font-size: .88rem; line-height: 1.55; }
    div.stButton > button[kind="primary"], div.stFormSubmitButton > button[kind="primary"] { background: #147d6e; border: 0; border-radius: 11px; font-weight: 700; padding: .68rem 1.15rem; }
    div[data-testid="stMetric"] { background: var(--secondary-background-color); color: var(--text-color); padding: 1rem 1.2rem; border-radius: 15px; border: 1px solid color-mix(in srgb, var(--text-color) 14%, transparent); }
    /* Streamlit's tabs inherit low-contrast defaults in custom themed apps. */
    button[data-baseweb="tab"] { color: var(--text-color); }
    @media (prefers-reduced-motion: no-preference) {
      .hero, .panel, .result { transition: background-color .2s ease, color .2s ease, border-color .2s ease; }
    }
    html[data-theme="dark"] .stApp { background: radial-gradient(ellipse at 15% 0%, #172c31 0, var(--background-color) 48%); }
    html[data-theme="dark"] .panel,
    html[data-theme="dark"] div[data-testid="stMetric"] { box-shadow: 0 8px 28px rgba(0,0,0,.18); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="hero">
      <div class="eyebrow">A data science demo · U.S. 2024 survey data</div>
      <h1>Understand a year of health-care spending.</h1>
      <p>Explore an educational estimate based on age, BMI, self-rated health, smoking frequency, sex, and Census region.</p>
    </section>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def _load_model(model_dir: str):
    return load_meps_artifacts(model_dir)


model_path = Path(MEPS_MODEL_DIR)
model_file = model_path / "annual_expenditure.joblib"
metadata_file = model_path / "metadata.json"

if not (model_file.exists() and metadata_file.exists()):
    st.info("The MEPS 2024 model has not been trained in this project folder yet.")
    st.markdown(
        """
        <div class="panel">
          <strong>One-time setup</strong><br>
          Train locally from AHRQ’s public-use file. The download is cached on this computer; no data is sent to a hosting service.
          The first run needs internet access and may take a few minutes.
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Download MEPS data and train model", type="primary"):
        try:
            with st.spinner("Preparing survey records, evaluating grouped folds, and fitting the model…"):
                outcome = train_meps_project()
            _load_model.clear()
            st.success(
                f"Done. Trained on {outcome['metadata']['valid_rows']:,} eligible adults. "
                f"Weighted holdout MAE: ${outcome['metadata']['holdout_weighted_metrics']['mae']:,.0f}."
            )
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    st.stop()

try:
    estimator, metadata = _load_model(str(model_path.resolve()))
except Exception as exc:
    st.error(f"Could not load the local MEPS model: {exc}")
    st.stop()

estimate_tab, methodology_tab = st.tabs(["Estimate", "How it works"])

with estimate_tab:
    left, right = st.columns([1.08, .92], gap="large")
    with left:
        st.markdown("### Person details")
        st.caption("All five inputs are required. MEPS features are adult survey fields.")
        with st.form("estimate_form"):
            age = st.slider("Age", min_value=18, max_value=85, value=35, help="The source survey top-codes age at 85.")
            bmi = st.number_input("Body mass index (BMI)", min_value=10.0, max_value=50.0, value=27.5, step=0.1)
            sex = st.selectbox("Sex", ["Female", "Male"])
            region = st.selectbox("U.S. Census region", ["Northeast", "Midwest", "South", "West"])
            smoking_frequency = st.selectbox(
                "Cigarette smoking frequency",
                ["Not at all", "Some days", "Every day"],
                help="Use the frequency reported in the MEPS survey categories.",
            )
            self_rated_health = st.selectbox(
                "General health, as the person would rate it",
                ["Excellent", "Very good", "Good", "Fair", "Poor"],
                help="This is a subjective survey measure, not a clinical assessment.",
            )
            submitted = st.form_submit_button("Estimate annual spending", type="primary", use_container_width=True)

    with right:
        st.markdown("### Your estimate")
        if submitted:
            features = pd.DataFrame(
                [{"age": age, "bmi": bmi, "sex": sex, "region": region,
                  "smoking_frequency": smoking_frequency, "self_rated_health": self_rated_health}],
                columns=MEPS_FEATURES,
            )
            try:
                estimate = float(predict_meps_expenditure(features, estimator, metadata)[0])
                st.markdown('<div class="result">', unsafe_allow_html=True)
                st.metric("Estimated total health-care expenditure", f"${estimate:,.0f}", help="Annual amount in 2024 U.S. dollars")
                st.markdown("</div>", unsafe_allow_html=True)
                st.caption("This estimates total annual health-care expenditures. It is not an insurance premium, a quote, or an individual financial recommendation.")
            except Exception as exc:
                st.error(str(exc))
        else:
            st.markdown(
                """
                <div class="panel">
                  <div class="eyebrow" style="color:#147d6e">Ready when you are</div>
                  <h3 style="margin:.65rem 0">Enter details to see an estimate</h3>
                  <p class="fineprint">The model uses grouped, survey-weighted training and keeps households together when evaluating its holdout.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

with methodology_tab:
    st.markdown("### A different target from the original FCC notebook")
    st.write(
        "This model uses the AHRQ Medical Expenditure Panel Survey (MEPS) 2024 Full Year Consolidated public-use file. "
        "It predicts reported total health-care expenditure for a year, not an insurance premium or the original FCC charges target."
    )
    st.write(
        f"Training used {metadata['valid_rows']:,} eligible adult records with complete BMI and cigarette-frequency responses. "
        "The model uses the MEPS expenditure survey weight, a household-grouped 20% holdout, and five-fold grouped cross-validation on development data."
    )
    metrics = metadata["holdout_weighted_metrics"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Weighted holdout MAE", f"${metrics['mae']:,.0f}")
    m2.metric("Weighted holdout RMSE", f"${metrics['rmse']:,.0f}")
    m3.metric("Weighted holdout R²", f"{metrics['r2']:.3f}")
    st.caption(
        "Survey results describe a population, not a specific person. The model omits many health and care-use factors, "
        "so predictions can be inaccurate. This teaching demo must not be used for medical, insurance, or underwriting decisions."
    )
    st.markdown("**Source:** AHRQ [MEPS 2024 public-use file](https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-256&prfricon=yes) and [documentation/codebook](https://meps.ahrq.gov/data_stats/download_data/pufs/h256/h256doc.shtml).")

st.markdown("<p class='fineprint' style='margin-top:2rem'>CareCost is an educational project. Predictions are not medical advice, insurance quotes, or underwriting decisions.</p>", unsafe_allow_html=True)
