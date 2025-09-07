# streamlit_app.py
# RothC-based Soil Organic Carbon (SOC) Simulator for Farmers — using pyRothC
# Author: ChatGPT (GPT-5 Thinking)
#
# Why this approach (high level):
# - We use the open-source **pyRothC** package, a faithful Python implementation of RothC-26.3, to avoid re‑coding core equations and to align with community-validated behavior.
# - Streamlit provides a farmer-friendly UI with immediate feedback, sliders, and side‑by‑side scenario comparison.
# - We expose practical levers (clay %, annual C inputs from residues/manure, plant cover via pE, monthly climate) and let farmers run Baseline vs. Alternative scenarios over 30 years.
# - We initialize Inert Organic Matter (IOM) using the standard Falloon et al. (1998) pedotransfer function and honor a user-provided current_soil_carbon.

import streamlit as st
import numpy as np
import pandas as pd

# External model
try:
    from pyRothC.RothC import RothC  # pip install pyRothC
except Exception as e:
    RothC = None

# -----------------------------
# User-provided current soil carbon (t C/ha)
# -----------------------------
# Expect a variable named `current_soil_carbon` to exist in the environment where this file runs.
# If not provided, we fall back to a sensible placeholder that the user can override in the UI.
try:
    current_soc_default = float(current_soil_carbon)  # type: ignore # noqa: F821
except Exception:
    current_soc_default = 50.0  # t C/ha placeholder

# -----------------------------
# Helpers
# -----------------------------


def iom_falloon(total_soc_t_ha: float) -> float:
    """Inert Organic Matter via Falloon et al. (1998)."""
    total_soc_t_ha = max(0.0, float(total_soc_t_ha))
    return 0.049 * (total_soc_t_ha**1.139)


def run_pyrothc(
    *,
    years: int,
    temperature_12,
    precip_12,
    evaporation_12,
    clay_pct: float,
    input_carbon_Mg_per_ha_yr: float,
    pE: float,
    C0_vec,
):
    """Run pyRothC for `years` with one climatology year (12 monthly values).
    pyRothC expects 12-length arrays for Temp/Precip/Evap. The model internally
    interpolates these across its time grid; we later slice the output to `years`.
    """
    # Validate 12-month inputs
    for label, arr in {
        "temperature": temperature_12,
        "precip": precip_12,
        "evaporation": evaporation_12,
    }.items():
        if len(arr) != 12:
            raise ValueError(f"{label} must have 12 monthly values; got {len(arr)}")

    Temp = np.array(temperature_12, dtype=float)
    Precip = np.array(precip_12, dtype=float)
    Evp = np.array(evaporation_12, dtype=float)

    model = RothC(
        temperature=Temp,
        precip=Precip,
        evaporation=Evp,
        clay=clay_pct,
        input_carbon=input_carbon_Mg_per_ha_yr,
        pE=pE,
        C0=np.array(C0_vec, dtype=float),
    )
    df = model.compute()
    try:
        df.index = np.array(model.t, dtype=float)
    except Exception:
        pass

    if "Total" in df.columns:
        df["SOC_total"] = df["Total"]
    else:
        pick = [c for c in ["DPM", "RPM", "BIO", "HUM", "IOM"] if c in df.columns]
        df["SOC_total"] = df[pick].sum(axis=1)

    # Slice to requested horizon (years) if index is time in years
    if np.issubdtype(df.index.dtype, np.number):
        return df[df.index <= years]
    return df


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="RothC SOC Simulator (pyRothC)", layout="wide")
st.title("RothC Soil Organic Carbon Simulator (pyRothC)")

colA, colB = st.columns([1, 1])
with colA:
    st.markdown("### Why pyRothC?")
    st.write(
        "Using the **pyRothC** package ensures we match the canonical RothC equations, "
        "benefiting from community validation and tests. We layer a farmer‑focused UI "
        "on top so you can try management changes and immediately see projected SOC."
    )
with colB:
    st.markdown("### Install the model library")
    st.code("pip install pyRothC  # requires numpy, pandas, scipy", language="bash")

if RothC is None:
    st.error(
        "pyRothC is not installed or failed to import. Please run: pip install pyRothC"
    )
    st.stop()

st.sidebar.header("Global Inputs")
years = st.sidebar.slider("Simulation years", 5, 50, 30, help="Projection horizon")
clay = st.sidebar.slider("Clay (%)", 0.0, 80.0, 25.0, step=0.5)
pE = st.sidebar.slider(
    "pE (pan evap to PET ratio)",
    0.5,
    1.5,
    1.0,
    0.01,
    help="Scales evaporation in the model; 1.0 is a common default.",
)

st.sidebar.header("Monthly Climate (12 values)")
defedit = st.sidebar.toggle("Use default temperate climate", value=True)
if defedit:
    t12 = [-0.4, 0.3, 4.2, 8.3, 13.0, 15.9, 18.0, 17.5, 13.4, 8.7, 3.9, 0.6]
    r12 = [49, 39, 44, 41, 61, 58, 71, 58, 51, 48, 50, 58]
    e12 = [12, 18, 35, 58, 82, 90, 97, 84, 54, 31, 14, 10]
else:
    with st.sidebar.expander("Temperature (°C)"):
        t12 = [st.number_input(f"T month {i+1}", value=10.0) for i in range(12)]
    with st.sidebar.expander("Rainfall (mm)"):
        r12 = [st.number_input(f"Rain month {i+1}", value=60.0) for i in range(12)]
    with st.sidebar.expander("Pan evaporation (mm)"):
        e12 = [st.number_input(f"PanEvap month {i+1}", value=60.0) for i in range(12)]

st.sidebar.header("Baseline vs. Alternative")
current_soc = st.sidebar.number_input(
    "Current Soil Carbon (t C/ha)", min_value=0.0, value=current_soc_default, step=0.1
)
base_inputs = st.sidebar.number_input(
    "Baseline annual C inputs (t C/ha/yr)",
    min_value=0.0,
    value=2.0,
    step=0.1,
    help="Residues + roots + manure as carbon.",
)
alt_inputs = st.sidebar.number_input(
    "Alternative annual C inputs (t C/ha/yr)",
    min_value=0.0,
    value=3.0,
    step=0.1,
    help="Try cover crops, higher residue returns, or added manure.",
)

# Initial pools (DPM, RPM, BIO, HUM, IOM)
IOM0 = iom_falloon(current_soc)
active = max(current_soc - IOM0, 0.0)
# Heuristic split if user does not provide pools: HUM 60%, BIO 2%, remaining split 50/50 DPM/RPM
HUM0 = 0.60 * active
BIO0 = 0.02 * active
DPM0 = 0.19 * active
RPM0 = active - HUM0 - BIO0 - DPM0

with st.expander("Advanced: set initial pools (DPM, RPM, BIO, HUM, IOM)"):
    use_custom_C0 = st.checkbox("Manually set initial pools", value=False)
    if use_custom_C0:
        DPM0 = st.number_input("DPM (t C/ha)", min_value=0.0, value=float(DPM0))
        RPM0 = st.number_input("RPM (t C/ha)", min_value=0.0, value=float(RPM0))
        BIO0 = st.number_input("BIO (t C/ha)", min_value=0.0, value=float(BIO0))
        HUM0 = st.number_input("HUM (t C/ha)", min_value=0.0, value=float(HUM0))
        IOM0 = st.number_input("IOM (t C/ha)", min_value=0.0, value=float(IOM0))
C0 = [DPM0, RPM0, BIO0, HUM0, IOM0]

# Run scenarios
base_df = run_pyrothc(
    years=years,
    temperature_12=t12,
    precip_12=r12,
    evaporation_12=e12,
    clay_pct=clay,
    input_carbon_Mg_per_ha_yr=base_inputs,
    pE=pE,
    C0_vec=C0,
)
alt_df = run_pyrothc(
    years=years,
    temperature_12=t12,
    precip_12=r12,
    evaporation_12=e12,
    clay_pct=clay,
    input_carbon_Mg_per_ha_yr=alt_inputs,
    pE=pE,
    C0_vec=C0,
)

# -----------------------------
# Display
# -----------------------------

left, right = st.columns(2)
with left:
    st.subheader("Baseline scenario")
    st.line_chart(base_df["SOC_total"], height=300)
    st.caption("Total SOC (t C/ha) over time")
with right:
    st.subheader("Alternative scenario")
    st.line_chart(alt_df["SOC_total"], height=300)
    st.caption("Total SOC (t C/ha) over time")

# Comparative table at decade marks
summary = pd.DataFrame(
    {
        "Year": list(range(0, years + 1, 10)),
    }
)
summary["SOC_baseline"] = [
    float(base_df["SOC_total"].iloc[min(len(base_df) - 1, y * 12)])
    for y in summary["Year"]
]
summary["SOC_alternative"] = [
    float(alt_df["SOC_total"].iloc[min(len(alt_df) - 1, y * 12)])
    for y in summary["Year"]
]
summary["Delta (Alt - Base)"] = summary["SOC_alternative"] - summary["SOC_baseline"]
st.dataframe(summary, hide_index=True, use_container_width=True)

# Download data
out = pd.DataFrame(
    {
        "time": base_df.index,
        "SOC_baseline": base_df["SOC_total"].values,
        "SOC_alternative": alt_df["SOC_total"]
        .reindex(base_df.index, method="nearest")
        .values,
    }
)
st.download_button(
    label="Download CSV",
    data=out.to_csv(index=False).encode("utf-8"),
    file_name=f"rothc_soc_{years}yr.csv",
    mime="text/csv",
)

with st.expander("About the model & assumptions"):
    st.markdown(
        """
        **RothC & pyRothC**  
        This app uses the [pyRothC](https://github.com/mishagrol/pyRothC) package, a Python implementation of the
        RothC-26.3 model from Rothamsted. We expose: **Clay %**, **annual carbon inputs** (residues, roots, manure),
        **monthly climate** (Temp, Rain, Pan Evap), and **pE** (scales evaporation).  
        **Initial pools** can be auto-estimated or set manually. IOM = 0.049·SOC^1.139 (Falloon 1998).

        **Why this design?**  
        - Leveraging a maintained library reduces implementation risk and matches published equations.  
        - Monthly inputs keep the UI farmer-oriented while staying true to RothC's time step.  
        - We repeat your monthly climate over the selected horizon to produce a 30‑year trajectory.

        **Outputs & units**  
        - SOC in **t C/ha (Mg C/ha)**.  
        - Downloadable CSV for record-keeping.
        """
    )
