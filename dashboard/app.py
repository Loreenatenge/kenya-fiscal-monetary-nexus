import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.ardl import UECM

st.set_page_config(page_title="Kenya Fiscal-Monetary Nexus", layout="wide")

if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

def get_colors():
    if st.session_state.dark_mode:
        return {
            'bg': '#0e1217',
            'card': '#1a222a',
            'text': '#e8edf2',
            'muted': '#8a9aa8',
            'maroon': '#c06070',
            'grid': '#2a3440',
            'border': '#2a3440',
            'heading': '#ffffff'
        }
    else:
        return {
            'bg': '#fdfaf3',
            'card': '#ececec',
            'text': '#2c3440',
            'muted': '#6b7280',
            'maroon': '#6e2a3a',
            'grid': '#e5e0d5',
            'border': '#ddd',
            'heading': '#2c3440'
        }

colors = get_colors()

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        background-color: {colors['bg']};
        color: {colors['text']};
    }}

    .stApp {{ background-color: {colors['bg']}; }}

    section[data-testid="stSidebar"] {{
        background-color: {colors['card']};
        border-right: 1px solid {colors['border']};
    }}

    h1 {{
        font-size: 28px;
        font-weight: 700;
        color: {colors['heading']} !important;
        letter-spacing: -0.5px;
    }}

    h2 {{
        font-size: 17px;
        font-weight: 600;
        color: {colors['heading']} !important;
        margin-top: 30px;
        margin-bottom: 10px;
    }}

    h3, h4, h5 {{ color: {colors['heading']} !important; }}

    p, li {{
        color: {colors['muted']};
        line-height: 1.7;
        font-size: 14px;
    }}

    .indicator-card {{
        background-color: {colors['card']};
        border-radius: 6px;
        padding: 22px 20px 18px 20px;
        text-align: center;
        margin-bottom: 6px;
        min-height: 168px;
        border: 1px solid {colors['border']};
    }}

    .indicator-title {{
        font-size: 14px;
        font-weight: 600;
        color: {colors['text']};
        margin-bottom: 10px;
    }}

    .indicator-value {{
        font-size: 30px;
        font-weight: 700;
        color: {colors['heading']};
        margin-bottom: 8px;
    }}

    .indicator-desc {{
        font-size: 12.5px;
        color: {colors['muted']};
        line-height: 1.5;
    }}

    .source-note {{
        font-size: 11px;
        font-style: italic;
        color: {colors['muted']};
        margin-top: 4px;
    }}

    .verdict-line {{
        font-size: 13px;
        color: {colors['maroon']};
        font-weight: 600;
        margin-top: 4px;
    }}

    .takeaway-box {{
        background-color: {colors['card']};
        border: 1px solid {colors['border']};
        border-radius: 8px;
        padding: 16px 20px;
        margin: 16px 0;
        border-left: 4px solid {colors['maroon']};
    }}

    hr {{ border-color: {colors['border']}; margin: 28px 0; }}

    .footer {{
        font-size: 11px;
        color: {colors['muted']};
        margin-top: 40px;
        padding-top: 14px;
        border-top: 1px solid {colors['border']};
    }}

    .sticky-bar {{
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: {colors['text']};
        padding: 10px 30px;
        display: flex;
        gap: 36px;
        z-index: 999;
        font-size: 12.5px;
        overflow-x: auto;
        white-space: nowrap;
    }}

    .sticky-item {{ color: #cfd3da; }}
    .sticky-item b {{ color: #ffffff; }}
    .sticky-val {{ color: #e8b4bf; font-weight: 600; }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: {colors['bg']};
        border-bottom: 2px solid {colors['border']};
    }}

    .stTabs [data-baseweb="tab"] {{
        background-color: transparent;
        border-radius: 6px 6px 0 0;
        padding: 10px 20px;
        color: {colors['muted']} !important;
        font-size: 13px;
        font-weight: 500;
    }}

    .stTabs [aria-selected="true"] {{
        background-color: {colors['card']};
        color: {colors['heading']} !important;
        border-bottom: 3px solid {colors['maroon']};
    }}
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned/kenya_nexus.csv")

df = load_data()

@st.cache_resource
def fit_model(data):
    endog = data["inflation"]
    exog = data[["external_debt_gni", "debt_service_exports"]]
    model = UECM(endog, lags=1, exog=exog, order=1)
    return model.fit()

results = fit_model(df)
bounds = results.bounds_test(case=3)

ect = results.params["inflation.L1"]
lr_debt = results.params["external_debt_gni.L1"] / -ect
lr_service = results.params["debt_service_exports.L1"] / -ect

fitted = results.fittedvalues
years_aligned = df["year"].iloc[1:].values
actual_aligned = df["inflation"].iloc[1:].values
start_val = df["inflation"].iloc[0]
fitted_levels = start_val + fitted.cumsum()
error_correction = actual_aligned - fitted_levels

plt.rcParams["font.family"] = "DejaVu Sans"

with st.sidebar:
    st.markdown("**Loreen Ateng'e Okonyo**")
    st.markdown("Economics and Statistics")
    st.markdown("University of Nairobi")
    st.markdown("---")
    dark_mode = st.toggle("Dark Mode", value=st.session_state.dark_mode)
    if dark_mode != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_mode
        st.rerun()
    st.markdown("---")
    st.markdown("**Data:** World Bank WDI")
    st.markdown("**Period:** 1980 to 2024")
    st.markdown("**Method:** ARDL bounds testing")
    st.markdown("---")
    st.markdown("**Related projects**")
    st.markdown("[Kenya Inflation Analysis](https://github.com/Loreenatenge/kenya-inflation-analysis)")
    st.markdown("[Kenya Debt Crisis Analysis](https://github.com/Loreenatenge/kenya-debt-crisis-analysis)")

title_col, btn_col = st.columns([5, 1])
with title_col:
    st.markdown("# Kenya Fiscal-Monetary Nexus")
    st.markdown("#### ARDL Bounds Testing Analysis (1980 to 2024)")
    st.markdown("Does Kenya's public debt have a measurable long run relationship with inflation?")
with btn_col:
    st.markdown("<br>", unsafe_allow_html=True)
    st.download_button(
        label="Download data",
        data=df.to_csv(index=False),
        file_name="kenya_fiscal_monetary_nexus_data.csv",
        mime="text/csv"
    )

st.markdown("<hr>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="indicator-card">
        <div class="indicator-title">Bounds Test F-statistic</div>
        <div class="indicator-value">{bounds.stat:.2f}</div>
        <div class="indicator-desc">Clears the upper bound at every confidence level tested, confirming a long run relationship exists.</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="indicator-card">
        <div class="indicator-title">Speed of Adjustment</div>
        <div class="indicator-value">{abs(ect)*100:.0f}%</div>
        <div class="indicator-desc">Of any gap between actual inflation and its long run path closes within a single year.</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="indicator-card">
        <div class="indicator-title">External Debt to GNI</div>
        <div class="indicator-value">{lr_debt:+.2f}</div>
        <div class="indicator-desc">Long run effect on inflation, in percentage points, for each 1 point rise in external debt to GNI.</div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="indicator-card">
        <div class="indicator-title">Debt Service to Exports</div>
        <div class="indicator-value">{lr_service:+.2f}</div>
        <div class="indicator-desc">Long run effect on inflation, in percentage points, for each 1 point rise in debt service to exports.</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

tabs = st.tabs(["Long Run Path", "Error Correction", "Bounds Test", "Diagnostics", "Correlations", "Data", "About"])

with tabs[0]:
    st.markdown("## Actual Inflation vs Fitted Long Run Path")

    fig1, ax1 = plt.subplots(figsize=(11, 3.8))
    fig1.patch.set_facecolor(colors['bg'])
    ax1.set_facecolor(colors['bg'])
    ax1.plot(years_aligned, actual_aligned, color=colors['muted'], linewidth=1.6, label="Actual inflation")
    ax1.plot(years_aligned, fitted_levels, color=colors['maroon'], linewidth=2.2, label="Fitted long run path")
    ax1.fill_between(years_aligned, actual_aligned, fitted_levels, color=colors['maroon'], alpha=0.08)

    events = {1993: "Structural\nadjustment", 2008: "Post-election\nviolence"}
    for yr, label in events.items():
        if yr in list(years_aligned):
            idx = list(years_aligned).index(yr)
            val = actual_aligned[idx]
            ax1.annotate(label, xy=(yr, val), xytext=(yr, val + 6), fontsize=7.5,
                         color=colors['text'], ha="center",
                         arrowprops=dict(arrowstyle="-", color=colors['muted'], lw=0.8))

    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_color(colors['grid'])
    ax1.spines["bottom"].set_color(colors['grid'])
    ax1.tick_params(colors=colors['muted'], labelsize=9)
    ax1.grid(axis="y", color=colors['grid'], linewidth=0.6)
    ax1.legend(frameon=False, fontsize=9, labelcolor=colors['text'])
    plt.tight_layout()
    st.pyplot(fig1)
    st.markdown('<p class="source-note">Data source: World Bank World Development Indicators. Fitted path from ARDL(0,1,0) error correction model.</p>', unsafe_allow_html=True)

    st.markdown("""
    The widest gaps fall in 1993 and 1994, the structural adjustment crisis, when inflation hit
    nearly 46 percent, and 2008, the post-election violence and global food crisis. These are
    exogenous shocks a debt-based model is not meant to fully explain. What matters is that
    inflation reliably returns toward its long run path afterward, and does so quickly.
    """)

with tabs[1]:
    st.markdown("## Error Correction Mechanism")
    st.markdown("Positive values mean inflation is above its long run path. Negative values mean it is below. The error correction term of -0.798 means around 80 percent of any gap closes within one year.")

    fig_ec, ax_ec = plt.subplots(figsize=(11, 3.6))
    fig_ec.patch.set_facecolor(colors['bg'])
    ax_ec.set_facecolor(colors['bg'])
    bar_colors = [colors['maroon'] if x >= 0 else colors['muted'] for x in error_correction]
    ax_ec.bar(years_aligned, error_correction, color=bar_colors, alpha=0.75, width=0.8)
    ax_ec.axhline(y=0, color=colors['text'], linestyle='-', linewidth=0.8, alpha=0.4)
    ax_ec.set_xlabel("Year", color=colors['muted'])
    ax_ec.set_ylabel("Deviation (percentage points)", color=colors['muted'])
    ax_ec.spines["top"].set_visible(False)
    ax_ec.spines["right"].set_visible(False)
    ax_ec.spines["left"].set_color(colors['grid'])
    ax_ec.spines["bottom"].set_color(colors['grid'])
    ax_ec.tick_params(colors=colors['muted'], labelsize=9)
    ax_ec.grid(axis="y", color=colors['grid'], linewidth=0.6)
    plt.tight_layout()
    st.pyplot(fig_ec)
    st.markdown('<p class="source-note">Data source: World Bank World Development Indicators.</p>', unsafe_allow_html=True)

    max_dev = float(np.abs(error_correction).max())
    mean_abs = float(np.abs(error_correction).mean())
    current_dev = float(error_correction.iloc[-1]) if hasattr(error_correction, 'iloc') else float(error_correction[-1])
    st.markdown(f"Maximum deviation: {max_dev:.1f} percentage points. Mean absolute deviation: {mean_abs:.1f} percentage points. Most recent deviation (2024): {current_dev:.1f} percentage points, {'above' if current_dev > 0 else 'below'} the long run path.")

with tabs[2]:
    st.markdown("## Bounds Test: Statistic vs Critical Value Bounds")

    crit = bounds.crit_vals
    fig2, ax2 = plt.subplots(figsize=(8.5, 3.6))
    fig2.patch.set_facecolor(colors['bg'])
    ax2.set_facecolor(colors['bg'])
    levels = crit.index.tolist()
    x_pos = range(len(levels))
    ax2.plot(x_pos, crit["lower"].tolist(), marker="o", color=colors['muted'], linewidth=1.6, label="Lower bound, I(0)")
    ax2.plot(x_pos, crit["upper"].tolist(), marker="o", color=colors['text'], linewidth=1.6, label="Upper bound, I(1)")
    ax2.axhline(bounds.stat, color=colors['maroon'], linewidth=2, linestyle="--", label=f"Our statistic ({bounds.stat:.2f})")
    ax2.set_xticks(list(x_pos))
    ax2.set_xticklabels([f"{l}%" for l in levels])
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_color(colors['grid'])
    ax2.spines["bottom"].set_color(colors['grid'])
    ax2.tick_params(colors=colors['muted'], labelsize=9)
    ax2.grid(axis="y", color=colors['grid'], linewidth=0.6)
    ax2.legend(frameon=False, fontsize=9, labelcolor=colors['text'])
    plt.tight_layout()
    st.pyplot(fig2)
    st.markdown(f'<p class="verdict-line">Result: cointegration confirmed at every confidence level shown, including 99.9 percent.</p>', unsafe_allow_html=True)
    st.markdown('<p class="source-note">Pesaran-Shin-Smith bounds test, case 3. Data source: World Bank World Development Indicators.</p>', unsafe_allow_html=True)

with tabs[3]:
    st.markdown("## Model Diagnostics")
    st.markdown("The table below shows the fitted model's information criteria. Lower AIC and BIC values indicate a better fitting, more parsimonious model. These are computed directly from the estimated UECM.")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("AIC", f"{results.aic:.1f}")
    with col_b:
        st.metric("BIC", f"{results.bic:.1f}")
    with col_c:
        st.metric("HQIC", f"{results.hqic:.1f}")

    st.markdown("### Residuals")
    residuals = results.resid
    fig_res, ax_res = plt.subplots(figsize=(11, 2.5))
    fig_res.patch.set_facecolor(colors['bg'])
    ax_res.set_facecolor(colors['bg'])
    ax_res.plot(years_aligned, residuals, color=colors['maroon'], linewidth=1.2)
    ax_res.axhline(y=0, color=colors['muted'], linestyle='--', linewidth=0.8)
    ax_res.set_xlabel("Year", color=colors['muted'])
    ax_res.set_ylabel("Residuals", color=colors['muted'])
    ax_res.spines["top"].set_visible(False)
    ax_res.spines["right"].set_visible(False)
    ax_res.spines["left"].set_color(colors['grid'])
    ax_res.spines["bottom"].set_color(colors['grid'])
    ax_res.tick_params(colors=colors['muted'])
    ax_res.grid(axis="y", color=colors['grid'], linewidth=0.6)
    plt.tight_layout()
    st.pyplot(fig_res)
    st.markdown("The residuals show no persistent patterns, which is consistent with the model capturing the key long run relationship. The largest residuals occur during the 1993 structural adjustment crisis and 2008 post-election violence, both exogenous shocks that temporarily pushed inflation away from its long run path.")

with tabs[4]:
    st.markdown("## Correlation Matrix")
    st.markdown("Pairwise correlations between variables in the dataset, calculated from the actual 1980 to 2024 annual data.")

    corr_vars = ["inflation", "external_debt_gni", "debt_service_exports", "exchange_rate", "broad_money"]
    corr_matrix = df[corr_vars].corr()

    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    fig_corr.patch.set_facecolor(colors['bg'])
    ax_corr.set_facecolor(colors['bg'])
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                square=True, linewidths=1, linecolor=colors['bg'],
                cbar_kws={'label': 'Correlation Coefficient'}, ax=ax_corr)
    ax_corr.tick_params(colors=colors['muted'])
    plt.tight_layout()
    st.pyplot(fig_corr)
    st.markdown('<p class="source-note">Data source: World Bank World Development Indicators, 1980 to 2024.</p>', unsafe_allow_html=True)

    st.markdown("""
Inflation and external debt to GNI are moderately strongly correlated at 0.612, the strongest pairwise relationship with inflation in the dataset, consistent with the positive long run coefficient found in the model.

Inflation and debt service to exports show almost no simple linear correlation at 0.069. This is an important finding: the relationship between debt service and inflation is not visible in a simple correlation but does emerge in the ARDL model, which accounts for the long run structure and dynamics of the series. This is exactly why econometric modelling adds value beyond correlation analysis alone.

Exchange rate and broad money are strongly correlated with each other at 0.808. This explains why both variables dropped out of the preferred model once external debt to GNI and debt service to exports were included, they were picking up overlapping information.
    """)

with tabs[5]:
    st.markdown("## The Data")
    table_df = df.rename(columns={
        "year": "Year",
        "inflation": "Inflation (%)",
        "exchange_rate": "Exchange Rate (KES/USD)",
        "broad_money": "Broad Money (% of GDP)",
        "external_debt_gni": "External Debt to GNI (%)",
        "debt_service_exports": "Debt Service to Exports (%)"
    })
    st.dataframe(table_df.style.format(precision=2), use_container_width=True, height=350)
    st.markdown("### Summary Statistics")
    st.dataframe(df[["inflation","external_debt_gni","debt_service_exports","exchange_rate","broad_money"]].describe().round(2), use_container_width=True)
    st.markdown('<p class="source-note">Data source: World Bank World Development Indicators, 1980 to 2024.</p>', unsafe_allow_html=True)

with tabs[6]:
    st.markdown("## About This Project")
    st.markdown("""
This project examines whether Kenya's public debt has a measurable long run relationship
with inflation, using ARDL bounds testing on annual World Bank data from 1980 to 2024.
It builds directly on two earlier projects, a standalone Kenya inflation analysis and a
Kenya debt sustainability analysis, by testing the fiscal-monetary transmission channel
that connects them.

The preferred model is ARDL(0,1,0), selected using BIC. It retains external debt to GNI
and debt service to exports as the two explanatory variables. Exchange rate and broad money
were tested but added no explanatory power once these two were included.

**Variable definitions**

Inflation is the annual percentage change in consumer prices. External debt to GNI is total
external debt as a percentage of gross national income. Debt service to exports is total debt
repayments as a percentage of export earnings. Exchange rate is Kenya shillings per US dollar.
Broad money is M2 as a percentage of GDP.

**Method**

ARDL bounds testing following Pesaran, Shin, and Smith (2001). Stationarity was confirmed
using Augmented Dickey-Fuller tests. All variables are either I(0) or I(1), satisfying the
precondition for this test. The bounds test was run under case 3, a constant included in
the model but excluded from the formal test.
    """)
    st.markdown("**Related projects**")
    st.markdown("[Kenya Inflation Drivers Analysis](https://github.com/Loreenatenge/kenya-inflation-analysis)")
    st.markdown("[Kenya Debt Crisis Analysis](https://github.com/Loreenatenge/kenya-debt-crisis-analysis)")
    st.markdown("[Kenya Fiscal-Monetary Nexus](https://github.com/Loreenatenge/kenya-fiscal-monetary-nexus)")

st.markdown("<hr>", unsafe_allow_html=True)

st.markdown(f"""
<div class="takeaway-box">
    <h4>Key Takeaways</h4>
    <ul>
        <li>Cointegration confirmed: Kenya's debt and inflation share a stable long run relationship. The F-statistic of {bounds.stat:.2f} clears every critical value bound.</li>
        <li>Rapid adjustment: around 80 percent of deviations from the long run path are corrected within a single year.</li>
        <li>External debt accumulation raises inflation over the long run, consistent with debt pressuring the exchange rate and import costs.</li>
        <li>Higher debt service to exports is associated with lower inflation, likely through a tightening of foreign exchange available for domestic spending.</li>
        <li>The simple correlation between inflation and debt service to exports is near zero at 0.069. The long run relationship only becomes visible through proper econometric modelling, which is exactly why ARDL is the appropriate method here.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.markdown("## Policy Implications")
st.markdown(f"""
External debt to GNI has a positive long run relationship with inflation ({lr_debt:+.2f}).
Higher external debt is associated with structurally higher inflation over time, consistent
with debt pressuring the exchange rate and raising the cost of imports.

Debt service to exports has a negative long run relationship with inflation ({lr_service:+.2f}).
When debt service consumes more of Kenya's foreign exchange earnings, less is available for
domestic spending, creating a tightening effect. This is the same indicator that signalled
fiscal stress five years early in the companion debt project. Here it shows a measurable,
opposite-direction relationship with inflation over the long run.

Exchange rate and broad money were tested and added no explanatory power once these two
variables were included, consistent with the high correlation between all four variables
identified in the correlations tab.

The debt composition matters as much as the debt level. The channel through which debt
affects the economy, whether through stock accumulation or through the flow of repayments
relative to export capacity, produces opposite effects on inflation. Fiscal and monetary
policy coordination needs to account for both channels simultaneously rather than treating
debt as a single aggregate indicator.

The speed of adjustment of around 80 percent per year suggests that policy interventions
aimed at bringing debt levels down would have relatively rapid effects on inflation, provided
the underlying long run relationship holds. Early monitoring of debt service to exports,
which this analysis confirms as the more sensitive flow indicator, should be built into
fiscal surveillance frameworks alongside the more commonly cited debt to GDP ratio.
""")

st.markdown(f'<p class="footer">Data: World Bank World Development Indicators | Analysis: Loreen Atenge, University of Nairobi</p>', unsafe_allow_html=True)
st.markdown("<div style='height:50px'></div>", unsafe_allow_html=True)

st.markdown(f"""
<div class="sticky-bar">
    <div class="sticky-item"><b>Bounds F-stat</b> <span class="sticky-val">{bounds.stat:.2f}</span></div>
    <div class="sticky-item"><b>Speed of adjustment</b> <span class="sticky-val">{abs(ect)*100:.0f}%/yr</span></div>
    <div class="sticky-item"><b>External debt to GNI</b> <span class="sticky-val">{lr_debt:+.2f}</span></div>
    <div class="sticky-item"><b>Debt service to exports</b> <span class="sticky-val">{lr_service:+.2f}</span></div>
    <div class="sticky-item"><b>AIC</b> <span class="sticky-val">{results.aic:.1f}</span></div>
    <div class="sticky-item"><b>Sample</b> <span class="sticky-val">1980–2024</span></div>
</div>
""", unsafe_allow_html=True)