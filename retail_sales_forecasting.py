
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ── Page config — MUST be first Streamlit call ────────────────────────────────
st.set_page_config(page_title="Retail Sales Forecasting", page_icon="🛒", layout="wide")

# ── Label encoding maps ────────────────────────────────────────────────────────
CATEGORY_MAP    = {0:'Beauty',1:'Clothing',2:'Electronics',3:'Home & Kitchen',4:'Sports'}
TRANSACTION_MAP = {0:'Card',1:'Cash',2:'UPI'}
CITY_TIER_MAP   = {0:'Tier 1',1:'Tier 2',2:'Tier 3'}
QUARTER_MAP     = {0:'Q1',1:'Q2',2:'Q3',3:'Q4'}
CITY_MAP = {
    0:'Agra',1:'Ahmedabad',2:'Amritsar',3:'Bengaluru',4:'Bhopal',5:'Chandigarh',
    6:'Chennai',7:'Coimbatore',8:'Delhi',9:'Guwahati',10:'Hyderabad',11:'Indore',
    12:'Jabalpur',13:'Jaipur',14:'Jodhpur',15:'Kochi',16:'Kolkata',17:'Lucknow',
    18:'Madurai',19:'Meerut',20:'Mumbai',21:'Mysuru',22:'Nagpur',23:'Nashik',
    24:'Patna',25:'Pune',26:'Raipur',27:'Ranchi',28:'Surat',29:'Thiruvananthapuram',
    30:'Vadodara',31:'Varanasi',32:'Visakhapatnam'
}
STATE_MAP = {
    0:'Andhra Pradesh',1:'Assam',2:'Bihar',3:'Chhattisgarh',4:'Delhi',5:'Gujarat',
    6:'Jharkhand',7:'Karnataka',8:'Kerala',9:'Madhya Pradesh',10:'Maharashtra',
    11:'Punjab',12:'Rajasthan',13:'Tamil Nadu',14:'Telangana',15:'Uttar Pradesh',
    16:'West Bengal'
}

INV_CATEGORY    = {v:k for k,v in CATEGORY_MAP.items()}
INV_TRANSACTION = {v:k for k,v in TRANSACTION_MAP.items()}
INV_CITY_TIER   = {v:k for k,v in CITY_TIER_MAP.items()}
INV_QUARTER     = {v:k for k,v in QUARTER_MAP.items()}
INV_CITY        = {v:k for k,v in CITY_MAP.items()}
INV_STATE       = {v:k for k,v in STATE_MAP.items()}

FEATURES = ['Year','Quarter','Category','State','City','Transaction_Type']

# ── Load data & model directly (no upload needed) ──────────────────────────────
@st.cache_data
def load_data():
    return pd.read_excel('cleaned_data (3).xlsx')

@st.cache_resource
def load_model():
    with open('RandomForestRegressor.pkl', 'rb') as f:
        return pickle.load(f)

df    = load_data()
model = load_model()


st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 700; color: #1F4E79;
        border-bottom: 3px solid #2E86AB; padding-bottom: 0.4rem; margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1F4E79 0%, #2E86AB 100%);
        border-radius: 12px; padding: 1.2rem; color: white; text-align: center;
    }
    .metric-value { font-size: 1.8rem; font-weight: 700; }
    .metric-label { font-size: 0.85rem; opacity: 0.85; margin-top: 4px; }
    .section-header { font-size: 1.1rem; font-weight: 600; color: #1F4E79; margin: 1rem 0 0.5rem; }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🛒 Retail Sales — Units Sold Forecasting</div>', unsafe_allow_html=True)

# ── Sidebar info only ──────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shop.png", width=60)
    st.markdown("## 📊 Dashboard Info")
    st.success(f"✅ Dataset loaded: **{len(df):,} rows**")
    st.success("✅ Model loaded: **RandomForestRegressor**")
    st.markdown("---")
    st.info("**Target:** Units Sold\n\n**Features:**\nYear, Quarter, Category,\nState, City, Transaction Type")
    st.markdown("---")
    st.markdown("**Forecast range:** 2022 – 2035")
    st.markdown("---")
    st.caption("Growth rate is calculated per city & quarter from training data")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈 Forecast Over Years", "🎯 Single Prediction", "📊 Data Overview"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Forecast Over Years
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Configure Forecast Parameters</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        sel_category = st.selectbox("🏷️ Category", list(INV_CATEGORY.keys()))
        sel_txn      = st.selectbox("💳 Transaction Type", list(INV_TRANSACTION.keys()))
    with col2:
        sel_city = st.selectbox("🏙️ City", list(INV_CITY.keys()))
        sel_tier = st.selectbox("🏢 City Tier", list(INV_CITY_TIER.keys()))
    with col3:
        sel_state      = st.selectbox("🗺️ State", list(INV_STATE.keys()))
        forecast_years = st.multiselect(
            "📅 Forecast Years",
            options=list(range(2022, 2036)),
            default=[2022, 2023, 2024, 2025, 2026]
        )

    if st.button("📈 Generate Forecast", use_container_width=True, type="primary"):
        if not forecast_years:
            st.error("Please select at least one year.")
        else:
            rows = []
            for yr in sorted(forecast_years):
                for q in range(4):
                    rows.append({
                        'Year':             yr,
                        'Quarter':          q,
                        'Category':         INV_CATEGORY[sel_category],
                        'State':            INV_STATE[sel_state],
                        'City':             INV_CITY[sel_city],
                        'Transaction_Type': INV_TRANSACTION[sel_txn],
                    })

            pred_df = pd.DataFrame(rows)
            base_preds = model.predict(pred_df[FEATURES])

            # City+Quarter specific: use actual avg units from training data
            # scaled by model's category/txn signal, then grown by city-quarter YoY rate
            TRAIN_MAX_YEAR = int(df['Year'].max())
            city_enc = INV_CITY[sel_city]

            # Step 1: actual avg units per city+quarter from training data
            city_qtr_actual = {}
            city_qtr_growth = {}
            for q in range(4):
                subset = df[(df['City'] == city_enc) & (df['Quarter'] == q)]
                if len(subset) > 0:
                    yavg = subset.groupby('Year')['Units_Sold'].mean().sort_index()
                    city_qtr_actual[q] = float(yavg.iloc[-1])  # last known year avg
                    rates = yavg.pct_change().dropna()
                    city_qtr_growth[q] = float(rates.mean()) if len(rates) > 0 else 0.05
                else:
                    city_qtr_actual[q] = float(df['Units_Sold'].mean())
                    city_qtr_growth[q] = 0.05

            # Step 2: model captures category+txn signal as a ratio vs overall mean
            overall_mean = float(df['Units_Sold'].mean())
            cat_txn_ratio = float(base_preds.mean()) / overall_mean if overall_mean > 0 else 1.0

            def get_prediction(year, quarter):
                base = city_qtr_actual.get(quarter, overall_mean) * cat_txn_ratio
                gap  = year - TRAIN_MAX_YEAR
                if gap <= 0:
                    return max(1, round(base))
                growth = city_qtr_growth.get(quarter, 0.05)
                return max(1, round(base * (1 + growth) ** gap))

            pred_df['Units_Sold_Predicted'] = [
                get_prediction(yr, q)
                for yr, q in zip(pred_df['Year'], pred_df['Quarter'])
            ]
            pred_df['Quarter_Label'] = pred_df['Quarter'].map(QUARTER_MAP)

            annual = pred_df.groupby('Year')['Units_Sold_Predicted'].sum().reset_index()
            annual.columns = ['Year', 'Total_Units']

            st.markdown("---")
            n_cols = min(len(forecast_years), 6)
            chunks = [forecast_years[i:i+n_cols] for i in range(0, len(forecast_years), n_cols)]
            for chunk in chunks:
                m_cols = st.columns(len(chunk))
                for i, yr in enumerate(chunk):
                    total = annual[annual['Year'] == yr]['Total_Units'].values
                    val   = f"{total[0]:,}" if len(total) else "—"
                    with m_cols[i]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{val}</div>
                            <div class="metric-label">📦 {yr} — Total Units</div>
                        </div>""", unsafe_allow_html=True)

            # Show city+quarter growth rates used
            st.markdown("---")
            st.markdown("**📊 Growth Rates Applied (City: {} | Per Quarter)**".format(sel_city))
            gr_cols = st.columns(4)
            for qi, qname in QUARTER_MAP.items():
                rate = city_qtr_growth.get(qi, 0.05)
                gr_cols[qi].metric(f"{qname} Growth", f"{rate*100:.2f}%")
            st.markdown("---")
            c1, c2 = st.columns(2)

            with c1:
                st.markdown("**📊 Quarterly Forecast by Year**")
                fig, ax = plt.subplots(figsize=(7, 4))
                colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(forecast_years)))
                for i, yr in enumerate(sorted(forecast_years)):
                    yr_data = pred_df[pred_df['Year'] == yr]
                    ax.plot(yr_data['Quarter_Label'], yr_data['Units_Sold_Predicted'],
                            marker='o', label=str(yr), color=colors[i], linewidth=2.2)
                ax.set_xlabel("Quarter")
                ax.set_ylabel("Units Sold")
                ax.set_title(f"{sel_category} | {sel_city} | {sel_txn}", fontsize=10, color='#1F4E79')
                ax.legend(title="Year", fontsize=7, ncol=2)
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
                ax.grid(axis='y', alpha=0.3)
                fig.tight_layout()
                st.pyplot(fig)

            with c2:
                st.markdown("**📈 Annual Total Units Sold**")
                fig2, ax2 = plt.subplots(figsize=(7, 4))
                bar_colors = ['#2E86AB' if y <= 2024 else '#E84855' for y in annual['Year']]
                bars = ax2.bar(annual['Year'].astype(str), annual['Total_Units'],
                               color=bar_colors, width=0.6, edgecolor='white', linewidth=1.2)
                for bar, val in zip(bars, annual['Total_Units']):
                    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                             f'{val:,}', ha='center', va='bottom', fontsize=7, fontweight='600')
                ax2.set_xlabel("Year")
                ax2.set_ylabel("Total Units Sold")
                ax2.set_title("Annual Forecast  (🔵 Historical | 🔴 Future)", fontsize=10, color='#1F4E79')
                ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
                plt.xticks(rotation=45, ha='right', fontsize=8)
                ax2.grid(axis='y', alpha=0.3)
                fig2.tight_layout()
                st.pyplot(fig2)

            st.markdown("**📋 Quarterly Forecast Table**")
            display_df = pred_df[['Year', 'Quarter_Label', 'Units_Sold_Predicted']].rename(columns={
                'Quarter_Label': 'Quarter', 'Units_Sold_Predicted': 'Predicted Units Sold'
            })
            st.dataframe(display_df.style.format({'Predicted Units Sold': '{:,}'}),
                         use_container_width=True, height=300)
            csv = display_df.to_csv(index=False).encode()
            st.download_button("⬇️ Download Forecast CSV", csv, "forecast.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Single Prediction
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Predict Units Sold for a Specific Record</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        p_year = st.selectbox("📅 Year", list(range(2022, 2036)), index=2)
        p_qtr  = st.selectbox("🗓️ Quarter", list(QUARTER_MAP.values()))
    with c2:
        p_cat  = st.selectbox("🏷️ Category", list(INV_CATEGORY.keys()), key="p_cat")
        p_txn  = st.selectbox("💳 Transaction Type", list(INV_TRANSACTION.keys()), key="p_txn")
    with c3:
        p_city  = st.selectbox("🏙️ City", list(INV_CITY.keys()), key="p_city")
        p_state = st.selectbox("🗺️ State", list(INV_STATE.keys()), key="p_state")

    if st.button("🔮 Predict", use_container_width=True, type="primary"):
        input_row = pd.DataFrame([{
            'Year':             p_year,
            'Quarter':          INV_QUARTER[p_qtr],
            'Category':         INV_CATEGORY[p_cat],
            'State':            INV_STATE[p_state],
            'City':             INV_CITY[p_city],
            'Transaction_Type': INV_TRANSACTION[p_txn],
        }])
        base_pred = float(model.predict(input_row)[0])
        TRAIN_MAX_YEAR = int(df['Year'].max())
        q_enc      = INV_QUARTER[p_qtr]
        city_enc_p = INV_CITY[p_city]

        # Use actual city+quarter avg from training data
        subset_p = df[(df['City'] == city_enc_p) & (df['Quarter'] == q_enc)]
        if len(subset_p) > 0:
            yavg_p   = subset_p.groupby('Year')['Units_Sold'].mean().sort_index()
            actual_p = float(yavg_p.iloc[-1])
            rates_p  = yavg_p.pct_change().dropna()
            growth_p = float(rates_p.mean()) if len(rates_p) > 0 else 0.05
        else:
            actual_p = float(df['Units_Sold'].mean())
            growth_p = 0.05

        # Scale by category+txn signal from model
        overall_mean_p = float(df['Units_Sold'].mean())
        cat_txn_ratio_p = base_pred / overall_mean_p if overall_mean_p > 0 else 1.0
        scaled_base = actual_p * cat_txn_ratio_p

        gap = p_year - TRAIN_MAX_YEAR
        multiplier = 1.0 if gap <= 0 else (1 + growth_p) ** gap
        prediction = max(1, int(round(scaled_base * multiplier)))

        st.markdown("---")
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{prediction:,}</div>
                <div class="metric-label">📦 Predicted Units Sold</div>
            </div>""", unsafe_allow_html=True)
        with mc2:
            st.markdown(f"""<div class="metric-card" style="background:linear-gradient(135deg,#1a6b4a,#28a16b)">
                <div class="metric-value">{p_year} {p_qtr}</div>
                <div class="metric-label">📅 Period</div>
            </div>""", unsafe_allow_html=True)
        with mc3:
            st.markdown(f"""<div class="metric-card" style="background:linear-gradient(135deg,#7b2d8b,#a44db5)">
                <div class="metric-value">{p_cat}</div>
                <div class="metric-label">🏷️ Category | {p_city}</div>
            </div>""", unsafe_allow_html=True)

        hist = df[
            (df['Category'] == INV_CATEGORY[p_cat]) &
            (df['Transaction_Type'] == INV_TRANSACTION[p_txn]) &
            (df['City'] == INV_CITY[p_city])
        ]['Units_Sold']
        if len(hist) > 0:
            avg  = int(hist.mean())
            diff = prediction - avg
            pct  = (diff / avg) * 100
            icon = "📈" if diff > 0 else "📉"
            st.info(f"{icon} Historical avg for this combination: **{avg:,} units**. "
                    f"Prediction is **{abs(pct):.1f}% {'above' if diff > 0 else 'below'}** average.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Data Overview
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)

    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.metric("Total Rows", f"{len(df):,}")
    with d2:
        st.metric("Years Covered", f"{df['Year'].min()} – {df['Year'].max()}")
    with d3:
        st.metric("Avg Units Sold", f"{df['Units_Sold'].mean():.0f}")
    with d4:
        st.metric("Model", "RandomForestRegressor")

    st.markdown("---")
    ca, cb = st.columns(2)

    with ca:
        st.markdown("**Units Sold by Category**")
        cat_group = df.groupby('Category')['Units_Sold'].sum().reset_index()
        cat_group['Category_Name'] = cat_group['Category'].map(CATEGORY_MAP)
        fig3, ax3 = plt.subplots(figsize=(5, 3.5))
        ax3.barh(cat_group['Category_Name'], cat_group['Units_Sold'],
                 color=['#1F4E79','#2E86AB','#52B2CF','#85C1E9','#AED6F1'])
        ax3.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x/1000)}K'))
        ax3.set_xlabel("Total Units Sold")
        ax3.grid(axis='x', alpha=0.3)
        fig3.tight_layout()
        st.pyplot(fig3)

    with cb:
        st.markdown("**Units Sold by Transaction Type**")
        txn_group = df.groupby('Transaction_Type')['Units_Sold'].sum().reset_index()
        txn_group['Txn_Name'] = txn_group['Transaction_Type'].map(TRANSACTION_MAP)
        fig4, ax4 = plt.subplots(figsize=(5, 3.5))
        ax4.pie(txn_group['Units_Sold'], labels=txn_group['Txn_Name'],
                autopct='%1.1f%%', colors=['#1F4E79','#2E86AB','#52B2CF'],
                startangle=90, wedgeprops={'edgecolor':'white','linewidth':2})
        fig4.tight_layout()
        st.pyplot(fig4)

    st.markdown("**📋 Sample Data (first 100 rows)**")
    display_sample = df.head(100).copy()
    for col, mapping in [('Category', CATEGORY_MAP), ('Transaction_Type', TRANSACTION_MAP),
                         ('City_Tier', CITY_TIER_MAP), ('City', CITY_MAP),
                         ('State', STATE_MAP), ('Quarter', QUARTER_MAP)]:
        if col in display_sample.columns:
            display_sample[col] = display_sample[col].map(mapping)
    st.dataframe(display_sample, use_container_width=True, height=300)
