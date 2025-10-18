import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import joblib
from tensorflow import keras
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="🏎️ F1 Prediction Dashboard",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
    color: white;
}
h1, h2, h3 {
    color: #FF1801;
    font-weight: bold;
}
.main-header {
    background: linear-gradient(135deg, #FF1801 0%, #8B0000 100%);
    padding: 2rem;
    border-radius: 15px;
    text-align: center;
    border: 2px solid #FF6B6B;
    margin-bottom: 1.5rem;
}
.prediction-card, .driver-card, .constructor-card {
    background: #2d2d44;
    border-radius: 15px;
    padding: 1.5rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    margin-bottom: 1.2rem;
}
.metric-highlight {
    background: linear-gradient(135deg, #FF1801 0%, #FF6B6B 100%);
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    color: white;
    font-weight: bold;
}
.stTabs [data-baseweb="tab-list"] button {
    background-color: #2d2d44;
    color: white;
    border-radius: 10px;
    margin: 4px;
}
.stTabs [aria-selected="true"] {
    background-color: #FF1801 !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD DATA AND MODELS
# ============================================================================
@st.cache_resource
def load_resources():
    resources = {}
    try:
        # Load datasets
        for key, file in {
            'final_df': 'f1_dashboard.csv',
            'driver_stats': 'driver_season_stats.csv',
            'constructor_stats': 'constructor_season_stats.csv'
        }.items():
            if os.path.exists(file):
                resources[key] = pd.read_csv(file)
            else:
                st.warning(f"⚠️ Missing file: {file}")
                resources[key] = pd.DataFrame()

        # Load model components
        model_base = "models/f1_models_20251018_230123"
        if not os.path.exists(model_base):
            st.error("Model directory not found.")
            return resources

        scaler_path = f"{model_base}/scalers_encoders/feature_scaler.pkl"
        feature_names_path = f"{model_base}/scalers_encoders/feature_names.pkl"
        resources['scaler'] = joblib.load(scaler_path)
        resources['feature_names'] = joblib.load(feature_names_path)

        # Load models
        model_files = {
            'rf_winner': f"{model_base}/sklearn_models/rf_winner.pkl",
            'gb_winner': f"{model_base}/sklearn_models/gb_winner.pkl",
            'rf_points': f"{model_base}/sklearn_models/rf_points.pkl"
        }
        for key, path in model_files.items():
            resources[key] = joblib.load(path)

        # Load deep learning models (optional)
        try:
            resources['nn_winner'] = keras.models.load_model(f"{model_base}/deep_learning/nn_winner_model.h5")
        except:
            resources['nn_winner'] = None

        return resources

    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return resources


resources = load_resources()

# Extract with safe fallbacks
final_df = resources.get('final_df') if resources.get('final_df') is not None else pd.DataFrame()
driver_stats = resources.get('driver_stats') if resources.get('driver_stats') is not None else pd.DataFrame()
constructor_stats = resources.get('constructor_stats') if resources.get('constructor_stats') is not None else pd.DataFrame()

scaler = resources.get('scaler')
rf_winner = resources.get('rf_winner')
gb_winner = resources.get('gb_winner')
rf_points = resources.get('rf_points')
nn_winner = resources.get('nn_winner')

if not final_df.empty:
    final_df['raceDate'] = pd.to_datetime(final_df['raceDate'], errors='coerce')
    final_df['year'] = final_df['year'].astype(int, errors='ignore')

# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.markdown("## 🏎️ F1 Predictor")
page = st.sidebar.radio("Navigate", ["🏠 Dashboard", "📊 Predictor", "👥 Drivers", "🏭 Constructors", "📈 Analytics"])

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def create_prediction_features(grid, qual, pit, avg_lap, consistency, pit_dur, dw, dp, pts, cw, cp, fr):
    return np.array([[
        grid, qual, grid - qual, avg_lap, consistency, pit, pit_dur,
        dw, dp, pts, cw, cp, fr
    ]])

def radar_chart(labels, values, title):
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values, theta=labels, fill='toself', name='Performance'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False, title=title, template='plotly_dark')
    return fig

# ============================================================================
# DASHBOARD PAGE
# ============================================================================
if page == "🏠 Dashboard":
    st.markdown("<div class='main-header'><h1>Formula 1 AI Dashboard</h1></div>", unsafe_allow_html=True)

    if final_df.empty:
        st.warning("Upload or connect your F1 dataset.")
    else:
        total_races = final_df['raceId'].nunique()
        total_drivers = final_df['driverRef'].nunique()
        total_wins = final_df['isWin'].sum()
        avg_points = final_df['points'].mean()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Races", total_races)
        col2.metric("Drivers", total_drivers)
        col3.metric("Wins", int(total_wins))
        col4.metric("Avg Points", f"{avg_points:.2f}")

        # Top Drivers Chart
        top_drivers = final_df.groupby('driverRef')['isWin'].sum().nlargest(10)
        fig = px.bar(top_drivers, x=top_drivers.values, y=top_drivers.index,
                     orientation='h', color=top_drivers.values,
                     title="Top 10 Drivers (Wins)", color_continuous_scale="Reds")
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# DRIVER ANALYTICS PAGE
# ============================================================================
elif page == "👥 Drivers":
    st.markdown("<div class='main-header'><h1>Driver Performance</h1></div>", unsafe_allow_html=True)

    if driver_stats.empty or final_df.empty:
        st.warning("Driver data not found.")
    else:
        search = st.text_input("🔍 Search Driver (type partial name)", "")
        matched_drivers = [d for d in sorted(final_df['driverRef'].unique()) if search.lower() in d.lower()]

        selected_driver = st.selectbox("Select Driver", matched_drivers)
        data = final_df[final_df['driverRef'] == selected_driver]

        if not data.empty:
            col1, col2 = st.columns(2)
            with col1:
                radar_labels = ["Wins", "Podiums", "Points", "Finish Rate", "Qualifying Avg"]
                radar_values = [
                    data['isWin'].mean() * 100,
                    data['isPodium'].mean() * 100,
                    data['points'].mean(),
                    (data['isFinished'].mean() * 100),
                    data['grid'].mean()
                ]
                st.plotly_chart(radar_chart(radar_labels, radar_values, f"{selected_driver} - Radar Performance"), use_container_width=True)

            with col2:
                points_by_season = data.groupby('year')['points'].sum()
                fig = px.line(points_by_season, x=points_by_season.index, y=points_by_season.values,
                              title="Season Points Trend", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# CONSTRUCTORS PAGE
# ============================================================================
elif page == "🏭 Constructors":
    st.markdown("<div class='main-header'><h1>Team Analytics</h1></div>", unsafe_allow_html=True)
    if constructor_stats.empty:
        st.warning("Constructor data missing.")
    else:
        team = st.selectbox("Select Team", sorted(final_df['constructorRef'].unique()))
        team_data = final_df[final_df['constructorRef'] == team]
        if not team_data.empty:
            win_rate = team_data['isWin'].mean() * 100
            podium_rate = team_data['isPodium'].mean() * 100
            avg_points = team_data['points'].mean()

            col1, col2, col3 = st.columns(3)
            col1.metric("Win Rate", f"{win_rate:.1f}%")
            col2.metric("Podium Rate", f"{podium_rate:.1f}%")
            col3.metric("Avg Points", f"{avg_points:.1f}")

            trend = team_data.groupby('year')['points'].sum()
            fig = px.line(trend, x=trend.index, y=trend.values, title=f"{team} - Season Points", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# ADVANCED ANALYTICS PAGE (REDESIGNED)
# ============================================================================
elif page == "📈 Analytics":
    st.markdown("<div class='main-header'><h1>Advanced F1 Analytics</h1></div>", unsafe_allow_html=True)

    if final_df.empty:
        st.warning("Data not available.")
    else:
        tab1, tab2, tab3 = st.tabs(["📊 Correlation Heatmap", "⚙️ Lap Consistency", "🚨 DNF Trends"])

        with tab1:
            numeric_cols = final_df.select_dtypes(include=np.number)
            corr = numeric_cols.corr()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
            st.pyplot(fig)

        with tab2:
            st.markdown("### Lap Time Consistency vs. Finishing Position")
            fig = px.scatter(final_df, x="lapTime_std" if "lapTime_std" in final_df.columns else "grid",
                             y="positionOrder", color="points",
                             title="Lap Consistency vs Finish",
                             template="plotly_dark", color_continuous_scale="Viridis")
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            dnf_trend = final_df.groupby('year')['isDNF'].mean() * 100
            fig = px.bar(dnf_trend, x=dnf_trend.index, y=dnf_trend.values,
                         title="DNF Rate per Season", color=dnf_trend.values, color_continuous_scale="Reds")
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("<p style='text-align:center;color:gray;'>🏎️ Formula 1 AI Dashboard • Human-Crafted • Data Science Powered</p>", unsafe_allow_html=True)
