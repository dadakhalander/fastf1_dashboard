# app.py - polished F1 Prediction Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
# PAGE CONFIG & GLOBAL CSS
# ============================================================================
st.set_page_config(
    page_title="F1 Race Prediction Dashboard",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        color: #ffffff;
    }
    .main-header {
        background: linear-gradient(135deg, #FF1801 0%, #8B0000 100%);
        padding: 1.4rem;
        border-radius: 12px;
        margin-bottom: 1.2rem;
        text-align: center;
        border: 2px solid #FF6B6B;
    }
    .metric-highlight {
        background: linear-gradient(135deg, #FF1801 0%, #FF6B6B 100%);
        padding: 12px;
        border-radius: 8px;
        text-align: center;
        color: white;
        font-weight: bold;
        margin: 4px;
    }
    h1, h2, h3 { color: #FF1801; font-weight: 700; }
    .stTabs [data-baseweb="tab-list"] button { background-color: #2d2d44; color: white; border-radius: 10px; padding: 10px 18px; margin: 6px; border: 1px solid #444; }
    .stTabs [aria-selected="true"] { background-color: #FF1801 !important; color: white !important; }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SAFE RESOURCE LOADER
# ============================================================================
@st.cache_resource
def load_resources(data_dir=".", model_base="models/f1_models_20251018_230123"):
    """
    Load CSVs and models where available. Never raise — return dict with None for missing items.
    """
    resources = {}
    # CSVs
    csv_map = {
        'final_df': os.path.join(data_dir, 'f1_dashboard.csv'),
        'driver_stats': os.path.join(data_dir, 'driver_season_stats.csv'),
        'constructor_stats': os.path.join(data_dir, 'constructor_season_stats.csv')
    }
    for k, p in csv_map.items():
        try:
            if os.path.exists(p):
                resources[k] = pd.read_csv(p)
            else:
                resources[k] = None
        except Exception:
            resources[k] = None

    # Scaler + feature names
    sc_path = os.path.join(model_base, 'scalers_encoders', 'feature_scaler.pkl')
    fn_path = os.path.join(model_base, 'scalers_encoders', 'feature_names.pkl')
    try:
        resources['scaler'] = joblib.load(sc_path) if os.path.exists(sc_path) else None
    except Exception:
        resources['scaler'] = None
    try:
        resources['feature_names'] = joblib.load(fn_path) if os.path.exists(fn_path) else None
    except Exception:
        resources['feature_names'] = None

    # sklearn models
    for name in ['rf_winner', 'gb_winner', 'rf_points']:
        p = os.path.join(model_base, 'sklearn_models', f'{name}.pkl')
        try:
            resources[name] = joblib.load(p) if os.path.exists(p) else None
        except Exception:
            resources[name] = None

    # keras models (optional)
    for name in ['nn_winner', 'nn_podium', 'nn_points']:
        p = os.path.join(model_base, 'deep_learning', f'{name}_model.h5')
        try:
            resources[name] = keras.models.load_model(p) if os.path.exists(p) else None
        except Exception:
            resources[name] = None

    # metadata
    mdp = os.path.join(model_base, 'metadata', 'model_metadata.pkl')
    try:
        resources['metadata'] = joblib.load(mdp) if os.path.exists(mdp) else {}
    except Exception:
        resources['metadata'] = {}

    return resources

resources = load_resources()

# Extract items with safe fallbacks
final_df = resources.get('final_df') or pd.DataFrame()
driver_stats = resources.get('driver_stats') or pd.DataFrame()
constructor_stats = resources.get('constructor_stats') or pd.DataFrame()
scaler = resources.get('scaler')
feature_names = resources.get('feature_names')
rf_winner = resources.get('rf_winner')
gb_winner = resources.get('gb_winner')
rf_points = resources.get('rf_points')
nn_winner = resources.get('nn_winner')
nn_podium = resources.get('nn_podium')
nn_points = resources.get('nn_points')
metadata = resources.get('metadata', {})

# If final_df empty, warn user (UI still runs)
if final_df.empty:
    st.warning("Dataset 'f1_dashboard.csv' not loaded or empty. UI will run but data visuals are disabled until CSV is added.")

# ============================================================================
# DATA SAFETY NORMALIZATION (defensive)
# ============================================================================
def safe_col(df, col, default):
    if col in df.columns:
        return df[col]
    else:
        return pd.Series([default] * len(df), index=df.index)

if not final_df.empty:
    # parse dates & numeric columns defensively
    if 'raceDate' in final_df.columns:
        final_df['raceDate'] = pd.to_datetime(final_df['raceDate'], errors='coerce')
    if 'year' in final_df.columns:
        final_df['year'] = pd.to_numeric(final_df['year'], errors='coerce').fillna(0).astype(int)
    # flags: isFinished/isDNF/isWin/isPodium/points/grid
    final_df['positionOrder'] = pd.to_numeric(final_df.get('positionOrder'), errors='coerce')
    final_df['isFinished'] = final_df.get('isFinished')
    if final_df['isFinished'].isnull().all():
        final_df['isFinished'] = final_df['positionOrder'].notna().astype(int)
    final_df['isDNF'] = final_df.get('isDNF')
    if final_df['isDNF'].isnull().all():
        final_df['isDNF'] = (~final_df['isFinished'].astype(bool)).astype(int)
    final_df['isWin'] = final_df.get('isWin')
    if final_df['isWin'].isnull().all():
        final_df['isWin'] = (final_df['positionOrder'] == 1).astype(int)
    final_df['isPodium'] = final_df.get('isPodium')
    if final_df['isPodium'].isnull().all():
        final_df['isPodium'] = final_df['positionOrder'].isin([1,2,3]).astype(int)

    final_df['points'] = pd.to_numeric(final_df.get('points'), errors='coerce').fillna(0)
    if 'grid' in final_df.columns:
        final_df['grid'] = pd.to_numeric(final_df['grid'], errors='coerce').fillna(-1).astype(int)
    else:
        final_df['grid'] = -1

# ============================================================================
# UTILITY: safe predict wrappers & feature builder
# ============================================================================
def safe_predict_proba(model, X):
    """Return probability for positive class if available, else try predict or return NaN"""
    try:
        if hasattr(model, 'predict_proba'):
            p = model.predict_proba(X)
            # second column usually positive class if classifier
            return float(p[0][1]) if p.shape[1] > 1 else float(p[0][0])
        else:
            # fallback to predict (may be 0/1 or continuous)
            return float(model.predict(X)[0])
    except Exception:
        return float('nan')

def safe_predict_reg(model, X):
    try:
        return float(model.predict(X)[0])
    except Exception:
        return float('nan')

def build_feature_array(inputs_map):
    """
    Build feature DataFrame/array for models.
    If saved feature_names exists, use it (fill missing with 0).
    Otherwise fallback to a sensible local ordering.
    inputs_map: dict of intuitive inputs (keys we set in the UI)
    """
    if feature_names and isinstance(feature_names, (list, tuple)):
        # create a DataFrame row with columns in feature_names order
        row = {f: inputs_map.get(f, 0.0) for f in feature_names}
        X_df = pd.DataFrame([row], columns=feature_names)
    else:
        # fallback ordering (must match training if you didn't save feature_names)
        fallback_order = [
            'grid', 'qual_pos', 'grid_minus_qual', 'avg_lap_ms', 'lap_std_ms',
            'pit_stops', 'pit_dur_sec', 'driver_wins', 'driver_podiums',
            'driver_points', 'constructor_wins', 'constructor_points', 'finish_rate_pct'
        ]
        row = {f: inputs_map.get(f, 0.0) for f in fallback_order}
        X_df = pd.DataFrame([row], columns=fallback_order)

    # scaling if scaler present
    if scaler is not None:
        try:
            X_scaled = scaler.transform(X_df.values.astype(float))
            return X_scaled, X_df
        except Exception:
            # fallback to raw numeric values if scaler fails
            return X_df.values.astype(float), X_df
    else:
        return X_df.values.astype(float), X_df

# ============================================================================
# SIDEBAR / NAVIGATION
# ============================================================================
st.sidebar.markdown("""
<div style='text-align:center'>
  <h2>🏎️ F1 PREDICTOR</h2>
  <div style='font-size:12px; color:#bbb'>AI-assisted race analytics</div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "NAVIGATION",
    ["🏠 Dashboard", "📊 Race Predictor", "👥 Drivers", "🏭 Constructors", "📈 Analytics", "🏆 Championships"],
    index=0
)

st.sidebar.markdown("---")
# quick metrics
if not final_df.empty:
    try:
        st.sidebar.metric("Races", f"{final_df['raceId'].nunique():,}")
        st.sidebar.metric("Drivers", f"{final_df['driverRef'].nunique():,}")
        st.sidebar.metric("Seasons", f"{final_df['year'].nunique():,}")
    except Exception:
        pass

# model status
models_active = sum([1 for m in (nn_winner, rf_winner, gb_winner) if m is not None])
st.sidebar.markdown("---")
st.sidebar.write(f"Models active: **{models_active}** (NN / RF / GB)")

# ============================================================================
# DASHBOARD PAGE
# ============================================================================
if page == "🏠 Dashboard":
    st.markdown("<div class='main-header'><h1>FORMULA 1 PREDICTION DASHBOARD</h1><h3>AI-Powered Race Analytics</h3></div>", unsafe_allow_html=True)

    if final_df.empty:
        st.info("Load `f1_dashboard.csv` to show data-driven metrics.")
    else:
        # Season overview metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"<div class='metric-highlight'><div style='font-size:20px'>🏆</div><div>Total Wins</div><div style='font-size:20px'>{int(final_df['isWin'].sum())}</div></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-highlight'><div style='font-size:20px'>💰</div><div>Points</div><div style='font-size:20px'>{int(final_df['points'].sum())}</div></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='metric-highlight'><div style='font-size:20px'>🥇</div><div>Podiums</div><div style='font-size:20px'>{int(final_df['isPodium'].sum())}</div></div>", unsafe_allow_html=True)
        with col4:
            finish_rate = final_df['isFinished'].mean() * 100 if len(final_df) > 0 else 0
            st.markdown(f"<div class='metric-highlight'><div style='font-size:20px'>✅</div><div>Finish Rate</div><div style='font-size:20px'>{finish_rate:.1f}%</div></div>", unsafe_allow_html=True)

        st.markdown("---")
        # top drivers and top constructors charts
        col1, col2 = st.columns(2)
        with col1:
            top_drivers = final_df.groupby('driverRef').agg({'isWin':'sum'}).sort_values('isWin', ascending=False).head(10)
            if not top_drivers.empty:
                fig = px.bar(top_drivers, x='isWin', y=top_drivers.index, orientation='h', title="Top Wins (All-time)", color='isWin', color_continuous_scale='Reds')
                fig.update_layout(template='plotly_dark', height=420, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            top_ctors = final_df.groupby('constructorRef').agg({'points':'sum'}).sort_values('points', ascending=False).head(10)
            if not top_ctors.empty:
                fig = px.bar(top_ctors, x='points', y=top_ctors.index, orientation='h', title="Top Constructors (Points)", color='points', color_continuous_scale='Blues')
                fig.update_layout(template='plotly_dark', height=420, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# RACE PREDICTOR PAGE
# ============================================================================
elif page == "📊 Race Predictor":
    st.markdown("<div class='main-header'><h1>RACE PREDICTION ENGINE</h1><h3>Enter conditions to produce predictions</h3></div>", unsafe_allow_html=True)

    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Race Inputs")
        grid_position = st.number_input("Starting Grid", 1, 20, 5)
        qual_position = st.number_input("Qualifying Position", 1, 20, 3)
        pit_stops = st.number_input("Planned Pit Stops", 0, 6, 2)
        avg_lap_time = st.number_input("Avg Lap Time (ms)", 50000, 200000, 90000)
        lap_time_consistency = st.number_input("Lap Time Std (ms)", 0, 3000, 500)
        pit_stop_duration = st.number_input("Pit Stop Time (s)", 5.0, 60.0, 22.5)
    with col2:
        st.subheader("Driver / Team Profile")
        driver_wins = st.number_input("Career Wins", 0, 300, 15)
        driver_podiums = st.number_input("Career Podiums", 0, 1000, 45)
        driver_points = st.number_input("Season Points", 0, 5000, 180)
        constructor_wins = st.number_input("Constructor Wins", 0, 1000, 80)
        constructor_points = st.number_input("Constructor Points", 0, 10000, 350)
        finish_rate_pct = st.number_input("Finish Rate %", 0.0, 100.0, 85.0)

    st.markdown("### Choose models")
    use_rf = st.checkbox("Random Forest", True)
    use_gb = st.checkbox("Gradient Boosting", True)
    use_nn = st.checkbox("Neural Network (if loaded)", nn_winner is not None)

    if st.button("🚀 Predict"):
        # build inputs_map (keys should match feature names if saved)
        inputs_map = {
            'grid': grid_position,
            'qual_pos': qual_position,
            'grid_minus_qual': grid_position - qual_position,
            'avg_lap_ms': avg_lap_time,
            'lap_std_ms': lap_time_consistency,
            'pit_stops': pit_stops,
            'pit_dur_sec': pit_stop_duration,
            'driver_wins': driver_wins,
            'driver_podiums': driver_podiums,
            'driver_points': driver_points,
            'constructor_wins': constructor_wins,
            'constructor_points': constructor_points,
            'finish_rate_pct': finish_rate_pct
        }

        X_scaled, X_df = build_feature_array(inputs_map)

        # generate predictions
        preds = {}
        # NN
        if use_nn and nn_winner is not None:
            try:
                p_win = float(nn_winner.predict(X_scaled, verbose=0).ravel()[0])
                p_podium = float(nn_podium.predict(X_scaled, verbose=0).ravel()[0]) if nn_podium else p_win * 0.7
                p_pts = float(nn_points.predict(X_scaled, verbose=0).ravel()[0]) if nn_points else p_win * 25
                preds['Neural Network'] = {'win': p_win, 'podium': p_podium, 'points': p_pts}
            except Exception as e:
                st.warning(f"NN prediction failed: {e}")
        # RF
        if use_rf and rf_winner is not None:
            p = safe_predict_proba(rf_winner, X_scaled)
            pts = safe_predict_reg(rf_points, X_scaled) if rf_points is not None else float('nan')
            preds['Random Forest'] = {'win': p, 'podium': p * 0.7 if not np.isnan(p) else float('nan'), 'points': pts}
        # GB
        if use_gb and gb_winner is not None:
            p = safe_predict_proba(gb_winner, X_scaled)
            preds['Gradient Boosting'] = {'win': p, 'podium': p * 0.7 if not np.isnan(p) else float('nan'), 'points': p * 25 if not np.isnan(p) else float('nan')}

        if not preds:
            st.warning("No predictions available. Models may not be loaded.")
        else:
            # ensemble
            wins = [v['win'] for v in preds.values() if not np.isnan(v.get('win', np.nan))]
            podiums = [v['podium'] for v in preds.values() if not np.isnan(v.get('podium', np.nan))]
            points = [v['points'] for v in preds.values() if not np.isnan(v.get('points', np.nan))]
            ensemble = {'win': np.mean(wins) if wins else float('nan'),
                        'podium': np.mean(podiums) if podiums else float('nan'),
                        'points': np.mean(points) if points else float('nan')}

            # show metrics
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Win Probability", f"{ensemble['win']:.1%}" if not np.isnan(ensemble['win']) else "N/A")
            with c2:
                st.metric("Podium Probability", f"{ensemble['podium']:.1%}" if not np.isnan(ensemble['podium']) else "N/A")
            with c3:
                st.metric("Expected Points", f"{ensemble['points']:.1f}" if not np.isnan(ensemble['points']) else "N/A")

            st.markdown("#### Model Breakdown")
            df_comp = pd.DataFrame([
                {'Model': m, 'Win%': v['win']*100 if not np.isnan(v.get('win', np.nan)) else np.nan,
                 'Podium%': v['podium']*100 if not np.isnan(v.get('podium', np.nan)) else np.nan,
                 'Points': v['points']} for m,v in preds.items()
            ]).set_index('Model')
            st.dataframe(df_comp.style.format({'Win%':'{:.1f}','Podium%':'{:.1f}','Points':'{:.1f}'}), use_container_width=True)

# ============================================================================
# DRIVERS PAGE (refined)
# ============================================================================
elif page == "👥 Drivers":
    st.markdown("<div class='main-header'><h1>Driver Analysis</h1><h3>Interactive, readable and compact driver insights</h3></div>", unsafe_allow_html=True)
    if final_df.empty:
        st.info("No data available. Upload 'f1_dashboard.csv' to use driver analysis.")
    else:
        current_year = final_df['year'].max()
        active_drivers = sorted(final_df[final_df['year'] == current_year]['driverRef'].unique())
        all_drivers = sorted(final_df['driverRef'].unique())

        show_all = st.checkbox("Show Historical Drivers (include retired)", value=False)
        search = st.text_input("Search driver (partial name/code)", value="")
        driver_list = all_drivers if show_all else active_drivers

        if search:
            driver_list = [d for d in driver_list if search.lower() in str(d).lower()]

        # limit list length for UX but allow "show more"
        if len(driver_list) > 80 and not show_all:
            st.info("Showing top active drivers. Toggle 'Show Historical Drivers' to expand.")
            driver_list = driver_list[:40]

        selected_driver = st.selectbox("Select driver", driver_list)

        if selected_driver:
            d_df = final_df[final_df['driverRef'] == selected_driver].sort_values(['year','raceDate'])
            # basic metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Career Wins", int(d_df['isWin'].sum()))
            with col2: st.metric("Career Podiums", int(d_df['isPodium'].sum()))
            with col3: st.metric("Career Points", int(d_df['points'].sum()))
            with col4: st.metric("Races", len(d_df))

            st.markdown("---")
            # Radar: normalized metrics for readability
            wins = d_df['isWin'].sum()
            podiums = d_df['isPodium'].sum()
            avg_points = d_df['points'].mean() if len(d_df) > 0 else 0
            finish_rate = d_df['isFinished'].mean() * 100 if len(d_df) > 0 else 0
            consistency = (1.0 / d_df['positionOrder'].std()) * 10 if d_df['positionOrder'].std() and d_df['positionOrder'].std() > 0 else 10

            radar_metrics = pd.DataFrame({
                'metric': ['Wins','Podiums','Avg Points','Finish Rate','Consistency'],
                'value': [wins, podiums, avg_points, finish_rate, consistency]
            })
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=radar_metrics['value'], theta=radar_metrics['metric'], fill='toself', name=selected_driver,
                line=dict(color="#FF1801", width=2)
            ))
            max_r = max(1, radar_metrics['value'].max() * 1.2)
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, max_r])), template='plotly_dark', height=420)
            st.plotly_chart(fig_radar, use_container_width=True)

            st.markdown("### Season Trends")
            c1, c2 = st.columns(2)
            with c1:
                pts_by_season = d_df.groupby('year')['points'].sum()
                if not pts_by_season.empty:
                    fig = px.bar(x=pts_by_season.index, y=pts_by_season.values, labels={'x':'Year','y':'Points'}, title="Points by Season")
                    fig.update_layout(template='plotly_dark', height=300)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No seasonal points to show.")
            with c2:
                avgpos = d_df.groupby('year')['positionOrder'].mean()
                if not avgpos.empty:
                    fig = px.line(x=avgpos.index, y=avgpos.values, labels={'x':'Year','y':'Avg Finish'}, title="Average Finish by Season")
                    fig.update_layout(template='plotly_dark', height=300)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No positional data to show.")

            st.markdown("### Recent Races")
            cols = ['year','raceName','grid','positionOrder','points','isWin','isPodium']
            available_cols = [c for c in cols if c in d_df.columns]
            st.dataframe(d_df[available_cols].sort_values('raceDate', ascending=False).head(12), use_container_width=True)

# ============================================================================
# CONSTRUCTORS PAGE (polished)
# ============================================================================
elif page == "🏭 Constructors":
    st.markdown("<div class='main-header'><h1>Constructor Analysis</h1><h3>Team performance & history</h3></div>", unsafe_allow_html=True)
    if final_df.empty:
        st.info("No data available.")
    else:
        all_ctors = sorted(final_df['constructorRef'].unique())
        selected_ctor = st.selectbox("Choose constructor", all_ctors)
        if selected_ctor:
            t_df = final_df[final_df['constructorRef'] == selected_ctor]
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Team Wins", int(t_df['isWin'].sum()))
            with col2: st.metric("Team Points", int(t_df['points'].sum()))
            with col3:
                wr = (t_df['isWin'].sum() / len(t_df) * 100) if len(t_df) > 0 else 0
                st.metric("Win Rate", f"{wr:.1f}%")
            with col4: st.metric("Seasons", t_df['year'].nunique())
            st.markdown("---")
            # points by season area
            pts = t_df.groupby('year')['points'].sum()
            if not pts.empty:
                fig = px.area(x=pts.index, y=pts.values, labels={'x':'Year','y':'Points'}, title="Points by Season")
                fig.update_layout(template='plotly_dark', height=380)
                st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# ANALYTICS PAGE (redesigned)
# ============================================================================
elif page == "📈 Analytics":
    st.markdown("<div class='main-header'><h1>Advanced Analytics</h1><h3>Grid behavior, pit strategies, and head-to-head comparisons</h3></div>", unsafe_allow_html=True)
    if final_df.empty:
        st.info("No data available.")
    else:
        tab1, tab2, tab3 = st.tabs(["🏁 Grid vs Finish", "🛠️ Pit Stop Analysis", "🎯 Driver Head-to-Head"])

        # Tab 1: Grid vs Finish
        with tab1:
            st.markdown("### Grid Position vs Finishing Position")
            min_year = int(final_df['year'].min())
            max_year = int(final_df['year'].max())
            year = st.slider("Year", min_year, max_year, max_year)
            dfy = final_df[final_df['year'] == year].dropna(subset=['grid','positionOrder'])
            if dfy.empty:
                st.info("No data for selected year.")
            else:
                fig = px.scatter(dfy, x='grid', y='positionOrder', size='points', color='constructorRef',
                                 hover_data=['driverRef','raceName'], trendline='ols',
                                 title=f"Grid → Finish relationship ({year})")
                fig.update_layout(template='plotly_dark', height=520)
                st.plotly_chart(fig, use_container_width=True)

        # Tab 2: Pit Stop Analysis
        with tab2:
            st.markdown("### Pit Stops vs Points")
            if 'pitStops' not in final_df.columns:
                st.info("No pitStops column found in data.")
            else:
                dfpit = final_df.copy()
                dfpit = dfpit.dropna(subset=['pitStops', 'points'])
                if dfpit.empty:
                    st.info("No usable pit-stop data.")
                else:
                    fig = px.scatter(dfpit, x='pitStops', y='points', color='constructorRef', size='points',
                                     trendline='ols', hover_data=['driverRef','raceName'],
                                     title="Pit Stop Count vs Points")
                    fig.update_layout(template='plotly_dark', height=520)
                    st.plotly_chart(fig, use_container_width=True)

        # Tab 3: Driver Head-to-Head
        with tab3:
            st.markdown("### Driver Consistency Comparison")
            drivers_unique = sorted(final_df['driverRef'].unique())
            da = st.selectbox("Driver A", drivers_unique, index=0)
            db = st.selectbox("Driver B", drivers_unique, index=min(1, len(drivers_unique)-1))
            def driver_metrics(d):
                df = final_df[final_df['driverRef'] == d]
                if df.empty:
                    return {'Avg Finish':0, 'Finish Rate':0, 'Consistency':0, 'Pts/Race':0, 'Podium Rate':0}
                avg_finish = df['positionOrder'].mean() if 'positionOrder' in df.columns else np.nan
                finish_rate = df['isFinished'].mean() * 100
                consistency = (1.0/(df['positionOrder'].std()))*10 if df['positionOrder'].std() and df['positionOrder'].std() > 0 else 10
                pts_per_race = df['points'].mean()
                podium_rate = df['isPodium'].mean() * 100
                return {'Avg Finish': (21 - avg_finish) if not np.isnan(avg_finish) else 0, 'Finish Rate': finish_rate,
                        'Consistency': consistency, 'Pts/Race': pts_per_race, 'Podium Rate': podium_rate}
            ma = driver_metrics(da)
            mb = driver_metrics(db)
            metrics = list(ma.keys())
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=list(ma.values()), theta=metrics, fill='toself', name=da))
            fig.add_trace(go.Scatterpolar(r=list(mb.values()), theta=metrics, fill='toself', name=db))
            max_val = max(max(ma.values()), max(mb.values())) * 1.2
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, max(1, max_val)])), template='plotly_dark', height=520,
                              title=f"{da} vs {db} - Performance Radar")
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# CHAMPIONSHIPS PAGE
# ============================================================================
elif page == "🏆 Championships":
    st.markdown("<div class='main-header'><h1>Championships</h1><h3>Seasonal standings & history</h3></div>", unsafe_allow_html=True)
    if final_df.empty:
        st.info("No data available.")
    else:
        seasons = sorted(final_df['year'].unique(), reverse=True)
        sel = st.selectbox("Select season", seasons, index=0)
        sdata = final_df[final_df['year'] == sel]
        if sdata.empty:
            st.info("No races for selected season.")
        else:
            standings = sdata.groupby('driverRef').agg({'points':'sum','isWin':'sum','isPodium':'sum','raceId':'count'}).sort_values('points', ascending=False)
            st.markdown(f"### {sel} Top 10")
            for i,(driver,row) in enumerate(standings.head(10).iterrows(), 1):
                emoji = "🥇" if i==1 else "🥈" if i==2 else "🥉" if i==3 else f"{i}."
                col1, col2, col3, col4 = st.columns([2,1,1,1])
                with col1: st.write(f"**{emoji} {driver}**")
                with col2: st.write(f"**{int(row['points'])}** pts")
                with col3: st.write(f"**{int(row['isWin'])}** wins")
                with col4: st.write(f"**{int(row['isPodium'])}** podiums")
                st.markdown("---")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("<div style='text-align:center; color:#888;'>Made for you — human-first design. • Data-driven F1 insights • Updated UI & safer model handling</div>", unsafe_allow_html=True)
