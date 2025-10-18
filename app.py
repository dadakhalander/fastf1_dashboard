import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# seaborn imported but not used — ok to keep for future
import seaborn as sns
from PIL import Image
import joblib
from tensorflow import keras
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="F1 Race Prediction Dashboard",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for stunning design
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        color: #ffffff;
    }
    .metric-card {
        background: linear-gradient(135deg, #FF1801 0%, #FF6B6B 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        font-weight: bold;
    }
    h1, h2, h3 {
        color: #FF1801;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] button {
        background-color: #2d2d44;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF1801;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD DATA & MODELS
# ============================================================================
@st.cache_resource
def load_all_resources():
    """Load all data and models"""
    try:
        # Load data files (adjust paths to your setup)
        final_df = pd.read_csv("f1_dashboard.csv")
        driver_stats = pd.read_csv("driver_season_stats.csv")
        constructor_stats = pd.read_csv("constructor_season_stats.csv")
        
        # Model paths
        model_base = "models/f1_models_20251018_230123"
        
        # Load scalers and feature names
        scaler = joblib.load(f"{model_base}/scalers_encoders/feature_scaler.pkl")
        feature_names = joblib.load(f"{model_base}/scalers_encoders/feature_names.pkl")
        
        # Load deep learning models
        nn_winner = keras.models.load_model(f"{model_base}/deep_learning/nn_winner_model.h5")
        nn_podium = keras.models.load_model(f"{model_base}/deep_learning/nn_podium_model.h5")
        nn_points = keras.models.load_model(f"{model_base}/deep_learning/nn_points_model.h5")
        
        # Load sklearn models
        rf_winner = joblib.load(f"{model_base}/sklearn_models/rf_winner.pkl")
        gb_winner = joblib.load(f"{model_base}/sklearn_models/gb_winner.pkl")
        rf_points = joblib.load(f"{model_base}/sklearn_models/rf_points.pkl")
        
        # Load metadata
        metadata = joblib.load(f"{model_base}/metadata/model_metadata.pkl")
        
        return {
            'final_df': final_df,
            'driver_stats': driver_stats,
            'constructor_stats': constructor_stats,
            'scaler': scaler,
            'feature_names': feature_names,
            'nn_winner': nn_winner,
            'nn_podium': nn_podium,
            'nn_points': nn_points,
            'rf_winner': rf_winner,
            'gb_winner': gb_winner,
            'rf_points': rf_points,
            'metadata': metadata
        }
    except Exception as e:
        st.error(f"❌ Error loading resources: {str(e)}")
        st.error("📁 Make sure models are in: models/f1_models_20251018_230123/")
        return None

resources = load_all_resources()

if resources is None:
    st.error("⚠️ Could not load models. Check file paths!")
    st.stop()

final_df = resources['final_df']
driver_stats = resources['driver_stats']
constructor_stats = resources['constructor_stats']
scaler = resources['scaler']
feature_names = resources['feature_names']
nn_winner = resources['nn_winner']
nn_podium = resources['nn_podium']
nn_points = resources['nn_points']
rf_winner = resources['rf_winner']
gb_winner = resources['gb_winner']
rf_points = resources['rf_points']

# Defensive: ensure date column parsed
if 'raceDate' in final_df.columns:
    final_df['raceDate'] = pd.to_datetime(final_df['raceDate'], errors='coerce')

# ============================================================================
# NORMALIZE COMMON BOOLEAN / FLAG COLUMNS (defensive)
# ============================================================================
# Some files use 'finished' or 'isFinished' etc. Create consistent aliases used below.
if 'isFinished' not in final_df.columns:
    if 'finished' in final_df.columns:
        final_df['isFinished'] = final_df['finished'].astype(int)
    else:
        # assume finished when positionOrder is present
        final_df['isFinished'] = final_df.get('positionOrder').notna().astype(int)

# isDNF: if explicit column exists use it, else infer from status or finished
if 'isDNF' not in final_df.columns:
    if 'status' in final_df.columns:
        final_df['isDNF'] = final_df['status'].str.contains('DNF|retired|collision|accident', case=False, na=False).astype(int)
        # If status does not include DNF terms, mark DNF when not finished
        final_df.loc[(final_df['isDNF'] == 0) & (final_df['isFinished'] == 0), 'isDNF'] = 1
    else:
        final_df['isDNF'] = (final_df['isFinished'] == 0).astype(int)

# isWin / isPodium normalization
if 'isWin' not in final_df.columns:
    final_df['isWin'] = (final_df.get('positionOrder') == 1).astype(int)
if 'isPodium' not in final_df.columns:
    final_df['isPodium'] = final_df.get('positionOrder').isin([1,2,3]).astype(int)

# Fill missing numeric columns that are used later
for col in ['points','grid','year','raceId','driverRef','driverId']:
    if col not in final_df.columns:
        final_df[col] = np.nan

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
st.sidebar.markdown("# 🏎️ F1 PREDICTION")
st.sidebar.markdown("### Advanced Analytics & ML Models")
st.sidebar.markdown("---")

# Navigation menu
page = st.sidebar.radio(
    "📍 SELECT PAGE:",
    [
        "🏠 Home",
        "📊 Data Analysis",
        "🎯 Single Race Prediction",
        "👥 Driver Analysis",
        "🏭 Constructor Analysis",
        "🔮 Advanced Predictions",
        "📈 Model Performance",
        "⚙️ Prediction Engine"
    ],
    key="page_navigation"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Quick Stats")
col1, col2 = st.sidebar.columns(2)
with col1:
    st.metric("Races", int(final_df['raceId'].nunique()) if 'raceId' in final_df.columns else 0)
with col2:
    st.metric("Drivers", int(final_df['driverRef'].nunique()) if 'driverRef' in final_df.columns else 0)
st.sidebar.metric("Seasons", int(final_df['year'].nunique()) if 'year' in final_df.columns else 0)

# ============================================================================
# HOME PAGE
# ============================================================================
if page == "🏠 Home":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("# 🏎️ Formula 1 Race Prediction Dashboard")
        st.markdown("""
        Welcome to the **Advanced F1 Prediction Dashboard** powered by:
        - 🧠 Deep Learning Neural Networks
        - 🌳 Random Forest & Gradient Boosting
        - 📊 Time Series Analysis
        - 🔮 Ensemble Predictions
        """)
    
    with col2:
        st.metric("Latest Season", int(final_df['year'].max()) if 'year' in final_df.columns else "N/A")
    
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        ### 🎯 Predictions
        - Race Winners
        - Podium Finishes
        - Points Scored
        """)
    
    with col2:
        st.markdown("""
        ### 📈 Analytics
        - Driver Performance
        - Constructor Stats
        - Historical Trends
        """)
    
    with col3:
        st.markdown("""
        ### 🤖 Models
        - Neural Networks
        - Random Forest
        - Gradient Boosting
        """)
    
    with col4:
        st.markdown("""
        ### 🔧 Features
        - Live Predictions
        - Model Comparison
        - Confidence Analysis
        """)
    
    st.markdown("---")
    
    # Key metrics dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        wins_by_driver = final_df[final_df['isWin'] == 1].groupby('driverRef').size()
        most_wins = int(wins_by_driver.max()) if not wins_by_driver.empty else 0
        st.metric("Most Wins", most_wins)
    
    with col2:
        recent_year = int(final_df['year'].max()) if 'year' in final_df.columns else None
        recent_races = int(final_df[final_df['year'] == recent_year]['raceId'].nunique()) if recent_year is not None else 0
        st.metric(f"Races ({recent_year})", recent_races)
    
    with col3:
        if 'points' in final_df.columns and not final_df[final_df['points'] > 0].empty:
            avg_points = final_df[final_df['points'] > 0]['points'].mean()
            st.metric("Avg Points", f"{avg_points:.1f}")
        else:
            st.metric("Avg Points", "N/A")
    
    with col4:
        finish_rate = (final_df['isFinished'].sum() / len(final_df) * 100) if len(final_df) > 0 else 0
        st.metric("Finish Rate", f"{finish_rate:.1f}%")
    
    st.markdown("---")
    st.markdown("### 🏁 Top Drivers (All Time)")
    top_drivers = wins_by_driver.sort_values(ascending=False).head(10)
    if not top_drivers.empty:
        fig = px.bar(x=top_drivers.values, y=top_drivers.index, orientation='h',
                     color=top_drivers.values, color_continuous_scale='Reds',
                     labels={'x': 'Wins', 'y': 'Driver'})
        fig.update_layout(template='plotly_dark', height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No win data available to show top drivers.")

# ============================================================================
# DATA ANALYSIS PAGE
# ============================================================================
elif page == "📊 Data Analysis":
    st.markdown("# 📊 F1 Data Analysis & Trends")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Races", "🎯 Results", "💰 Points", "🔴 DNF", "🏆 Podiums"])
    
    with tab1:
        st.subheader("Races Per Season")
        if 'year' in final_df.columns and 'raceId' in final_df.columns:
            races_by_year = final_df.groupby('year')['raceId'].nunique()
            fig = px.bar(x=races_by_year.index, y=races_by_year.values,
                         labels={'x': 'Year', 'y': 'Number of Races'},
                         color_discrete_sequence=['#FF1801'],
                         title="F1 Races Per Season")
            fig.update_layout(template='plotly_dark', height=450, hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Max Races/Season", int(races_by_year.max()))
            with col2:
                st.metric("Min Races/Season", int(races_by_year.min()))
            with col3:
                st.metric("Avg Races/Season", f"{races_by_year.mean():.1f}")
        else:
            st.info("Year or raceId column missing in dataset.")
    
    with tab2:
        st.subheader("Result Distribution (All Time)")
        # ensure columns exist
        wins = int(final_df['isWin'].sum()) if 'isWin' in final_df.columns else 0
        podiums = int(final_df['isPodium'].sum()) if 'isPodium' in final_df.columns else 0
        finishes = int(final_df['isFinished'].sum()) if 'isFinished' in final_df.columns else 0
        dnfs = int(final_df['isDNF'].sum()) if 'isDNF' in final_df.columns else 0
        
        results_dist = pd.DataFrame({
            'Category': ['Wins', 'Podiums', 'Finishes', 'DNF'],
            'Count': [wins, podiums, finishes, dnfs]
        })
        
        fig = px.pie(results_dist, values='Count', names='Category',
                     color_discrete_sequence=['#FFD700', '#C0C0C0', '#CD7F32', '#FF6B6B'],
                     title="Race Results Distribution")
        fig.update_layout(template='plotly_dark', height=450)
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Wins", wins)
        with col2:
            st.metric("Podiums", podiums)
        with col3:
            st.metric("Finishes", finishes)
        with col4:
            st.metric("DNF", dnfs)
    
    with tab3:
        st.subheader("Points Distribution")
        if 'points' in final_df.columns:
            points_data = final_df[final_df['points'] > 0]['points'].value_counts().sort_index(ascending=False).head(15)
            if not points_data.empty:
                fig = px.bar(x=points_data.values, y=points_data.index,
                             orientation='h', color_discrete_sequence=['#0082FA'],
                             labels={'x': 'Frequency', 'y': 'Points'},
                             title="Top 15 Points Distributions")
                fig.update_layout(template='plotly_dark', height=450)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No points > 0 found in dataset.")
        else:
            st.info("No points column found.")
    
    with tab4:
        st.subheader("DNF (Did Not Finish) Trends")
        if 'year' in final_df.columns and 'isDNF' in final_df.columns and 'raceId' in final_df.columns:
            dnf_by_year = final_df.groupby('year').agg({
                'isDNF': 'sum',
                'raceId': 'count'
            }).reset_index()
            dnf_by_year['dnf_rate'] = (dnf_by_year['isDNF'] / dnf_by_year['raceId'] * 100)
            
            fig = px.line(dnf_by_year, x='year', y='dnf_rate', 
                          markers=True, color_discrete_sequence=['#FF1801'],
                          labels={'year': 'Year', 'dnf_rate': 'DNF Rate (%)'},
                          title="DNF Rate Trend Over Years")
            fig.update_layout(template='plotly_dark', height=450, hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Required columns for DNF trend are missing.")
    
    with tab5:
        st.subheader("Podium Rate by Year")
        if 'year' in final_df.columns and 'isPodium' in final_df.columns and 'raceId' in final_df.columns:
            podium_by_year = final_df.groupby('year').agg({
                'isPodium': 'sum',
                'raceId': 'count'
            }).reset_index()
            podium_by_year['podium_rate'] = (podium_by_year['isPodium'] / podium_by_year['raceId'] * 100)
            
            fig = px.area(podium_by_year, x='year', y='podium_rate',
                          color_discrete_sequence=['#C0C0C0'],
                          labels={'year': 'Year', 'podium_rate': 'Podium Rate (%)'},
                          title="Podium Rate Trend")
            fig.update_layout(template='plotly_dark', height=450, hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Required columns for podium trend are missing.")

# ============================================================================
# SINGLE RACE PREDICTION
# ============================================================================
elif page == "🎯 Single Race Prediction":
    st.markdown("# 🎯 Single Race Prediction Engine")
    
    st.info("📌 Enter driver and race conditions to predict outcomes using ML models")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        grid_position = st.number_input("🏁 Grid Position", 1, 20, 5)
    
    with col2:
        qual_position = st.number_input("🏎️ Qualifying Position", 1, 20, 6)
    
    with col3:
        pit_stops = st.number_input("🔧 Expected Pit Stops", 0, 10, 2)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_lap_time = st.number_input("⏱️ Avg Lap Time (ms)", 50000, 200000, 85000)
    
    with col2:
        lap_time_consistency = st.number_input("📊 Lap Time Consistency (ms)", 0, 5000, 500)
    
    with col3:
        pit_stop_duration = st.number_input("⏸️ Avg Pit Stop Duration (s)", 5.0, 60.0, 25.0)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        driver_wins = st.number_input("🏆 Driver Career Wins", 0, 100, 5)
    
    with col2:
        driver_podiums = st.number_input("🥇 Driver Career Podiums", 0, 200, 20)
    
    with col3:
        driver_points = st.number_input("💰 Driver Season Points", 0, 2000, 100)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        constructor_wins = st.number_input("🏭 Constructor Wins", 0, 2000, 50)
    
    with col2:
        constructor_points = st.number_input("🏭 Constructor Points", 0, 10000, 300)
    
    with col3:
        finish_rate_input = st.number_input("✅ Finish Rate (%)", 0.0, 100.0, 75.0)
    
    # Model selection
    st.markdown("---")
    st.markdown("### 🤖 Select Models to Use")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        use_nn = st.checkbox("🧠 Neural Network", True)
    with col2:
        use_rf = st.checkbox("🌳 Random Forest", True)
    with col3:
        use_gb = st.checkbox("⚡ Gradient Boosting", True)
    
    if st.button("🚀 PREDICT", use_container_width=True, key="predict_btn"):
        # Create feature array (match to your trained feature order)
        # NOTE: you must ensure the features and order match scaler/feature_names used during training.
        X_new = np.array([[
            grid_position,
            qual_position,
            grid_position - qual_position,
            avg_lap_time,
            lap_time_consistency,
            pit_stops,
            pit_stop_duration,
            driver_wins,
            driver_podiums,
            driver_points,
            constructor_wins,
            constructor_points,
            finish_rate_input
        ]], dtype=float)
        
        # Scale features (use the same scaler used during training)
        try:
            X_scaled = scaler.transform(X_new)
        except Exception as e:
            st.error(f"Failed to scale features: {e}")
            st.stop()
        
        predictions = {}
        
        # Neural nets (expect scaled features)
        if use_nn:
            try:
                pred_nn_win = float(nn_winner.predict(X_scaled, verbose=0).ravel()[0])
                pred_nn_podium = float(nn_podium.predict(X_scaled, verbose=0).ravel()[0])
                pred_nn_points = float(nn_points.predict(X_scaled, verbose=0).ravel()[0])
                predictions['Neural Network'] = {
                    'win': pred_nn_win,
                    'podium': pred_nn_podium,
                    'points': pred_nn_points
                }
            except Exception as e:
                st.error(f"NN prediction failed: {e}")
        
        # Scikit-learn models — use scaled features (X_scaled)
        if use_rf:
            try:
                # rf_winner may expose predict_proba; handle gracefully if not
                if hasattr(rf_winner, 'predict_proba'):
                    pred_rf_win = float(rf_winner.predict_proba(X_scaled)[0][1])
                else:
                    # fallback to decision function or predict
                    pred_rf_win = float(rf_winner.predict(X_scaled)[0])
                    # convert to probability-like if necessary (not ideal)
                pred_rf_points = float(rf_points.predict(X_scaled)[0])
                predictions['Random Forest'] = {
                    'win': pred_rf_win,
                    'podium': pred_rf_win * 0.7,
                    'points': pred_rf_points
                }
            except Exception as e:
                st.error(f"Random Forest prediction failed: {e}")
        
        if use_gb:
            try:
                if hasattr(gb_winner, 'predict_proba'):
                    pred_gb_win = float(gb_winner.predict_proba(X_scaled)[0][1])
                else:
                    pred_gb_win = float(gb_winner.predict(X_scaled)[0])
                predictions['Gradient Boosting'] = {
                    'win': pred_gb_win,
                    'podium': pred_gb_win * 0.7,
                    'points': pred_gb_win * 20
                }
            except Exception as e:
                st.error(f"Gradient Boosting prediction failed: {e}")
        
        if not predictions:
            st.warning("No model produced predictions. Check model loading / feature compatibility.")
        else:
            # Display ensemble metrics
            st.markdown("---")
            st.markdown("## 🎯 PREDICTION RESULTS")
            col1, col2, col3 = st.columns(3)
            
            ensemble_win = np.mean([predictions[m]['win'] for m in predictions])
            ensemble_podium = np.mean([predictions[m]['podium'] for m in predictions])
            ensemble_points = np.mean([predictions[m]['points'] for m in predictions])
            
            with col1:
                st.metric("🏆 Win Probability", f"{ensemble_win:.1%}", delta=f"{(ensemble_win-0.5)*100:+.1f}%")
            with col2:
                st.metric("🥇 Podium Probability", f"{ensemble_podium:.1%}")
            with col3:
                st.metric("📊 Expected Points", f"{ensemble_points:.1f}")
            
            # Detailed predictions
            st.markdown("---")
            st.markdown("## 📋 Model-by-Model Breakdown")
            df_predictions = pd.DataFrame(predictions).T
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Prediction Scores")
                st.dataframe(df_predictions.style.format({
                    'win': '{:.2%}',
                    'podium': '{:.2%}',
                    'points': '{:.1f}'
                }), use_container_width=True)
            
            with col2:
                st.markdown("### Confidence Levels")
                confidence_data = {
                    'Model': list(predictions.keys()),
                    'Win%': [predictions[m]['win']*100 for m in predictions]
                }
                fig = px.bar(confidence_data, x='Model', y='Win%',
                            color_discrete_sequence=['#FF1801'],
                            title="Win Probability by Model")
                fig.update_layout(template='plotly_dark', height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            # comparison plots
            col1, col2 = st.columns(2)
            with col1:
                fig = go.Figure()
                for i, model in enumerate(predictions):
                    fig.add_trace(go.Bar(
                        name=model,
                        x=[model],
                        y=[predictions[model]['win']],
                        marker_color=['#FF1801', '#0082FA', '#FFD700'][i % 3]
                    ))
                fig.update_layout(title="Win Probability Comparison", template='plotly_dark', height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = go.Figure()
                for i, model in enumerate(predictions):
                    fig.add_trace(go.Bar(
                        name=model,
                        x=[model],
                        y=[predictions[model]['points']],
                        marker_color=['#FF1801', '#0082FA', '#FFD700'][i % 3]
                    ))
                fig.update_layout(title="Expected Points Comparison", template='plotly_dark', height=400)
                st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# DRIVER ANALYSIS
# ============================================================================
elif page == "👥 Driver Analysis":
    st.markdown("# 👥 Driver Performance Analysis")
    
    # defensive: check expected columns in driver_stats
    if 'driver' in driver_stats.columns:
        driver_list = sorted(driver_stats['driver'].unique())
    elif 'driverName' in driver_stats.columns:
        driver_list = sorted(driver_stats['driverName'].unique())
    else:
        st.error("Driver stats file missing expected 'driver' or 'driverName' column.")
        driver_list = []
    
    selected_driver = st.selectbox("🏎️ Select Driver", driver_list)
    
    # support both naming conventions
    if 'driver' in driver_stats.columns:
        driver_data = driver_stats[driver_stats['driver'] == selected_driver].sort_values('year')
    else:
        driver_data = driver_stats[driver_stats['driverName'] == selected_driver].sort_values('year')
    
    if len(driver_data) > 0:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🏆 Total Wins", int(driver_data['wins'].sum()) if 'wins' in driver_data.columns else 0)
        with col2:
            st.metric("🥇 Total Podiums", int(driver_data['podiums'].sum()) if 'podiums' in driver_data.columns else 0)
        with col3:
            st.metric("💰 Total Points", int(driver_data['totalPoints'].sum()) if 'totalPoints' in driver_data.columns else 0)
        with col4:
            st.metric("📅 Seasons", len(driver_data))
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            ycols = [c for c in ['wins','podiums'] if c in driver_data.columns]
            if ycols:
                fig = px.line(driver_data, x='year', y=ycols,
                             markers=True, title="Wins & Podiums Trend")
                fig.update_layout(template='plotly_dark', height=450, hovermode='x unified')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No wins/podiums columns available for plotting.")
        
        with col2:
            if 'totalPoints' in driver_data.columns:
                fig = px.line(driver_data, x='year', y='totalPoints',
                             markers=True, title="Season Points Trend")
                fig.update_layout(template='plotly_dark', height=450, hovermode='x unified')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No totalPoints column available for plotting.")
        
        st.markdown("---")
        st.markdown("### 📊 Detailed Statistics")
        st.dataframe(driver_data.fillna("N/A"), use_container_width=True)

# ============================================================================
# CONSTRUCTOR ANALYSIS
# ============================================================================
elif page == "🏭 Constructor Analysis":
    st.markdown("# 🏭 Constructor Performance Analysis")
    
    if 'constructor' in constructor_stats.columns:
        constructor_list = sorted(constructor_stats['constructor'].unique())
    elif 'constructorName' in constructor_stats.columns:
        constructor_list = sorted(constructor_stats['constructorName'].unique())
    else:
        st.error("Constructor stats missing expected column.")
        constructor_list = []
    
    selected_constructor = st.selectbox("🏭 Select Constructor", constructor_list)
    
    if 'constructor' in constructor_stats.columns:
        const_data = constructor_stats[constructor_stats['constructor'] == selected_constructor].sort_values('year')
    else:
        const_data = constructor_stats[constructor_stats['constructorName'] == selected_constructor].sort_values('year')
    
    if len(const_data) > 0:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🏆 Total Wins", int(const_data['wins'].sum()) if 'wins' in const_data.columns else 0)
        with col2:
            st.metric("💰 Total Points", int(const_data['totalPoints'].sum()) if 'totalPoints' in const_data.columns else 0)
        with col3:
            st.metric("🎯 Avg Win Rate", f\"{const_data['winRate'].mean():.1f}%\" if 'winRate' in const_data.columns else "N/A")
        with col4:
            st.metric("📅 Seasons", len(const_data))
        
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        if 'totalPoints' in const_data.columns:
            with col1:
                fig = px.bar(const_data, x='year', y='totalPoints', title="Points by Season")
                fig.update_layout(template='plotly_dark', height=400)
                st.plotly_chart(fig, use_container_width=True)
        if 'wins' in const_data.columns:
            with col2:
                fig = px.line(const_data, x='year', y='wins', markers=True, title="Wins Trend")
                fig.update_layout(template='plotly_dark', height=400)
                st.plotly_chart(fig, use_container_width=True)
        if 'finishRate' in const_data.columns:
            with col3:
                fig = px.line(const_data, x='year', y='finishRate', markers=True, title="Finish Rate Trend")
                fig.update_layout(template='plotly_dark', height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 📊 Detailed Statistics")
        st.dataframe(const_data.fillna("N/A"), use_container_width=True)

# ============================================================================
# ADVANCED PREDICTIONS
# ============================================================================
elif page == "🔮 Advanced Predictions":
    st.markdown("# 🔮 Advanced Predictions & Insights")
    
    tab1, tab2, tab3 = st.tabs(["📊 Grid Analysis", "👥 Driver Form", "🏆 Championship Race"])
    
    with tab1:
        st.subheader("Grid Position Performance Analysis")
        if {'grid','isWin','isPodium','points','driverId'}.issubset(final_df.columns):
            grid_analysis = final_df.groupby('grid').agg({
                'isWin': 'sum',
                'isPodium': 'sum',
                'points': 'mean',
                'driverId': 'count'
            }).reset_index()
            grid_analysis.columns = ['Grid', 'Wins', 'Podiums', 'Avg_Points', 'Races']
            grid_analysis = grid_analysis[grid_analysis['Grid'] <= 20]
            grid_analysis['win_rate'] = (grid_analysis['Wins'] / grid_analysis['Races'] * 100)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(grid_analysis, x='Grid', y='win_rate',
                            color='win_rate', color_continuous_scale='Reds',
                            title="Win Rate by Grid Position")
                fig.update_layout(template='plotly_dark', height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.scatter(grid_analysis, x='Grid', y='Avg_Points',
                               size='Races', color='win_rate',
                               color_continuous_scale='RdYlGn',
                               title="Points vs Grid Position")
                fig.update_layout(template='plotly_dark', height=400)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Required columns for grid analysis are missing (grid/isWin/isPodium/points/driverId).")
    
    with tab2:
        st.subheader("Current Driver Form Analysis")
        if 'year' in final_df.columns:
            recent_year = final_df['year'].max()
            # defensive aggregation
            agg_cols = {}
            if 'isWin' in final_df.columns: agg_cols['isWin'] = 'sum'
            if 'isPodium' in final_df.columns: agg_cols['isPodium'] = 'sum'
            agg_cols['points'] = 'sum'
            agg_cols['raceId'] = 'count'
            
            driver_form = final_df[final_df['year'] == recent_year].groupby('driverRef').agg(agg_cols).reset_index()
            driver_form.columns = ['Driver'] + [c for c in driver_form.columns if c != 'Driver'][1:]
            # sort safely
            if 'points' in driver_form.columns:
                driver_form = driver_form.sort_values('points', ascending=False).head(15)
            
            if not driver_form.empty:
                fig = px.bar(driver_form, x='points' if 'points' in driver_form.columns else driver_form.columns[1],
                            y='Driver', orientation='h', color='points' if 'points' in driver_form.columns else None,
                            title=f"Top Drivers - Season {int(recent_year)}")
                fig.update_layout(template='plotly_dark', height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No driver form data available for the recent year.")
        else:
            st.info("Year column missing in dataset.")
