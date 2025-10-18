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
from datetime import datetime, timedelta
import warnings
import os
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
    .main-header {
        background: linear-gradient(135deg, #FF1801 0%, #8B0000 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        border: 2px solid #FF6B6B;
    }
    .prediction-card {
        background: linear-gradient(135deg, #2d2d44 0%, #3d3d5c 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #FF1801;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .driver-card {
        background: linear-gradient(135deg, #2d2d44 0%, #3d3d5c 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #0082FA;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .constructor-card {
        background: linear-gradient(135deg, #2d2d44 0%, #3d3d5c 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #FFD700;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .metric-highlight {
        background: linear-gradient(135deg, #FF1801 0%, #FF6B6B 100%);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-weight: bold;
        margin: 5px;
    }
    h1, h2, h3 {
        color: #FF1801;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] button {
        background-color: #2d2d44;
        color: white;
        border-radius: 10px;
        padding: 12px 24px;
        margin: 5px;
        border: 1px solid #444;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF1801 !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD DATA & MODELS
# ============================================================================
@st.cache_resource
def load_all_resources():
    """Load all data and models"""
    resources = {}
    
    try:
        # Load data files
        data_files = {
            'final_df': 'f1_dashboard.csv',
            'driver_stats': 'driver_season_stats.csv', 
            'constructor_stats': 'constructor_season_stats.csv'
        }
        
        for key, file_path in data_files.items():
            if os.path.exists(file_path):
                resources[key] = pd.read_csv(file_path)
            else:
                st.error(f"❌ Missing data file: {file_path}")
                return None
        
        # Define model base path
        model_base = "models/f1_models_20251018_230123"
        
        if not os.path.exists(model_base):
            st.error(f"❌ Model directory not found: {model_base}")
            return None
        
        # Load scalers and feature names
        scaler_path = f"{model_base}/scalers_encoders/feature_scaler.pkl"
        feature_names_path = f"{model_base}/scalers_encoders/feature_names.pkl"
        
        if os.path.exists(scaler_path):
            resources['scaler'] = joblib.load(scaler_path)
            resources['feature_names'] = joblib.load(feature_names_path)
        else:
            st.error("❌ Scaler files not found")
            return None
        
        # Load sklearn models
        sklearn_models = {}
        sk_models_to_load = {
            'rf_winner': f"{model_base}/sklearn_models/rf_winner.pkl",
            'gb_winner': f"{model_base}/sklearn_models/gb_winner.pkl", 
            'rf_points': f"{model_base}/sklearn_models/rf_points.pkl"
        }
        
        for model_name, model_path in sk_models_to_load.items():
            if os.path.exists(model_path):
                sklearn_models[model_name] = joblib.load(model_path)
            else:
                st.error(f"❌ Model file not found: {model_path}")
                return None
        
        resources.update(sklearn_models)
        
        # Try to load deep learning models (optional)
        try:
            nn_winner = keras.models.load_model(f"{model_base}/deep_learning/nn_winner_model.h5")
            resources['nn_winner'] = nn_winner
        except:
            resources['nn_winner'] = None
            
        try:
            nn_podium = keras.models.load_model(f"{model_base}/deep_learning/nn_podium_model.h5")
            resources['nn_podium'] = nn_podium
        except:
            resources['nn_podium'] = None
            
        try:
            nn_points = keras.models.load_model(f"{model_base}/deep_learning/nn_points_model.h5")
            resources['nn_points'] = nn_points
        except:
            resources['nn_points'] = None
        
        # Load metadata
        metadata_path = f"{model_base}/metadata/model_metadata.pkl"
        if os.path.exists(metadata_path):
            resources['metadata'] = joblib.load(metadata_path)
        else:
            resources['metadata'] = {}
        
        return resources
        
    except Exception as e:
        st.error(f"❌ Error loading resources: {str(e)}")
        return None

# Initialize resources
resources = load_all_resources()

if resources is None:
    st.error("🚨 Failed to load required resources. Please check your data and model files.")
    st.stop()

# Extract resources
final_df = resources['final_df']
driver_stats = resources['driver_stats']
constructor_stats = resources['constructor_stats']
scaler = resources['scaler']
feature_names = resources['feature_names']
rf_winner = resources['rf_winner']
gb_winner = resources['gb_winner']
rf_points = resources['rf_points']
nn_winner = resources.get('nn_winner')
nn_podium = resources.get('nn_podium')
nn_points = resources.get('nn_points')
metadata = resources['metadata']

# Data preprocessing
final_df['raceDate'] = pd.to_datetime(final_df['raceDate'])
final_df['year'] = final_df['year'].astype(int)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def create_prediction_features(grid_position, qual_position, pit_stops, avg_lap_time, 
                              lap_time_consistency, pit_stop_duration, driver_wins, 
                              driver_podiums, driver_points, constructor_wins, 
                              constructor_points, finish_rate):
    """Create feature array for model prediction"""
    features = np.array([[
        grid_position, qual_position, grid_position - qual_position, 
        qual_position - grid_position, avg_lap_time, lap_time_consistency, 
        pit_stops, pit_stop_duration, 0,
        driver_points, driver_wins, driver_podiums, finish_rate,
        driver_points/10, 0,
        driver_wins/2, driver_podiums/2, driver_points*0.8,
        constructor_points, constructor_wins, finish_rate,
        finish_rate*0.9
    ]])
    return features

def make_predictions(X_new, use_nn=True, use_rf=True, use_gb=True):
    """Make predictions using selected models"""
    predictions = {}
    X_scaled = scaler.transform(X_new)
    
    if use_nn and nn_winner is not None:
        try:
            pred_nn_win = nn_winner.predict(X_scaled, verbose=0)[0][0]
            pred_nn_podium = nn_podium.predict(X_scaled, verbose=0)[0][0] if nn_podium else pred_nn_win * 0.7
            pred_nn_points = nn_points.predict(X_scaled, verbose=0)[0][0] if nn_points else pred_nn_win * 25
            predictions['Neural Network'] = {
                'win': pred_nn_win,
                'podium': pred_nn_podium,
                'points': pred_nn_points
            }
        except:
            pass
    
    if use_rf:
        pred_rf_win = rf_winner.predict_proba(X_new)[0][1]
        pred_rf_points = rf_points.predict(X_new)[0]
        predictions['Random Forest'] = {
            'win': pred_rf_win,
            'podium': pred_rf_win * 0.7,
            'points': pred_rf_points
        }
    
    if use_gb:
        pred_gb_win = gb_winner.predict_proba(X_new)[0][1]
        pred_gb_points = pred_gb_win * 25
        predictions['Gradient Boosting'] = {
            'win': pred_gb_win,
            'podium': pred_gb_win * 0.7,
            'points': pred_gb_points
        }
    
    return predictions

def calculate_ensemble_predictions(predictions):
    """Calculate ensemble predictions from multiple models"""
    if not predictions:
        return None
    
    ensemble_win = np.mean([predictions[m]['win'] for m in predictions])
    ensemble_podium = np.mean([predictions[m]['podium'] for m in predictions])
    ensemble_points = np.mean([predictions[m]['points'] for m in predictions])
    
    return {
        'win': ensemble_win,
        'podium': ensemble_podium,
        'points': ensemble_points
    }

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
st.sidebar.markdown("""
<div style='text-align: center;'>
    <h1>🏎️ F1 PREDICTOR</h1>
    <p><em>Advanced Race Analytics</em></p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# Navigation menu
page = st.sidebar.radio(
    "**NAVIGATION**",
    [
        "🏠 Dashboard",
        "📊 Race Predictor", 
        "👥 Drivers",
        "🏭 Constructors",
        "📈 Analytics",
        "🏆 Championships"
    ],
    key="page_navigation"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 QUICK STATS")

# Calculate key metrics
total_races = final_df['raceId'].nunique()
total_drivers = final_df['driverRef'].nunique()
latest_season = final_df['year'].max()
current_drivers = final_df[final_df['year'] == latest_season]['driverRef'].nunique()

col1, col2 = st.sidebar.columns(2)
with col1:
    st.metric("Total Races", f"{total_races:,}")
with col2:
    st.metric("Total Drivers", total_drivers)

st.sidebar.metric(f"Active Drivers ({latest_season})", current_drivers)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 MODEL STATUS")
models_loaded = sum([nn_winner is not None, rf_winner is not None, gb_winner is not None])
st.sidebar.write(f"**Active Models:** {models_loaded}/3")

# ============================================================================
# DASHBOARD PAGE
# ============================================================================
if page == "🏠 Dashboard":
    # Header Section
    st.markdown("""
    <div class='main-header'>
        <h1>FORMULA 1 PREDICTION DASHBOARD</h1>
        <h3>AI-Powered Race Analytics & Predictions</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Metrics Row
    st.markdown("## 🏆 SEASON OVERVIEW")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_wins = final_df['isWin'].sum()
        st.markdown(f"""
        <div class='metric-highlight'>
            <div style='font-size: 24px;'>🏆</div>
            <div>Total Wins</div>
            <div style='font-size: 28px;'>{int(total_wins):,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_points = final_df['points'].sum()
        st.markdown(f"""
        <div class='metric-highlight'>
            <div style='font-size: 24px;'>💰</div>
            <div>Points Scored</div>
            <div style='font-size: 28px;'>{int(total_points):,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        podium_finishes = final_df['isPodium'].sum()
        st.markdown(f"""
        <div class='metric-highlight'>
            <div style='font-size: 24px;'>🥇</div>
            <div>Podium Finishes</div>
            <div style='font-size: 28px;'>{int(podium_finishes):,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        finish_rate = (final_df['isFinished'].sum() / len(final_df)) * 100
        st.markdown(f"""
        <div class='metric-highlight'>
            <div style='font-size: 24px;'>✅</div>
            <div>Finish Rate</div>
            <div style='font-size: 28px;'>{finish_rate:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Current Season Highlights
    st.markdown("## 📈 CURRENT SEASON HIGHLIGHTS")
    
    current_season = final_df[final_df['year'] == latest_season]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Current Championship Standings
        championship = current_season.groupby('driverRef').agg({
            'points': 'sum',
            'isWin': 'sum',
            'isPodium': 'sum'
        }).sort_values('points', ascending=False).head(10)
        
        st.markdown("### 🏆 Championship Standings")
        for i, (driver, data) in enumerate(championship.head(5).iterrows(), 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "🏁"
            st.write(f"{emoji} **{driver}** - {int(data['points'])} pts ({int(data['isWin'])} wins)")
    
    with col2:
        # Recent Race Winners
        recent_races = current_season.sort_values('raceDate', ascending=False).head(5)
        st.markdown("### 🏁 Recent Race Winners")
        for _, race in recent_races.iterrows():
            if race['isWin'] == 1:
                st.write(f"🏆 **{race['driverRef']}** - {race['raceName']}")
    
    st.markdown("---")
    
    # Performance Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Top Drivers All-Time
        top_drivers = final_df.groupby('driverRef').agg({
            'isWin': 'sum',
            'points': 'sum'
        }).nlargest(10, 'isWin')
        
        fig = px.bar(top_drivers, x='isWin', y=top_drivers.index, orientation='h',
                    title="🏆 All-Time Wins Leaders",
                    color='isWin', color_continuous_scale='Reds')
        fig.update_layout(template='plotly_dark', height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Constructor Performance
        top_constructors = final_df.groupby('constructorRef').agg({
            'isWin': 'sum',
            'points': 'sum'
        }).nlargest(10, 'points')
        
        fig = px.bar(top_constructors, x='points', y=top_constructors.index, orientation='h',
                    title="🏭 All-Time Points Leaders",
                    color='points', color_continuous_scale='Blues')
        fig.update_layout(template='plotly_dark', height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# RACE PREDICTOR PAGE
# ============================================================================
elif page == "📊 Race Predictor":
    st.markdown("""
    <div class='main-header'>
        <h1>RACE PREDICTION ENGINE</h1>
        <h3>AI-Powered Race Outcome Predictions</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='prediction-card'>
        <h3>🎯 Predict Race Outcomes</h3>
        <p>Enter race conditions and driver statistics to get AI-powered predictions for win probability, 
        podium chances, and expected points.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input Section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.container():
            st.markdown("### 🏁 Race Conditions")
            
            col1a, col2a, col3a = st.columns(3)
            with col1a:
                grid_position = st.slider("Starting Grid", 1, 20, 5, help="Driver's starting position")
            with col2a:
                qual_position = st.slider("Qualifying Position", 1, 20, 3, help="Position in qualifying session")
            with col3a:
                pit_stops = st.slider("Planned Pit Stops", 0, 5, 2, help="Number of planned pit stops")
            
            st.markdown("### ⏱️ Performance Metrics")
            col1b, col2b, col3b = st.columns(3)
            with col1b:
                avg_lap_time = st.number_input("Avg Lap Time (ms)", 80000, 120000, 90000)
            with col2b:
                lap_time_consistency = st.slider("Lap Consistency", 100, 2000, 500, help="Lower = more consistent")
            with col3b:
                pit_stop_duration = st.slider("Pit Stop Time (s)", 18.0, 35.0, 22.5, 0.5)
    
    with col2:
        with st.container():
            st.markdown("### 👤 Driver Profile")
            
            driver_wins = st.number_input("Career Wins", 0, 103, 15)
            driver_podiums = st.number_input("Career Podiums", 0, 200, 45)
            driver_points = st.number_input("Season Points", 0, 500, 180)
            
            st.markdown("### 🏭 Team Profile")
            constructor_wins = st.number_input("Team Wins", 0, 250, 80)
            constructor_points = st.number_input("Team Points", 0, 1000, 350)
            finish_rate = st.slider("Finish Rate %", 50.0, 100.0, 85.0, 1.0)
    
    # Model Selection
    st.markdown("### 🤖 Prediction Models")
    col1, col2, col3 = st.columns(3)
    with col1:
        use_rf = st.checkbox("Random Forest", True)
    with col2:
        use_gb = st.checkbox("Gradient Boosting", True)
    with col3:
        use_nn = st.checkbox("Neural Network", nn_winner is not None, disabled=nn_winner is None)
    
    # Prediction Button
    if st.button("🚀 GENERATE PREDICTIONS", use_container_width=True, type="primary"):
        with st.spinner("🤖 Analyzing race conditions..."):
            # Create features and make predictions
            X_new = create_prediction_features(
                grid_position, qual_position, pit_stops, avg_lap_time,
                lap_time_consistency, pit_stop_duration, driver_wins,
                driver_podiums, driver_points, constructor_wins,
                constructor_points, finish_rate
            )
            
            predictions = make_predictions(X_new, use_nn, use_rf, use_gb)
            ensemble = calculate_ensemble_predictions(predictions)
        
        # Display Results
        if ensemble:
            st.markdown("---")
            st.markdown("## 📊 PREDICTION RESULTS")
            
            # Main Metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                win_color = "normal" if ensemble['win'] > 0.3 else "off"
                st.metric("🏆 WIN PROBABILITY", f"{ensemble['win']:.1%}", 
                         delta=f"{(ensemble['win']-0.1)*100:+.1f}%", delta_color=win_color)
            
            with col2:
                podium_color = "normal" if ensemble['podium'] > 0.4 else "off"
                st.metric("🥇 PODIUM CHANCE", f"{ensemble['podium']:.1%}", delta_color=podium_color)
            
            with col3:
                points_color = "normal" if ensemble['points'] > 10 else "off"
                st.metric("💰 EXPECTED POINTS", f"{ensemble['points']:.1f}", delta_color=points_color)
            
            # Model Comparison
            st.markdown("### 📈 Model Consensus")
            if predictions:
                model_data = []
                for model_name, preds in predictions.items():
                    model_data.append({
                        'Model': model_name,
                        'Win %': preds['win'] * 100,
                        'Podium %': preds['podium'] * 100,
                        'Points': preds['points']
                    })
                
                df_comparison = pd.DataFrame(model_data)
                st.dataframe(df_comparison.style.format({
                    'Win %': '{:.1f}%',
                    'Podium %': '{:.1f}%',
                    'Points': '{:.1f}'
                }), use_container_width=True)
            
            # Strategy Recommendations
            st.markdown("### 💡 Race Strategy Advice")
            if ensemble['win'] > 0.6:
                st.success("**AGGRESSIVE STRATEGY** - High win probability suggests going for victory with bold tire choices and aggressive overtaking.")
            elif ensemble['podium'] > 0.5:
                st.warning("**BALANCED APPROACH** - Strong podium chances suggest focusing on consistent pace and strategic pit stops.")
            else:
                st.info("**POINTS FOCUS** - Target points finish with conservative strategy and tire management.")

# ============================================================================
# DRIVERS PAGE
# ============================================================================
elif page == "👥 Drivers":
    st.markdown("""
    <div class='main-header'>
        <h1>DRIVER ANALYSIS</h1>
        <h3>Comprehensive Driver Performance & Statistics</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # All drivers across all seasons
    all_drivers = sorted(final_df['driverRef'].unique())
    selected_driver = st.selectbox("🎯 SELECT DRIVER", all_drivers)
    
    if selected_driver:
        driver_data = final_df[final_df['driverRef'] == selected_driver]
        
        # Driver Overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_wins = driver_data['isWin'].sum()
            st.metric("🏆 Career Wins", int(total_wins))
        
        with col2:
            total_podiums = driver_data['isPodium'].sum()
            st.metric("🥇 Career Podiums", int(total_podiums))
        
        with col3:
            total_points = driver_data['points'].sum()
            st.metric("💰 Career Points", int(total_points))
        
        with col4:
            total_races = len(driver_data)
            st.metric("🏁 Races Entered", total_races)
        
        st.markdown("---")
        
        # Performance Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Wins by Season
            wins_by_season = driver_data.groupby('year')['isWin'].sum()
            if len(wins_by_season) > 0:
                fig = px.line(x=wins_by_season.index, y=wins_by_season.values,
                             title=f"🏆 Wins by Season - {selected_driver}",
                             labels={'x': 'Season', 'y': 'Wins'})
                fig.update_layout(template='plotly_dark', height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Points by Season
            points_by_season = driver_data.groupby('year')['points'].sum()
            if len(points_by_season) > 0:
                fig = px.bar(x=points_by_season.index, y=points_by_season.values,
                            title=f"💰 Points by Season - {selected_driver}",
                            color=points_by_season.values,
                            color_continuous_scale='Viridis')
                fig.update_layout(template='plotly_dark', height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        # Recent Performance
        st.markdown("### 🏁 Recent Race Performance")
        recent_races = driver_data.sort_values('raceDate', ascending=False).head(10)[
            ['year', 'raceName', 'grid', 'positionOrder', 'points', 'isWin', 'isPodium']
        ]
        st.dataframe(recent_races, use_container_width=True)

# ============================================================================
# CONSTRUCTORS PAGE
# ============================================================================
elif page == "🏭 Constructors":
    st.markdown("""
    <div class='main-header'>
        <h1>CONSTRUCTOR ANALYSIS</h1>
        <h3>Team Performance & Championship History</h3>
    </div>
    """, unsafe_allow_html=True)
    
    all_constructors = sorted(final_df['constructorRef'].unique())
    selected_constructor = st.selectbox("🏭 SELECT TEAM", all_constructors)
    
    if selected_constructor:
        team_data = final_df[final_df['constructorRef'] == selected_constructor]
        
        # Team Overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            team_wins = team_data['isWin'].sum()
            st.metric("🏆 Team Wins", int(team_wins))
        
        with col2:
            team_points = team_data['points'].sum()
            st.metric("💰 Total Points", int(team_points))
        
        with col3:
            win_rate = (team_wins / len(team_data)) * 100
            st.metric("🎯 Win Rate", f"{win_rate:.1f}%")
        
        with col4:
            seasons = team_data['year'].nunique()
            st.metric("📅 Seasons", seasons)
        
        st.markdown("---")
        
        # Team Performance
        col1, col2 = st.columns(2)
        
        with col1:
            # Championship Performance
            points_by_year = team_data.groupby('year')['points'].sum()
            fig = px.area(points_by_year, x=points_by_year.index, y=points_by_year.values,
                         title=f"📈 Championship Points by Season",
                         color_discrete_sequence=['#FF1801'])
            fig.update_layout(template='plotly_dark', height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Drivers in Team
            team_drivers = team_data.groupby('driverRef').agg({
                'points': 'sum',
                'isWin': 'sum',
                'raceId': 'count'
            }).sort_values('points', ascending=False)
            
            st.markdown("### 👥 Team Drivers History")
            for driver, stats in team_drivers.head(5).iterrows():
                st.write(f"**{driver}** - {int(stats['points'])} pts ({int(stats['isWin'])} wins)")

# ============================================================================
# ANALYTICS PAGE
# ============================================================================
elif page == "📈 Analytics":
    st.markdown("""
    <div class='main-header'>
        <h1>ADVANCED ANALYTICS</h1>
        <h3>Deep Dive into F1 Performance Data</h3>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🏁 Grid Analysis", "📊 Performance Trends", "🎯 Championship Battles"])
    
    with tab1:
        st.markdown("### 🏁 Starting Grid Impact")
        
        grid_analysis = final_df.groupby('grid').agg({
            'isWin': 'mean',
            'isPodium': 'mean', 
            'points': 'mean'
        }).reset_index()
        grid_analysis = grid_analysis[grid_analysis['grid'] <= 20]
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(grid_analysis, x='grid', y='isWin',
                         title="Win Probability by Starting Position",
                         labels={'grid': 'Grid Position', 'isWin': 'Win Probability'})
            fig.update_layout(template='plotly_dark', height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(grid_analysis, x='grid', y='points',
                           title="Average Points by Grid Position",
                           size='isPodium', color='isWin',
                           color_continuous_scale='RdYlGn')
            fig.update_layout(template='plotly_dark', height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### 📊 Historical Performance Trends")
        
        # DNF Analysis
        dnf_trends = final_df.groupby('year').agg({
            'isDNF': 'mean',
            'raceId': 'count'
        })
        dnf_trends['dnf_rate'] = dnf_trends['isDNF'] * 100
        
        fig = px.line(dnf_trends, x=dnf_trends.index, y='dnf_rate',
                     title="DNF Rate Trend Over Seasons",
                     labels={'x': 'Season', 'dnf_rate': 'DNF Rate (%)'})
        fig.update_layout(template='plotly_dark', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### 🏆 Championship Analysis")
        
        # Current season championship
        current_season = final_df[final_df['year'] == latest_season]
        championship = current_season.groupby('driverRef')['points'].sum().sort_values(ascending=False).head(10)
        
        fig = px.bar(x=championship.values, y=championship.index, orientation='h',
                    title=f"Current Championship Standings - {latest_season}",
                    color=championship.values, color_continuous_scale='Viridis')
        fig.update_layout(template='plotly_dark', height=500)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# CHAMPIONSHIPS PAGE
# ============================================================================
elif page == "🏆 Championships":
    st.markdown("""
    <div class='main-header'>
        <h1>CHAMPIONSHIP HISTORY</h1>
        <h3>World Champions & Season Reviews</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Season selector
    seasons = sorted(final_df['year'].unique(), reverse=True)
    selected_season = st.selectbox("📅 SELECT SEASON", seasons)
    
    if selected_season:
        season_data = final_df[final_df['year'] == selected_season]
        
        # Championship standings for selected season
        standings = season_data.groupby('driverRef').agg({
            'points': 'sum',
            'isWin': 'sum',
            'isPodium': 'sum',
            'raceId': 'count'
        }).sort_values('points', ascending=False).head(10)
        
        st.markdown(f"### 🏆 {selected_season} Championship Standings")
        
        # Display standings
        for i, (driver, stats) in enumerate(standings.iterrows(), 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            with st.container():
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                with col1:
                    st.write(f"**{emoji} {driver}**")
                with col2:
                    st.write(f"**{int(stats['points'])}** pts")
                with col3:
                    st.write(f"**{int(stats['isWin'])}** wins")
                with col4:
                    st.write(f"**{int(stats['isPodium'])}** podiums")
            st.markdown("---")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888; padding: 20px;'>
        <p>🏎️ <strong>F1 Prediction Dashboard</strong> • Powered by Machine Learning • 
        Advanced Race Analytics</p>
        <p style='font-size: 0.8em;'>Data Source: Historical F1 Data • Models: Neural Networks, Random Forest, Gradient Boosting</p>
    </div>
    """,
    unsafe_allow_html=True
)
