
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
        margin: 5px;
    }
    .prediction-card {
        background: linear-gradient(135deg, #2d2d44 0%, #3d3d5c 100%);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #FF1801;
        margin: 10px 0;
    }
    h1, h2, h3 {
        color: #FF1801;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #2d2d44;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        border: 1px solid #444;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF1801;
        color: white;
    }
    .stButton button {
        background: linear-gradient(135deg, #FF1801 0%, #FF6B6B 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: bold;
    }
    .stButton button:hover {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF1801 100%);
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD DATA & MODELS
# ============================================================================
@st.cache_resource
def load_all_resources():
    """Load all data and models with enhanced error handling"""
    try:
        # Load data files
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
        st.info("💡 If models are missing, you can still use the dashboard for data analysis")
        return None

# Load resources
with st.spinner("🚀 Loading F1 Prediction Engine..."):
    resources = load_all_resources()

# Initialize variables
final_df = pd.DataFrame()
driver_stats = pd.DataFrame()
constructor_stats = pd.DataFrame()
models_loaded = False

if resources is not None:
    final_df = resources['final_df']
    driver_stats = resources['driver_stats']
    constructor_stats = resources['constructor_stats']
    models_loaded = True

# Convert date column if data exists
if not final_df.empty:
    final_df['raceDate'] = pd.to_datetime(final_df['raceDate'])

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

if not final_df.empty:
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Races", final_df['raceId'].nunique())
    with col2:
        st.metric("Drivers", final_df['driverRef'].nunique())
    st.sidebar.metric("Seasons", final_df['year'].nunique())
else:
    st.sidebar.warning("No data loaded")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 System Status")
if models_loaded:
    st.sidebar.success("✅ Models Loaded")
else:
    st.sidebar.error("❌ Models Missing")

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
        if not final_df.empty:
            st.metric("Latest Season", int(final_df['year'].max()))
        else:
            st.metric("Latest Season", "N/A")
    
    st.markdown("---")
    
    # Feature highlights
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        ### 🎯 Predictions
        - Race Winners
        - Podium Finishes  
        - Points Scored
        - DNF Probability
        """)
    
    with col2:
        st.markdown("""
        ### 📈 Analytics
        - Driver Performance
        - Constructor Stats
        - Historical Trends
        - Real-time Insights
        """)
    
    with col3:
        st.markdown("""
        ### 🤖 Models
        - Neural Networks
        - Random Forest
        - Gradient Boosting
        - Ensemble Methods
        """)
    
    with col4:
        st.markdown("""
        ### 🔧 Features
        - Live Predictions
        - Model Comparison
        - Confidence Analysis
        - Performance Metrics
        """)
    
    st.markdown("---")
    
    # Key metrics dashboard
    if not final_df.empty:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            all_wins = final_df[final_df['isWin'] == 1].groupby('driverRef').size().max()
            st.metric("Most Wins", all_wins)
        
        with col2:
            recent_year = final_df['year'].max()
            recent_races = final_df[final_df['year'] == recent_year]['raceId'].nunique()
            st.metric(f"Races ({int(recent_year)})", recent_races)
        
        with col3:
            avg_points = final_df[final_df['points'] > 0]['points'].mean()
            st.metric("Avg Points", f"{avg_points:.1f}")
        
        with col4:
            finish_rate = (final_df['isFinished'].sum() / len(final_df) * 100)
            st.metric("Finish Rate", f"{finish_rate:.1f}%")
        
        st.markdown("---")
        st.markdown("### 🏁 Top Drivers (All Time)")
        
        top_drivers = final_df[final_df['isWin'] == 1].groupby('driverRef').size().sort_values(ascending=False).head(10)
        
        fig = px.bar(
            x=top_drivers.values, 
            y=top_drivers.index, 
            orientation='h',
            color=top_drivers.values, 
            color_continuous_scale='Reds',
            labels={'x': 'Wins', 'y': 'Driver'},
            title="All-Time Wins Leaderboard"
        )
        fig.update_layout(template='plotly_dark', height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("📊 No data available. Please check your data files.")

# ============================================================================
# DATA ANALYSIS PAGE
# ============================================================================
elif page == "📊 Data Analysis":
    st.markdown("# 📊 F1 Data Analysis & Trends")
    
    if final_df.empty:
        st.error("No data available for analysis")
        st.stop()
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Races", "🎯 Results", "💰 Points", "🔴 DNF", "🏆 Podiums"])
    
    with tab1:
        st.subheader("Races Per Season")
        races_by_year = final_df.groupby('year')['raceId'].nunique()
        
        fig = px.bar(
            x=races_by_year.index, 
            y=races_by_year.values,
            labels={'x': 'Year', 'y': 'Number of Races'},
            color_discrete_sequence=['#FF1801'],
            title="F1 Races Per Season"
        )
        fig.update_layout(template='plotly_dark', height=450, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Max Races/Season", int(races_by_year.max()))
        with col2:
            st.metric("Min Races/Season", int(races_by_year.min()))
        with col3:
            st.metric("Avg Races/Season", f"{races_by_year.mean():.1f}")
    
    with tab2:
        st.subheader("Result Distribution (All Time)")
        results_dist = pd.DataFrame({
            'Category': ['Wins', 'Podiums', 'Finishes', 'DNF'],
            'Count': [
                final_df['isWin'].sum(), 
                final_df['isPodium'].sum(), 
                final_df['isFinished'].sum(), 
                final_df['isDNF'].sum()
            ]
        })
        
        fig = px.pie(
            results_dist, 
            values='Count', 
            names='Category',
            color_discrete_sequence=['#FFD700', '#C0C0C0', '#CD7F32', '#FF6B6B'],
            title="Race Results Distribution"
        )
        fig.update_layout(template='plotly_dark', height=450)
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Wins", int(final_df['isWin'].sum()))
        with col2:
            st.metric("Podiums", int(final_df['isPodium'].sum()))
        with col3:
            st.metric("Finishes", int(final_df['isFinished'].sum()))
        with col4:
            st.metric("DNF", int(final_df['isDNF'].sum()))
    
    with tab3:
        st.subheader("Points Distribution")
        points_data = final_df[final_df['points'] > 0]['points'].value_counts().sort_index(ascending=False).head(15)
        
        fig = px.bar(
            x=points_data.values, 
            y=points_data.index,
            orientation='h', 
            color_discrete_sequence=['#0082FA'],
            labels={'x': 'Frequency', 'y': 'Points'},
            title="Top 15 Points Distributions"
        )
        fig.update_layout(template='plotly_dark', height=450)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("DNF (Did Not Finish) Trends")
        dnf_by_year = final_df.groupby('year').agg({
            'isDNF': 'sum',
            'raceId': 'count'
        }).reset_index()
        dnf_by_year['dnf_rate'] = (dnf_by_year['isDNF'] / dnf_by_year['raceId'] * 100)
        
        fig = px.line(
            dnf_by_year, 
            x='year', 
            y='dnf_rate', 
            markers=True, 
            color_discrete_sequence=['#FF1801'],
            labels={'year': 'Year', 'dnf_rate': 'DNF Rate (%)'},
            title="DNF Rate Trend Over Years"
        )
        fig.update_layout(template='plotly_dark', height=450, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.subheader("Podium Rate by Year")
        podium_by_year = final_df.groupby('year').agg({
            'isPodium': 'sum',
            'raceId': 'count'
        }).reset_index()
        podium_by_year['podium_rate'] = (podium_by_year['isPodium'] / podium_by_year['raceId'] * 100)
        
        fig = px.area(
            podium_by_year, 
            x='year', 
            y='podium_rate',
            color_discrete_sequence=['#C0C0C0'],
            labels={'year': 'Year', 'podium_rate': 'Podium Rate (%)'},
            title="Podium Rate Trend"
        )
        fig.update_layout(template='plotly_dark', height=450, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# SINGLE RACE PREDICTION
# ============================================================================
elif page == "🎯 Single Race Prediction":
    st.markdown("# 🎯 Single Race Prediction Engine")
    
    if not models_loaded:
        st.error("❌ Models not loaded. Cannot make predictions.")
        st.info("💡 Please check that all model files are available in the models directory")
        st.stop()
    
    st.info("📌 Enter driver and race conditions to predict outcomes using ML models")
    
    # Input sections in expanders for better organization
    with st.expander("🏁 Race Conditions", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            grid_position = st.slider("Grid Position", 1, 20, 5, help="Starting position on the grid")
        
        with col2:
            qual_position = st.slider("Qualifying Position", 1, 20, 6, help="Position achieved in qualifying")
        
        with col3:
            pit_stops = st.slider("Expected Pit Stops", 0, 5, 2, help="Number of planned pit stops")
    
    with st.expander("⏱️ Performance Metrics"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_lap_time = st.number_input("Avg Lap Time (ms)", 70000, 100000, 85000, 
                                         help="Average lap time in milliseconds")
        
        with col2:
            lap_time_consistency = st.number_input("Lap Time Consistency (ms)", 0, 5000, 500,
                                                 help="Standard deviation of lap times")
        
        with col3:
            pit_stop_duration = st.number_input("Avg Pit Stop Duration (s)", 15.0, 40.0, 25.0, step=0.5,
                                              help="Average pit stop duration in seconds")
    
    with st.expander("👤 Driver Statistics"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            driver_wins = st.number_input("Driver Career Wins", 0, 100, 5)
        
        with col2:
            driver_podiums = st.number_input("Driver Career Podiums", 0, 200, 20)
        
        with col3:
            driver_points = st.number_input("Driver Season Points", 0, 500, 100)
    
    with st.expander("🏭 Constructor Statistics"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            constructor_wins = st.number_input("Constructor Wins", 0, 500, 50)
        
        with col2:
            constructor_points = st.number_input("Constructor Points", 0, 1000, 300)
        
        with col3:
            finish_rate = st.number_input("Finish Rate (%)", 0.0, 100.0, 75.0, step=1.0)
    
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
    
    if not any([use_nn, use_rf, use_gb]):
        st.warning("⚠️ Please select at least one model for prediction")
    
    if st.button("🚀 PREDICT", use_container_width=True, key="predict_btn"):
        with st.spinner("🤖 Running ML models..."):
            # Create feature array (matching training features)
            X_new = np.array([[
                grid_position, qual_position, grid_position - qual_position, 
                qual_position - grid_position, avg_lap_time, lap_time_consistency, 
                pit_stops, pit_stop_duration, 0,  # overtakesCount
                driver_points, driver_wins, driver_podiums, finish_rate,
                driver_points/10, 0,  # avgPointsPerRace, totalOvertakes
                driver_wins/2, driver_podiums/2, driver_points*0.8,  # lag features
                constructor_wins, constructor_wins, finish_rate,  # constructor stats
                finish_rate*0.9  # finishRate_lag_1
            ]])
            
            # Scale features
            X_scaled = scaler.transform(X_new)
            
            # Make predictions
            predictions = {}
            
            if use_nn:
                pred_nn_win = nn_winner.predict(X_scaled, verbose=0)[0][0]
                pred_nn_podium = nn_podium.predict(X_scaled, verbose=0)[0][0]
                pred_nn_points = nn_points.predict(X_scaled, verbose=0)[0][0]
                predictions['Neural Network'] = {
                    'win': pred_nn_win,
                    'podium': pred_nn_podium,
                    'points': pred_nn_points
                }
            
            if use_rf:
                pred_rf_win = rf_winner.predict_proba(X_new)[0][1]
                pred_rf_points = rf_points.predict(X_new)[0]
                predictions['Random Forest'] = {
                    'win': pred_rf_win,
                    'podium': pred_rf_win * 0.7,  # Estimate podium from win probability
                    'points': pred_rf_points
                }
            
            if use_gb:
                pred_gb_win = gb_winner.predict_proba(X_new)[0][1]
                predictions['Gradient Boosting'] = {
                    'win': pred_gb_win,
                    'podium': pred_gb_win * 0.7,  # Estimate podium from win probability
                    'points': pred_gb_win * 20  # Estimate points from win probability
                }
            
            # Calculate ensemble predictions
            ensemble_win = np.mean([predictions[m]['win'] for m in predictions])
            ensemble_podium = np.mean([predictions[m]['podium'] for m in predictions])
            ensemble_points = np.mean([predictions[m]['points'] for m in predictions])
            
            # Display results
            st.markdown("---")
            st.markdown("## 🎯 PREDICTION RESULTS")
            
            # Main metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                win_delta = f"{(ensemble_win-0.5)*100:+.1f}%" if ensemble_win > 0.5 else None
                st.metric("🏆 Win Probability", f"{ensemble_win:.1%}", delta=win_delta)
            
            with col2:
                st.metric("🥇 Podium Probability", f"{ensemble_podium:.1%}")
            
            with col3:
                st.metric("📊 Expected Points", f"{ensemble_points:.1f}")
            
            # Confidence indicator
            st.markdown("### 📈 Confidence Level")
            confidence_score = (ensemble_win + ensemble_podium) / 2
            
            if confidence_score > 0.7:
                confidence_text = "High Confidence"
                confidence_color = "green"
            elif confidence_score > 0.4:
                confidence_text = "Medium Confidence" 
                confidence_color = "orange"
            else:
                confidence_text = "Low Confidence"
                confidence_color = "red"
            
            st.progress(float(confidence_score), text=f"{confidence_text} ({confidence_score:.1%})")
            
            # Detailed predictions
            st.markdown("---")
            st.markdown("## 📋 Model-by-Model Breakdown")
            
            df_predictions = pd.DataFrame(predictions).T
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Prediction Scores")
                styled_df = df_predictions.style.format({
                    'win': '{:.2%}',
                    'podium': '{:.2%}', 
                    'points': '{:.1f}'
                })
                st.dataframe(styled_df, use_container_width=True)
            
            with col2:
                st.markdown("### 📈 Win Probability Comparison")
                fig = px.bar(
                    x=list(predictions.keys()),
                    y=[predictions[m]['win'] for m in predictions],
                    color=list(predictions.keys()),
                    color_discrete_sequence=['#FF1801', '#0082FA', '#FFD700'],
                    labels={'x': 'Model', 'y': 'Win Probability'},
                    title="Win Probability by Model"
                )
                fig.update_layout(template='plotly_dark', height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Win probability comparison
                fig = go.Figure()
                colors = ['#FF1801', '#0082FA', '#FFD700']
                for i, (model, values) in enumerate(predictions.items()):
                    fig.add_trace(go.Bar(
                        name=model,
                        x=[model],
                        y=[values['win']],
                        marker_color=colors[i % len(colors)],
                        text=[f"{values['win']:.1%}"],
                        textposition='auto'
                    ))
                fig.update_layout(
                    title="Win Probability Comparison", 
                    template='plotly_dark', 
                    height=400,
                    yaxis_tickformat='.0%'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Points prediction comparison
                fig = go.Figure()
                for i, (model, values) in enumerate(predictions.items()):
                    fig.add_trace(go.Bar(
                        name=model,
                        x=[model],
                        y=[values['points']],
                        marker_color=colors[i % len(colors)],
                        text=[f"{values['points']:.1f}"],
                        textposition='auto'
                    ))
                fig.update_layout(
                    title="Expected Points Comparison", 
                    template='plotly_dark', 
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Recommendation
            st.markdown("---")
            st.markdown("## 💡 Race Strategy Recommendation")
            
            if ensemble_win > 0.3:
                st.success(f"**Aggressive Strategy Recommended** - High win probability ({ensemble_win:.1%}) suggests going for the win with optimal pit strategy.")
            elif ensemble_podium > 0.4:
                st.info(f"**Balanced Strategy Recommended** - Good podium chance ({ensemble_podium:.1%}) suggests focusing on consistent performance.")
            else:
                st.warning(f"**Conservative Strategy Recommended** - Focus on points finish with safe pit strategy.")

# ============================================================================
# DRIVER ANALYSIS
# ============================================================================
elif page == "👥 Driver Analysis":
    st.markdown("# 👥 Driver Performance Analysis")
    
    if driver_stats.empty:
        st.error("No driver statistics available")
        st.stop()
    
    driver_list = sorted(driver_stats['driver'].unique())
    selected_driver = st.selectbox("🏎️ Select Driver", driver_list)
    
    driver_data = driver_stats[driver_stats['driver'] == selected_driver].sort_values('year')
    
    if len(driver_data) > 0:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🏆 Total Wins", int(driver_data['wins'].sum()))
        with col2:
            st.metric("🥇 Total Podiums", int(driver_data['podiums'].sum()))
        with col3:
            st.metric("💰 Total Points", int(driver_data['totalPoints'].sum()))
        with col4:
            st.metric("📅 Seasons", len(driver_data))
        
        st.markdown("---")
        
        # Performance trends
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(
                driver_data, 
                x='year', 
                y=['wins', 'podiums'],
                markers=True, 
                color_discrete_map={
                    'wins': '#FFD700',
                    'podiums': '#C0C0C0'
                },
                title=f"{selected_driver} - Wins & Podiums Trend"
            )
            fig.update_layout(template='plotly_dark', height=450, hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.line(
                driver_data, 
                x='year', 
                y='totalPoints',
                markers=True, 
                color_discrete_sequence=['#0082FA'],
                title=f"{selected_driver} - Season Points Trend"
            )
            fig.update_layout(template='plotly_dark', height=450, hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
        
        # Additional metrics
        col1, col2 = st.columns(2)
        
        with col1:
            if 'finishRate' in driver_data.columns:
                fig = px.line(
                    driver_data, 
                    x='year', 
                    y='finishRate',
                    markers=True,
                    color_discrete_sequence=['#00FF00'],
                    title=f"{selected_driver} - Finish Rate Trend"
                )
                fig.update_layout(template='plotly_dark', height=400, hovermode='x unified')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'avgPointsPerRace' in driver_data.columns:
                fig = px.bar(
                    driver_data, 
                    x='year', 
                    y='avgPointsPerRace',
                    color_discrete_sequence=['#FF6B6B'],
                    title=f"{selected_driver} - Average Points Per Race"
                )
                fig.update_layout(template='plotly_dark', height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 📊 Detailed Statistics")
        
        # Format and display dataframe
        display_cols = [col for col in driver_data.columns if col != 'driver']
        formatted_data = driver_data[display_cols].copy()
        
        # Apply formatting
        formatting_rules = {}
        if 'totalPoints' in formatted_data.columns:
            formatting_rules['totalPoints'] = '{:.0f}'
        if 'avgLapTime' in formatted_data.columns:
            formatting_rules['avgLapTime'] = '{:.0f}'
        if 'finishRate' in formatted_data.columns:
            formatting_rules['finishRate'] = '{:.1f}%'
        if 'podiumRate' in formatted_data.columns:
            formatting_rules['podiumRate'] = '{:.1f}%'
        if 'avgPointsPerRace' in formatted_data.columns:
            formatting_rules['avgPointsPerRace'] = '{:.2f}'
        
        st.dataframe(
            formatted_data.style.format(formatting_rules), 
            use_container_width=True,
            height=400
        )

    else:
        st.warning(f"No data available for driver: {selected_driver}")

# ============================================================================
# CONSTRUCTOR ANALYSIS
# ============================================================================
elif page == "🏭 Constructor Analysis":
    st.markdown("# 🏭 Constructor Performance Analysis")
    
    if constructor_stats.empty:
        st.error("No constructor statistics available")
        st.stop()
    
    constructor_list = sorted(constructor_stats['constructor'].unique())
    selected_constructor = st.selectbox("🏭 Select Constructor", constructor_list)
    
    const_data = constructor_stats[constructor_stats['constructor'] == selected_constructor].sort_values('year')
    
    if len(const_data) > 0:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🏆 Total Wins", int(const_data['wins'].sum()))
        with col2:
            st.metric("💰 Total Points", int(const_data['totalPoints'].sum()))
        with col3:
            win_rate = const_data['winRate'].mean() if 'winRate' in const_data.columns else 0
            st.metric("🎯 Avg Win Rate", f"{win_rate:.1f}%")
        with col4:
            st.metric("📅 Seasons", len(const_data))
        
        st.markdown("---")
        
        # Performance trends
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig = px.bar(
                const_data, 
                x='year', 
                y='totalPoints',
                color_discrete_sequence=['#0082FA'],
                title=f"{selected_constructor} - Points by Season"
            )
            fig.update_layout(template='plotly_dark', height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.line(
                const_data, 
                x='year', 
                y='wins',
                markers=True, 
                color_discrete_sequence=['#FFD700'],
                title=f"{selected_constructor} - Wins Trend"
            )
            fig.update_layout(template='plotly_dark', height=400, hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            if 'finishRate' in const_data.columns:
                fig = px.line(
                    const_data, 
                    x='year', 
                    y='finishRate',
                    markers=True, 
                    color_discrete_sequence=['#FF1801'],
                    title=f"{selected_constructor} - Finish Rate Trend"
                )
                fig.update_layout(template='plotly_dark', height=400, hovermode='x unified')
                st.plotly_chart(fig, use_container_width=True)
        
        # Championship performance
        if 'championshipPosition' in const_data.columns:
            st.markdown("---")
            st.markdown("### 🏆 Championship Performance")
            
            fig = px.line(
                const_data, 
                x='year', 
                y='championshipPosition',
                markers=True,
                color_discrete_sequence=['#FFD700'],
                title=f"{selected_constructor} - Championship Position"
            )
            fig.update_yaxis(autorange="reversed")  # Lower number = better position
            fig.update_layout(template='plotly_dark', height=400, hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 📊 Detailed Statistics")
        
        # Format and display dataframe
        display_cols = [col for col in const_data.columns if col != 'constructor']
        formatted_data = const_data[display_cols].copy()
        
        # Apply formatting
        formatting_rules = {}
        if 'totalPoints' in formatted_data.columns:
            formatting_rules['totalPoints'] = '{:.0f}'
        if 'finishRate' in formatted_data.columns:
            formatting_rules['finishRate'] = '{:.1f}%'
        if 'winRate' in formatted_data.columns:
            formatting_rules['winRate'] = '{:.1f}%'
        if 'avgPointsPerRace' in formatted_data.columns:
            formatting_rules['avgPointsPerRace'] = '{:.2f}'
        if 'championshipPosition' in formatted_data.columns:
            formatting_rules['championshipPosition'] = '{:.0f}'
        
        st.dataframe(
            formatted_data.style.format(formatting_rules), 
            use_container_width=True,
            height=400
        )

    else:
        st.warning(f"No data available for constructor: {selected_constructor}")

# ============================================================================
# ADVANCED PREDICTIONS
# ============================================================================
elif page == "🔮 Advanced Predictions":
    st.markdown("# 🔮 Advanced Predictions & Insights")
    
    if final_df.empty:
        st.error("No data available for advanced predictions")
        st.stop()
    
    tab1, tab2, tab3 = st.tabs(["📊 Grid Analysis", "👥 Driver Form", "🏆 Championship Race"])
    
    with tab1:
        st.subheader("Grid Position Performance Analysis")
        
        with st.spinner("Analyzing grid performance..."):
            grid_analysis = final_df.groupby('grid').agg({
                'isWin': 'sum',
                'isPodium': 'sum',
                'points': 'mean',
                'driverId': 'count'
            }).reset_index()
            grid_analysis.columns = ['Grid', 'Wins', 'Podiums', 'Avg_Points', 'Races']
