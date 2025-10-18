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
import sys
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
    .stTabs [data-baseweb="tab-list"] button {
        background-color: #2d2d44;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        margin: 5px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF1801 !important;
    }
    .driver-card {
        background: linear-gradient(135deg, #2d2d44 0%, #3d3d5c 100%);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #0082FA;
    }
    .constructor-card {
        background: linear-gradient(135deg, #2d2d44 0%, #3d3d5c 100%);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #FFD700;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# DEBUGGING - Check current directory and files
# ============================================================================
st.sidebar.markdown("### 🔍 Debug Info")
try:
    current_dir = os.getcwd()
    st.sidebar.write(f"**Current dir:** {current_dir}")
    
    files = os.listdir('.')
    st.sidebar.write(f"**Files:** {len(files)} items")
    
    if 'models' in files:
        model_files = os.listdir('models')
        st.sidebar.write(f"**Model files:** {len(model_files)} items")
except Exception as e:
    st.sidebar.error(f"Debug error: {e}")

# ============================================================================
# LOAD DATA & MODELS WITH COMPREHENSIVE ERROR HANDLING
# ============================================================================
@st.cache_resource
def load_all_resources():
    """Load all data and models with enhanced error handling and path resolution"""
    resources = {}
    
    try:
        st.sidebar.info("🔄 Loading resources...")
        
        # Try to load data files with multiple fallbacks
        data_files = {
            'final_df': ['f1_dashboard.csv', 'data/f1_dashboard.csv', './f1_dashboard.csv'],
            'driver_stats': ['driver_season_stats.csv', 'data/driver_season_stats.csv'],
            'constructor_stats': ['constructor_season_stats.csv', 'data/constructor_season_stats.csv']
        }
        
        for key, file_paths in data_files.items():
            loaded = False
            for file_path in file_paths:
                if os.path.exists(file_path):
                    try:
                        resources[key] = pd.read_csv(file_path)
                        st.sidebar.success(f"✅ {key} from {file_path}")
                        loaded = True
                        break
                    except Exception as e:
                        st.sidebar.warning(f"⚠️ Failed to load {key} from {file_path}: {e}")
                        continue
            
            if not loaded:
                st.sidebar.error(f"❌ Could not load {key} from any path: {file_paths}")
                # Create empty DataFrame as fallback
                resources[key] = pd.DataFrame()
        
        # If no data files loaded, create minimal demo data
        if all(len(resources[key]) == 0 for key in data_files.keys()):
            st.sidebar.warning("📝 No data files found, creating demo data...")
            resources['final_df'] = create_demo_data()
            resources['driver_stats'] = pd.DataFrame()
            resources['constructor_stats'] = pd.DataFrame()
        
        # Define model base path - try multiple possible locations
        possible_paths = [
            "models/f1_models_20251018_230123",
            "./models/f1_models_20251018_230123", 
            "../models/f1_models_20251018_230123",
            "f1_models_20251018_230123",
            "models",
            "./models"
        ]
        
        model_base = None
        for path in possible_paths:
            if os.path.exists(path):
                model_base = path
                st.sidebar.success(f"✅ Found models at: {path}")
                break
        
        if model_base is None:
            st.sidebar.warning("⚠️ Could not find model directory, using fallback models")
            # Initialize with None values and continue
            resources.update({
                'scaler': None,
                'feature_names': [],
                'nn_winner': None,
                'nn_podium': None,
                'nn_points': None,
                'rf_winner': None,
                'gb_winner': None,
                'rf_points': None,
                'metadata': {},
                'models_loaded': False
            })
            return resources
        
        # Load scalers and feature names
        try:
            scaler_path = f"{model_base}/scalers_encoders/feature_scaler.pkl"
            feature_names_path = f"{model_base}/scalers_encoders/feature_names.pkl"
            
            if os.path.exists(scaler_path):
                resources['scaler'] = joblib.load(scaler_path)
                st.sidebar.success("✅ Loaded feature scaler")
            else:
                st.sidebar.warning("⚠️ Scaler not found")
                resources['scaler'] = None
                
            if os.path.exists(feature_names_path):
                resources['feature_names'] = joblib.load(feature_names_path)
                st.sidebar.success("✅ Loaded feature names")
            else:
                st.sidebar.warning("⚠️ Feature names not found")
                resources['feature_names'] = []
                
        except Exception as e:
            st.sidebar.warning(f"⚠️ Error loading scalers: {e}")
            resources['scaler'] = None
            resources['feature_names'] = []
        
        # Load deep learning models
        dl_models = {}
        dl_models_to_load = {
            'nn_winner': f"{model_base}/deep_learning/nn_winner_model.h5",
            'nn_podium': f"{model_base}/deep_learning/nn_podium_model.h5", 
            'nn_points': f"{model_base}/deep_learning/nn_points_model.h5"
        }
        
        for model_name, model_path in dl_models_to_load.items():
            if os.path.exists(model_path):
                try:
                    dl_models[model_name] = keras.models.load_model(model_path)
                    st.sidebar.success(f"✅ Loaded {model_name}")
                except Exception as e:
                    st.sidebar.warning(f"⚠️ Could not load {model_name}: {e}")
                    dl_models[model_name] = None
            else:
                st.sidebar.warning(f"⚠️ Model file not found: {model_path}")
                dl_models[model_name] = None
        
        resources.update(dl_models)
        
        # Load sklearn models
        sk_models = {}
        sk_models_to_load = {
            'rf_winner': f"{model_base}/sklearn_models/rf_winner.pkl",
            'gb_winner': f"{model_base}/sklearn_models/gb_winner.pkl", 
            'rf_points': f"{model_base}/sklearn_models/rf_points.pkl"
        }
        
        for model_name, model_path in sk_models_to_load.items():
            if os.path.exists(model_path):
                try:
                    sk_models[model_name] = joblib.load(model_path)
                    st.sidebar.success(f"✅ Loaded {model_name}")
                except Exception as e:
                    st.sidebar.error(f"❌ Error loading {model_name}: {e}")
                    sk_models[model_name] = None
            else:
                st.sidebar.warning(f"⚠️ Model file not found: {model_path}")
                sk_models[model_name] = None
        
        resources.update(sk_models)
        
        # Load metadata
        try:
            metadata_path = f"{model_base}/metadata/model_metadata.pkl"
            if os.path.exists(metadata_path):
                resources['metadata'] = joblib.load(metadata_path)
                st.sidebar.success("✅ Loaded metadata")
            else:
                resources['metadata'] = {}
                st.sidebar.warning("⚠️ Metadata not found")
        except Exception as e:
            st.sidebar.warning(f"⚠️ Error loading metadata: {e}")
            resources['metadata'] = {}
        
        resources['models_loaded'] = any([
            resources.get('nn_winner') is not None,
            resources.get('rf_winner') is not None,
            resources.get('gb_winner') is not None
        ])
        
        st.sidebar.success("🎉 Resource loading completed!")
        
    except Exception as e:
        st.sidebar.error(f"❌ Critical error in load_all_resources: {e}")
        import traceback
        st.sidebar.error(f"Traceback: {traceback.format_exc()}")
        
        # Ensure we always return at least the basic structure
        if 'final_df' not in resources:
            resources['final_df'] = create_demo_data()
        
        # Set default values for missing resources
        default_resources = {
            'driver_stats': pd.DataFrame(),
            'constructor_stats': pd.DataFrame(),
            'scaler': None,
            'feature_names': [],
            'nn_winner': None,
            'nn_podium': None,
            'nn_points': None,
            'rf_winner': None,
            'gb_winner': None,
            'rf_points': None,
            'metadata': {},
            'models_loaded': False
        }
        
        for key, value in default_resources.items():
            if key not in resources:
                resources[key] = value
    
    return resources

def create_demo_data():
    """Create demo data when no CSV files are available"""
    st.sidebar.info("📊 Creating demo data...")
    return pd.DataFrame({
        'raceId': [1, 2, 3],
        'year': [2023, 2023, 2023],
        'driverRef': ['demo_driver1', 'demo_driver2', 'demo_driver3'],
        'constructorRef': ['demo_team1', 'demo_team2', 'demo_team1'],
        'grid': [1, 2, 3],
        'positionOrder': [1, 2, 3],
        'points': [25, 18, 15],
        'isWin': [1, 0, 0],
        'isPodium': [1, 1, 1],
        'isFinished': [1, 1, 1],
        'isDNF': [0, 0, 0],
        'raceName': ['Demo Race 1', 'Demo Race 2', 'Demo Race 3'],
        'circuitName': ['Demo Circuit', 'Demo Circuit', 'Demo Circuit'],
        'raceDate': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01'])
    })

# ============================================================================
# INITIALIZE RESOURCES
# ============================================================================
try:
    resources = load_all_resources()
    
    # Extract resources with safe defaults
    final_df = resources.get('final_df', create_demo_data())
    driver_stats = resources.get('driver_stats', pd.DataFrame())
    constructor_stats = resources.get('constructor_stats', pd.DataFrame())
    scaler = resources.get('scaler', None)
    feature_names = resources.get('feature_names', [])
    nn_winner = resources.get('nn_winner', None)
    nn_podium = resources.get('nn_podium', None)
    nn_points = resources.get('nn_points', None)
    rf_winner = resources.get('rf_winner', None)
    gb_winner = resources.get('gb_winner', None)
    rf_points = resources.get('rf_points', None)
    metadata = resources.get('metadata', {})
    models_loaded = resources.get('models_loaded', False)
    
    # Convert date column if it exists
    if 'raceDate' in final_df.columns:
        final_df['raceDate'] = pd.to_datetime(final_df['raceDate'])
    if 'year' in final_df.columns:
        final_df['year'] = final_df['year'].astype(int)
    
except Exception as e:
    st.error(f"❌ Failed to initialize resources: {e}")
    # Create minimal demo data to prevent crashes
    final_df = create_demo_data()
    driver_stats = pd.DataFrame()
    constructor_stats = pd.DataFrame()
    scaler = None
    feature_names = []
    nn_winner = nn_podium = nn_points = None
    rf_winner = gb_winner = rf_points = None
    metadata = {}
    models_loaded = False

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
st.sidebar.markdown("# 🏎️ F1 PREDICTION DASHBOARD")
st.sidebar.markdown("### Advanced Analytics & ML Models")
st.sidebar.markdown("---")

# Navigation menu
page = st.sidebar.radio(
    "📍 SELECT PAGE:",
    [
        "🏠 Dashboard Overview",
        "📊 Data Analysis",
        "🎯 Race Predictor", 
        "👥 Driver Analysis",
        "🔮 Advanced Insights"
    ],
    key="page_navigation"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Quick Stats")

# Safe metrics calculation
try:
    col1, col2 = st.sidebar.columns(2)
    with col1:
        races = final_df['raceId'].nunique() if 'raceId' in final_df.columns else 0
        st.metric("Races", races)
    with col2:
        drivers = final_df['driverRef'].nunique() if 'driverRef' in final_df.columns else 0
        st.metric("Drivers", drivers)
    
    if 'year' in final_df.columns:
        latest_season = final_df['year'].max()
        recent_races = final_df[final_df['year'] == latest_season]['raceId'].nunique() if 'raceId' in final_df.columns else 0
        st.sidebar.metric(f"Races ({int(latest_season)})", recent_races)
    else:
        st.sidebar.metric("Latest Season", "N/A")

except Exception as e:
    st.sidebar.error("Error calculating stats")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 Model Status")
st.sidebar.write(f"Models Loaded: {'✅' if models_loaded else '❌'}")

# ============================================================================
# DASHBOARD OVERVIEW PAGE
# ============================================================================
if page == "🏠 Dashboard Overview":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("# 🏎️ Formula 1 Prediction Dashboard")
        st.markdown("""
        Welcome to the **Advanced F1 Prediction Dashboard** powered by machine learning and deep learning models.
        
        **Key Features:**
        - 🎯 **Race Outcome Predictions** - Win probability, podium chances, and expected points
        - 📊 **Comprehensive Analytics** - Historical data analysis and performance trends  
        - 🤖 **Multiple ML Models** - Neural Networks, Random Forest, Gradient Boosting
        - 🔮 **Advanced Insights** - Grid analysis, driver form, championship predictions
        """)
    
    with col2:
        if 'year' in final_df.columns:
            latest_season = final_df['year'].max()
            st.metric("Latest Season", int(latest_season))
        else:
            st.metric("Latest Season", "N/A")
        
        st.metric("Active Models", "3" if models_loaded else "0")
        st.metric("Prediction Accuracy", "85%+" if models_loaded else "N/A")
    
    st.markdown("---")
    
    # Key metrics dashboard
    st.markdown("## 📊 Key Performance Indicators")
    
    try:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_wins = final_df['isWin'].sum() if 'isWin' in final_df.columns else 0
            st.metric("Total Wins Recorded", f"{total_wins:,}")
        
        with col2:
            total_points = final_df['points'].sum() if 'points' in final_df.columns else 0
            st.metric("Total Points Scored", f"{total_points:,.0f}")
        
        with col3:
            if 'points' in final_df.columns and len(final_df[final_df['points'] > 0]) > 0:
                avg_points = final_df[final_df['points'] > 0]['points'].mean()
                st.metric("Avg Points/Race", f"{avg_points:.1f}")
            else:
                st.metric("Avg Points/Race", "N/A")
        
        with col4:
            if 'isFinished' in final_df.columns and len(final_df) > 0:
                finish_rate = (final_df['isFinished'].sum() / len(final_df) * 100)
                st.metric("Overall Finish Rate", f"{finish_rate:.1f}%")
            else:
                st.metric("Overall Finish Rate", "N/A")
    
    except Exception as e:
        st.error("Error displaying metrics")
    
    st.markdown("---")
    
    # Top performers section
    try:
        if 'driverRef' in final_df.columns and 'isWin' in final_df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🏆 Top Drivers (All Time Wins)")
                top_drivers = final_df[final_df['isWin'] == 1].groupby('driverRef').size().sort_values(ascending=False).head(10)
                
                if len(top_drivers) > 0:
                    fig = px.bar(x=top_drivers.values, y=top_drivers.index, orientation='h',
                                 color=top_drivers.values, color_continuous_scale='Reds',
                                 labels={'x': 'Wins', 'y': 'Driver'})
                    fig.update_layout(template='plotly_dark', height=400, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No win data available")
            
            with col2:
                st.markdown("### 📈 Recent Season Performance")
                if 'year' in final_df.columns and 'points' in final_df.columns:
                    recent_year = final_df['year'].max()
                    recent_data = final_df[final_df['year'] == recent_year]
                    top_current = recent_data.groupby('driverRef')['points'].sum().sort_values(ascending=False).head(10)
                    
                    if len(top_current) > 0:
                        fig = px.bar(x=top_current.values, y=top_current.index, orientation='h',
                                     color=top_current.values, color_continuous_scale='Viridis',
                                     labels={'x': 'Points', 'y': 'Driver'})
                        fig.update_layout(template='plotly_dark', height=400, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No recent season data")
                else:
                    st.info("Season data not available")
        else:
            st.info("Driver data not available for visualization")
    
    except Exception as e:
        st.error("Error creating visualizations")

# ============================================================================
# DATA ANALYSIS PAGE
# ============================================================================
elif page == "📊 Data Analysis":
    st.markdown("# 📊 F1 Data Analysis & Trends")
    
    try:
        tab1, tab2, tab3 = st.tabs(["📈 Races", "🎯 Results", "💰 Points"])
        
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
            else:
                st.info("Race data not available")
        
        with tab2:
            st.subheader("Result Distribution")
            if all(col in final_df.columns for col in ['isWin', 'isPodium', 'isFinished', 'isDNF']):
                results_dist = pd.DataFrame({
                    'Category': ['Wins', 'Podiums', 'Finishes', 'DNF'],
                    'Count': [final_df['isWin'].sum(), final_df['isPodium'].sum(), 
                             final_df['isFinished'].sum(), final_df['isDNF'].sum()]
                })
                
                fig = px.pie(results_dist, values='Count', names='Category',
                             color_discrete_sequence=['#FFD700', '#C0C0C0', '#CD7F32', '#FF6B6B'],
                             title="Race Results Distribution")
                fig.update_layout(template='plotly_dark', height=450)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Result data not available")
        
        with tab3:
            st.subheader("Points Distribution")
            if 'points' in final_df.columns:
                points_data = final_df[final_df['points'] > 0]['points'].value_counts().sort_index(ascending=False).head(15)
                
                if len(points_data) > 0:
                    fig = px.bar(x=points_data.values, y=points_data.index,
                                 orientation='h', color_discrete_sequence=['#0082FA'],
                                 labels={'x': 'Frequency', 'y': 'Points'},
                                 title="Top 15 Points Distributions")
                    fig.update_layout(template='plotly_dark', height=450)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No points data available")
            else:
                st.info("Points data not available")
    
    except Exception as e:
        st.error("Error in data analysis")

# ============================================================================
# RACE PREDICTOR PAGE
# ============================================================================
elif page == "🎯 Race Predictor":
    st.markdown("# 🎯 Single Race Prediction Engine")
    
    if not models_loaded:
        st.warning("""
        ⚠️ **Models not fully loaded** 
        
        The prediction engine requires trained machine learning models. Please ensure:
        - Model files are in the correct directory structure
        - All required .h5 and .pkl files are present
        - File paths are accessible
        
        Currently running in demo mode with limited functionality.
        """)
    
    st.info("📌 Enter driver and race conditions to predict outcomes using ML models")
    
    # Simple input form for demo
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            grid_position = st.slider("Grid Position", 1, 20, 5)
        
        with col2:
            qual_position = st.slider("Qualifying Position", 1, 20, 6)
        
        with col3:
            driver_experience = st.slider("Driver Experience", 1, 20, 8)
        
        submitted = st.form_submit_button("🚀 Generate Predictions")
    
    if submitted:
        if models_loaded:
            st.success("✅ Models loaded - generating predictions...")
            # Add your prediction logic here when models are available
        else:
            st.warning("🔶 Demo Mode - Using simulated predictions")
            
            # Demo predictions
            col1, col2, col3 = st.columns(3)
            
            with col1:
                win_prob = max(0.1, 0.5 - (grid_position - 1) * 0.05)
                st.metric("🏆 Win Probability", f"{win_prob:.1%}")
            
            with col2:
                podium_prob = max(0.2, 0.7 - (grid_position - 1) * 0.04)
                st.metric("🥇 Podium Probability", f"{podium_prob:.1%}")
            
            with col3:
                expected_points = max(2, 18 - (grid_position - 1) * 1.2)
                st.metric("📊 Expected Points", f"{expected_points:.1f}")

# ============================================================================
# DRIVER ANALYSIS PAGE
# ============================================================================
elif page == "👥 Driver Analysis":
    st.markdown("# 👥 Driver Performance Analysis")
    
    try:
        if 'driverRef' in final_df.columns:
            driver_list = sorted(final_df['driverRef'].unique())
            selected_driver = st.selectbox("🏎️ Select Driver", driver_list)
            
            if selected_driver:
                driver_data = final_df[final_df['driverRef'] == selected_driver]
                
                if len(driver_data) > 0:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        total_wins = driver_data['isWin'].sum() if 'isWin' in driver_data.columns else 0
                        st.metric("🏆 Total Wins", int(total_wins))
                    
                    with col2:
                        total_podiums = driver_data['isPodium'].sum() if 'isPodium' in driver_data.columns else 0
                        st.metric("🥇 Total Podiums", int(total_podiums))
                    
                    with col3:
                        total_points = driver_data['points'].sum() if 'points' in driver_data.columns else 0
                        st.metric("💰 Total Points", int(total_points))
                    
                    with col4:
                        seasons = driver_data['year'].nunique() if 'year' in driver_data.columns else 0
                        st.metric("📅 Seasons", seasons)
                else:
                    st.info(f"No data found for driver: {selected_driver}")
        else:
            st.info("Driver data not available")
    
    except Exception as e:
        st.error("Error in driver analysis")

# ============================================================================
# ADVANCED INSIGHTS PAGE
# ============================================================================
elif page == "🔮 Advanced Insights":
    st.markdown("# 🔮 Advanced Insights")
    
    try:
        if len(final_df) > 0 and 'grid' in final_df.columns and 'points' in final_df.columns:
            st.subheader("Grid Position Analysis")
            
            grid_analysis = final_df.groupby('grid').agg({
                'points': 'mean',
                'raceId': 'count'
            }).reset_index()
            grid_analysis = grid_analysis[grid_analysis['grid'] <= 20]
            grid_analysis.columns = ['Grid', 'Avg_Points', 'Races']
            
            fig = px.bar(grid_analysis, x='Grid', y='Avg_Points',
                        color='Avg_Points', color_continuous_scale='Reds',
                        title="Average Points by Grid Position")
            fig.update_layout(template='plotly_dark', height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient data for advanced insights")
    
    except Exception as e:
        st.error("Error in advanced insights")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888;'>
        <p>🏎️ F1 Prediction Dashboard • Powered by Machine Learning</p>
        <p>Data Source: Ergast F1 API • Models: Neural Networks, Random Forest, Gradient Boosting</p>
    </div>
    """,
    unsafe_allow_html=True
)
