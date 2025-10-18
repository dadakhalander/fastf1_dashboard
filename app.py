import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
from tensorflow import keras
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

# PAGE CONFIG
st.set_page_config(
    page_title="F1 Race Prediction Dashboard",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# STYLING
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        color: #ffffff;
    }
    h1, h2, h3 {
        color: #FF1801;
        font-weight: bold;
    }
    .metric-box {
        background: linear-gradient(135deg, #2d2d44 0%, #3d3d5c 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #FF1801;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# LOAD RESOURCES
@st.cache_resource
def load_resources():
    try:
        final_df = pd.read_csv('f1_dashboard.csv')
        driver_stats = pd.read_csv('driver_season_stats.csv')
        constructor_stats = pd.read_csv('constructor_season_stats.csv')
        
        model_base = "models/f1_models_20251018_230123"
        
        scaler = joblib.load(f"{model_base}/scalers_encoders/feature_scaler.pkl")
        rf_winner = joblib.load(f"{model_base}/sklearn_models/rf_winner.pkl")
        gb_winner = joblib.load(f"{model_base}/sklearn_models/gb_winner.pkl")
        rf_points = joblib.load(f"{model_base}/sklearn_models/rf_points.pkl")
        
        nn_winner = None
        try:
            nn_winner = keras.models.load_model(f"{model_base}/deep_learning/nn_winner_model.h5")
        except:
            pass
        
        return {
            'final_df': final_df,
            'driver_stats': driver_stats,
            'constructor_stats': constructor_stats,
            'scaler': scaler,
            'rf_winner': rf_winner,
            'gb_winner': gb_winner,
            'rf_points': rf_points,
            'nn_winner': nn_winner
        }
    except Exception as e:
        st.error(f"Error loading resources: {str(e)}")
        return None

resources = load_resources()
if resources is None:
    st.stop()

final_df = resources['final_df']
driver_stats = resources['driver_stats']
constructor_stats = resources['constructor_stats']
scaler = resources['scaler']
rf_winner = resources['rf_winner']
gb_winner = resources['gb_winner']
rf_points = resources['rf_points']
nn_winner = resources['nn_winner']

final_df['raceDate'] = pd.to_datetime(final_df['raceDate'])

# SIDEBAR
st.sidebar.markdown("<h1>🏎️ F1 PREDICTOR</h1>", unsafe_allow_html=True)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    ["Dashboard", "Race Predictor", "Driver Analysis", "Constructor Analysis", "Advanced Analytics", "Championships"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Statistics")
st.sidebar.metric("Total Races", final_df['raceId'].nunique())
st.sidebar.metric("Drivers", final_df['driverRef'].nunique())
st.sidebar.metric("Latest Season", int(final_df['year'].max()))

# HELPER FUNCTIONS
def get_top_drivers_by_season(season):
    """Get top 10 drivers for a season"""
    season_data = final_df[final_df['year'] == season]
    return season_data.groupby('driverRef')['points'].sum().sort_values(ascending=False).head(10)

def create_radar_chart(driver_name, metrics_dict):
    """Create interactive radar chart for driver comparison"""
    categories = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=driver_name,
        line_color='#FF1801'
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        template='plotly_dark',
        title=f"Performance Radar - {driver_name}",
        height=500
    )
    return fig

def normalize_value(value, min_val, max_val):
    """Normalize value to 0-100 scale"""
    if max_val == min_val:
        return 50
    return ((value - min_val) / (max_val - min_val)) * 100

# PAGE: DASHBOARD
if page == "Dashboard":
    st.markdown("<div class='main-header'><h1>F1 PREDICTION DASHBOARD</h1></div>", unsafe_allow_html=True)
    
    latest_season = final_df['year'].max()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Races", final_df['raceId'].nunique())
    with col2:
        st.metric("Career Wins", int(final_df['isWin'].sum()))
    with col3:
        st.metric("Podium Finishes", int(final_df['isPodium'].sum()))
    with col4:
        finish_rate = (final_df['isFinished'].sum() / len(final_df)) * 100
        st.metric("Finish Rate", f"{finish_rate:.1f}%")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top Drivers (All-Time)")
        top_drivers = final_df.groupby('driverRef')['isWin'].sum().sort_values(ascending=False).head(10)
        fig = px.bar(x=top_drivers.values, y=top_drivers.index, orientation='h',
                    color=top_drivers.values, color_continuous_scale='Reds',
                    title="Wins Leaders")
        fig.update_layout(template='plotly_dark', height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Races Per Season")
        races_by_year = final_df.groupby('year')['raceId'].nunique()
        fig = px.line(x=races_by_year.index, y=races_by_year.values, markers=True,
                     title="Race Count Trend", labels={'x': 'Season', 'y': 'Races'})
        fig.update_layout(template='plotly_dark', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader(f"Championship Standing {latest_season}")
    
    season_standings = final_df[final_df['year'] == latest_season].groupby('driverRef')['points'].sum().sort_values(ascending=False).head(10)
    
    for i, (driver, points) in enumerate(season_standings.items(), 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        st.write(f"{emoji} **{driver}** - {int(points)} pts")

# PAGE: RACE PREDICTOR
elif page == "Race Predictor":
    st.markdown("<div class='main-header'><h1>RACE PREDICTION ENGINE</h1></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Race Conditions")
        grid_pos = st.slider("Starting Grid Position", 1, 20, 5)
        qual_pos = st.slider("Qualifying Position", 1, 20, 3)
        pit_stops = st.slider("Planned Pit Stops", 0, 5, 2)
        avg_lap = st.number_input("Avg Lap Time (ms)", 80000, 120000, 90000)
    
    with col2:
        st.subheader("Driver Profile")
        d_wins = st.number_input("Career Wins", 0, 103, 15)
        d_podiums = st.number_input("Career Podiums", 0, 200, 45)
        d_points = st.number_input("Season Points", 0, 500, 180)
        finish_rate = st.slider("Finish Rate", 50.0, 100.0, 85.0)
    
    st.subheader("Team Profile")
    col1, col2, col3 = st.columns(3)
    with col1:
        t_wins = st.number_input("Team Wins", 0, 250, 80)
    with col2:
        t_points = st.number_input("Team Points", 0, 1000, 350)
    with col3:
        lap_consistency = st.slider("Lap Consistency", 100, 2000, 500)
    
    st.subheader("Model Selection")
    col1, col2, col3 = st.columns(3)
    with col1:
        use_rf = st.checkbox("Random Forest", True)
    with col2:
        use_gb = st.checkbox("Gradient Boosting", True)
    with col3:
        use_nn = st.checkbox("Neural Network", nn_winner is not None, disabled=nn_winner is None)
    
    if st.button("PREDICT", use_container_width=True, type="primary"):
        X_new = np.array([[
            grid_pos, qual_pos, grid_pos - qual_pos, qual_pos - grid_pos,
            avg_lap, lap_consistency, pit_stops, 22.5, 0,
            d_points, d_wins, d_podiums, finish_rate,
            d_points/10, 0, d_wins/2, d_podiums/2, d_points*0.8,
            t_points, t_wins, finish_rate, finish_rate*0.9
        ]])
        
        X_scaled = scaler.transform(X_new)
        predictions = {}
        
        if use_rf:
            pred_rf = rf_winner.predict_proba(X_new)[0][1]
            predictions['Random Forest'] = {'win': pred_rf, 'points': rf_points.predict(X_new)[0]}
        
        if use_gb:
            pred_gb = gb_winner.predict_proba(X_new)[0][1]
            predictions['Gradient Boosting'] = {'win': pred_gb, 'points': pred_gb * 25}
        
        if use_nn and nn_winner:
            pred_nn = nn_winner.predict(X_scaled, verbose=0)[0][0]
            predictions['Neural Network'] = {'win': pred_nn, 'points': pred_nn * 25}
        
        ensemble_win = np.mean([p['win'] for p in predictions.values()])
        ensemble_points = np.mean([p['points'] for p in predictions.values()])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Win Probability", f"{ensemble_win:.1%}")
        with col2:
            st.metric("Expected Points", f"{ensemble_points:.1f}")
        with col3:
            podium_prob = ensemble_win * 0.7 + 0.2
            st.metric("Podium Chance", f"{podium_prob:.1%}")
        
        st.subheader("Model Breakdown")
        df_pred = pd.DataFrame([
            {'Model': k, 'Win%': f"{v['win']:.1%}", 'Points': f"{v['points']:.1f}"}
            for k, v in predictions.items()
        ])
        st.dataframe(df_pred, use_container_width=True)

# PAGE: DRIVER ANALYSIS
elif page == "Driver Analysis":
    st.markdown("<div class='main-header'><h1>DRIVER ANALYSIS</h1></div>", unsafe_allow_html=True)
    
    # Better driver selection with search and filtering
    st.subheader("Select Driver")
    
    # Get top drivers for easier selection
    latest_season = final_df['year'].max()
    top_drivers_current = get_top_drivers_by_season(latest_season).index.tolist()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selection_type = st.radio("Filter", ["Current Season", "All-Time Wins", "Search"])
    
    with col2:
        if selection_type == "Current Season":
            selected_driver = st.selectbox("Top Drivers", top_drivers_current)
        elif selection_type == "All-Time Wins":
            all_time_wins = final_df.groupby('driverRef')['isWin'].sum().sort_values(ascending=False).head(20).index.tolist()
            selected_driver = st.selectbox("Top Winners", all_time_wins)
        else:
            all_drivers = sorted(final_df['driverRef'].unique())
            selected_driver = st.selectbox("Search", all_drivers)
    
    if selected_driver:
        driver_data = final_df[final_df['driverRef'] == selected_driver]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Wins", int(driver_data['isWin'].sum()))
        with col2:
            st.metric("Podiums", int(driver_data['isPodium'].sum()))
        with col3:
            st.metric("Points", int(driver_data['points'].sum()))
        with col4:
            st.metric("Races", len(driver_data))
        
        st.markdown("---")
        
        # Calculate normalized metrics for radar chart
        all_drivers_agg = final_df.groupby('driverRef').agg({
            'isWin': 'sum',
            'isPodium': 'sum',
            'points': 'sum',
            'avgLapTime': 'mean',
            'finishRate': 'mean'
        })
        
        driver_metrics = {
            'Wins': normalize_value(driver_data['isWin'].sum(), 
                                   all_drivers_agg['isWin'].min(), 
                                   all_drivers_agg['isWin'].max()),
            'Podiums': normalize_value(driver_data['isPodium'].sum(), 
                                      all_drivers_agg['isPodium'].min(), 
                                      all_drivers_agg['isPodium'].max()),
            'Points': normalize_value(driver_data['points'].sum(), 
                                     all_drivers_agg['points'].min(), 
                                     all_drivers_agg['points'].max()),
            'Consistency': normalize_value(driver_data['avgLapTime'].mean() if 'avgLapTime' in driver_data.columns else 0,
                                          0, 100),
            'Finish Rate': driver_data['isFinished'].sum() / len(driver_data) * 100 if len(driver_data) > 0 else 0
        }
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            fig_radar = create_radar_chart(selected_driver, driver_metrics)
            st.plotly_chart(fig_radar, use_container_width=True)
        
        with col2:
            st.subheader("Performance Summary")
            for metric, value in driver_metrics.items():
                st.write(f"**{metric}:** {value:.1f}/100")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            wins_by_year = driver_data.groupby('year')['isWin'].sum()
            fig = px.line(x=wins_by_year.index, y=wins_by_year.values, markers=True,
                         title="Wins by Season")
            fig.update_layout(template='plotly_dark', height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            points_by_year = driver_data.groupby('year')['points'].sum()
            fig = px.bar(x=points_by_year.index, y=points_by_year.values,
                        title="Points by Season", color=points_by_year.values,
                        color_continuous_scale='Viridis')
            fig.update_layout(template='plotly_dark', height=350)
            st.plotly_chart(fig, use_container_width=True)

# PAGE: CONSTRUCTOR ANALYSIS
elif page == "Constructor Analysis":
    st.markdown("<div class='main-header'><h1>CONSTRUCTOR ANALYSIS</h1></div>", unsafe_allow_html=True)
    
    latest_season = final_df['year'].max()
    top_constructors = final_df[final_df['year'] == latest_season].groupby('constructorRef')['points'].sum().sort_values(ascending=False).head(12).index.tolist()
    
    selected_const = st.selectbox("Select Team", top_constructors)
    
    if selected_const:
        const_data = final_df[final_df['constructorRef'] == selected_const]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Wins", int(const_data['isWin'].sum()))
        with col2:
            st.metric("Points", int(const_data['points'].sum()))
        with col3:
            st.metric("Win Rate", f"{(const_data['isWin'].sum() / len(const_data) * 100):.1f}%")
        with col4:
            st.metric("Seasons", const_data['year'].nunique())
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            points_by_year = const_data.groupby('year')['points'].sum()
            fig = px.bar(x=points_by_year.index, y=points_by_year.values,
                        title="Points by Season", color=points_by_year.values,
                        color_continuous_scale='Blues')
            fig.update_layout(template='plotly_dark', height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            drivers_in_team = const_data.groupby('driverRef')['points'].sum().sort_values(ascending=False).head(5)
            fig = px.bar(x=drivers_in_team.values, y=drivers_in_team.index, orientation='h',
                        title="Top Drivers in Team")
            fig.update_layout(template='plotly_dark', height=350)
            st.plotly_chart(fig, use_container_width=True)

# PAGE: ADVANCED ANALYTICS
elif page == "Advanced Analytics":
    st.markdown("<div class='main-header'><h1>ADVANCED ANALYTICS</h1></div>", unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Grid Analysis", "Historical Trends", "Driver Comparisons", "Predictive Insights"])
    
    with tab1:
        st.subheader("Starting Position Impact")
        
        grid_stats = final_df.groupby('grid').agg({
            'isWin': lambda x: (x == 1).sum() / len(x) * 100,
            'isPodium': lambda x: (x == 1).sum() / len(x) * 100,
            'points': 'mean'
        }).reset_index()
        grid_stats = grid_stats[grid_stats['grid'] <= 15]
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(grid_stats, x='grid', y='isWin',
                        title="Win Rate by Grid Position",
                        labels={'grid': 'Grid Position', 'isWin': 'Win Rate (%)'})
            fig.update_layout(template='plotly_dark', height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(grid_stats, x='grid', y='points', size='isPodium',
                           color='isWin', color_continuous_scale='RdYlGn',
                           title="Grid Impact on Average Points")
            fig.update_layout(template='plotly_dark', height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Historical Performance Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            dnf_by_year = final_df.groupby('year').apply(lambda x: (x['isDNF'] == 1).sum() / len(x) * 100)
            fig = px.line(x=dnf_by_year.index, y=dnf_by_year.values, markers=True,
                         title="DNF Rate Trend", labels={'x': 'Season', 'y': 'DNF Rate (%)'})
            fig.update_layout(template='plotly_dark', height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            podium_by_year = final_df.groupby('year').apply(lambda x: (x['isPodium'] == 1).sum() / len(x) * 100)
            fig = px.area(x=podium_by_year.index, y=podium_by_year.values,
                         title="Podium Rate Trend", labels={'x': 'Season', 'y': 'Podium Rate (%)'})
            fig.update_layout(template='plotly_dark', height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Compare Two Drivers")
        
        col1, col2 = st.columns(2)
        
        all_drivers = sorted(final_df['driverRef'].unique())
        with col1:
            driver1 = st.selectbox("Driver 1", all_drivers, key="d1")
        with col2:
            driver2 = st.selectbox("Driver 2", all_drivers, key="d2")
        
        if driver1 and driver2:
            d1_data = final_df[final_df['driverRef'] == driver1]
            d2_data = final_df[final_df['driverRef'] == driver2]
            
            comparison = pd.DataFrame({
                'Metric': ['Wins', 'Podiums', 'Total Points', 'Avg Grid', 'Finish Rate'],
                driver1: [
                    int(d1_data['isWin'].sum()),
                    int(d1_data['isPodium'].sum()),
                    int(d1_data['points'].sum()),
                    f"{d1_data['grid'].mean():.1f}",
                    f"{(d1_data['isFinished'].sum() / len(d1_data) * 100):.1f}%"
                ],
                driver2: [
                    int(d2_data['isWin'].sum()),
                    int(d2_data['isPodium'].sum()),
                    int(d2_data['points'].sum()),
                    f"{d2_data['grid'].mean():.1f}",
                    f"{(d2_data['isFinished'].sum() / len(d2_data) * 100):.1f}%"
                ]
            })
            
            st.dataframe(comparison, use_container_width=True)
    
    with tab4:
        st.subheader("Predictive Insights")
        
        latest_season = final_df['year'].max()
        recent_data = final_df[final_df['year'] == latest_season]
        
        # Best starting position converters
        converters = recent_data[recent_data['grid'] > 10].groupby('driverRef').agg({
            'gridToFinish': 'mean',
            'isWin': 'sum'
        }).sort_values('gridToFinish')
        
        st.write("**Best Overtakers (Grid 10+):**")
        if len(converters) > 0:
            for driver, data in converters.head(5).iterrows():
                st.write(f"  {driver}: {data['gridToFinish']:.1f} positions gained")

# PAGE: CHAMPIONSHIPS
elif page == "Championships":
    st.markdown("<div class='main-header'><h1>CHAMPIONSHIP HISTORY</h1></div>", unsafe_allow_html=True)
    
    seasons = sorted(final_df['year'].unique(), reverse=True)
    selected_season = st.selectbox("Select Season", seasons)
    
    season_data = final_df[final_df['year'] == selected_season]
    
    st.subheader(f"Driver Championship - {selected_season}")
    
    standings = season_data.groupby('driverRef').agg({
        'points': 'sum',
        'isWin': 'sum',
        'isPodium': 'sum'
    }).sort_values('points', ascending=False)
    
    for i, (driver, stats) in enumerate(standings.head(10).iterrows(), 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        with col1:
            st.write(f"**{emoji} {driver}**")
        with col2:
            st.write(f"{int(stats['points'])} pts")
        with col3:
            st.write(f"{int(stats['isWin'])} wins")
        with col4:
            st.write(f"{int(stats['isPodium'])} podiums")

st.markdown("---")
st.markdown("<div style='text-align: center; color: #888;'><p>F1 Prediction Dashboard | Powered by ML</p></div>", unsafe_allow_html=True)
