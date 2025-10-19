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

# PAGE CONFIGURATION
st.set_page_config(
    page_title="F1 Race Prediction Dashboard",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM STYLING
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

# LOAD ALL RESOURCES
@st.cache_resource
def load_resources():
    try:
        final_df = pd.read_csv('f1_dashboard.csv')
        driver_stats = pd.read_csv('driver_season_stats.csv')
        constructor_stats = pd.read_csv('constructor_season_stats.csv')
        
        model_path = "models/f1_models_20251018_230123"
        
        scaler = joblib.load(f"{model_path}/scalers_encoders/feature_scaler.pkl")
        rf_winner = joblib.load(f"{model_path}/sklearn_models/rf_winner.pkl")
        gb_winner = joblib.load(f"{model_path}/sklearn_models/gb_winner.pkl")
        rf_points = joblib.load(f"{model_path}/sklearn_models/rf_points.pkl")
        
        nn_winner = None
        try:
            nn_winner = keras.models.load_model(f"{model_path}/deep_learning/nn_winner_model.h5")
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

# Preprocess final_df to add missing columns
final_df['raceDate'] = pd.to_datetime(final_df['raceDate'])

# Create derived columns if they don't exist
if 'isWin' not in final_df.columns and 'positionOrder' in final_df.columns:
    final_df['isWin'] = (final_df['positionOrder'] == 1).astype(int)

if 'isPodium' not in final_df.columns and 'positionOrder' in final_df.columns:
    final_df['isPodium'] = (final_df['positionOrder'].isin([1, 2, 3])).astype(int)

if 'isFinished' not in final_df.columns and 'status' in final_df.columns:
    final_df['isFinished'] = final_df['status'].apply(lambda x: 1 if x == 'Finished' or x == 1 else 0)

if 'finishRate' not in final_df.columns and 'isFinished' in final_df.columns:
    finish_rates = final_df.groupby('driverRef')['isFinished'].mean() * 100
    final_df['finishRate'] = final_df['driverRef'].map(finish_rates)

if 'isDNF' not in final_df.columns and 'isFinished' in final_df.columns:
    final_df['isDNF'] = (~final_df['isFinished'].astype(bool)).astype(int)

# Debug: Show available columns (optional, can be commented out after debugging)
# st.write("Available columns in final_df:", final_df.columns.tolist())

# Check for critical columns
required_columns = ['driverRef', 'constructorRef', 'year', 'raceId', 'points', 'grid', 'isWin', 'isPodium', 'isFinished', 'finishRate']
missing_columns = [col for col in required_columns if col not in final_df.columns]
if missing_columns:
    st.error(f"Critical columns missing from dataset: {missing_columns}. Please check the data source.")
    st.stop()

# SIDEBAR NAVIGATION
st.sidebar.title("Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select a section:",
    ["Dashboard", "Race Predictor", "Driver Analysis", "Constructor Analysis", "Advanced Analytics", "Championships"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Quick Stats")
st.sidebar.metric("Total Races", final_df['raceId'].nunique())
st.sidebar.metric("Total Drivers", final_df['driverRef'].nunique())
st.sidebar.metric("Current Season", int(final_df['year'].max()))

# HELPER FUNCTIONS
def get_top_drivers_by_season(season):
    """Get the top 10 drivers for a specific season"""
    season_data = final_df[final_df['year'] == season]
    return season_data.groupby('driverRef')['points'].sum().sort_values(ascending=False).head(10)

def create_radar_chart(driver_name, metrics_dict):
    """Create a radar chart showing driver performance metrics"""
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

def normalize_metric(value, minimum, maximum):
    """Convert a value to a 0-100 scale for comparison"""
    if maximum == minimum:
        return 50
    return ((value - minimum) / (maximum - minimum)) * 100

# MAIN DASHBOARD
if page == "Dashboard":
    st.title("Formula 1 Prediction Dashboard")
    st.markdown("Get insights into F1 racing data and make race predictions")
    
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
        st.subheader("All-Time Top Winners")
        top_drivers = final_df.groupby('driverRef')['isWin'].sum().sort_values(ascending=False).head(10)
        fig = px.bar(x=top_drivers.values, y=top_drivers.index, orientation='h',
                    color=top_drivers.values, color_continuous_scale='Reds',
                    title="Drivers with Most Wins")
        fig.update_layout(template='plotly_dark', height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Number of Races Per Season")
        races_by_year = final_df.groupby('year')['raceId'].nunique()
        fig = px.line(x=races_by_year.index, y=races_by_year.values, markers=True,
                     title="Race Count Over Time", labels={'x': 'Year', 'y': 'Number of Races'})
        fig.update_layout(template='plotly_dark', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader(f"Current Season Standings - {latest_season}")
    
    season_standings = final_df[final_df['year'] == latest_season].groupby('driverRef')['points'].sum().sort_values(ascending=False).head(10)
    
    for position, (driver, points) in enumerate(season_standings.items(), 1):
        if position == 1:
            badge = "First"
        elif position == 2:
            badge = "Second"
        elif position == 3:
            badge = "Third"
        else:
            badge = f"Position {position}"
        
        st.write(f"**{badge}:** {driver} - {int(points)} points")

# RACE PREDICTION PAGE
elif page == "Race Predictor":
    st.title("Race Prediction Engine")
    st.markdown("Input race and driver details to predict the outcome")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Race Conditions")
        grid_position = st.slider("Starting Grid Position", 1, 20, 5)
        qualifying_position = st.slider("Qualifying Position", 1, 20, 3)
        planned_pit_stops = st.slider("Planned Pit Stops", 0, 5, 2)
        average_lap_time = st.number_input("Average Lap Time (milliseconds)", 80000, 120000, 90000)
    
    with col2:
        st.subheader("Driver Profile")
        career_wins = st.number_input("Career Wins", 0, 103, 15)
        career_podiums = st.number_input("Career Podiums", 0, 200, 45)
        season_points = st.number_input("Points This Season", 0, 500, 180)
        driver_finish_rate = st.slider("Driver Finish Rate (%)", 50.0, 100.0, 85.0)
    
    st.subheader("Team Profile")
    col1, col2, col3 = st.columns(3)
    with col1:
        team_wins = st.number_input("Team Wins (Career)", 0, 250, 80)
    with col2:
        team_points = st.number_input("Team Points (Career)", 0, 1000, 350)
    with col3:
        lap_consistency = st.slider("Lap Consistency Score", 100, 2000, 500)
    
    st.subheader("Select Prediction Models")
    col1, col2, col3 = st.columns(3)
    with col1:
        use_random_forest = st.checkbox("Use Random Forest", True)
    with col2:
        use_gradient_boosting = st.checkbox("Use Gradient Boosting", True)
    with col3:
        use_neural_network = st.checkbox("Use Neural Network", nn_winner is not None, disabled=nn_winner is None)
    
    if st.button("Generate Prediction", use_container_width=True, type="primary"):
        features = np.array([[
            grid_position, qualifying_position, grid_position - qualifying_position, qualifying_position - grid_position,
            average_lap_time, lap_consistency, planned_pit_stops, 22.5, 0,
            season_points, career_wins, career_podiums, driver_finish_rate,
            season_points / 10, 0, career_wins / 2, career_podiums / 2, season_points * 0.8,
            team_points, team_wins, driver_finish_rate, driver_finish_rate * 0.9
        ]])
        
        features_scaled = scaler.transform(features)
        all_predictions = {}
        
        if use_random_forest:
            rf_prob = rf_winner.predict_proba(features)[0][1]
            predicted_points = rf_points.predict(features)[0]
            all_predictions['Random Forest'] = {'win_prob': rf_prob, 'points': predicted_points}
        
        if use_gradient_boosting:
            gb_prob = gb_winner.predict_proba(features)[0][1]
            all_predictions['Gradient Boosting'] = {'win_prob': gb_prob, 'points': gb_prob * 25}
        
        if use_neural_network and nn_winner:
            nn_prob = nn_winner.predict(features_scaled, verbose=0)[0][0]
            all_predictions['Neural Network'] = {'win_prob': nn_prob, 'points': nn_prob * 25}
        
        ensemble_win_probability = np.mean([p['win_prob'] for p in all_predictions.values()])
        ensemble_expected_points = np.mean([p['points'] for p in all_predictions.values()])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Winning Probability", f"{ensemble_win_probability:.1%}")
        with col2:
            st.metric("Expected Points", f"{ensemble_expected_points:.1f}")
        with col3:
            estimated_podium = ensemble_win_probability * 0.7 + 0.2
            st.metric("Podium Probability", f"{estimated_podium:.1%}")
        
        st.subheader("Individual Model Results")
        results_df = pd.DataFrame([
            {'Model Name': k, 'Win Probability': f"{v['win_prob']:.1%}", 'Expected Points': f"{v['points']:.1f}"}
            for k, v in all_predictions.items()
        ])
        st.dataframe(results_df, use_container_width=True)

# DRIVER ANALYSIS PAGE
elif page == "Driver Analysis":
    st.title("Driver Analysis")
    st.markdown("Detailed performance metrics and statistics for individual drivers")
    
    st.subheader("Select a Driver")
    
    latest_season = final_df['year'].max()
    top_drivers_current = get_top_drivers_by_season(latest_season).index.tolist()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        filter_type = st.radio("Filter by:", ["Current Season", "All-Time Wins", "Search All"])
    
    with col2:
        if filter_type == "Current Season":
            selected_driver = st.selectbox("Choose from top drivers:", top_drivers_current)
        elif filter_type == "All-Time Wins":
            all_time_winners = final_df.groupby('driverRef')['isWin'].sum().sort_values(ascending=False).head(20).index.tolist()
            selected_driver = st.selectbox("Choose from top winners:", all_time_winners)
        else:
            all_drivers_list = sorted(final_df['driverRef'].unique())
            selected_driver = st.selectbox("Search for any driver:", all_drivers_list)
    
    if selected_driver:
        driver_info = final_df[final_df['driverRef'] == selected_driver]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Wins", int(driver_info['isWin'].sum()))
        with col2:
            st.metric("Podium Finishes", int(driver_info['isPodium'].sum()))
        with col3:
            st.metric("Career Points", int(driver_info['points'].sum()))
        with col4:
            st.metric("Races Entered", len(driver_info))
        
        st.markdown("---")
        
        # Define aggregation dictionary based on available columns
        agg_dict = {
            'isWin': 'sum',
            'isPodium': 'sum',
            'points': 'sum',
            'finishRate': 'mean'
        }
        if 'avgLapTime' in final_df.columns:
            agg_dict['avgLapTime'] = 'mean'
        
        # Calculate performance metrics
        all_drivers_stats = final_df.groupby('driverRef').agg(agg_dict)
        
        driver_performance = {
            'Wins': normalize_metric(driver_info['isWin'].sum(), 
                                   all_drivers_stats['isWin'].min(), 
                                   all_drivers_stats['isWin'].max()),
            'Podiums': normalize_metric(driver_info['isPodium'].sum(), 
                                      all_drivers_stats['isPodium'].min(), 
                                      all_drivers_stats['isPodium'].max()),
            'Points': normalize_metric(driver_info['points'].sum(), 
                                     all_drivers_stats['points'].min(), 
                                     all_drivers_stats['points'].max()),
            'Reliability': driver_info['isFinished'].sum() / len(driver_info) * 100 if len(driver_info) > 0 else 0
        }
        if 'avgLapTime' in driver_info.columns:
            driver_performance['Speed'] = normalize_metric(driver_info['avgLapTime'].mean(), 0, 100)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            fig_radar = create_radar_chart(selected_driver, driver_performance)
            st.plotly_chart(fig_radar, use_container_width=True)
        
        with col2:
            st.subheader("Performance Breakdown")
            for metric_name, metric_value in driver_performance.items():
                st.write(f"**{metric_name}:** {metric_value:.1f}/100")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            wins_timeline = driver_info.groupby('year')['isWin'].sum()
            fig = px.line(x=wins_timeline.index, y=wins_timeline.values, markers=True,
                         title="Wins Over Years")
            fig.update_layout(template='plotly_dark', height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            points_timeline = driver_info.groupby('year')['points'].sum()
            fig = px.bar(x=points_timeline.index, y=points_timeline.values,
                        title="Points Earned Per Season", color=points_timeline.values,
                        color_continuous_scale='Viridis')
            fig.update_layout(template='plotly_dark', height=350)
            st.plotly_chart(fig, use_container_width=True)

# CONSTRUCTOR ANALYSIS PAGE
elif page == "Constructor Analysis":
    st.title("Team (Constructor) Analysis")
    st.markdown("Performance metrics for Formula 1 teams")
    
    latest_season = final_df['year'].max()
    top_teams = final_df[final_df['year'] == latest_season].groupby('constructorRef')['points'].sum().sort_values(ascending=False).head(12).index.tolist()
    
    selected_team = st.selectbox("Select a Team:", top_teams)
    
    if selected_team:
        team_info = final_df[final_df['constructorRef'] == selected_team]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Wins", int(team_info['isWin'].sum()))
        with col2:
            st.metric("Career Points", int(team_info['points'].sum()))
        with col3:
            win_percentage = (team_info['isWin'].sum() / len(team_info) * 100)
            st.metric("Win Rate", f"{win_percentage:.1f}%")
        with col4:
            st.metric("Seasons Active", team_info['year'].nunique())
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            points_by_season = team_info.groupby('year')['points'].sum()
            fig = px.bar(x=points_by_season.index, y=points_by_season.values,
                        title="Points Earned Per Season", color=points_by_season.values,
                        color_continuous_scale='Blues')
            fig.update_layout(template='plotly_dark', height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            top_drivers_in_team = team_info.groupby('driverRef')['points'].sum().sort_values(ascending=False).head(5)
            fig = px.bar(x=top_drivers_in_team.values, y=top_drivers_in_team.index, orientation='h',
                        title="Top Drivers in This Team")
            fig.update_layout(template='plotly_dark', height=350)
            st.plotly_chart(fig, use_container_width=True)

# ADVANCED ANALYTICS PAGE
elif page == "Advanced Analytics":
    st.title("Advanced Analytics")
    st.markdown("Deep dive into F1 statistics and performance analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Grid Analysis", "Historical Trends", "Driver Comparisons", "Predictive Insights"])
    
    with tab1:
        st.subheader("How Grid Position Affects Race Outcome")
        
        grid_performance = final_df.groupby('grid').agg({
            'isWin': lambda x: (x == 1).sum() / len(x) * 100,
            'isPodium': lambda x: (x == 1).sum() / len(x) * 100,
            'points': 'mean'
        }).reset_index()
        grid_performance = grid_performance[grid_performance['grid'] <= 15]
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(grid_performance, x='grid', y='isWin',
                        title="Winning Probability by Starting Position",
                        labels={'grid': 'Grid Position', 'isWin': 'Win Rate (%)'})
            fig.update_layout(template='plotly_dark', height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(grid_performance, x='grid', y='points', size='isPodium',
                           color='isWin', color_continuous_scale='RdYlGn',
                           title="Average Points Based on Starting Position")
            fig.update_layout(template='plotly_dark', height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Historical Trends in Formula 1")
        
        col1, col2 = st.columns(2)
        
        with col1:
            dnf_rate = final_df.groupby('year').apply(lambda x: (x['isDNF'] == 1).sum() / len(x) * 100)
            fig = px.line(x=dnf_rate.index, y=dnf_rate.values, markers=True,
                         title="Did Not Finish Rate Over Time", labels={'x': 'Year', 'y': 'DNF Rate (%)'})
            fig.update_layout(template='plotly_dark', height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            podium_rate = final_df.groupby('year').apply(lambda x: (x['isPodium'] == 1).sum() / len(x) * 100)
            fig = px.area(x=podium_rate.index, y=podium_rate.values,
                         title="Podium Finish Rate Over Time", labels={'x': 'Year', 'y': 'Podium Rate (%)'})
            fig.update_layout(template='plotly_dark', height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Compare Two Drivers")
        
        col1, col2 = st.columns(2)
        
        all_drivers_list = sorted(final_df['driverRef'].unique())
        with col1:
            first_driver = st.selectbox("First Driver:", all_drivers_list, key="driver1")
        with col2:
            second_driver = st.selectbox("Second Driver:", all_drivers_list, key="driver2")
        
        if first_driver and second_driver:
            first_data = final_df[final_df['driverRef'] == first_driver]
            second_data = final_df[final_df['driverRef'] == second_driver]
            
            comparison_table = pd.DataFrame({
                'Statistic': ['Career Wins', 'Podium Finishes', 'Total Points', 'Average Starting Position', 'Finish Rate'],
                first_driver: [
                    int(first_data['isWin'].sum()),
                    int(first_data['isPodium'].sum()),
                    int(first_data['points'].sum()),
                    f"{first_data['grid'].mean():.1f}",
                    f"{(first_data['isFinished'].sum() / len(first_data) * 100):.1f}%"
                ],
                second_driver: [
                    int(second_data['isWin'].sum()),
                    int(second_data['isPodium'].sum()),
                    int(second_data['points'].sum()),
                    f"{second_data['grid'].mean():.1f}",
                    f"{(second_data['isFinished'].sum() / len(second_data) * 100):.1f}%"
                ]
            })
            
            st.dataframe(comparison_table, use_container_width=True)
    
    with tab4:
        st.subheader("Performance Predictions")
        
        latest_season = final_df['year'].max()
        recent_racing = final_df[final_df['year'] == latest_season]
        
        overtakers = recent_racing[recent_racing['grid'] > 10].groupby('driverRef').agg({
            'gridToFinish': 'mean',
            'isWin': 'sum'
        }).sort_values('gridToFinish')
        
        st.write("**Best Overtakers (Starting from position 10+):**")
        if len(overtakers) > 0:
            for driver, stats in overtakers.head(5).iterrows():
                st.write(f"  {driver}: gains an average of {stats['gridToFinish']:.1f} positions")

# CHAMPIONSHIP HISTORY PAGE
elif page == "Championships":
    st.title("Championship History")
    st.markdown("View final standings from any season")
    
    all_seasons = sorted(final_df['year'].unique(), reverse=True)
    chosen_season = st.selectbox("Choose a Season:", all_seasons)
    
    season_info = final_df[final_df['year'] == chosen_season]
    
    st.subheader(f"Driver Championship Final Standings - {chosen_season}")
    
    final_standings = season_info.groupby('driverRef').agg({
        'points': 'sum',
        'isWin': 'sum',
        'isPodium': 'sum'
    }).sort_values('points', ascending=False)
    
    for rank, (driver_name, driver_stats) in enumerate(final_standings.head(10).iterrows(), 1):
        if rank == 1:
            rank_label = "Champion"
        elif rank == 2:
            rank_label = "Runner-up"
        elif rank == 3:
            rank_label = "Third Place"
        else:
            rank_label = f"Position {rank}"
        
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        with col1:
            st.write(f"**{rank_label}:** {driver_name}")
        with col2:
            st.write(f"{int(driver_stats['points'])} points")
        with col3:
            st.write(f"{int(driver_stats['isWin'])} wins")
        with col4:
            st.write(f"{int(driver_stats['isPodium'])} podiums")

st.markdown("---")
st.markdown("<div style='text-align: center; color: #888;'><p>Formul<p></div>", unsafe_allow_html=True)
