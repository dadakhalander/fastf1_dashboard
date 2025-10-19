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
    page_title="Advanced F1 Race Prediction Dashboard",
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
        driver_stats = None
        constructor_stats = None
        try:
            driver_stats = pd.read_csv('driver_season_stats.csv', on_bad_lines='skip')
            constructor_stats = pd.read_csv('constructor_season_stats.csv', on_bad_lines='skip')
        except:
            pass
        
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
        
        race_predictor = None
        try:
            race_predictor = joblib.load("models/race_predictor.pkl")
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
            'nn_winner': nn_winner,
            'race_predictor': race_predictor
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
race_predictor = resources['race_predictor']

# Preprocess final_df to add missing columns
final_df['raceDate'] = pd.to_datetime(final_df['raceDate'])
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
if 'gridToFinish' not in final_df.columns and 'grid' in final_df.columns and 'positionOrder' in final_df.columns:
    final_df['gridToFinish'] = final_df['grid'] - final_df['positionOrder']

# Check for critical columns in final_df
required_columns = ['driverRef', 'constructorRef', 'year', 'raceId', 'points', 'grid']
missing_columns = [col for col in required_columns if col not in final_df.columns]
if missing_columns:
    st.error(f"Critical columns missing from f1_dashboard.csv: {missing_columns}. Please check the data source.")
    st.stop()

# Check for EDA-specific columns
eda_required_columns = ['driverRef', 'constructorRef', 'year', 'points']
eda_missing_columns = [col for col in eda_required_columns if col not in final_df.columns]
if eda_missing_columns:
    st.error(f"EDA page cannot function. Missing columns in f1_dashboard.csv: {eda_missing_columns}.")
    st.stop()

# SIDEBAR NAVIGATION
st.sidebar.title("Navigation")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Select a section:",
    ["Dashboard", "EDA", "Race Predictor", "Driver Analysis", "Constructor Analysis", "Advanced Analytics", "Championships", "Clustering", "Simulation"]
)
st.sidebar.markdown("---")
st.sidebar.subheader("Quick Stats")
st.sidebar.metric("Total Races", final_df['raceId'].nunique())
st.sidebar.metric("Total Drivers", final_df['driverRef'].nunique())
st.sidebar.metric("Current Season", int(final_df['year'].max()))

# HELPER FUNCTIONS
def get_top_drivers_by_season(season, df=final_df):
    """Get the top 10 drivers for a specific season"""
    season_data = df[df['year'] == season]
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
    st.markdown("Get insights into F1 racing data, explore analytics, and predict race outcomes")
    latest_season = final_df['year'].max()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Races", final_df['raceId'].nunique())
    with col2:
        st.metric("Career Wins", int(final_df['isWin'].sum()) if 'isWin' in final_df.columns else 0)
    with col3:
        st.metric("Podium Finishes", int(final_df['isPodium'].sum()) if 'isPodium' in final_df.columns else 0)
    with col4:
        finish_rate = (final_df['isFinished'].sum() / len(final_df)) * 100 if 'isFinished' in final_df.columns else 0
        st.metric("Finish Rate", f"{finish_rate:.1f}%" if finish_rate > 0 else "N/A")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("All-Time Top Winners")
        if 'isWin' in final_df.columns:
            top_drivers = final_df.groupby('driverRef')['isWin'].sum().sort_values(ascending=False).head(10)
            fig = px.bar(x=top_drivers.values, y=top_drivers.index, orientation='h',
                         color=top_drivers.values, color_continuous_scale='Reds',
                         title="Drivers with Most Wins")
            fig.update_layout(template='plotly_dark', height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Cannot display top winners: 'isWin' column missing.")
    
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
        badge = "First" if position == 1 else "Second" if position == 2 else "Third" if position == 3 else f"Position {position}"
        st.write(f"**{badge}:** {driver} - {int(points)} points")

# EDA PAGE
elif page == "EDA":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("Use the filters below to explore performance across seasons, drivers, and circuits.")
    
    if 'circuitName' not in final_df.columns:
        st.error("The 'circuitName' column is missing from f1_dashboard.csv. The EDA page requires circuit information.")
        st.stop()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_year = st.selectbox("Select Year", sorted(final_df["year"].unique()), index=len(final_df["year"].unique())-1)
    with col2:
        selected_circuit = st.selectbox("Select Circuit", sorted(final_df["circuitName"].dropna().unique()))
    with col3:
        selected_driver = st.selectbox("Select Driver", sorted(final_df["driverRef"].dropna().unique()))
    
    filtered_df = final_df[(final_df["year"] == selected_year) & (final_df["circuitName"] == selected_circuit)]
    
    st.markdown(f"### {selected_driver}'s Performance at {selected_circuit} ({selected_year})")
    
    driver_laps = filtered_df[filtered_df["driverRef"] == selected_driver]
    if not driver_laps.empty and "avgLapTime" in driver_laps.columns:
        fig1 = px.line(driver_laps, x="positionOrder", y="avgLapTime",
                       title="Average Lap Time Trend (by Position Order)",
                       markers=True, color_discrete_sequence=["#E10600"])
        fig1.update_layout(template='plotly_dark')
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.warning("No lap time data available for this driver and race.")
    
    if 'positionOrder' in final_df.columns:
        fig2 = px.box(final_df[final_df["driverRef"] == selected_driver], x="year", y="positionOrder",
                      color="constructorRef", title=f"Race Position Distribution for {selected_driver}")
        fig2.update_layout(template='plotly_dark')
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Cannot display position distribution: 'positionOrder' column missing.")
    
    fig3 = px.bar(final_df[final_df["year"] == selected_year], x="constructorRef", y="points",
                  color="constructorRef", barmode="group",
                  title=f"Constructor Points in {selected_year}")
    fig3.update_layout(template='plotly_dark')
    st.plotly_chart(fig3, use_container_width=True)
    
    st.markdown("### 📈 Summary Statistics")
    agg_columns = ['points']
    if 'pitStops' in final_df.columns:
        agg_columns.append('pitStops')
    if 'laps' in final_df.columns:
        agg_columns.append('laps')
    stats = final_df.groupby("constructorRef")[agg_columns].mean().round(2)
    st.dataframe(stats, use_container_width=True)

# RACE PREDICTOR PAGE
elif page == "Race Predictor":
    st.title("🎯 Race Prediction Engine")
    st.markdown("Input race and driver details to predict the outcome")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Race Conditions")
        grid_position = st.slider("Starting Grid Position", 1, 30, 5)
        qualifying_position = st.slider("Qualifying Position", 1, 30, 3)
        planned_pit_stops = st.slider("Planned Pit Stops", 0, 5, 2)
        average_lap_time = st.number_input("Average Lap Time (milliseconds)", 80000, 120000, 90000, disabled='avgLapTime' not in final_df.columns)
    
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
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        use_random_forest = st.checkbox("Use Random Forest (Winner)", True)
    with col2:
        use_gradient_boosting = st.checkbox("Use Gradient Boosting", True)
    with col3:
        use_neural_network = st.checkbox("Use Neural Network", nn_winner is not None, disabled=nn_winner is None)
    with col4:
        use_simple_predictor = st.checkbox("Use Simple Predictor (Position)", race_predictor is not None, disabled=race_predictor is None)
    
    if st.button("Generate Prediction", use_container_width=True, type="primary"):
        all_predictions = {}
        
        # Advanced Prediction (Random Forest, Gradient Boosting, Neural Network)
        if use_random_forest or use_gradient_boosting or (use_neural_network and nn_winner):
            features = np.array([[
                grid_position, qualifying_position, grid_position - qualifying_position, qualifying_position - grid_position,
                average_lap_time if 'avgLapTime' in final_df.columns else 90000, lap_consistency, planned_pit_stops, 22.5, 0,
                season_points, career_wins, career_podiums, driver_finish_rate,
                season_points / 10, 0, career_wins / 2, career_podiums / 2, season_points * 0.8,
                team_points, team_wins, driver_finish_rate, driver_finish_rate * 0.9
            ]])
            features_scaled = scaler.transform(features)
            
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
        
        # Simple Prediction (Random Forest Regressor for Position)
        if use_simple_predictor and race_predictor:
            simple_features = [[grid_position, 70, planned_pit_stops, season_points]]
            position_prediction = race_predictor.predict(simple_features)[0]
            all_predictions['Simple Predictor'] = {'position': position_prediction}
        
        # Display Results
        if all_predictions:
            col1, col2, col3 = st.columns(3)
            with col1:
                if any(k in all_predictions for k in ['Random Forest', 'Gradient Boosting', 'Neural Network']):
                    ensemble_win_probability = np.mean([p['win_prob'] for p in all_predictions.values() if 'win_prob' in p])
                    st.metric("Winning Probability", f"{ensemble_win_probability:.1%}")
            with col2:
                if any(k in all_predictions for k in ['Random Forest', 'Gradient Boosting', 'Neural Network']):
                    ensemble_expected_points = np.mean([p['points'] for p in all_predictions.values() if 'points' in p])
                    st.metric("Expected Points", f"{ensemble_expected_points:.1f}")
            with col3:
                if 'Simple Predictor' in all_predictions:
                    st.metric("Predicted Position", f"{all_predictions['Simple Predictor']['position']:.1f}")
                elif any(k in all_predictions for k in ['Random Forest', 'Gradient Boosting', 'Neural Network']):
                    estimated_podium = ensemble_win_probability * 0.7 + 0.2
                    st.metric("Podium Probability", f"{estimated_podium:.1%}")
            
            st.subheader("Individual Model Results")
            results_df = pd.DataFrame([
                {'Model Name': k, 
                 'Win Probability': f"{v['win_prob']:.1%}" if 'win_prob' in v else '-', 
                 'Expected Points': f"{v['points']:.1f}" if 'points' in v else '-',
                 'Predicted Position': f"{v['position']:.1f}" if 'position' in v else '-'}
                for k, v in all_predictions.items()
            ])
            st.dataframe(results_df, use_container_width=True)
            st.balloons()
        else:
            st.warning("No predictions generated. Please select at least one model.")

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
            st.metric("Total Wins", int(driver_info['isWin'].sum()) if 'isWin' in driver_info.columns else 0)
        with col2:
            st.metric("Podium Finishes", int(driver_info['isPodium'].sum()) if 'isPodium' in driver_info.columns else 0)
        with col3:
            st.metric("Career Points", int(driver_info['points'].sum()))
        with col4:
            st.metric("Races Entered", len(driver_info))
        
        st.markdown("---")
        
        agg_dict = {
            'isWin': 'sum',
            'isPodium': 'sum',
            'points': 'sum',
            'finishRate': 'mean'
        }
        if 'avgLapTime' in final_df.columns:
            agg_dict['avgLapTime'] = 'mean'
        
        all_drivers_stats = final_df.groupby('driverRef').agg(agg_dict)
        
        driver_performance = {
            'Wins': normalize_metric(driver_info['isWin'].sum(), 
                                   all_drivers_stats['isWin'].min(), 
                                   all_drivers_stats['isWin'].max()) if 'isWin' in driver_info.columns else 0,
            'Podiums': normalize_metric(driver_info['isPodium'].sum(), 
                                      all_drivers_stats['isPodium'].min(), 
                                      all_drivers_stats['isPodium'].max()) if 'isPodium' in driver_info.columns else 0,
            'Points': normalize_metric(driver_info['points'].sum(), 
                                     all_drivers_stats['points'].min(), 
                                     all_drivers_stats['points'].max()),
            'Reliability': driver_info['isFinished'].sum() / len(driver_info) * 100 if 'isFinished' in driver_info.columns and len(driver_info) > 0 else 0
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
            if 'isWin' in driver_info.columns:
                wins_timeline = driver_info.groupby('year')['isWin'].sum()
                fig = px.line(x=wins_timeline.index, y=wins_timeline.values, markers=True,
                             title="Wins Over Years")
                fig.update_layout(template='plotly_dark', height=350)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Cannot display wins timeline: 'isWin' column missing.")
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
            st.metric("Total Wins", int(team_info['isWin'].sum()) if 'isWin' in team_info.columns else 0)
        with col2:
            st.metric("Career Points", int(team_info['points'].sum()))
        with col3:
            win_percentage = (team_info['isWin'].sum() / len(team_info) * 100) if 'isWin' in team_info.columns else 0
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
        if 'isWin' in final_df.columns and 'isPodium' in final_df.columns:
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
        else:
            st.warning("Cannot display grid analysis: 'isWin' or 'isPodium' columns missing.")
    
    with tab2:
        st.subheader("Historical Trends in Formula 1")
        col1, col2 = st.columns(2)
        with col1:
            if 'isDNF' in final_df.columns:
                dnf_rate = final_df.groupby('year').apply(lambda x: (x['isDNF'] == 1).sum() / len(x) * 100)
                fig = px.line(x=dnf_rate.index, y=dnf_rate.values, markers=True,
                             title="Did Not Finish Rate Over Time", labels={'x': 'Year', 'y': 'DNF Rate (%)'})
                fig.update_layout(template='plotly_dark', height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Cannot display DNF rate: 'isDNF' column missing.")
        with col2:
            if 'isPodium' in final_df.columns:
                podium_rate = final_df.groupby('year').apply(lambda x: (x['isPodium'] == 1).sum() / len(x) * 100)
                fig = px.area(x=podium_rate.index, y=podium_rate.values,
                             title="Podium Finish Rate Over Time", labels={'x': 'Year', 'y': 'Podium Rate (%)'})
                fig.update_layout(template='plotly_dark', height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Cannot display podium rate: 'isPodium' column missing.")
    
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
                    int(first_data['isWin'].sum()) if 'isWin' in first_data.columns else 0,
                    int(first_data['isPodium'].sum()) if 'isPodium' in first_data.columns else 0,
                    int(first_data['points'].sum()),
                    f"{first_data['grid'].mean():.1f}",
                    f"{(first_data['isFinished'].sum() / len(first_data) * 100):.1f}%" if 'isFinished' in first_data.columns else "N/A"
                ],
                second_driver: [
                    int(second_data['isWin'].sum()) if 'isWin' in second_data.columns else 0,
                    int(second_data['isPodium'].sum()) if 'isPodium' in second_data.columns else 0,
                    int(second_data['points'].sum()),
                    f"{second_data['grid'].mean():.1f}",
                    f"{(second_data['isFinished'].sum() / len(second_data) * 100):.1f}%" if 'isFinished' in second_data.columns else "N/A"
                ]
            })
            st.dataframe(comparison_table, use_container_width=True)
    
    with tab4:
        st.subheader("Performance Predictions")
        latest_season = final_df['year'].max()
        recent_racing = final_df[final_df['year'] == latest_season]
        
        if 'gridToFinish' in final_df.columns:
            overtakers = recent_racing[recent_racing['grid'] > 10].groupby('driverRef').agg({
                'gridToFinish': 'mean',
                'isWin': 'sum'
            }).sort_values('gridToFinish')
            
            st.write("**Best Overtakers (Starting from position 10+):**")
            if len(overtakers) > 0:
                for driver, stats in overtakers.head(5).iterrows():
                    st.write(f"  {driver}: gains an average of {stats['gridToFinish']:.1f} positions")
            else:
                st.write("No overtaking data available for drivers starting beyond position 10.")
        else:
            st.warning("Cannot display overtakers: 'gridToFinish' column missing.")

# CHAMPIONSHIPS PAGE
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
        rank_label = "Champion" if rank == 1 else "Runner-up" if rank == 2 else "Third Place" if rank == 3 else f"Position {rank}"
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        with col1:
            st.write(f"**{rank_label}:** {driver_name}")
        with col2:
            st.write(f"{int(driver_stats['points'])} points")
        with col3:
            st.write(f"{int(driver_stats['isWin']) if 'isWin' in driver_stats else 0} wins")
        with col4:
            st.write(f"{int(driver_stats['isPodium']) if 'isPodium' in driver_stats else 0} podiums")

# CLUSTERING PAGE
elif page == "Clustering":
    st.title("Driver Clustering Analysis")
    st.markdown("Analyze driver performance clusters based on key metrics")
    st.warning("Clustering analysis is not yet implemented. Please provide a clustering model or logic to enable this feature.")
    # Placeholder for clustering logic
    # Example: Implement K-means clustering on driver stats
    # st.write("Clustering analysis will group drivers based on performance metrics like points, wins, and finish rate.")

# SIMULATION PAGE
elif page == "Simulation":
    st.title("Race Simulation")
    st.markdown("Simulate race outcomes based on historical data and model predictions")
    st.warning("Race simulation is not yet implemented. Please provide a simulation model or logic to enable this feature.")
    # Placeholder for simulation logic
    # Example: Monte Carlo simulation of race outcomes
    # st.write("Race simulation will predict race outcomes using probabilistic models.")

st.markdown("---")
st.markdown("<div style='text-align: center; color: #888;'><p>Advanced Formula 1 Prediction Dashboard | Machine Learning Powered</p></div>", unsafe_allow_html=True)
