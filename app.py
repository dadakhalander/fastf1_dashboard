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
    .metric-card {
        background: linear-gradient(135deg, #2d2d44 0%, #3d3d5c 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #FF1801;
        margin: 10px 0;
    }
    .podium-container {
        display: flex;
        justify-content: center;
        align-items: flex-end;
        gap: 20px;
        margin: 30px 0;
    }
    .podium-box {
        text-align: center;
        padding: 20px;
        border-radius: 10px;
        color: white;
        font-weight: bold;
    }
    .first-place {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        height: 250px;
        width: 120px;
        font-size: 48px;
    }
    .second-place {
        background: linear-gradient(135deg, #C0C0C0 0%, #A9A9A9 100%);
        height: 200px;
        width: 120px;
        font-size: 36px;
    }
    .third-place {
        background: linear-gradient(135deg, #CD7F32 0%, #8B4513 100%);
        height: 150px;
        width: 120px;
        font-size: 32px;
    }
    .stat-box {
        background: rgba(255, 24, 1, 0.1);
        border: 2px solid #FF1801;
        padding: 15px;
        border-radius: 10px;
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

final_df['raceDate'] = pd.to_datetime(final_df['raceDate'])

# SIDEBAR NAVIGATION
with st.sidebar:
    st.title("F1 Dashboard")
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["Dashboard", "Race Predictor", "Driver Analysis", "Constructor Analysis", "Advanced Analytics", "Championships"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.subheader("Quick Statistics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Races", final_df['raceId'].nunique())
    with col2:
        st.metric("Drivers", final_df['driverRef'].nunique())
    
    st.metric("Current Season", int(final_df['year'].max()))

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
        line_color='#FF1801',
        fillcolor='rgba(255, 24, 1, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        template='plotly_dark',
        title=f"Performance Radar - {driver_name}",
        height=500,
        showlegend=False
    )
    return fig

def normalize_metric(value, minimum, maximum):
    """Convert a value to a 0-100 scale for comparison"""
    if maximum == minimum:
        return 50
    return ((value - minimum) / (maximum - minimum)) * 100

def show_podium(positions_dict):
    """Display podium visualization"""
    st.markdown("""
    <div class="podium-container">
        <div class="podium-box second-place">
            <div>2️⃣</div>
            <div style="margin-top: 10px; font-size: 14px;">SECOND</div>
        </div>
        <div class="podium-box first-place">
            <div>1️⃣</div>
            <div style="margin-top: 10px; font-size: 14px;">FIRST</div>
        </div>
        <div class="podium-box third-place">
            <div>3️⃣</div>
            <div style="margin-top: 10px; font-size: 14px;">THIRD</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# MAIN DASHBOARD
if page == "Dashboard":
    st.title("🏎️ Formula 1 Prediction Dashboard")
    st.markdown("Get real-time insights into F1 racing data and predictive analytics")
    
    latest_season = final_df['year'].max()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_races = final_df['raceId'].nunique()
        st.metric("Total Races", total_races)
    with col2:
        career_wins = int(final_df['isWin'].sum())
        st.metric("Career Wins", career_wins)
    with col3:
        podiums = int(final_df['isPodium'].sum())
        st.metric("Podium Finishes", podiums)
    with col4:
        finish_rate = (final_df['isFinished'].sum() / len(final_df)) * 100
        st.metric("Finish Rate", f"{finish_rate:.1f}%")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 All-Time Top Winners")
        top_drivers = final_df.groupby('driverRef')['isWin'].sum().sort_values(ascending=False).head(10)
        
        fig = go.Figure(data=[
            go.Bar(
                x=top_drivers.values,
                y=top_drivers.index,
                orientation='h',
                marker=dict(
                    color=top_drivers.values,
                    colorscale='Reds',
                    showscale=False
                ),
                text=top_drivers.values,
                textposition='auto'
            )
        ])
        fig.update_layout(template='plotly_dark', height=400, showlegend=False, 
                         xaxis_title="Wins", yaxis_title="Driver")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Races Per Season")
        races_by_year = final_df.groupby('year')['raceId'].nunique()
        
        fig = px.line(x=races_by_year.index, y=races_by_year.values, markers=True,
                     title="Race Count Trend")
        fig.update_traces(line=dict(color='#FF1801', width=3), marker=dict(size=8))
        fig.update_layout(template='plotly_dark', height=400, 
                         xaxis_title="Year", yaxis_title="Number of Races")
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader(f"🏆 Championship Standings - {latest_season}")
    
    season_standings = final_df[final_df['year'] == latest_season].groupby('driverRef')['points'].sum().sort_values(ascending=False).head(10)
    
    for position, (driver, points) in enumerate(season_standings.items(), 1):
        if position == 1:
            badge = "🥇"
            color = "🟡"
        elif position == 2:
            badge = "🥈"
            color = "⚪"
        elif position == 3:
            badge = "🥉"
            color = "🟠"
        else:
            badge = f"{position}."
            color = ""
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"{badge} **{driver}** {color}")
        with col2:
            st.write(f"**{int(points)}** points")

# RACE PREDICTION PAGE
elif page == "Race Predictor":
    st.title("🎯 Race Prediction Engine")
    st.markdown("Input race and driver details to predict the outcome with multiple AI models")
    
    with st.expander("📋 How to use this predictor", expanded=False):
        st.markdown("""
        - **Race Conditions:** Set the starting position and race variables
        - **Driver Profile:** Input the driver's career statistics
        - **Team Profile:** Add team performance metrics
        - **Model Selection:** Choose which AI models to use
        - The system will average predictions from all selected models
        """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏁 Race Conditions")
        grid_position = st.slider("Starting Grid Position", 1, 20, 5)
        qualifying_position = st.slider("Qualifying Position", 1, 20, 3)
        planned_pit_stops = st.slider("Planned Pit Stops", 0, 5, 2)
        average_lap_time = st.number_input("Average Lap Time (milliseconds)", 80000, 120000, 90000)
    
    with col2:
        st.subheader("👤 Driver Profile")
        career_wins = st.number_input("Career Wins", 0, 103, 15)
        career_podiums = st.number_input("Career Podiums", 0, 200, 45)
        season_points = st.number_input("Points This Season", 0, 500, 180)
        driver_finish_rate = st.slider("Driver Finish Rate (%)", 50.0, 100.0, 85.0)
    
    st.subheader("🏢 Team Profile")
    col1, col2, col3 = st.columns(3)
    with col1:
        team_wins = st.number_input("Team Wins (Career)", 0, 250, 80)
    with col2:
        team_points = st.number_input("Team Points (Career)", 0, 1000, 350)
    with col3:
        lap_consistency = st.slider("Lap Consistency Score", 100, 2000, 500)
    
    st.subheader("🤖 Select AI Models")
    col1, col2, col3 = st.columns(3)
    with col1:
        use_random_forest = st.checkbox("Random Forest", True, help="Ensemble learning method")
    with col2:
        use_gradient_boosting = st.checkbox("Gradient Boosting", True, help="Iterative boosting method")
    with col3:
        use_neural_network = st.checkbox("Neural Network", nn_winner is not None, 
                                        disabled=nn_winner is None, help="Deep learning model")
    
    if st.button("🚀 Generate Prediction", use_container_width=True, type="primary"):
        with st.spinner("Analyzing race conditions..."):
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
            
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Winning Probability", f"{ensemble_win_probability:.1%}")
            with col2:
                st.metric("Expected Points", f"{ensemble_expected_points:.1f}")
            with col3:
                estimated_podium = ensemble_win_probability * 0.7 + 0.2
                st.metric("Podium Probability", f"{estimated_podium:.1%}")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Individual Model Results")
                results_df = pd.DataFrame([
                    {'Model': k, 'Win Probability': f"{v['win_prob']:.1%}", 'Points': f"{v['points']:.1f}"}
                    for k, v in all_predictions.items()
                ])
                st.dataframe(results_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.subheader("Confidence Analysis")
                
                # Calculate confidence
                model_values = [p['win_prob'] for p in all_predictions.values()]
                if len(model_values) > 1:
                    variance = np.var(model_values)
                    confidence = max(0, 100 - (variance * 100))
                else:
                    confidence = 75
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=confidence,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Model Consensus"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#FF1801"},
                        'steps': [
                            {'range': [0, 50], 'color': "#400000"},
                            {'range': [50, 100], 'color': "#800000"}
                        ]
                    }
                ))
                fig.update_layout(template='plotly_dark', height=350)
                st.plotly_chart(fig, use_container_width=True)

# DRIVER ANALYSIS PAGE
elif page == "Driver Analysis":
    st.title("👤 Driver Analysis")
    st.markdown("Detailed performance metrics and career statistics")
    
    st.subheader("Select a Driver")
    
    latest_season = final_df['year'].max()
    top_drivers_current = get_top_drivers_by_season(latest_season).index.tolist()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        filter_type = st.radio("Filter:", ["Current Season", "All-Time Wins", "Search All"])
    
    with col2:
        if filter_type == "Current Season":
            selected_driver = st.selectbox("Top drivers this season:", top_drivers_current, label_visibility="collapsed")
        elif filter_type == "All-Time Wins":
            all_time_winners = final_df.groupby('driverRef')['isWin'].sum().sort_values(ascending=False).head(20).index.tolist()
            selected_driver = st.selectbox("Top winners all-time:", all_time_winners, label_visibility="collapsed")
        else:
            all_drivers_list = sorted(final_df['driverRef'].unique())
            selected_driver = st.selectbox("Find any driver:", all_drivers_list, label_visibility="collapsed")
    
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
            st.metric("Races", len(driver_info))
        
        st.markdown("---")
        
        # Calculate performance metrics
        agg_dict = {
            'isWin': 'sum',
            'isPodium': 'sum',
            'points': 'sum'
        }
        
        if 'avgLapTime' in final_df.columns:
            agg_dict['avgLapTime'] = 'mean'
        if 'isFinished' in final_df.columns:
            agg_dict['isFinished'] = 'sum'
        
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
            'Consistency': 75.0,
            'Reliability': (driver_info['isFinished'].sum() / len(driver_info) * 100) if 'isFinished' in driver_info.columns and len(driver_info) > 0 else 80.0
        }
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            fig_radar = create_radar_chart(selected_driver, driver_performance)
            st.plotly_chart(fig_radar, use_container_width=True)
        
        with col2:
            st.subheader("Performance Metrics")
            for metric_name, metric_value in driver_performance.items():
                progress_value = metric_value / 100
                st.write(f"**{metric_name}**")
                st.progress(progress_value)
        
        st.markdown("---")
        
        tab1, tab2, tab3 = st.tabs(["📈 Career Trends", "🏁 Season Stats", "📊 Record Holder"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                wins_timeline = driver_info.groupby('year')['isWin'].sum()
                fig = px.line(x=wins_timeline.index, y=wins_timeline.values, markers=True,
                             title="Wins Over Years", labels={'x': 'Year', 'y': 'Wins'})
                fig.update_traces(line=dict(color='#FF1801', width=3), marker=dict(size=8))
                fig.update_layout(template='plotly_dark', height=350)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                points_timeline = driver_info.groupby('year')['points'].sum()
                fig = px.bar(x=points_timeline.index, y=points_timeline.values,
                            title="Points Per Season", color=points_timeline.values,
                            color_continuous_scale='Reds')
                fig.update_layout(template='plotly_dark', height=350)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            latest_season_driver = driver_info[driver_info['year'] == latest_season]
            if len(latest_season_driver) > 0:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Wins This Season", int(latest_season_driver['isWin'].sum()))
                with col2:
                    st.metric("Podiums This Season", int(latest_season_driver['isPodium'].sum()))
                
                st.metric("Points This Season", int(latest_season_driver['points'].sum()))
            else:
                st.info("No data available for this season")
        
        with tab3:
            all_drivers_agg = final_df.groupby('driverRef').agg({'isWin': 'sum', 'points': 'sum', 'isPodium': 'sum'})
            
            col1, col2, col3 = st.columns(3)
            with col1:
                rank_wins = (all_drivers_agg['isWin'] >= driver_info['isWin'].sum()).sum()
                st.metric("Wins Rank", f"#{rank_wins}")
            with col2:
                rank_podiums = (all_drivers_agg['isPodium'] >= driver_info['isPodium'].sum()).sum()
                st.metric("Podiums Rank", f"#{rank_podiums}")
            with col3:
                rank_points = (all_drivers_agg['points'] >= driver_info['points'].sum()).sum()
                st.metric("Points Rank", f"#{rank_points}")

# CONSTRUCTOR ANALYSIS PAGE
elif page == "Constructor Analysis":
    st.title("🏢 Constructor Analysis")
    st.markdown("Team performance and achievement tracking")
    
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
            st.metric("Seasons", team_info['year'].nunique())
        
        st.markdown("---")
        
        tab1, tab2, tab3 = st.tabs(["📊 Performance", "👥 Drivers", "🏆 Rankings"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                points_by_season = team_info.groupby('year')['points'].sum()
                fig = px.bar(x=points_by_season.index, y=points_by_season.values,
                            title="Points Per Season", color=points_by_season.values,
                            color_continuous_scale='Blues')
                fig.update_layout(template='plotly_dark', height=350)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                wins_by_season = team_info.groupby('year')['isWin'].sum()
                fig = px.line(x=wins_by_season.index, y=wins_by_season.values, markers=True,
                             title="Wins Over Seasons")
                fig.update_traces(line=dict(color='#FF1801', width=3), marker=dict(size=8))
                fig.update_layout(template='plotly_dark', height=350)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Top Drivers in This Team")
            top_drivers_in_team = team_info.groupby('driverRef')['points'].sum().sort_values(ascending=False).head(5)
            fig = px.bar(x=top_drivers_in_team.values, y=top_drivers_in_team.index, orientation='h',
                        title="Driver Points Contribution", color=top_drivers_in_team.values,
                        color_continuous_scale='Reds')
            fig.update_layout(template='plotly_dark', height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            all_teams = final_df.groupby('constructorRef').agg({'isWin': 'sum', 'points': 'sum'})
            rank_wins = (all_teams['isWin'] >= team_info['isWin'].sum()).sum()
            rank_points = (all_teams['points'] >= team_info['points'].sum()).sum()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Wins Ranking", f"#{rank_wins}")
            with col2:
                st.metric("Points Ranking", f"#{rank_points}")

# ADVANCED ANALYTICS PAGE
elif page == "Advanced Analytics":
    st.title("📊 Advanced Analytics")
    st.markdown("Deep insights into F1 performance patterns")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Grid Analysis", "Historical Trends", "Driver Comparisons", "Predictive Insights"])
    
    with tab1:
        st.subheader("Grid Position Impact Analysis")
        
        grid_performance = final_df.groupby('grid').agg({
            'isWin': lambda x: (x == 1).sum() / len(x) * 100,
            'isPodium': lambda x: (x == 1).sum() / len(x) * 100,
            'points': 'mean'
        }).reset_index()
        grid_performance = grid_performance[grid_performance['grid'] <= 15]
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(grid_performance, x='grid', y='isWin',
                        title="Win Rate by Starting Position",
                        labels={'grid': 'Grid Position', 'isWin': 'Win Rate (%)'},
                        color='isWin', color_continuous_scale='Reds')
            fig.update_layout(template='plotly_dark', height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(grid_performance, x='grid', y='points', size='isPodium',
                           color='isWin', color_continuous_scale='RdYlGn',
                           title="Grid Position vs Average Points",
                           labels={'grid': 'Grid Position', 'points': 'Avg Points'})
            fig.update_layout(template='plotly_dark', height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Historical Performance Trends")
        
        col1, col2 = st.columns(2)
        
        with col1:
            dnf_rate = final_df.groupby('year').apply(lambda x: (x['isDNF'] == 1).sum() / len(x) * 100)
            fig = px.line(x=dnf_rate.index, y=dnf_rate.values, markers=True,
                         title="Did Not Finish Rate Trend")
            fig.update_traces(line=dict(color='#FF1801', width=3), marker=dict(size=8))
            fig.update_layout(template='plotly_dark', height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            podium_rate = final_df.groupby('year').apply(lambda x: (x['isPodium'] == 1).sum() / len(x) * 100)
            fig = px.area(x=podium_rate.index, y=podium_rate.values,
                         title="Podium Finish Rate Trend", fill='tozeroy')
            fig.update_traces(fillcolor='rgba(255, 24, 1, 0.3)', line=dict(color='#FF1801', width=3))
            fig.update_layout(template='plotly_dark', height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Head-to-Head Driver Comparison")
        
        col1, col2 = st.columns(2)
        
        all_drivers_list = sorted(final_df['driverRef'].unique())
        with col1:
            first_driver = st.selectbox("First Driver:", all_drivers_list, key="driver1")
        with col2:
            second_driver = st.selectbox("Second Driver:", all_drivers_list, key="driver2")
        
        if first_driver and second_driver and first_driver != second_driver:
            first_data = final_df[final_df['driverRef'] == first_driver]
            second_data = final_df[final_df['driverRef'] == second_driver]
            
            comparison_metrics = {
                'Statistic': ['Career Wins', 'Podium Finishes', 'Total Points', 'Avg Grid Position', 'Finish Rate (%)'],
                first_driver: [
                    int(first_data['isWin'].sum()),
                    int(first_data['isPodium'].sum()),
                    int(first_data['points'].sum()),
                    f"{first_data['grid'].mean():.1f}",
                    f"{(first_data['isFinished'].sum() / len(first_data) * 100):.1f}" if 'isFinished' in first_data.columns else "N/A"
                ],
                second_driver: [
                    int(second_data['isWin'].sum()),
                    int(second_data['isPodium'].sum()),
                    int(second_data['points'].sum()),
                    f"{second_data['grid'].mean():.1f}",
                    f"{(second_data['isFinished'].sum() / len(second_data) * 100):.1f}" if 'isFinished' in second_data.columns else "N/A"
                ]
            }
            
            comparison_df = pd.DataFrame(comparison_metrics)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            # Comparison visualization
            col1, col2 = st.columns(2)
            
            with col1:
                comparison_data = pd.DataFrame({
                    'Driver': [first_driver, second_driver],
                    'Wins': [int(first_data['isWin'].sum()), int(second_data['isWin'].sum())],
                    'Podiums': [int(first_data['isPodium'].sum()), int(second_data['isPodium'].sum())]
                })
                
                fig = go.Figure()
                fig.add_trace(go.Bar(x=comparison_data['Driver'], y=comparison_data['Wins'], 
                                    name='Wins', marker_color='#FF1801'))
                fig.add_trace(go.Bar(x=comparison_data['Driver'], y=comparison_data['Podiums'], 
                                    name='Podiums', marker_color='#FFB800'))
                fig.update_layout(template='plotly_dark', height=350, barmode='group')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                points_data = pd.DataFrame({
                    'Driver': [first_driver, second_driver],
                    'Points': [int(first_data['points'].sum()), int(second_data['points'].sum())]
                })
                
                fig = px.pie(points_data, values='Points', names='Driver', 
                            title="Total Points Distribution",
                            color_discrete_sequence=['#FF1801', '#FFB800'])
                fig.update_layout(template='plotly_dark', height=350)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select two different drivers to compare")
    
    with tab4:
        st.subheader("Predictive Insights & Patterns")
        
        latest_season = final_df['year'].max()
        recent_racing = final_df[final_df['year'] == latest_season]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Best Overtakers (Starting from Grid 10+)**")
            overtakers = recent_racing[recent_racing['grid'] > 10].groupby('driverRef').agg({
                'gridToFinish': 'mean',
                'isWin': 'sum'
            }).sort_values('gridToFinish', ascending=False)
            
            if len(overtakers) > 0:
                for i, (driver, stats) in enumerate(overtakers.head(5).iterrows(), 1):
                    st.write(f"{i}. **{driver}** - gains {stats['gridToFinish']:.1f} positions avg")
            else:
                st.info("No overtaking data available")
        
        with col2:
            st.markdown("**Consistency Leaders (Lowest DNF Rate)**")
            dnf_stats = final_df.groupby('driverRef').apply(
                lambda x: (x['isDNF'] == 0).sum() / len(x) * 100 if 'isDNF' in x.columns else 0
            ).sort_values(ascending=False)
            
            if len(dnf_stats) > 0:
                for i, (driver, rate) in enumerate(dnf_stats.head(5).items(), 1):
                    st.write(f"{i}. **{driver}** - {rate:.1f}% finish rate")

# CHAMPIONSHIP HISTORY PAGE
elif page == "Championships":
    st.title("🏆 Championship History")
    st.markdown("View final standings and historical records")
    
    all_seasons = sorted(final_df['year'].unique(), reverse=True)
    chosen_season = st.selectbox("Select a Season:", all_seasons)
    
    season_info = final_df[final_df['year'] == chosen_season]
    
    st.subheader(f"Championship Standings - {chosen_season}")
    
    final_standings = season_info.groupby('driverRef').agg({
        'points': 'sum',
        'isWin': 'sum',
        'isPodium': 'sum'
    }).sort_values('points', ascending=False)
    
    # Display podium visualization
    show_podium({})
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    # Top 3 drivers with details
    top_3 = final_standings.head(3)
    
    with col1:
        if len(top_3) >= 1:
            driver1, stats1 = list(top_3.iterrows())[0]
            st.markdown(f"""
            <div class="stat-box">
                <h3 style="text-align: center; margin: 0;">🥇 CHAMPION</h3>
                <h2 style="text-align: center; color: #FFD700; margin: 10px 0;">{driver1}</h2>
                <p style="text-align: center; font-size: 18px;"><b>{int(stats1['points'])} Points</b></p>
                <p style="text-align: center; color: #999;">Wins: {int(stats1['isWin'])} | Podiums: {int(stats1['isPodium'])}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if len(top_3) >= 2:
            driver2, stats2 = list(top_3.iterrows())[1]
            st.markdown(f"""
            <div class="stat-box" style="border-color: #C0C0C0;">
                <h3 style="text-align: center; margin: 0;">🥈 RUNNER-UP</h3>
                <h2 style="text-align: center; color: #C0C0C0; margin: 10px 0;">{driver2}</h2>
                <p style="text-align: center; font-size: 18px;"><b>{int(stats2['points'])} Points</b></p>
                <p style="text-align: center; color: #999;">Wins: {int(stats2['isWin'])} | Podiums: {int(stats2['isPodium'])}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if len(top_3) >= 3:
            driver3, stats3 = list(top_3.iterrows())[2]
            st.markdown(f"""
            <div class="stat-box" style="border-color: #CD7F32;">
                <h3 style="text-align: center; margin: 0;">🥉 THIRD</h3>
                <h2 style="text-align: center; color: #CD7F32; margin: 10px 0;">{driver3}</h2>
                <p style="text-align: center; font-size: 18px;"><b>{int(stats3['points'])} Points</b></p>
                <p style="text-align: center; color: #999;">Wins: {int(stats3['isWin'])} | Podiums: {int(stats3['isPodium'])}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("Full Championship Standings")
    
    standings_display = pd.DataFrame({
        'Position': range(1, len(final_standings.head(15)) + 1),
        'Driver': final_standings.head(15).index,
        'Points': final_standings.head(15)['points'].values.astype(int),
        'Wins': final_standings.head(15)['isWin'].values.astype(int),
        'Podiums': final_standings.head(15)['isPodium'].values.astype(int)
    })
    
    st.dataframe(standings_display, use_container_width=True, hide_index=True)
    
    # Championship statistics
    st.markdown("---")
    st.subheader("Season Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Races Held", season_info['raceId'].nunique())
    with col2:
        st.metric("Total Drivers", season_info['driverRef'].nunique())
    with col3:
        st.metric("Total Teams", season_info['constructorRef'].nunique())
    with col4:
        avg_dnf = (season_info['isDNF'].sum() / len(season_info) * 100) if 'isDNF' in season_info.columns else 0
        st.metric("DNF Rate", f"{avg_dnf:.1f}%")

st.markdown("---")
st.markdown("<div style='text-align: center; color: #888; padding: 20px;'><p>🏎️ Formula 1 Prediction Dashboard | Powered by Machine Learning | Data Analysis & Visualization</p></div>", unsafe_allow_html=True)
