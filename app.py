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
    .stat-box {
        background: rgba(255, 24, 1, 0.1);
        border: 2px solid #FF1801;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

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

def get_top_drivers_by_season(season):
    season_data = final_df[final_df['year'] == season]
    return season_data.groupby('driverRef')['points'].sum().sort_values(ascending=False).head(10)

def create_radar_chart(driver_name, metrics_dict):
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
    if maximum == minimum:
        return 50
    return ((value - minimum) / (maximum - minimum)) * 100

if page == "Dashboard":
    st.title("F1 Prediction Dashboard")
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
        finish_rate = (final_df['isFinished'].sum() / len(final_df)) * 100 if 'isFinished' in final_df.columns else 0
        st.metric("Finish Rate", f"{finish_rate:.1f}%")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("All-Time Top Winners")
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
        fig.update_layout(template='plotly_dark', height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Races Per Season")
        races_by_year = final_df.groupby('year')['raceId'].nunique()
        
        fig = px.line(x=races_by_year.index, y=races_by_year.values, markers=True)
        fig.update_traces(line=dict(color='#FF1801', width=3), marker=dict(size=8))
        fig.update_layout(template='plotly_dark', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader(f"Championship Standings - {latest_season}")
    
    season_standings = final_df[final_df['year'] == latest_season].groupby('driverRef')['points'].sum().sort_values(ascending=False).head(10)
    
    for position, (driver, points) in enumerate(season_standings.items(), 1):
        if position == 1:
            badge = "1st"
            color = "gold"
        elif position == 2:
            badge = "2nd"
            color = "silver"
        elif position == 3:
            badge = "3rd"
            color = "orange"
        else:
            badge = f"{position}."
            color = "white"
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**{badge} {driver}**")
        with col2:
            st.write(f"**{int(points)}** pts")

elif page == "Race Predictor":
    st.title("Race Prediction Engine")
    st.markdown("Input race and driver details to predict the outcome")
    
    with st.expander("How to use", expanded=False):
        st.markdown("Set race conditions, driver profile, and team metrics to get predictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Race Conditions")
        grid_position = st.slider("Starting Grid Position", 1, 20, 5)
        qualifying_position = st.slider("Qualifying Position", 1, 20, 3)
        planned_pit_stops = st.slider("Planned Pit Stops", 0, 5, 2)
        average_lap_time = st.number_input("Average Lap Time (ms)", 80000, 120000, 90000)
    
    with col2:
        st.subheader("Driver Profile")
        career_wins = st.number_input("Career Wins", 0, 103, 15)
        career_podiums = st.number_input("Career Podiums", 0, 200, 45)
        season_points = st.number_input("Points This Season", 0, 500, 180)
        driver_finish_rate = st.slider("Driver Finish Rate (%)", 50.0, 100.0, 85.0)
    
    st.subheader("Team Profile")
    col1, col2, col3 = st.columns(3)
    with col1:
        team_wins = st.number_input("Team Wins", 0, 250, 80)
    with col2:
        team_points = st.number_input("Team Points", 0, 1000, 350)
    with col3:
        lap_consistency = st.slider("Lap Consistency", 100, 2000, 500)
    
    st.subheader("AI Models")
    col1, col2, col3 = st.columns(3)
    with col1:
        use_random_forest = st.checkbox("Random Forest", True)
    with col2:
        use_gradient_boosting = st.checkbox("Gradient Boosting", True)
    with col3:
        use_neural_network = st.checkbox("Neural Network", nn_winner is not None, disabled=nn_winner is None)
    
    if st.button("Generate Prediction", use_container_width=True, type="primary"):
        with st.spinner("Analyzing..."):
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
            
            ensemble_win = np.mean([p['win_prob'] for p in all_predictions.values()])
            ensemble_points = np.mean([p['points'] for p in all_predictions.values()])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Win Probability", f"{ensemble_win:.1%}")
            with col2:
                st.metric("Expected Points", f"{ensemble_points:.1f}")
            with col3:
                podium = ensemble_win * 0.7 + 0.2
                st.metric("Podium Chance", f"{podium:.1%}")
            
            st.subheader("Model Results")
            results_df = pd.DataFrame([
                {'Model': k, 'Win Prob': f"{v['win_prob']:.1%}", 'Points': f"{v['points']:.1f}"}
                for k, v in all_predictions.items()
            ])
            st.dataframe(results_df, use_container_width=True, hide_index=True)

elif page == "Driver Analysis":
    st.title("Driver Analysis")
    st.markdown("Detailed performance metrics")
    
    latest_season = final_df['year'].max()
    top_drivers_current = get_top_drivers_by_season(latest_season).index.tolist()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        filter_type = st.radio("Filter:", ["Current Season", "All-Time Wins", "Search All"])
    
    with col2:
        if filter_type == "Current Season":
            selected_driver = st.selectbox("Drivers:", top_drivers_current, label_visibility="collapsed")
        elif filter_type == "All-Time Wins":
            winners = final_df.groupby('driverRef')['isWin'].sum().sort_values(ascending=False).head(20).index.tolist()
            selected_driver = st.selectbox("Winners:", winners, label_visibility="collapsed")
        else:
            all_drivers = sorted(final_df['driverRef'].unique())
            selected_driver = st.selectbox("Driver:", all_drivers, label_visibility="collapsed")
    
    if selected_driver:
        driver_info = final_df[final_df['driverRef'] == selected_driver]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Wins", int(driver_info['isWin'].sum()))
        with col2:
            st.metric("Podiums", int(driver_info['isPodium'].sum()))
        with col3:
            st.metric("Points", int(driver_info['points'].sum()))
        with col4:
            st.metric("Races", len(driver_info))
        
        st.markdown("---")
        
        agg_dict = {'isWin': 'sum', 'isPodium': 'sum', 'points': 'sum'}
        
        if 'avgLapTime' in final_df.columns:
            agg_dict['avgLapTime'] = 'mean'
        if 'isFinished' in final_df.columns:
            agg_dict['isFinished'] = 'sum'
        
        all_drivers_stats = final_df.groupby('driverRef').agg(agg_dict)
        
        driver_performance = {
            'Wins': normalize_metric(driver_info['isWin'].sum(), all_drivers_stats['isWin'].min(), all_drivers_stats['isWin'].max()),
            'Podiums': normalize_metric(driver_info['isPodium'].sum(), all_drivers_stats['isPodium'].min(), all_drivers_stats['isPodium'].max()),
            'Points': normalize_metric(driver_info['points'].sum(), all_drivers_stats['points'].min(), all_drivers_stats['points'].max()),
            'Consistency': 75.0,
            'Reliability': (driver_info['isFinished'].sum() / len(driver_info) * 100) if 'isFinished' in driver_info.columns else 80.0
        }
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            fig_radar = create_radar_chart(selected_driver, driver_performance)
            st.plotly_chart(fig_radar, use_container_width=True)
        
        with col2:
            st.subheader("Performance")
            for metric_name, metric_value in driver_performance.items():
                st.write(f"**{metric_name}:** {metric_value:.1f}/100")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            wins_data = driver_info.groupby('year')['isWin'].sum()
            fig = px.line(x=wins_data.index, y=wins_data.values, markers=True, title="Wins Over Years")
            fig.update_traces(line=dict(color='#FF1801', width=3))
            fig.update_layout(template='plotly_dark', height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            points_data = driver_info.groupby('year')['points'].sum()
            fig = px.bar(x=points_data.index, y=points_data.values, title="Points Per Season", color=points_data.values, color_continuous_scale='Reds')
            fig.update_layout(template='plotly_dark', height=350)
            st.plotly_chart(fig, use_container_width=True)

elif page == "Constructor Analysis":
    st.title("Team Analysis with 3D Visualization")
    st.markdown("Explore team performance in 3D space")
    
    latest_season = final_df['year'].max()
    top_teams = final_df[final_df['year'] == latest_season].groupby('constructorRef')['points'].sum().sort_values(ascending=False).head(12).index.tolist()
    
    selected_team = st.selectbox("Select Team:", top_teams)
    
    if selected_team:
        team_info = final_df[final_df['constructorRef'] == selected_team]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Wins", int(team_info['isWin'].sum()))
        with col2:
            st.metric("Points", int(team_info['points'].sum()))
        with col3:
            win_pct = (team_info['isWin'].sum() / len(team_info) * 100)
            st.metric("Win Rate", f"{win_pct:.1f}%")
        with col4:
            st.metric("Seasons", team_info['year'].nunique())
        
        st.markdown("---")
        
        tab1, tab2, tab3, tab4 = st.tabs(["3D Map", "Performance", "Drivers", "Rankings"])
        
        with tab1:
            st.subheader("3D Performance Map")
            st.markdown("Interactive 3D visualization of all teams")
            
            all_teams_data = final_df.groupby('constructorRef').agg({
                'isWin': 'sum',
                'points': 'sum',
                'year': 'max'
            }).reset_index()
            
            team_counts = final_df.groupby('constructorRef').size()
            all_teams_data['avgPtsPerRace'] = all_teams_data.apply(
                lambda row: row['points'] / team_counts.get(row['constructorRef'], 1),
                axis=1
            )
            
            fig_3d = go.Figure(data=[go.Scatter3d(
                x=all_teams_data['isWin'],
                y=all_teams_data['points'],
                z=all_teams_data['year'],
                mode='markers+text',
                marker=dict(
                    size=all_teams_data['avgPtsPerRace'] / 50 + 5,
                    color=all_teams_data['points'],
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title="Points"),
                    opacity=0.8,
                    line=dict(color='white', width=1)
                ),
                text=all_teams_data['constructorRef'],
                textposition='top center',
                textfont=dict(size=9, color='white'),
                hovertemplate='<b>%{text}</b><br>Wins: %{x}<br>Points: %{y}<br>Season: %{z}<extra></extra>'
            )])
            
            fig_3d.update_layout(
                title="Team Performance 3D Map",
                scene=dict(
                    xaxis=dict(title='Wins', backgroundcolor="rgb(30,30,46)", gridcolor="gray"),
                    yaxis=dict(title='Points', backgroundcolor="rgb(30,30,46)", gridcolor="gray"),
                    zaxis=dict(title='Latest Season', backgroundcolor="rgb(30,30,46)", gridcolor="gray"),
                ),
                template='plotly_dark',
                height=600
            )
            
            st.plotly_chart(fig_3d, use_container_width=True)
            
            st.markdown("---")
            st.subheader("Team Evolution 3D Timeline")
            
            top_5 = all_teams_data.nlargest(5, 'points')['constructorRef'].tolist()
            fig_evolution = go.Figure()
            
            colors = ['#FF1801', '#FFB800', '#00D9FF', '#9D00FF', '#00FF9D']
            
            for i, team in enumerate(top_5):
                team_timeline = final_df[final_df['constructorRef'] == team].groupby('year').agg({
                    'points': 'sum',
                    'isWin': 'sum'
                }).reset_index().sort_values('year')
                
                fig_evolution.add_trace(go.Scatter3d(
                    x=team_timeline['year'],
                    y=[team] * len(team_timeline),
                    z=team_timeline['points'],
                    mode='lines+markers',
                    name=team,
                    line=dict(width=4, color=colors[i]),
                    marker=dict(size=6)
                ))
            
            fig_evolution.update_layout(
                title="Top 5 Teams Performance Over Years",
                scene=dict(
                    xaxis=dict(title='Year'),
                    yaxis=dict(title='Team'),
                    zaxis=dict(title='Points'),
                ),
                template='plotly_dark',
                height=600,
                showlegend=True
            )
            
            st.plotly_chart(fig_evolution, use_container_width=True)
            
            st.markdown("---")
            st.subheader("3D Bubble Analysis")
            
            consistency_list = []
            for team in all_teams_data['constructorRef']:
                races = final_df[final_df['constructorRef'] == team].groupby('raceId')['points'].sum()
                consistency_list.append(races.std() if len(races) > 1 else 0)
            
            all_teams_data['consistency'] = consistency_list
            
            fig_bubble = go.Figure(data=[go.Scatter3d(
                x=all_teams_data['isWin'],
                y=all_teams_data['points'],
                z=all_teams_data['consistency'],
                mode='markers',
                marker=dict(
                    size=7,
                    color=all_teams_data['points'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Points"),
                    opacity=0.8,
                    line=dict(color='white', width=2)
                ),
                text=all_teams_data['constructorRef'],
                hovertemplate='<b>%{text}</b><br>Wins: %{x}<br>Points: %{y}<br>Consistency: %{z:.2f}<extra></extra>'
            )])
            
            fig_bubble.update_layout(
                title="3D: Wins vs Points vs Consistency",
                scene=dict(
                    xaxis=dict(title='Wins'),
                    yaxis=dict(title='Points'),
                    zaxis=dict(title='Consistency'),
                ),
                template='plotly_dark',
                height=600
            )
            
            st.plotly_chart(fig_bubble, use_container_width=True)
        
        with tab2:
            st.subheader("2D Performance")
            col1, col2 = st.columns(2)
            
            with col1:
                pts_season = team_info.groupby('year')['points'].sum()
                fig = px.bar(x=pts_season.index, y=pts_season.values, title="Points Per Season", color=pts_season.values, color_continuous_scale='Blues')
                fig.update_layout(template='plotly_dark', height=350)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                wins_season = team_info.groupby('year')['isWin'].sum()
                fig = px.line(x=wins_season.index, y=wins_season.values, markers=True, title="Wins Over Seasons")
                fig.update_traces(line=dict(color='#FF1801', width=3), marker=dict(size=8))
                fig.update_layout(template='plotly_dark', height=350)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Top Drivers")
            top_drivers = team_info.groupby('driverRef')['points'].sum().sort_values(ascending=False).head(5)
            fig = px.bar(x=top_drivers.values, y=top_drivers.index, orientation='h', title="Driver Contribution", color=top_drivers.values, color_continuous_scale='Reds')
            fig.update_layout(template='plotly_dark', height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("Rankings")
            all_teams = final_df.groupby('constructorRef').agg({'isWin': 'sum', 'points': 'sum'})
            rank_wins = (all_teams['isWin'] >= team_info['isWin'].sum()).sum()
            rank_pts = (all_teams['points'] >= team_info['points'].sum()).sum()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Wins Rank", f"#{rank_wins}")
            with col2:
                st.metric("Points Rank", f"#{rank_pts}")

elif page == "Advanced Analytics":
    st.title("Advanced Analytics")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Grid Analysis", "Trends", "Comparisons", "Insights"])
    
    with tab1:
        st.subheader("Grid Position Impact")
        
        grid_stats = final_df.groupby('grid').agg({
            'isWin': lambda x: (x == 1).sum() / len(x) * 100,
            'isPodium': lambda x: (x == 1).sum() / len(x) * 100,
            'points': 'mean'
        }).reset_index()
        grid_stats = grid_stats[grid_stats['grid'] <= 15]
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(grid_stats, x='grid', y='isWin', title="Win Rate by Position", color='isWin', color_continuous_scale='Reds')
            fig.update_layout(template='plotly_dark', height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(grid_stats, x='grid', y='points', size='isPodium', color='isWin', color_continuous_scale='RdYlGn', title="Grid vs Points")
            fig.update_layout(template='plotly_dark', height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Historical Trends")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'isDNF' in final_df.columns:
                dnf = final_df.groupby('year').apply(lambda x: (x['isDNF'] == 1).sum() / len(x) * 100)
                fig = px.line(x=dnf.index, y=dnf.values, markers=True, title="DNF Rate Trend")
                fig.update_traces(line=dict(color='#FF1801', width=3))
                fig.update_layout(template='plotly_dark', height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            podium = final_df.groupby('year').apply(lambda x: (x['isPodium'] == 1).sum() / len(x) * 100)
            fig = px.area(x=podium.index, y=podium.values, title="Podium Rate Trend", fill='tozeroy')
            fig.update_traces(fillcolor='rgba(255, 24, 1, 0.3)', line=dict(color='#FF1801', width=3))
            fig.update_layout(template='plotly_dark', height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Driver Comparison")
        
        all_drivers = sorted(final_df['driverRef'].unique())
        col1, col2 = st.columns(2)
        
        with col1:
            driver1 = st.selectbox("Driver 1:", all_drivers, key="d1")
        with col2:
            driver2 = st.selectbox("Driver 2:", all_drivers, key="d2")
        
        if driver1 and driver2 and driver1 != driver2:
            d1 = final_df[final_df['driverRef'] == driver1]
            d2 = final_df[final_df['driverRef'] == driver2]
            
            comparison = pd.DataFrame({
                'Stat': ['Wins', 'Podiums', 'Points', 'Avg Grid', 'Finish Rate'],
                driver1: [
                    int(d1['isWin'].sum()),
                    int(d1['isPodium'].sum()),
                    int(d1['points'].sum()),
                    f"{d1['grid'].mean():.1f}",
                    f"{(d1['isFinished'].sum() / len(d1) * 100):.1f}%" if 'isFinished' in d1.columns else "N/A"
                ],
                driver2: [
                    int(d2['isWin'].sum()),
                    int(d2['isPodium'].sum()),
                    int(d2['points'].sum()),
                    f"{d2['grid'].mean():.1f}",
                    f"{(d2['isFinished'].sum() / len(d2) * 100):.1f}%" if 'isFinished' in d2.columns else "N/A"
                ]
            })
            
            st.dataframe(comparison, use_container_width=True, hide_index=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                comp_data = pd.DataFrame({
                    'Driver': [driver1, driver2],
                    'Wins': [int(d1['isWin'].sum()), int(d2['isWin'].sum())],
                    'Podiums': [int(d1['isPodium'].sum()), int(d2['isPodium'].sum())]
                })
                
                fig = go.Figure()
                fig.add_trace(go.Bar(x=comp_data['Driver'], y=comp_data['Wins'], name='Wins', marker_color='#FF1801'))
                fig.add_trace(go.Bar(x=comp_data['Driver'], y=comp_data['Podiums'], name='Podiums', marker_color='#FFB800'))
                fig.update_layout(template='plotly_dark', height=350, barmode='group')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                pts_data = pd.DataFrame({
                    'Driver': [driver1, driver2],
                    'Points': [int(d1['points'].sum()), int(d2['points'].sum())]
                })
                
                fig = px.pie(pts_data, values='Points', names='Driver', color_discrete_sequence=['#FF1801', '#FFB800'])
                fig.update_layout(template='plotly_dark', height=350)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select two different drivers")
    
    with tab4:
        st.subheader("Performance Insights")
        
        latest_season = final_df['year'].max()
        recent = final_df[final_df['year'] == latest_season]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Best Overtakers**")
            if 'gridToFinish' in recent.columns:
                overtakers = recent[recent['grid'] > 10].groupby('driverRef')['gridToFinish'].mean().sort_values(ascending=False)
                if len(overtakers) > 0:
                    for i, (driver, gain) in enumerate(overtakers.head(5).items(), 1):
                        st.write(f"{i}. {driver}: +{gain:.1f} positions")
        
        with col2:
            st.markdown("**Reliability Leaders**")
            if 'isDNF' in recent.columns:
                reliability = recent.groupby('driverRef').apply(lambda x: (x['isDNF'] == 0).sum() / len(x) * 100)
                if len(reliability) > 0:
                    for i, (driver, rate) in enumerate(reliability.sort_values(ascending=False).head(5).items(), 1):
                        st.write(f"{i}. {driver}: {rate:.1f}% finish")

elif page == "Championships":
    st.title("Championship History")
    
    all_seasons = sorted(final_df['year'].unique(), reverse=True)
    chosen_season = st.selectbox("Season:", all_seasons)
    
    season_data = final_df[final_df['year'] == chosen_season]
    
    st.subheader(f"Championship - {chosen_season}")
    
    standings = season_data.groupby('driverRef').agg({
        'points': 'sum',
        'isWin': 'sum',
        'isPodium': 'sum'
    }).sort_values('points', ascending=False)
    
    col1, col2, col3 = st.columns(3)
    
    if len(standings) >= 1:
        d1, s1 = list(standings.iterrows())[0]
        with col1:
            st.markdown(f"""
            <div class="stat-box" style="border-color: #FFD700;">
                <h3 style="text-align: center; margin: 0;">1st Champion</h3>
                <h2 style="text-align: center; color: #FFD700; margin: 10px 0;">{d1}</h2>
                <p style="text-align: center;"><b>{int(s1['points'])} Points</b></p>
                <p style="text-align: center; font-size: 12px;">{int(s1['isWin'])} W | {int(s1['isPodium'])} P</p>
            </div>
            """, unsafe_allow_html=True)
    
    if len(standings) >= 2:
        d2, s2 = list(standings.iterrows())[1]
        with col2:
            st.markdown(f"""
            <div class="stat-box" style="border-color: #C0C0C0;">
                <h3 style="text-align: center; margin: 0;">2nd Runner-up</h3>
                <h2 style="text-align: center; color: #C0C0C0; margin: 10px 0;">{d2}</h2>
                <p style="text-align: center;"><b>{int(s2['points'])} Points</b></p>
                <p style="text-align: center; font-size: 12px;">{int(s2['isWin'])} W | {int(s2['isPodium'])} P</p>
            </div>
            """, unsafe_allow_html=True)
    
    if len(standings) >= 3:
        d3, s3 = list(standings.iterrows())[2]
        with col3:
            st.markdown(f"""
            <div class="stat-box" style="border-color: #CD7F32;">
                <h3 style="text-align: center; margin: 0;">3rd Third</h3>
                <h2 style="text-align: center; color: #CD7F32; margin: 10px 0;">{d3}</h2>
                <p style="text-align: center;"><b>{int(s3['points'])} Points</b></p>
                <p style="text-align: center; font-size: 12px;">{int(s3['isWin'])} W | {int(s3['isPodium'])} P</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("Full Standings")
    
    display = pd.DataFrame({
        'Pos': range(1, len(standings.head(15)) + 1),
        'Driver': standings.head(15).index,
        'Points': standings.head(15)['points'].values.astype(int),
        'Wins': standings.head(15)['isWin'].values.astype(int),
        'Podiums': standings.head(15)['isPodium'].values.astype(int)
    })
    
    st.dataframe(display, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.subheader("Season Stats")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Races", season_data['raceId'].nunique())
    with col2:
        st.metric("Drivers", season_data['driverRef'].nunique())
    with col3:
        st.metric("Teams", season_data['constructorRef'].nunique())
    with col4:
        if 'isDNF' in season_data.columns:
            dnf_pct = (season_data['isDNF'].sum() / len(season_data) * 100)
            st.metric("DNF Rate", f"{dnf_pct:.1f}%")

st.markdown("---")
st.markdown("<div style='text-align: center; color: #888; padding: 20px;'><p>F1 Dashboard | ML Powered | 2025</p></div>", unsafe_allow_html=True)
