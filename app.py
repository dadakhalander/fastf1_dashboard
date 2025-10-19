import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Add this to your CONSTRUCTOR ANALYSIS PAGE section
# Replace the entire Constructor Analysis page with this updated version

# CONSTRUCTOR ANALYSIS PAGE
elif page == "Constructor Analysis":
    st.title("Team (Constructor) Analysis")
    st.markdown("Performance metrics for Formula 1 teams with 3D visualization")
    
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
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["3D Performance Map", "2D Performance", "Drivers", "Rankings"])
        
        with tab1:
            st.subheader("3D Constructor Performance Map")
            st.markdown("Explore team performance across wins, points, and seasons in 3D space")
            
            # Prepare data for all teams
            all_teams_3d = final_df.groupby('constructorRef').agg({
                'isWin': 'sum',
                'points': 'sum',
                'year': 'max'
            }).reset_index()
            
            # Calculate average points per race
            team_race_counts = final_df.groupby('constructorRef').size()
            all_teams_3d['avgPointsPerRace'] = all_teams_3d.apply(
                lambda row: row['points'] / team_race_counts[row['constructorRef']] if row['constructorRef'] in team_race_counts.index else 0,
                axis=1
            )
            all_teams_3d['numRaces'] = all_teams_3d['constructorRef'].map(team_race_counts)
            
            # Create 3D scatter plot
            fig_3d_scatter = go.Figure(data=[go.Scatter3d(
                x=all_teams_3d['isWin'],
                y=all_teams_3d['points'],
                z=all_teams_3d['year'],
                mode='markers+text',
                marker=dict(
                    size=all_teams_3d['avgPointsPerRace'] / 50 + 5,
                    color=all_teams_3d['points'],
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title="Total Points"),
                    opacity=0.8,
                    line=dict(color='white', width=1)
                ),
                text=all_teams_3d['constructorRef'],
                textposition='top center',
                textfont=dict(size=9, color='white'),
                hovertemplate='<b>%{text}</b><br>Wins: %{x}<br>Points: %{y}<br>Latest Season: %{z}<extra></extra>'
            )])
            
            fig_3d_scatter.update_layout(
                title="Team Performance in 3D Space (Wins vs Points vs Season)",
                scene=dict(
                    xaxis=dict(title='Total Wins', backgroundcolor="rgb(30, 30, 46)", gridcolor="gray"),
                    yaxis=dict(title='Total Points', backgroundcolor="rgb(30, 30, 46)", gridcolor="gray"),
                    zaxis=dict(title='Latest Season', backgroundcolor="rgb(30, 30, 46)", gridcolor="gray"),
                    bgcolor="rgb(30, 30, 46)"
                ),
                template='plotly_dark',
                height=600,
                hovermode='closest'
            )
            
            st.plotly_chart(fig_3d_scatter, use_container_width=True)
            
            st.markdown("---")
            
            # 3D Line plot for performance over time
            st.subheader("Top Teams Performance Evolution (3D Timeline)")
            st.markdown("Track how top teams have performed over the years")
            
            top_5_teams = all_teams_3d.nlargest(5, 'points')['constructorRef'].tolist()
            
            fig_3d_evolution = go.Figure()
            
            colors = ['#FF1801', '#FFB800', '#00D9FF', '#9D00FF', '#00FF9D']
            
            for i, team in enumerate(top_5_teams):
                team_data = final_df[final_df['constructorRef'] == team].groupby('year').agg({
                    'points': 'sum',
                    'isWin': 'sum'
                }).reset_index().sort_values('year')
                
                fig_3d_evolution.add_trace(go.Scatter3d(
                    x=team_data['year'],
                    y=[team] * len(team_data),
                    z=team_data['points'],
                    mode='lines+markers',
                    name=team,
                    line=dict(width=4, color=colors[i]),
                    marker=dict(size=6, color=colors[i])
                ))
            
            fig_3d_evolution.update_layout(
                title="Top 5 Teams: Performance Evolution Over Years",
                scene=dict(
                    xaxis=dict(title='Year'),
                    yaxis=dict(title='Team'),
                    zaxis=dict(title='Points Per Season'),
                    bgcolor="rgb(30, 30, 46)"
                ),
                template='plotly_dark',
                height=600,
                showlegend=True
            )
            
            st.plotly_chart(fig_3d_evolution, use_container_width=True)
            
            st.markdown("---")
            
            # 3D Bubble chart for advanced comparison
            st.subheader("3D Team Analysis: Wins vs Points vs Consistency")
            st.markdown("Bubble size and position show performance consistency patterns")
            
            consistency_data = []
            for team in all_teams_3d['constructorRef']:
                team_races = final_df[final_df['constructorRef'] == team].groupby('raceId')['points'].sum()
                consistency = team_races.std() if len(team_races) > 1 else 0
                consistency_data.append(consistency)
            
            all_teams_3d['consistency'] = consistency_data
            
            fig_3d_bubble = go.Figure(data=[go.Scatter3d(
                x=all_teams_3d['isWin'],
                y=all_teams_3d['points'],
                z=all_teams_3d['consistency'],
                mode='markers',
                marker=dict(
                    size=7,
                    color=all_teams_3d['points'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Total Points"),
                    opacity=0.8,
                    line=dict(color='white', width=2)
                ),
                text=all_teams_3d['constructorRef'],
                hovertemplate='<b>%{text}</b><br>Wins: %{x}<br>Points: %{y}<br>Consistency (Std): %{z:.2f}<extra></extra>'
            )])
            
            fig_3d_bubble.update_layout(
                title="3D Team Comparison: Wins vs Points vs Consistency",
                scene=dict(
                    xaxis=dict(title='Total Wins'),
                    yaxis=dict(title='Total Points'),
                    zaxis=dict(title='Performance Consistency (Std Dev)'),
                    bgcolor="rgb(30, 30, 46)"
                ),
                template='plotly_dark',
                height=600
            )
            
            st.plotly_chart(fig_3d_bubble, use_container_width=True)
        
        with tab2:
            st.subheader("2D Performance Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                points_by_season = team_info.groupby('year')['points'].sum()
                fig = px.bar(x=points_by_season.index, y=points_by_season.values,
                            title="Points Earned Per Season", color=points_by_season.values,
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
        
        with tab3:
            st.subheader("Top Drivers in This Team")
            top_drivers_in_team = team_info.groupby('driverRef')['points'].sum().sort_values(ascending=False).head(5)
            fig = px.bar(x=top_drivers_in_team.values, y=top_drivers_in_team.index, orientation='h',
                        title="Driver Points Contribution", color=top_drivers_in_team.values,
                        color_continuous_scale='Reds')
            fig.update_layout(template='plotly_dark', height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("Team Rankings")
            all_teams = final_df.groupby('constructorRef').agg({'isWin': 'sum', 'points': 'sum'})
            rank_wins = (all_teams['isWin'] >= team_info['isWin'].sum()).sum()
            rank_points = (all_teams['points'] >= team_info['points'].sum()).sum()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Wins Ranking", f"#{rank_wins}")
            with col2:
                st.metric("Points Ranking", f"#{rank_points}")
        
        st.markdown("---")
        
        # NEW: 3D VISUALIZATION TABS
        tab1, tab2, tab3, tab4 = st.tabs(["3D Performance Map", "Performance", "Drivers", "Rankings"])
        
        with tab1:
            st.subheader("3D Constructor Performance Map")
            st.markdown("Explore team performance across wins, points, and seasons in 3D space")
            
            # Prepare data for all teams
            all_teams_3d = final_df.groupby('constructorRef').agg({
                'isWin': 'sum',
                'points': 'sum',
                'year': 'max'
            }).reset_index()
            
            # Calculate average points per race
            all_teams_3d['avgPointsPerRace'] = all_teams_3d['points'] / final_df.groupby('constructorRef').size()
            all_teams_3d['numRaces'] = final_df.groupby('constructorRef').size()
            
            # Create 3D scatter plot
            fig_3d_scatter = go.Figure(data=[go.Scatter3d(
                x=all_teams_3d['isWin'],
                y=all_teams_3d['points'],
                z=all_teams_3d['year'],
                mode='markers+text',
                marker=dict(
                    size=all_teams_3d['avgPointsPerRace'] / 50,
                    color=all_teams_3d['points'],
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title="Total Points"),
                    opacity=0.8,
                    line=dict(
                        color='white',
                        width=1
                    )
                ),
                text=all_teams_3d['constructorRef'],
                textposition='top center',
                textfont=dict(size=10, color='white'),
                hovertemplate='<b>%{text}</b><br>Wins: %{x}<br>Points: %{y}<br>Last Season: %{z}<extra></extra>'
            )])
            
            fig_3d_scatter.update_layout(
                title="Team Performance in 3D Space",
                scene=dict(
                    xaxis=dict(title='Total Wins', backgroundcolor="rgb(30, 30, 46)", gridcolor="gray"),
                    yaxis=dict(title='Total Points', backgroundcolor="rgb(30, 30, 46)", gridcolor="gray"),
                    zaxis=dict(title='Latest Season', backgroundcolor="rgb(30, 30, 46)", gridcolor="gray"),
                    bgcolor="rgb(30, 30, 46)"
                ),
                template='plotly_dark',
                height=600,
                hovermode='closest'
            )
            
            st.plotly_chart(fig_3d_scatter, use_container_width=True)
            
            st.markdown("---")
            
            # 3D Surface plot for performance over time
            st.subheader("Team Performance Evolution (3D Surface)")
            
            # Prepare time series data for top teams
            top_5_teams = all_teams_3d.nlargest(5, 'points')['constructorRef'].tolist()
            
            team_timeline = []
            for team in top_5_teams:
                team_data = final_df[final_df['constructorRef'] == team].groupby('year').agg({
                    'points': 'sum',
                    'isWin': 'sum'
                }).reset_index()
                team_timeline.append(team_data)
            
            if team_timeline:
                fig_3d_surface = go.Figure()
                
                for i, team_data in enumerate(team_timeline):
                    fig_3d_surface.add_trace(go.Scatter3d(
                        x=team_data['year'],
                        y=[top_5_teams[i]] * len(team_data),
                        z=team_data['points'],
                        mode='lines+markers',
                        name=top_5_teams[i],
                        line=dict(width=5),
                        marker=dict(size=8)
                    ))
                
                fig_3d_surface.update_layout(
                    title="Top 5 Teams Performance Evolution",
                    scene=dict(
                        xaxis=dict(title='Year'),
                        yaxis=dict(title='Team'),
                        zaxis=dict(title='Points'),
                        bgcolor="rgb(30, 30, 46)"
                    ),
                    template='plotly_dark',
                    height=600,
                    showlegend=True
                )
                
                st.plotly_chart(fig_3d_surface, use_container_width=True)
            
            st.markdown("---")
            
            # 3D Bubble chart comparing teams
            st.subheader("Team Comparison: Wins vs Points vs Consistency")
            
            # Calculate consistency (standard deviation of points across races)
            consistency_data = []
            for team in all_teams_3d['constructorRef']:
                team_races = final_df[final_df['constructorRef'] == team].groupby('raceId')['points'].sum()
                consistency = team_races.std() if len(team_races) > 1 else 0
                consistency_data.append(consistency)
            
            all_teams_3d['consistency'] = consistency_data
            
            fig_3d_bubble = go.Figure(data=[go.Scatter3d(
                x=all_teams_3d['isWin'],
                y=all_teams_3d['points'],
                z=all_teams_3d['consistency'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=all_teams_3d['points'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Total Points"),
                    opacity=0.8,
                    line=dict(color='white', width=2)
                ),
                text=all_teams_3d['constructorRef'],
                hovertemplate='<b>%{text}</b><br>Wins: %{x}<br>Points: %{y}<br>Consistency (Std): %{z:.1f}<extra></extra>'
            )])
            
            fig_3d_bubble.update_layout(
                title="3D Team Comparison: Wins vs Points vs Consistency",
                scene=dict(
                    xaxis=dict(title='Total Wins'),
                    yaxis=dict(title='Total Points'),
                    zaxis=dict(title='Performance Consistency'),
                    bgcolor="rgb(30, 30, 46)"
                ),
                template='plotly_dark',
                height=600
            )
            
            st.plotly_chart(fig_3d_bubble, use_container_width=True)
        
        with tab2:
            st.subheader("2D Performance Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                points_by_season = team_info.groupby('year')['points'].sum()
                fig = px.bar(x=points_by_season.index, y=points_by_season.values,
                            title="Points Earned Per Season", color=points_by_season.values,
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
        
        with tab3:
            st.subheader("Top Drivers in This Team")
            top_drivers_in_team = team_info.groupby('driverRef')['points'].sum().sort_values(ascending=False).head(5)
            fig = px.bar(x=top_drivers_in_team.values, y=top_drivers_in_team.index, orientation='h',
                        title="Driver Points Contribution", color=top_drivers_in_team.values,
                        color_continuous_scale='Reds')
            fig.update_layout(template='plotly_dark', height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("Team Rankings")
            all_teams = final_df.groupby('constructorRef').agg({'isWin': 'sum', 'points': 'sum'})
            rank_wins = (all_teams['isWin'] >= team_info['isWin'].sum()).sum()
            rank_points = (all_teams['points'] >= team_info['points'].sum()).sum()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Wins Ranking", f"#{rank_wins}")
            with col2:
                st.metric("Points Ranking", f"#{rank_points}")
