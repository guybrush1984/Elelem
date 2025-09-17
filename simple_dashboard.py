#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import numpy as np

# Configuration
ELELEM_API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Elelem Metrics Dashboard",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Elelem Advanced Metrics Dashboard")

# Get overall metrics
try:
    response = requests.get(f"{ELELEM_API_URL}/v1/metrics/summary", timeout=10)
    metrics = response.json()

    # Display overall stats with delta indicators
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Requests", metrics.get('total_calls', 0), delta=None)
    with col2:
        success_rate = metrics.get('success_rate', 0) * 100
        st.metric("Success Rate", f"{success_rate:.1f}%",
                 delta=f"{success_rate - 95:.1f}%" if success_rate < 95 else None)
    with col3:
        total_cost = metrics.get('total_cost_usd', {}).get('total', 0)
        st.metric("Total Cost", f"${total_cost:.4f}", delta=None)
    with col4:
        total_output = metrics.get('output_tokens', {}).get('total', 0)
        total_duration = metrics.get('duration_seconds', {}).get('total', 1)
        speed = total_output / max(total_duration, 0.001)
        st.metric("Avg Speed", f"{speed:.1f} tok/s", delta=None)

except Exception as e:
    st.error(f"Could not fetch metrics: {e}")

# Get model-specific data for fancy visualizations
st.subheader("📊 Advanced Model Analytics")

try:
    # Get available models from tags
    tags_response = requests.get(f"{ELELEM_API_URL}/v1/metrics/tags", timeout=5)
    tags_data = tags_response.json()

    model_tags = [tag for tag in tags_data.get("tags", []) if tag.startswith("model:")]
    models = [tag.replace("model:", "") for tag in model_tags]

    if models:
        # Fetch metrics for each model
        model_data = []

        for model in models:
            model_response = requests.get(
                f"{ELELEM_API_URL}/v1/metrics/summary?tags=model:{model}",
                timeout=5
            )
            model_metrics = model_response.json()

            if model_metrics.get('total_calls', 0) > 0:
                input_tokens = model_metrics.get('input_tokens', {}).get('total', 0)
                output_tokens = model_metrics.get('output_tokens', {}).get('total', 0)
                duration = model_metrics.get('duration_seconds', {}).get('total', 1)
                cost = model_metrics.get('total_cost_usd', {}).get('total', 0)

                model_data.append({
                    'Model': model,
                    'Calls': model_metrics.get('total_calls', 0),
                    'Success Rate': model_metrics.get('success_rate', 0) * 100,
                    'Tokens/sec': output_tokens / max(duration, 0.001),
                    'Cost': cost,
                    'Input Tokens': input_tokens,
                    'Output Tokens': output_tokens,
                    'Avg Duration': duration / max(model_metrics.get('total_calls', 1), 1),
                    'Cost per Call': cost / max(model_metrics.get('total_calls', 1), 1)
                })

        if model_data:
            df = pd.DataFrame(model_data)

            # Create tabs for different visualizations
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Performance Overview", "💰 Cost Analysis", "⚡ Speed Metrics", "📈 Token Usage", "📅 Timeline View"])

            with tab1:
                # Multi-metric comparison chart
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Requests per Model', 'Success Rate', 'Average Response Time', 'Cost per Call'),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"secondary_y": False}]]
                )

                # Bar chart for requests
                fig.add_trace(
                    go.Bar(x=df['Model'], y=df['Calls'], name='Requests',
                          marker_color='steelblue', showlegend=False),
                    row=1, col=1
                )

                # Success rate gauge-style bar
                colors = ['red' if x < 90 else 'orange' if x < 95 else 'green' for x in df['Success Rate']]
                fig.add_trace(
                    go.Bar(x=df['Model'], y=df['Success Rate'], name='Success %',
                          marker_color=colors, showlegend=False),
                    row=1, col=2
                )

                # Response time
                fig.add_trace(
                    go.Bar(x=df['Model'], y=df['Avg Duration'], name='Avg Duration',
                          marker_color='purple', showlegend=False),
                    row=2, col=1
                )

                # Cost per call
                fig.add_trace(
                    go.Bar(x=df['Model'], y=df['Cost per Call'], name='Cost/Call',
                          marker_color='orange', showlegend=False),
                    row=2, col=2
                )

                fig.update_layout(height=600, title_text="📊 Comprehensive Model Performance Dashboard")
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                # Cost analysis with pie chart and scatter
                col1, col2 = st.columns(2)

                with col1:
                    # Pie chart of cost distribution
                    fig_pie = px.pie(df, values='Cost', names='Model',
                                    title='💰 Cost Distribution by Model')
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_pie, use_container_width=True)

                with col2:
                    # Scatter plot: calls vs cost
                    fig_scatter = px.scatter(df, x='Calls', y='Cost', size='Cost per Call',
                                           color='Model', title='💸 Requests vs Total Cost',
                                           hover_data=['Cost per Call'])
                    fig_scatter.update_traces(marker=dict(sizemode='diameter', sizeref=0.1))
                    st.plotly_chart(fig_scatter, use_container_width=True)

                # Cost efficiency table
                st.subheader("💵 Cost Efficiency Rankings")
                cost_df = df[['Model', 'Cost per Call', 'Tokens/sec', 'Success Rate']].copy()
                cost_df = cost_df.sort_values('Cost per Call')
                cost_df['Cost per Call'] = cost_df['Cost per Call'].apply(lambda x: f"${x:.6f}")
                cost_df['Success Rate'] = cost_df['Success Rate'].apply(lambda x: f"{x:.1f}%")
                cost_df['Tokens/sec'] = cost_df['Tokens/sec'].apply(lambda x: f"{x:.1f}")
                st.dataframe(cost_df, use_container_width=True)

            with tab3:
                # Speed and performance metrics
                col1, col2 = st.columns(2)

                with col1:
                    # Tokens per second comparison
                    fig_speed = go.Figure()
                    fig_speed.add_trace(go.Bar(
                        x=df['Model'],
                        y=df['Tokens/sec'],
                        marker_color=px.colors.sequential.Plasma_r,
                        text=[f"{x:.1f}" for x in df['Tokens/sec']],
                        textposition='auto'
                    ))
                    fig_speed.update_layout(title='⚡ Processing Speed (Tokens/Second)')
                    st.plotly_chart(fig_speed, use_container_width=True)

                with col2:
                    # Response time vs success rate
                    fig_perf = px.scatter(df, x='Avg Duration', y='Success Rate',
                                        size='Calls', color='Model',
                                        title='🎯 Response Time vs Success Rate',
                                        labels={'Avg Duration': 'Average Response Time (s)',
                                               'Success Rate': 'Success Rate (%)'})
                    fig_perf.add_hline(y=95, line_dash="dash", line_color="green",
                                     annotation_text="95% Target")
                    st.plotly_chart(fig_perf, use_container_width=True)

            with tab4:
                # Token usage analysis
                col1, col2 = st.columns(2)

                with col1:
                    # Stacked bar chart for token types
                    fig_tokens = go.Figure()
                    fig_tokens.add_trace(go.Bar(
                        name='Input Tokens',
                        x=df['Model'],
                        y=df['Input Tokens'],
                        marker_color='lightblue'
                    ))
                    fig_tokens.add_trace(go.Bar(
                        name='Output Tokens',
                        x=df['Model'],
                        y=df['Output Tokens'],
                        marker_color='darkblue'
                    ))
                    fig_tokens.update_layout(barmode='stack', title='📊 Token Usage Breakdown')
                    st.plotly_chart(fig_tokens, use_container_width=True)

                with col2:
                    # Token ratio analysis
                    df_ratio = df.copy()
                    df_ratio['I/O Ratio'] = df_ratio['Input Tokens'] / (df_ratio['Output Tokens'] + 1)
                    fig_ratio = px.bar(df_ratio, x='Model', y='I/O Ratio',
                                     title='📈 Input/Output Token Ratio',
                                     color='I/O Ratio',
                                     color_continuous_scale='RdYlBu_r')
                    st.plotly_chart(fig_ratio, use_container_width=True)

                # Detailed token statistics
                st.subheader("📋 Token Usage Statistics")
                token_df = df[['Model', 'Input Tokens', 'Output Tokens', 'Calls']].copy()
                token_df['Avg Input/Call'] = token_df['Input Tokens'] / token_df['Calls']
                token_df['Avg Output/Call'] = token_df['Output Tokens'] / token_df['Calls']
                token_df['Total Tokens'] = token_df['Input Tokens'] + token_df['Output Tokens']
                st.dataframe(token_df, use_container_width=True)

            with tab5:
                # Timeline view of requests
                st.subheader("📅 Request Timeline")

                try:
                    # Fetch detailed metrics data with timestamps
                    timeline_response = requests.get(f"{ELELEM_API_URL}/v1/metrics/data", timeout=10)
                    timeline_data = timeline_response.json()

                    if timeline_data:
                        # Convert to DataFrame
                        timeline_df = pd.DataFrame(timeline_data)
                        timeline_df['timestamp'] = pd.to_datetime(timeline_df['timestamp'])

                        # Time range selector for timeline
                        col1, col2 = st.columns(2)
                        with col1:
                            hours_back = st.slider("Hours to show", 1, 24, 6)
                        with col2:
                            bin_size = st.selectbox("Time bins", ["1min", "5min", "15min", "1hour"], index=1)

                        # Filter by time range
                        cutoff_time = timeline_df['timestamp'].max() - pd.Timedelta(hours=hours_back)
                        recent_df = timeline_df[timeline_df['timestamp'] >= cutoff_time].copy()

                        if not recent_df.empty:
                            # Requests over time
                            fig_timeline = px.histogram(
                                recent_df,
                                x='timestamp',
                                color='model' if 'model' in recent_df.columns else None,
                                title=f'📊 Requests Over Time (Last {hours_back}h)',
                                nbins=min(50, len(recent_df)),
                                marginal="rug"
                            )
                            fig_timeline.update_layout(
                                xaxis_title="Time",
                                yaxis_title="Number of Requests"
                            )
                            st.plotly_chart(fig_timeline, use_container_width=True)

                            # Success/failure timeline
                            if 'success' in recent_df.columns:
                                success_timeline = recent_df.groupby([
                                    pd.Grouper(key='timestamp', freq=bin_size),
                                    'success'
                                ]).size().unstack(fill_value=0).reset_index()

                                if not success_timeline.empty:
                                    fig_success_timeline = go.Figure()

                                    if True in success_timeline.columns:
                                        fig_success_timeline.add_trace(go.Scatter(
                                            x=success_timeline['timestamp'],
                                            y=success_timeline[True],
                                            mode='lines+markers',
                                            name='Successful',
                                            line=dict(color='green'),
                                            fill='tonexty'
                                        ))

                                    if False in success_timeline.columns:
                                        fig_success_timeline.add_trace(go.Scatter(
                                            x=success_timeline['timestamp'],
                                            y=success_timeline[False],
                                            mode='lines+markers',
                                            name='Failed',
                                            line=dict(color='red'),
                                            fill='tozeroy'
                                        ))

                                    fig_success_timeline.update_layout(
                                        title='✅❌ Success/Failure Timeline',
                                        xaxis_title='Time',
                                        yaxis_title='Requests',
                                        hovermode='x unified'
                                    )
                                    st.plotly_chart(fig_success_timeline, use_container_width=True)

                            # Response time over time
                            if 'duration_seconds' in recent_df.columns:
                                # Bin by time and calculate average response time
                                response_time_bins = recent_df.groupby(
                                    pd.Grouper(key='timestamp', freq=bin_size)
                                )['duration_seconds'].agg(['mean', 'median', 'std']).reset_index()

                                fig_response_time = go.Figure()

                                # Add mean line
                                fig_response_time.add_trace(go.Scatter(
                                    x=response_time_bins['timestamp'],
                                    y=response_time_bins['mean'],
                                    mode='lines+markers',
                                    name='Average',
                                    line=dict(color='blue')
                                ))

                                # Add median line
                                fig_response_time.add_trace(go.Scatter(
                                    x=response_time_bins['timestamp'],
                                    y=response_time_bins['median'],
                                    mode='lines+markers',
                                    name='Median',
                                    line=dict(color='orange')
                                ))

                                # Add std deviation as fill
                                if not response_time_bins['std'].isna().all():
                                    upper_bound = response_time_bins['mean'] + response_time_bins['std']
                                    lower_bound = response_time_bins['mean'] - response_time_bins['std']

                                    fig_response_time.add_trace(go.Scatter(
                                        x=list(response_time_bins['timestamp']) + list(reversed(response_time_bins['timestamp'])),
                                        y=list(upper_bound) + list(reversed(lower_bound)),
                                        fill='toself',
                                        fillcolor='rgba(0,100,80,0.2)',
                                        line=dict(color='rgba(255,255,255,0)'),
                                        name='±1 Std Dev'
                                    ))

                                fig_response_time.update_layout(
                                    title='⏱️ Response Time Over Time',
                                    xaxis_title='Time',
                                    yaxis_title='Response Time (seconds)',
                                    hovermode='x unified'
                                )
                                st.plotly_chart(fig_response_time, use_container_width=True)

                            # Cost accumulation over time
                            if 'cost_usd' in recent_df.columns:
                                recent_df_sorted = recent_df.sort_values('timestamp')
                                recent_df_sorted['cumulative_cost'] = recent_df_sorted['cost_usd'].cumsum()

                                fig_cost_timeline = px.line(
                                    recent_df_sorted,
                                    x='timestamp',
                                    y='cumulative_cost',
                                    title='💰 Cumulative Cost Over Time',
                                    color='model' if 'model' in recent_df.columns else None
                                )
                                fig_cost_timeline.update_layout(
                                    xaxis_title='Time',
                                    yaxis_title='Cumulative Cost ($)'
                                )
                                st.plotly_chart(fig_cost_timeline, use_container_width=True)

                            # Recent requests table
                            st.subheader("📋 Recent Requests")
                            display_columns = ['timestamp', 'model', 'success', 'duration_seconds']
                            if 'cost_usd' in recent_df.columns:
                                display_columns.append('cost_usd')

                            recent_requests = recent_df[display_columns].tail(20).sort_values('timestamp', ascending=False)
                            recent_requests['timestamp'] = recent_requests['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

                            st.dataframe(recent_requests, use_container_width=True)
                        else:
                            st.info(f"No data available for the last {hours_back} hours")
                    else:
                        st.info("No timeline data available yet")

                except Exception as e:
                    st.error(f"Could not fetch timeline data: {e}")
                    st.info("Timeline requires individual request records from /v1/metrics/data endpoint")

        else:
            st.info("No model-specific data available")
    else:
        st.info("No models found in metrics")

except Exception as e:
    st.error(f"Could not fetch model data: {e}")

# Enhanced raw data section
st.subheader("🔍 Advanced Data Explorer")
with st.expander("Show detailed metrics breakdown"):
    try:
        response = requests.get(f"{ELELEM_API_URL}/v1/metrics/summary", timeout=5)
        raw_data = response.json()

        # Create formatted display
        col1, col2 = st.columns(2)
        with col1:
            st.json({"Summary Stats": {
                "total_calls": raw_data.get('total_calls', 0),
                "success_rate": raw_data.get('success_rate', 0),
                "total_cost": raw_data.get('total_cost_usd', {})
            }})
        with col2:
            st.json({"Token & Duration Stats": {
                "input_tokens": raw_data.get('input_tokens', {}),
                "output_tokens": raw_data.get('output_tokens', {}),
                "duration_seconds": raw_data.get('duration_seconds', {})
            }})

    except Exception as e:
        st.error(f"Error: {e}")

# Add some styling
st.markdown("""
<style>
.metric-container {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)