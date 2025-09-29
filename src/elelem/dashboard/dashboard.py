#!/usr/bin/env python3
"""
🚀 Elelem Dashboard - Standalone

A completely independent dashboard for Elelem metrics.
Connects directly to PostgreSQL database without any Elelem dependencies.

Environment Variables:
- DATABASE_URL: PostgreSQL connection string (required)
- DASHBOARD_TITLE: Custom dashboard title (optional)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import os
import json
from typing import Optional, List
import sqlalchemy
from sqlalchemy import create_engine, text

from schema import REQUEST_METRICS_COLUMNS, DISPLAY_COLUMNS, COLUMN_TYPES

# Configure Streamlit
st.set_page_config(
    page_title="🚀 Elelem Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def get_database_connection():
    """Get database connection using environment variable."""
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        st.error("❌ DATABASE_URL environment variable not set!")
        st.stop()

    try:
        engine = create_engine(database_url)
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as e:
        st.error(f"❌ Database connection failed: {e}")
        st.stop()

def load_data(engine, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None, limit: int = 1000) -> pd.DataFrame:
    """Load data from database with optional time filtering."""
    try:
        # Build query
        query = "SELECT * FROM request_metrics"
        params = {}
        conditions = []

        if start_time:
            conditions.append("timestamp >= :start_time")
            params['start_time'] = start_time

        if end_time:
            conditions.append("timestamp <= :end_time")
            params['end_time'] = end_time

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY timestamp DESC"

        if limit:
            query += f" LIMIT {limit}"

        # Load data using SQLAlchemy text() for proper parameter binding
        df = pd.read_sql(text(query), engine, params=params)

        # Parse JSON tags
        if 'tags' in df.columns and not df.empty:
            df['tags'] = df['tags'].apply(
                lambda x: json.loads(x) if isinstance(x, str) and x.strip() else []
            )

        return df

    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return pd.DataFrame()

def calculate_stats(df: pd.DataFrame) -> dict:
    """Calculate summary statistics from DataFrame."""
    if df.empty:
        return {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'success_rate': 0.0,
            'total_cost': 0.0,
            'avg_duration': 0.0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_reasoning_tokens': 0
        }

    successful = df[df['status'] == 'success']
    failed = df[df['status'] == 'failed']

    return {
        'total_requests': len(df),
        'successful_requests': len(successful),
        'failed_requests': len(failed),
        'success_rate': len(successful) / len(df) if len(df) > 0 else 0.0,
        'total_cost': successful['total_cost_usd'].sum() if not successful.empty else 0.0,
        'avg_duration': df['total_duration_seconds'].mean() if not df.empty else 0.0,
        'total_input_tokens': successful['input_tokens'].sum() if not successful.empty else 0,
        'total_output_tokens': successful['output_tokens'].sum() if not successful.empty else 0,
        'total_reasoning_tokens': successful['reasoning_tokens'].sum() if not successful.empty else 0
    }

def main():
    """Main dashboard function."""

    # Get database connection
    engine = get_database_connection()

    # Header
    dashboard_title = os.getenv('DASHBOARD_TITLE', '🚀 Elelem Dashboard')
    st.title(dashboard_title)
    st.caption("Standalone metrics dashboard")

    # Sidebar controls
    st.sidebar.header("📅 Controls")

    # Time range selector
    time_options = {
        "Last 1 minute": (datetime.now() - timedelta(minutes=1), datetime.now()),
        "Last 15 minutes": (datetime.now() - timedelta(minutes=15), datetime.now()),
        "Last 1 hour": (datetime.now() - timedelta(hours=1), datetime.now()),
        "Last 6 hours": (datetime.now() - timedelta(hours=6), datetime.now()),
        "Last 24 hours": (datetime.now() - timedelta(hours=24), datetime.now()),
        "All time": (None, None)
    }

    selected_range = st.sidebar.selectbox(
        "Time Range:",
        list(time_options.keys()),
        index=2  # Default to 1 hour
    )

    start_time, end_time = time_options[selected_range]

    # Limit selector
    max_rows = st.sidebar.number_input(
        "Max rows to display:",
        min_value=10,
        max_value=10000,
        value=500,
        step=50
    )

    # Auto-refresh control
    auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (30s)", value=True)

    if auto_refresh:
        # Auto-refresh every 30 seconds
        time.sleep(0.1)

        if "last_refresh" not in st.session_state:
            st.session_state.last_refresh = time.time()

        if time.time() - st.session_state.last_refresh > 30:
            st.session_state.last_refresh = time.time()
            st.rerun()

        st.sidebar.write(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")

    # Load data
    with st.spinner("Loading data..."):
        df = load_data(engine, start_time, end_time, max_rows)

    if df.empty:
        st.warning("📭 No data available for the selected time range")
        return

    # Calculate statistics
    stats = calculate_stats(df)

    # Display summary metrics
    st.subheader("📊 Summary Metrics")

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric("🎯 Total Requests", stats['total_requests'])

    with col2:
        st.metric("✅ Success Rate", f"{stats['success_rate']:.1%}")

    with col3:
        st.metric("💰 Total Cost", f"${stats['total_cost']:.6f}")

    with col4:
        st.metric("⏱️ Avg Duration", f"{stats['avg_duration']:.2f}s")

    with col5:
        st.metric("🎫 Input Tokens", f"{stats['total_input_tokens']:,}")

    with col6:
        st.metric("📝 Output Tokens", f"{stats['total_output_tokens']:,}")

    # Charts section
    if len(df) > 1:
        st.subheader("📈 Analytics")

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            # Cost per model
            successful = df[df['status'] == 'success']
            if not successful.empty and 'actual_model' in successful.columns:
                st.write("💰 **Cost by Model**")
                model_costs = successful.groupby('actual_model')['total_cost_usd'].sum().reset_index()
                model_costs.columns = ['Model', 'Cost']

                if not model_costs.empty:
                    fig_cost = px.bar(
                        model_costs,
                        x='Model',
                        y='Cost',
                        title="Total Cost by Model"
                    )
                    fig_cost.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_cost, use_container_width=True)

        with chart_col2:
            # Provider distribution
            if not successful.empty and 'actual_provider' in successful.columns:
                st.write("🏢 **Usage by Provider**")
                provider_costs = successful.groupby('actual_provider')['total_cost_usd'].sum().reset_index()
                provider_costs.columns = ['Provider', 'Cost']

                if not provider_costs.empty:
                    fig_provider = px.pie(
                        provider_costs,
                        values='Cost',
                        names='Provider',
                        title="Cost Distribution by Provider"
                    )
                    st.plotly_chart(fig_provider, use_container_width=True)

        # Requests over time
        if len(df) > 5:
            st.write("📊 **Requests Over Time**")
            df_time = df.copy()
            df_time['timestamp'] = pd.to_datetime(df_time['timestamp'])

            # Group by 5-minute intervals
            df_time = df_time.set_index('timestamp').resample('5min').agg({
                'request_id': 'count',
                'total_cost_usd': 'sum',
                'status': lambda x: (x == 'success').sum() / len(x) if len(x) > 0 else 0
            }).reset_index()

            df_time.columns = ['timestamp', 'requests', 'cost', 'success_rate']

            fig_time = go.Figure()

            fig_time.add_trace(go.Scatter(
                x=df_time['timestamp'],
                y=df_time['requests'],
                mode='lines+markers',
                name='Requests',
                yaxis='y'
            ))

            fig_time.add_trace(go.Scatter(
                x=df_time['timestamp'],
                y=df_time['cost'],
                mode='lines+markers',
                name='Cost ($)',
                yaxis='y2'
            ))

            fig_time.update_layout(
                title="Requests and Cost Over Time (5min intervals)",
                xaxis_title="Time",
                yaxis=dict(title="Requests", side="left"),
                yaxis2=dict(title="Cost ($)", side="right", overlaying="y")
            )

            st.plotly_chart(fig_time, use_container_width=True)

    # Raw data table
    st.subheader("📋 Raw Request Data")

    # Filter to display columns
    display_df = df[DISPLAY_COLUMNS].copy() if all(col in df.columns for col in DISPLAY_COLUMNS) else df

    # Format columns for better display
    if 'timestamp' in display_df.columns:
        display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

    if 'total_duration_seconds' in display_df.columns:
        display_df['total_duration_seconds'] = display_df['total_duration_seconds'].round(3)

    if 'total_cost_usd' in display_df.columns:
        display_df['total_cost_usd'] = display_df['total_cost_usd'].apply(
            lambda x: f"${x:.6f}" if pd.notna(x) else ""
        )

    # Display table
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=400
    )

    st.caption(f"Showing {len(display_df)} records")

    # Recent errors section
    error_df = df[df['status'] == 'failed']
    if not error_df.empty:
        st.subheader("❌ Recent Errors")

        error_display = error_df[
            ['timestamp', 'requested_model', 'final_error_type', 'final_error_message']
        ].head(10).copy()

        if 'timestamp' in error_display.columns:
            error_display['timestamp'] = pd.to_datetime(error_display['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

        st.dataframe(error_display, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()