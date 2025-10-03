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

def load_data(engine, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None, tags: Optional[List[str]] = None, limit: int = 1000) -> pd.DataFrame:
    """Load data from database with optional time and tag filtering."""
    try:
        params = {}

        # Build base query - conditionally join with tags if filtering by tags
        if tags:
            # When filtering by tags with AND logic (must have ALL tags)
            # Use GROUP BY and HAVING COUNT to ensure all tags are present
            if len(tags) == 1:
                tag_filter = f"rt.tag = '{tags[0]}'"
            else:
                tag_filter = f"rt.tag IN {tuple(tags)}"

            # Build time filter conditions
            time_conditions = []
            if start_time:
                time_conditions.append("rm.timestamp >= :start_time")
                params['start_time'] = start_time
            if end_time:
                time_conditions.append("rm.timestamp <= :end_time")
                params['end_time'] = end_time

            time_filter = " AND " + " AND ".join(time_conditions) if time_conditions else ""

            query = f"""
            SELECT rm.*
            FROM request_metrics rm
            WHERE rm.request_id IN (
                SELECT rt.request_id
                FROM request_tags rt
                WHERE {tag_filter}
                GROUP BY rt.request_id
                HAVING COUNT(DISTINCT rt.tag) = {len(tags)}
            ){time_filter}
            ORDER BY rm.timestamp DESC
            """
        else:
            # When not filtering by tags, just get all metrics
            query = "SELECT * FROM request_metrics rm"
            conditions = []

            if start_time:
                conditions.append("rm.timestamp >= :start_time")
                params['start_time'] = start_time

            if end_time:
                conditions.append("rm.timestamp <= :end_time")
                params['end_time'] = end_time

            # Add WHERE clause only if we have conditions
            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY rm.timestamp DESC"

        if limit:
            query += f" LIMIT {limit}"

        # Load main dataframe
        df = pd.read_sql(text(query), engine, params=params)

        # ALWAYS append tags column (fetch tags for all requests in result)
        if not df.empty:
            request_ids = df['request_id'].tolist()
            if request_ids:
                # Build query to get tags for these requests
                if len(request_ids) == 1:
                    tags_query = f"SELECT request_id, tag FROM request_tags WHERE request_id = '{request_ids[0]}'"
                else:
                    tags_query = f"SELECT request_id, tag FROM request_tags WHERE request_id IN {tuple(request_ids)}"

                tags_df = pd.read_sql(text(tags_query), engine)

                # Group tags by request_id into lists
                if not tags_df.empty:
                    tags_by_request = tags_df.groupby('request_id')['tag'].apply(list).to_dict()
                    df['tags'] = df['request_id'].map(lambda rid: tags_by_request.get(rid, []))
                else:
                    df['tags'] = [[] for _ in range(len(df))]
            else:
                df['tags'] = [[] for _ in range(len(df))]

        return df

    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        raise

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

    # Load tag prefixes (keys) for autocomplete suggestions
    try:
        tags_query = "SELECT DISTINCT tag FROM request_tags ORDER BY tag"
        available_tags_df = pd.read_sql(text(tags_query), engine)
        all_tags = available_tags_df['tag'].tolist() if not available_tags_df.empty else []

        # Extract unique tag prefixes (e.g., "phase", "generation_id" from "phase:blurb")
        tag_prefixes = set()
        for tag in all_tags:
            if ':' in tag:
                prefix = tag.split(':', 1)[0]
                tag_prefixes.add(prefix)
        tag_prefixes = sorted(tag_prefixes)
    except:
        all_tags = []
        tag_prefixes = []

    # Tag filter with text input
    st.sidebar.markdown("### 🏷️ Filter by Tags")
    st.sidebar.caption("Enter tags as comma-separated (e.g., `phase:blurb, generation_id:abc123`)")

    # Show common tag prefixes as hints
    if tag_prefixes:
        st.sidebar.caption(f"Available tag types: {', '.join(tag_prefixes)}")

    tag_input = st.sidebar.text_input(
        "Tags (comma-separated):",
        value="",
        placeholder="phase:blurb, generation_id:abc123",
        help="Enter one or more tags separated by commas (AND logic - must have ALL tags)"
    )

    # Parse tag input
    selected_tags = []
    if tag_input.strip():
        selected_tags = [tag.strip() for tag in tag_input.split(',') if tag.strip()]

    # Group by selector
    group_by = st.sidebar.selectbox(
        "📊 Group by:",
        options=["None", "Model", "Provider", "Status", "Tags"],
        index=0,
        help="Group data for aggregated view"
    )

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

    # Load data with tag filtering
    with st.spinner("Loading data..."):
        df = load_data(engine, start_time, end_time, selected_tags if selected_tags else None, max_rows)

    if df.empty:
        st.warning("📭 No data available for the selected filters")
        return

    # Calculate statistics
    stats = calculate_stats(df)

    # Display filter info if tags are selected
    if selected_tags:
        st.info(f"🏷️ Filtered by tags: {', '.join(selected_tags)} ({len(df)} requests)")

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

    # Grouped/Pivot view if group_by is selected
    if group_by != "None" and len(df) > 0:
        st.subheader(f"📊 Aggregated View - Grouped by {group_by}")

        successful = df[df['status'] == 'success']

        if group_by == "Model":
            grouped = successful.groupby('actual_model').agg({
                'request_id': 'count',
                'total_cost_usd': ['sum', 'mean'],
                'total_duration_seconds': 'mean',
                'input_tokens': 'sum',
                'output_tokens': 'sum',
                'reasoning_tokens': 'sum'
            }).round(6)
            grouped.columns = ['Requests', 'Total Cost', 'Avg Cost', 'Avg Duration (s)', 'Input Tokens', 'Output Tokens', 'Reasoning Tokens']

        elif group_by == "Provider":
            grouped = successful.groupby('actual_provider').agg({
                'request_id': 'count',
                'total_cost_usd': ['sum', 'mean'],
                'total_duration_seconds': 'mean',
                'input_tokens': 'sum',
                'output_tokens': 'sum'
            }).round(6)
            grouped.columns = ['Requests', 'Total Cost', 'Avg Cost', 'Avg Duration (s)', 'Input Tokens', 'Output Tokens']

        elif group_by == "Status":
            grouped = df.groupby('status').agg({
                'request_id': 'count',
                'total_cost_usd': 'sum',
                'total_duration_seconds': 'mean'
            }).round(6)
            grouped.columns = ['Requests', 'Total Cost', 'Avg Duration (s)']

        elif group_by == "Tags":
            # Explode tags into individual rows for grouping
            df_exploded = df.explode('tags')
            df_exploded = df_exploded[df_exploded['status'] == 'success']
            if not df_exploded.empty and 'tags' in df_exploded.columns:
                grouped = df_exploded.groupby('tags').agg({
                    'request_id': 'count',
                    'total_cost_usd': ['sum', 'mean'],
                    'total_duration_seconds': 'mean'
                }).round(6)
                grouped.columns = ['Requests', 'Total Cost', 'Avg Cost', 'Avg Duration (s)']
            else:
                st.warning("No tags available for grouping")
                grouped = pd.DataFrame()

        if not grouped.empty:
            st.dataframe(grouped, use_container_width=True)

            # Add a chart for the grouped view
            if 'Total Cost' in grouped.columns:
                fig_grouped = px.bar(
                    grouped.reset_index(),
                    x=grouped.index.name,
                    y='Total Cost',
                    title=f"Total Cost by {group_by}"
                )
                fig_grouped.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_grouped, use_container_width=True)

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