# app.py — Polished UI + Light/Dark Toggle + Tooltips + Summary Parser (table + subtitles)
import os, json, pickle, io, re
from typing import Any, Dict, Optional, List, Tuple
import numpy as np

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from st_aggrid import JsCode

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from pm4py.objects.petri_net.obj import PetriNet, Marking
import graphviz
import hashlib
from difflib import SequenceMatcher

# Compatibility for components.html: pass 'key' only if it exists in the signature
import inspect
# Add after existing functions and before the sidebar

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import uuid
import time
from datetime import datetime, timedelta


import requests
import base64
import tempfile
import threading

# ------------------------------------------------------------
# Base configuration
# ------------------------------------------------------------
st.set_page_config(page_title="Interface • Process Mining", page_icon="🧭", layout="wide")


def is_backend_ready() -> bool:
    """True only when the backend has delivered the state.pkl and it exists in tmp disk."""
    path = st.session_state.get('state_file_path')
    return bool(path) and os.path.exists(path)

# Backend configuration
BACKEND_URL = "https://agentes-service-947456414948.us-central1.run.app"

# --- Client identity and session (without login) ---
def get_or_make_client_id() -> str:
    if 'client_id' not in st.session_state:
        st.session_state.client_id = f"cli_{uuid.uuid4().hex[:12]}"
    return st.session_state.client_id

def get_or_make_session_token() -> str:
    if 'session_token' not in st.session_state:
        seed = f"{get_or_make_client_id()}_{time.time()}"
        st.session_state.session_token = hashlib.md5(seed.encode()).hexdigest()[:10]
    return st.session_state.session_token

CLIENT_ID = get_or_make_client_id()
SESSION_ID = get_or_make_session_token()

# --- Async endpoints (adjust if your routes differ) ---
UPLOAD_ENDPOINT = f"{BACKEND_URL}/upload-build-state-pkl"  # Use the direct endpoint
UPLOAD_START    = f"{BACKEND_URL}/upload-build-state-pkl/start"
UPLOAD_PROGRESS = f"{BACKEND_URL}/upload-build-state-pkl/progress"
UPLOAD_RESULT   = f"{BACKEND_URL}/upload-build-state-pkl/result"

POLL_INTERVAL_SEC = 0.6
POLL_TIMEOUT_SEC  = 60 * 60  # 60 min max (large processes)


def start_heartbeat(client_id: str, interval_minutes: int = 10):
    """Sends periodic heartbeat - IMPROVED VERSION with error handling"""
    def heartbeat_loop():
        last_success = time.time()
        consecutive_failures = 0
        max_consecutive_failures = 3
        
        while st.session_state.get('heartbeat_active', True):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/qdrant/heartbeat",
                    params={
                        'client_id': client_id,
                        'ttl_seconds': 7200  # 2 hours
                    },
                    timeout=30  # Increase timeout for batch operations
                )
                
                if response.status_code == 200:
                    if consecutive_failures > 0:
                        print(f"✅ Heartbeat recovered for {client_id}")
                    consecutive_failures = 0
                    last_success = time.time()
                else:
                    consecutive_failures += 1
                    print(f"❌ Heartbeat error: {response.status_code}")
                    
            except requests.exceptions.Timeout:
                consecutive_failures += 1
                print(f"⏰ Heartbeat timeout")
            except Exception as e:
                consecutive_failures += 1
                print(f"❌ Error sending heartbeat: {e}")
            
            # Exponential backoff in case of failures
            if consecutive_failures >= max_consecutive_failures:
                wait_time = min(interval_minutes * 4, 60) * 60
                print(f"⚠️ Many failures. Waiting {wait_time/60} min...")
                time.sleep(wait_time)
                consecutive_failures = 0
            else:
                time.sleep(interval_minutes * 60)
    
    thread = threading.Thread(target=heartbeat_loop, daemon=True)
    thread.start()
    return thread

def _progress_overlay_html(percent: int, domain: str, p: Dict[str,str]) -> str:
    percent = max(0, min(100, int(percent or 0)))
    return f"""
    <div style="
      position: fixed; inset: 0; z-index: 99999;
      background: rgba(0,0,0,{0.75 if p['MODE']=='dark' else 0.55});
      display:flex;align-items:center;justify-content:center;">
      <div style="
        width:min(900px,92vw); border-radius:16px; padding:28px;
        background:{p['CARD']}; border:1px solid {p['BORDER']}; box-shadow:0 30px 80px rgba(0,0,0,.35)">
        <div style="display:flex; gap:16px; align-items:center; margin-bottom:14px;">
          <div style="font-size:40px">🗎</div>
          <div>
            <div style="font-weight:800;color:{p['TEXT']};font-size:20px;margin-bottom:2px">Fetching report…</div>
            <div style="color:{p['SOFT']};font-size:13px">Domain: <b style="color:{p['ACCENT']}">{domain}</b></div>
          </div>
        </div>
        <div style="height:14px;background:{p['SURFACE']};border:1px solid {p['BORDER']};
                    border-radius:999px;overflow:hidden;margin:10px 0 8px">
          <div style="height:100%;width:{percent}%;
                      background: linear-gradient(90deg, {p['ACCENT']} 0%, {p['ACCENT2']} 100%);
                      transition:width .25s ease"></div>
        </div>
        <div style="display:flex;align-items:center;justify-content:space-between;color:{p['SOFT']};font-size:12.5px">
          <span>Building <code>state.pkl</code> on backend…</span>
          <span><b style="color:{p['TEXT']}">{percent}</b> %</span>
        </div>
      </div>
    </div>
    """

def process_file_backend(uploaded_file, domain, upsert_qdrant=False, ttl_seconds=7200, use_preprocessed_state=False) -> bool:
    """
    Simplified version without complex overlays
    """
    try:
        files = {
            'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
        }
        data = {
            'domain_filter': domain,
            'upsert_to_qdrant': str(upsert_qdrant).lower(),
            'ttl_seconds': str(ttl_seconds),
            'use_preprocessed_state': str(use_preprocessed_state).lower()  # ✅ NEW FIELD
        }

        headers = {
            'x-client-id': CLIENT_ID,
            'x-session-id': SESSION_ID
        }

        # Only show a simple Streamlit status
        with st.status(f"📤 Processing {uploaded_file.name}...", expanded=True) as status:
            st.write("🔄 Uploading file to backend...")
            
            response = requests.post(
                UPLOAD_ENDPOINT,
                files=files,
                data=data,
                headers=headers,
                timeout=3600
            )

            if response.status_code == 200:
                st.write("✅ File received by backend")
        
                if use_preprocessed_state:
                    st.write("⚡ Loading preprocessed state...")
                else:
                    st.write("⚙️ Building state.pkl from scratch...")
                
                # Save to temporary file
                state_content = response.content
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
                    tmp_file.write(state_content)
                    new_state_path = tmp_file.name

                # Clear cache
                st.cache_data.clear()
                st.session_state['state_file_path'] = new_state_path
                st.write("📥 Downloading results...")
        
                if use_preprocessed_state:
                    status.update(label="Fast processing completed", state="complete", expanded=False)
                    st.success("✅ Preprocessed state loaded successfully")
                else:
                    status.update(label="Full processing completed", state="complete", expanded=False)
                    st.success("✅ File processed successfully from scratch")
        
                st.rerun()
                return True
            else:
                status.update(label="Processing error", state="error", expanded=False)
                st.error(f"❌ Backend error: {response.status_code} - {response.text}")
                return False

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return False


def html_safe(html_str, *, height=0, scrolling=False, key=None):
    params = inspect.signature(components.html).parameters
    kwargs = dict(height=height, scrolling=scrolling)
    if "key" in params and key is not None:
        kwargs["key"] = key
    return components.html(html_str, **kwargs)



def create_activity_frequency_chart(df: pd.DataFrame, top_n: int = 8, show_all: bool = False, 
                                  p: Dict[str, str] = None) -> go.Figure:
    """Corrected version - precise calculations of cases vs events"""
    
    # Calculate frequencies CORRECTLY
    case_freq = df.groupby('concept:name')['case:concept:name'].nunique().reset_index()
    case_freq.columns = ['Activity', 'Number of cases']
    event_freq = df['concept:name'].value_counts().reset_index()
    event_freq.columns = ['Activity', 'Number of events']
    
    chart_df = pd.merge(case_freq, event_freq, on='Activity', how='outer').fillna(0)
    chart_df = chart_df.sort_values('Number of cases', ascending=False)

    if not show_all and top_n > 0:
        chart_df = chart_df.head(top_n)
    
    chart_df = chart_df.sort_values('Number of cases', ascending=True)
    
    # Create figure
    fig = go.Figure()
    
    bar_color = p['ACCENT'] if p else '#2563eb'
    
    fig.add_trace(go.Bar(
        y=chart_df['Activity'],
        x=chart_df['Number of cases'],
        orientation='h',
        marker=dict(
            color=bar_color,
            line=dict(color=bar_color, width=1)
        ),
        hovertemplate=(
            "<b>%{y}</b><br>" +
            "Cases: %{x:,.0f}<br>" +
            "Events: %{customdata:,.0f}<br>" +
            "<extra></extra>"
        ),
        customdata=chart_df['Number of events'],
        name=''
    ))
    
    total_cases = chart_df['Number of cases'].sum()
    total_events = chart_df['Number of events'].sum()
    
    fig.update_layout(
        height=max(400, len(chart_df) * 35),
        plot_bgcolor=p['SURFACE'] if p else 'white',
        paper_bgcolor=p['SURFACE'] if p else 'white',
        font=dict(color=p['TEXT'] if p else '#0f172a', size=12),
        margin=dict(l=10, r=10, t=60, b=10),
        showlegend=False,
        xaxis=dict(
            title='Number of Cases',
            gridcolor=p['BORDER'] if p else '#e5e7eb',
            gridwidth=1,
            showline=True,
            linecolor=p['BORDER'] if p else '#e5e7eb',
            tickformat=',.0f'
        ),
        yaxis=dict(
            title='Activity',
            gridcolor=p['BORDER'] if p else '#e5e7eb',
            showline=True,
            linecolor=p['BORDER'] if p else '#e5e7eb',
            tickfont=dict(size=11)
        ),
        hoverlabel=dict(
            bgcolor='rgba(15,23,42,0.9)' if p and p['MODE'] == 'dark' else 'rgba(255,255,255,0.95)',
            font_size=12,
            font_color=p['TEXT'] if p else '#0f172a'
        ),
        title=dict(
            text=f"Activity Frequency (Total: {total_cases:,.0f} cases, {total_events:,.0f} events)",
            x=0.5,
            y=0.95,
            xanchor='center',
            font=dict(size=14, color=p['TEXT'] if p else '#0f172a')
        )
    )
    
    return fig

def create_resource_role_heatmap(
    df: pd.DataFrame,
    p: Dict[str, str] = None,
    top_resources: int = 12,
    top_roles: int = 10,
    normalize: str = "none",   # "none" | "row" | "col"
    metric: str = "events",    # "events" | "cases"
    include_unknown: bool = True,
    others_bucket: bool = True,
    swap_axes: bool = False,
    show_values: bool = True,
    text_min: Optional[float] = None,   # None -> auto (5 if counts, 2.0 if %)
    zrange: Optional[Tuple[float, float]] = None,
    colorscale: Optional[str] = None,
) -> go.Figure:
    # Minimal validations
    need_cols = {"org:resource", "org:role"}
    if df is None or df.empty or not need_cols.issubset(df.columns):
        fig = go.Figure()
        fig.update_layout(
            title="Missing columns 'org:resource' and/or 'org:role'",
            height=320,
            plot_bgcolor=(p['SURFACE'] if p else 'white'),
            paper_bgcolor=(p['SURFACE'] if p else 'white'),
            font=dict(color=(p['TEXT'] if p else '#0f172a'))
        )
        return fig

    # Base
    if metric == "cases":
        if "case:concept:name" not in df.columns:
            metric = "events"  # fallback
        else:
            base = df[["case:concept:name", "org:resource", "org:role"]].copy()
            if include_unknown:
                base["org:resource"] = base["org:resource"].fillna("No resource")
                base["org:role"] = base["org:role"].fillna("No role")
            else:
                base = base.dropna(subset=["org:resource", "org:role"])
            base = base.drop_duplicates()  # 1 row per (case,resource,role)
    if metric != "cases":
        base = df[["org:resource", "org:role"]].copy()
        if include_unknown:
            base["org:resource"] = base["org:resource"].fillna("No resource")
            base["org:role"] = base["org:role"].fillna("No role")
        else:
            base = base.dropna(subset=["org:resource", "org:role"])

    # Crosstab
    ct = pd.crosstab(base["org:resource"].astype(str), base["org:role"].astype(str))
    if ct.size == 0:
        fig = go.Figure()
        fig.update_layout(
            title="No data to build Resources × Roles matrix",
            height=320,
            plot_bgcolor=(p['SURFACE'] if p else 'white'),
            paper_bgcolor=(p['SURFACE'] if p else 'white'),
            font=dict(color=(p['TEXT'] if p else '#0f172a'))
        )
        return fig

    # Top-K + "Others" bucket
    row_tot = ct.sum(axis=1).sort_values(ascending=False)
    col_tot = ct.sum(axis=0).sort_values(ascending=False)
    keep_rows = row_tot.head(max(1, top_resources)).index
    keep_cols = col_tot.head(max(1, top_roles)).index

    ct = ct.loc[keep_rows, :]

    ct = ct.loc[:, keep_cols]

    # Swap axes if requested
    if swap_axes:
        ct = ct.T

    # Normalization
    title_suffix = " (counts)"
    if normalize == "row":
        ct = ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0) * 100.0
        title_suffix = " (% by row)"
    elif normalize == "col":
        ct = ct.div(ct.sum(axis=0).replace(0, np.nan), axis=1) * 100.0
        title_suffix = " (% by column)"

    # Text: automatic threshold if not defined
    if text_min is None:
        text_min = 2.0 if normalize in ("row", "col") else 5.0

    zvals = ct.values
    show_text = None
    if show_values:
        if normalize in ("row", "col"):
            show_text = [[f"{v:.1f}%" if (pd.notna(v) and v >= text_min) else "" for v in row] for row in zvals]
        else:
            show_text = [[f"{int(v):,}" if (pd.notna(v) and v >= text_min) else "" for v in row] for row in zvals]

    # Dynamic height
    height = max(360, 26 * ct.shape[0] + 140)

    # Colors
    cs = colorscale or ("Blues" if (not p or p.get("MODE") == "light") else "Viridis")

    # Heatmap
    hm = go.Heatmap(
        z=zvals,
        x=ct.columns.tolist(),
        y=ct.index.tolist(),
        colorscale=cs,
        zmin=(zrange[0] if zrange else None),
        zmax=(zrange[1] if zrange else None),
        colorbar=dict(title=""),
        text=show_text,
        texttemplate="%{text}" if show_values else None,
        hovertemplate=(
            "<b>Row:</b> %{y}<br>"
            "<b>Column:</b> %{x}<br>"
            f"<b>Value</b>: %{ 'z:.1f' if normalize in ('row','col') else 'z' }"
            f"{'%' if normalize in ('row','col') else ''}"
            "<extra></extra>"
        ),
    )

    # Title
    metric_name = "cases" if metric == "cases" else "events"
    fig = go.Figure(data=hm)
    fig.update_layout(
        title=f"👥 Resources × Roles — {metric_name}{title_suffix}",
        height=height,
        plot_bgcolor=(p['SURFACE'] if p else 'white'),
        paper_bgcolor=(p['SURFACE'] if p else 'white'),
        font=dict(color=(p['TEXT'] if p else '#0f172a'), size=12),
        margin=dict(l=10, r=10, t=48, b=10),
        xaxis=dict(
            title=("Role" if not swap_axes else "Resource"),
            showgrid=False,
            tickangle=45,
            tickfont=dict(size=10, color=(p['SOFT'] if p else '#64748b')),
            title_font=dict(size=12, color=(p['SOFT'] if p else '#64748b')),
        ),
        yaxis=dict(
            title=("Resource" if not swap_axes else "Role"),
            showgrid=False,
            automargin=True,
            tickfont=dict(size=10, color=(p['SOFT'] if p else '#64748b')),
            title_font=dict(size=12, color=(p['SOFT'] if p else '#64748b')),
        ),
        hoverlabel=dict(
            bgcolor=(p['CARD'] if p else 'white'),
            font_size=12,
            font_color=(p['TEXT'] if p else '#0f172a'),
            bordercolor=(p['BORDER'] if p else '#e5e7eb'),
        ),
        showlegend=False,
    )
    return fig


def create_monthly_trend_chart(df: pd.DataFrame, p: Dict[str, str] = None, selected_month: str = None, show_summary: bool = True) -> go.Figure:
    """Creates a DAILY temporal trend chart with expandable summary"""
    
    # Verify we have timestamp column
    if 'time:timestamp' not in df.columns:
        fig = go.Figure()
        fig.update_layout(
            title="No timestamp data available",
            height=400,
            plot_bgcolor=p['SURFACE'] if p else 'white',
            paper_bgcolor=p['SURFACE'] if p else 'white',
            font=dict(color=p['TEXT'] if p else '#0f172a')
        )
        return fig
    
    # Convert to datetime and clean
    df_temp = df.copy()
    df_temp['time:timestamp'] = pd.to_datetime(df_temp['time:timestamp'], errors='coerce')
    df_temp = df_temp.dropna(subset=['time:timestamp'])
    
    if df_temp.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No valid timestamp data",
            height=400,
            plot_bgcolor=p['SURFACE'] if p else 'white',
            paper_bgcolor=p['SURFACE'] if p else 'white',
            font=dict(color=p['TEXT'] if p else '#0f172a')
        )
        return fig
    
    # Filter by selected month
    df_filtered = pd.DataFrame()
    month_name = "Unknown month"
    
    try:
        if selected_month:
            selected_date = pd.to_datetime(selected_month + "-01")
            start_date = selected_date.replace(day=1)
            end_date = (selected_date + pd.DateOffset(months=1)) - pd.DateOffset(days=1)
            
            mask = (df_temp['time:timestamp'] >= start_date) & (df_temp['time:timestamp'] <= end_date)
            df_filtered = df_temp[mask]
            month_name = selected_date.strftime("%B %Y")
        else:
            most_recent_month = df_temp['time:timestamp'].max().replace(day=1)
            start_date = most_recent_month
            end_date = (most_recent_month + pd.DateOffset(months=1)) - pd.DateOffset(days=1)
            mask = (df_temp['time:timestamp'] >= start_date) & (df_temp['time:timestamp'] <= end_date)
            df_filtered = df_temp[mask]
            selected_month = most_recent_month.strftime("%Y-%m")
            month_name = most_recent_month.strftime("%B %Y")
            
    except Exception as e:
        st.error(f"Error filtering by month: {str(e)}")
        df_filtered = df_temp
        month_name = "All months"
    
    # Group by DAY
    if not df_filtered.empty:
        daily_events = df_filtered.set_index('time:timestamp').resample('D').size()
    else:
        daily_events = pd.Series(dtype=int)
    
    if daily_events.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"No data for {month_name}",
            height=400,
            plot_bgcolor=p['SURFACE'] if p else 'white',
            paper_bgcolor=p['SURFACE'] if p else 'white',
            font=dict(color=p['TEXT'] if p else '#0f172a')
        )
        return fig
    
    # Create labels
    date_labels = []
    full_dates = []
    day_names = []
    
    for date in daily_events.index:
        label = str(date.day)
        date_labels.append(label)
        full_dates.append(date.strftime("%Y-%m-%d"))
        day_names.append(date.strftime("%a"))
    
    # Calculate statistics
    total_events_month = daily_events.sum()
    
    if len(daily_events) > 0:
        max_day = daily_events.max()
        avg_daily = daily_events.mean()
        max_day_idx = daily_events.argmax()
        max_day_date = daily_events.index[max_day_idx]
        working_days = len(daily_events[daily_events > 0])
    else:
        max_day = 0
        avg_daily = 0
        max_day_date = None
        working_days = 0
    
    # Adjust width
    num_days = len(daily_events)
    base_width = 800
    dynamic_width = max(base_width, num_days * 60)
    
    # Create line chart
    fig = go.Figure()
    
    if not daily_events.empty:
        # Main line
        fig.add_trace(go.Scatter(
            x=date_labels,
            y=daily_events.values,
            mode='lines+markers',
            line=dict(
                color=p['ACCENT'] if p else '#2563eb',
                width=4,
                shape='spline',
                smoothing=1.3
            ),
            marker=dict(
                size=8,
                color=p['ACCENT'] if p else '#2563eb',
                line=dict(width=2, color='white'),
                symbol='circle'
            ),
            fill='tozeroy',
            fillcolor=f"rgba({int(p['ACCENT'][1:3], 16) if p else 37}, {int(p['ACCENT'][3:5], 16) if p else 99}, {int(p['ACCENT'][5:7], 16) if p else 235}, 0.1)" if p else 'rgba(37, 99, 235, 0.1)',
            hovertemplate=(
                "<b>%{customdata[1]} %{x} (%{customdata[0]})</b><br>" +
                "Events: <b>%{y:,.0f}</b><br>" +
                "<extra></extra>"
            ),
            customdata=list(zip(day_names, full_dates)),
            name='Daily events'
        ))
    
    # Layout configuration
    grid_color = p['BORDER'] if p else '#e5e7eb'
    
    fig.update_layout(
        height=500,
        width=dynamic_width,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=p['TEXT'] if p else '#0f172a', size=12, family="Arial, sans-serif"),
        margin=dict(l=60, r=40, t=80, b=60),
        showlegend=False,
        xaxis=dict(
            title='Day of Month',
            gridcolor=grid_color,
            gridwidth=1,
            showline=True,
            linecolor=grid_color,
            linewidth=2,
            tickmode='array',
            tickvals=date_labels,
            ticktext=date_labels,
            tickfont=dict(size=11, color=p['SOFT'] if p else '#64748b'),
            title_font=dict(size=12, color=p['SOFT'] if p else '#64748b')
        ),
        yaxis=dict(
            title='Number of Events',
            gridcolor=grid_color,
            gridwidth=1,
            showline=True,
            linecolor=grid_color,
            linewidth=2,
            tickformat=',.0f',
            tickfont=dict(size=11, color=p['SOFT'] if p else '#64748b'),
            title_font=dict(size=12, color=p['SOFT'] if p else '#64748b')
        ),
        hoverlabel=dict(
            bgcolor=p['CARD'] if p else 'white',
            font_size=12,
            font_color=p['TEXT'] if p else '#0f172a',
            bordercolor=p['BORDER'] if p else '#e5e7eb'
        ),
        title=dict(
            text=f"📈 {month_name}",
            x=0.03,
            y=0.95,
            xanchor='left',
            yanchor='top',
            font=dict(size=18, color=p['TEXT'] if p else '#0f172a')
        )
    )
    
    # IMPROVEMENT: Expandable summary with better design
    annotations = []
    
    if show_summary:
        # Improved summary - more visible and attractive
        summary_text = (
            f"📊 <b>Monthly Summary</b><br>"
            f"• Total: <b>{total_events_month:,.0f}</b> events<br>"
            f"• Average/day: <b>{avg_daily:,.1f}</b><br>"
            f"• Peak day: <b>{max_day:,.0f}</b> events<br>"
            f"• Active days: <b>{working_days}</b>"
        )
        
        annotations.append(dict(
            x=0.97, y=0.95,
            xref="paper", yref="paper",
            text=summary_text,
            showarrow=False,
            font=dict(size=12, color=p['TEXT'] if p else '#0f172a', family="Arial, sans-serif"),
            align="right",
            bgcolor=p['ACCENT'] if p else '#2563eb',
            bordercolor=p['ACCENT2'] if p else '#1d4ed8',
            borderwidth=2,
            borderpad=12
        ))
    
    fig.update_layout(annotations=annotations)
    
    return fig

def create_semester_trend_chart(df: pd.DataFrame, p: Dict[str, str] = None, 
                              semester: str = "January-June", selected_year: int = None, 
                              show_summary: bool = True) -> go.Figure:
    """Creates a SEMESTER DAILY trend chart with expandable summary"""
    
    month_names_en = {
        1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
        7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"
    }
    
    if 'time:timestamp' not in df.columns:
        fig = go.Figure()
        fig.update_layout(
            title="No timestamp data available",
            height=400,
            plot_bgcolor=p['SURFACE'] if p else 'white',
            paper_bgcolor=p['SURFACE'] if p else 'white',
            font=dict(color=p['TEXT'] if p else '#0f172a')
        )
        return fig
    
    # Convert to datetime and clean
    df_temp = df.copy()
    df_temp['time:timestamp'] = pd.to_datetime(df_temp['time:timestamp'], errors='coerce')
    df_temp = df_temp.dropna(subset=['time:timestamp'])
    
    if df_temp.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No valid timestamp data",
            height=400,
            plot_bgcolor=p['SURFACE'] if p else 'white',
            paper_bgcolor=p['SURFACE'] if p else 'white',
            font=dict(color=p['TEXT'] if p else '#0f172a')
        )
        return fig
    
    # Determine semester months
    if semester == "January-June":
        target_months = [1, 2, 3, 4, 5, 6]
        semester_name = "January-June"
    else:
        target_months = [7, 8, 9, 10, 11, 12]
        semester_name = "July-December"
    
    # Use selected year or determine automatically
    available_years = get_available_years(df_temp)
    
    if selected_year is None:
        if available_years:
            selected_year = available_years[0]
        else:
            selected_year = datetime.now().year
    
    # Filter data by specific semester and year
    try:
        mask = (df_temp['time:timestamp'].dt.year == selected_year) & (df_temp['time:timestamp'].dt.month.isin(target_months))
        df_semester = df_temp[mask]
        
        if not df_semester.empty:
            daily_events = df_semester.set_index('time:timestamp').resample('D').size()
            
            # Create detailed labels
            month_day_labels = []
            full_dates = []
            day_names = []
            
            for date in daily_events.index:
                month_day = f"{date.day} {month_names_en[date.month][:3]}"
                month_day_labels.append(month_day)
                full_dates.append(date.strftime("%d %B %Y"))
                day_names.append(date.strftime("%A"))
            
            total_events_semester = daily_events.sum()
            num_days = len(daily_events)
            
            # Calculate statistics
            if len(daily_events) > 0:
                max_day = daily_events.max()
                avg_daily = daily_events.mean()
                max_day_idx = daily_events.argmax()
                max_day_date = daily_events.index[max_day_idx]
                working_days = len(daily_events[daily_events > 0])
                weekly_avg = daily_events.resample('W').mean().mean()
            else:
                max_day = 0
                avg_daily = 0
                max_day_date = None
                working_days = 0
                weekly_avg = 0
            
        else:
            daily_events = pd.Series(dtype=int)
            month_day_labels = []
            total_events_semester = 0
            num_days = 0
            max_day = 0
            avg_daily = 0
            max_day_date = None
            working_days = 0
            weekly_avg = 0
        
    except Exception as e:
        st.error(f"Error processing semester data: {str(e)}")
        daily_events = pd.Series(dtype=int)
        month_day_labels = []
        total_events_semester = 0
        num_days = 0
        max_day = 0
        avg_daily = 0
        max_day_date = None
        working_days = 0
        weekly_avg = 0
    
    # Create chart
    fig = go.Figure()
    
    if not daily_events.empty:
        # Main line
        fig.add_trace(go.Scatter(
            x=month_day_labels,
            y=daily_events.values,
            mode='lines+markers',
            line=dict(
                color=p['ACCENT'] if p else '#2563eb',
                width=3,
                shape='spline',
                smoothing=1.2
            ),
            marker=dict(
                size=4,
                color=p['ACCENT'] if p else '#2563eb',
                line=dict(width=1, color='white'),
                symbol='circle'
            ),
            fill='tozeroy',
            fillcolor=f"rgba({int(p['ACCENT'][1:3], 16) if p else 37}, {int(p['ACCENT'][3:5], 16) if p else 99}, {int(p['ACCENT'][5:7], 16) if p else 235}, 0.08)" if p else 'rgba(37, 99, 235, 0.08)',
            hovertemplate=(
                "<b>%{customdata[1]}</b><br>" +
                "Day: %{customdata[0]}<br>" +
                "Events: <b>%{y:,.0f}</b><br>" +
                "<extra></extra>"
            ),
            customdata=list(zip(day_names, full_dates)),
            name='Daily events'
        ))
        
        # Moving average line
        if len(daily_events) > 7:
            moving_avg = daily_events.rolling(window=7, center=True).mean()
            fig.add_trace(go.Scatter(
                x=month_day_labels,
                y=moving_avg.values,
                mode='lines',
                line=dict(
                    color=p['ACCENT2'] if p else '#1d4ed8',
                    width=2,
                    dash='dash'
                ),
                name='Moving average (7 days)',
                hovertemplate=(
                    "<b>Moving average (7 days)</b><br>" +
                    "Events: <b>%{y:,.1f}</b><br>" +
                    "<extra></extra>"
                )
            ))
    
    # Layout configuration
    grid_color = p['BORDER'] if p else '#e5e7eb'
    
    num_days = len(daily_events)
    base_width = 1000
    dynamic_width = max(base_width, num_days * 25)
    
    fig.update_layout(
        height=550,
        width=dynamic_width,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=p['TEXT'] if p else '#0f172a', size=11, family="Arial, sans-serif"),
        margin=dict(l=60, r=40, t=80, b=80),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255,255,255,0.8)' if p and p['MODE'] == 'light' else 'rgba(15,23,42,0.8)',
            bordercolor=p['BORDER'] if p else '#e5e7eb'
        ),
        xaxis=dict(
            title='Days of Semester',
            gridcolor=grid_color,
            gridwidth=1,
            showline=True,
            linecolor=grid_color,
            linewidth=1,
            tickangle=45,
            tickfont=dict(size=9, color=p['SOFT'] if p else '#64748b'),
            title_font=dict(size=11, color=p['SOFT'] if p else '#64748b')
        ),
        yaxis=dict(
            title='Number of Daily Events',
            gridcolor=grid_color,
            gridwidth=1,
            showline=True,
            linecolor=grid_color,
            linewidth=1,
            tickformat=',.0f',
            tickfont=dict(size=10, color=p['SOFT'] if p else '#64748b'),
            title_font=dict(size=11, color=p['SOFT'] if p else '#64748b')
        ),
        hoverlabel=dict(
            bgcolor=p['CARD'] if p else 'white',
            font_size=11,
            font_color=p['TEXT'] if p else '#0f172a',
            bordercolor=p['BORDER'] if p else '#e5e7eb'
        ),
        title=dict(
            text=f"📊 {semester_name} {selected_year}",
            x=0.03,
            y=0.97,
            xanchor='left',
            yanchor='top',
            font=dict(size=18, color=p['TEXT'] if p else '#0f172a')
        )
    )
    
    # IMPROVEMENT: Expandable summary with better design
    annotations = []
    
    if show_summary:
        # Improved summary - more visible and attractive
        summary_text = (
            f"📈 <b>Semester Summary</b><br>"
            f"• Total: <b>{total_events_semester:,.0f}</b> events<br>"
            f"• Average/day: <b>{avg_daily:,.1f}</b><br>"
            f"• Peak day: <b>{max_day:,.0f}</b> events<br>"
            f"• Active days: <b>{working_days}/{num_days}</b>"
        )
        
        annotations.append(dict(
            x=0.97, y=0.95,
            xref="paper", yref="paper",
            text=summary_text,
            showarrow=False,
            font=dict(size=12, color=p['TEXT'] if p else '#0f172a', family="Arial, sans-serif"),
            align="right",
            bgcolor=p['ACCENT'] if p else '#2563eb',
            bordercolor=p['ACCENT2'] if p else '#1d4ed8',
            borderwidth=2,
            borderpad=12
        ))
            
    fig.update_layout(annotations=annotations)
    
    return fig

    
def get_available_months(df: pd.DataFrame) -> List[str]:
    """Gets the list of available months in YYYY-MM format - IMPROVED"""
    if df is None or 'time:timestamp' not in df.columns:
        return []
    
    df_temp = df.copy()
    df_temp['time:timestamp'] = pd.to_datetime(df_temp['time:timestamp'], errors='coerce')
    df_temp = df_temp.dropna(subset=['time:timestamp'])
    
    if df_temp.empty:
        return []
    
    # Extract year-month and get unique values
    available_months = df_temp['time:timestamp'].dt.to_period('M').unique()
    available_months = sorted(available_months, reverse=True)  # Most recent first
    
    # Convert to string format "YYYY-MM"
    month_options = [f"{period.year}-{period.month:02d}" for period in available_months]
    
    return month_options

def get_available_years(df: pd.DataFrame) -> List[int]:
    """Gets the list of available years in the data - CORRECTED VERSION"""
    if df is None or 'time:timestamp' not in df.columns:
        return []
    
    df_temp = df.copy()
    df_temp['time:timestamp'] = pd.to_datetime(df_temp['time:timestamp'], errors='coerce')
    df_temp = df_temp.dropna(subset=['time:timestamp'])
    
    if df_temp.empty:
        return []
    
    # Extract year and get unique values
    available_years = df_temp['time:timestamp'].dt.year.unique()
    
    # Convert to Python list and sort
    if hasattr(available_years, 'tolist'):
        available_years = available_years.tolist()
    else:
        # If already a list or doesn't have tolist method
        available_years = list(available_years)
    
    available_years = sorted(available_years, reverse=True)  # Most recent first
    
    return available_years

def get_semester_data_for_year(df: pd.DataFrame, semester: str, year: int) -> pd.Series:
    """Gets semester data for a specific year"""
    month_names_en = {
        1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
        7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"
    }
    
    df_temp = df.copy()
    df_temp['time:timestamp'] = pd.to_datetime(df_temp['time:timestamp'], errors='coerce')
    df_temp = df_temp.dropna(subset=['time:timestamp'])
    
    if semester == "January-June":
        target_months = [1, 2, 3, 4, 5, 6]
    else:
        target_months = [7, 8, 9, 10, 11, 12]
    
    # Filter by year and specific months
    mask = (df_temp['time:timestamp'].dt.year == year) & (df_temp['time:timestamp'].dt.month.isin(target_months))
    df_semester = df_temp[mask]
    
    if not df_semester.empty:
        daily_events = df_semester.set_index('time:timestamp').resample('D').size()
        
        # Create detailed labels
        month_day_labels = []
        full_dates = []
        day_names = []
        
        for date in daily_events.index:
            month_day = f"{date.day} {month_names_en[date.month][:3]}"
            month_day_labels.append(month_day)
            full_dates.append(date.strftime("%d %B %Y"))
            day_names.append(date.strftime("%A"))
        
        return daily_events, month_day_labels, full_dates, day_names
    else:
        return pd.Series(dtype=int), [], [], []

def get_month_display_names(month_list: List[str]) -> List[str]:
    """Converts YYYY-MM list to readable names"""
    display_names = []
    for month_str in month_list:
        try:
            date_obj = pd.to_datetime(month_str + "-01")
            # Format in English
            month_names_en = {
                1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
                7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"
            }
            month_name = month_names_en[date_obj.month]
            display_name = f"{month_name} {date_obj.year}"
            display_names.append(display_name)
        except:
            display_names.append(month_str)
    return display_names

    
def render_interactive_activity_chart(fig: go.Figure, p: Dict[str, str], 
                                    height: int = 500, instance_token: str = ""):
    """
    Renders activity chart with interactive controls
    """
    import hashlib
    import json
    
    # Convert figure to JSON
    plot_json = fig.to_json()
    instance_id = hashlib.md5((plot_json + instance_token).encode()).hexdigest()[:8]
    
    wrap_id = f"activity-chart-{instance_id}"
    chart_id = f"chart-{instance_id}"
    controls_id = f"controls-{instance_id}"
    
    html = f"""
    <div id="{wrap_id}" style="position:relative; height:{height}px; background:{p['SURFACE']};
         border:1px solid {p['BORDER']}; border-radius:14px; overflow:hidden;">
      <style>
        #{wrap_id} {{
          box-shadow: 0 10px 30px rgba(0,0,0,.25);
          border-color: {p['ACCENT']};
        }}
        
        #{controls_id} {{
          position: absolute;
          top: 12px;
          right: 12px;
          z-index: 10;
          background: {'rgba(15,23,42,0.94)' if p['MODE']=='dark' else 'rgba(255,255,255,0.96)'};
          border: 1px solid {p['BORDER']};
          border-radius: 12px;
          padding: 8px 12px;
          display: flex;
          gap: 8px;
          align-items: center;
          box-shadow: 0 10px 30px rgba(0,0,0,.45);
          user-select: none;
        }}
        
        #{controls_id} label {{
          color: {p['TEXT']};
          font-size: 12px;
          font-weight: 600;
          white-space: nowrap;
        }}
        
        #{controls_id} select, #{controls_id} input {{
          background: {p['CARD']};
          color: {p['TEXT']};
          border: 1px solid {p['BORDER']};
          border-radius: 6px;
          padding: 4px 8px;
          font-size: 12px;
        }}
        
        #{controls_id} select:focus, #{controls_id} input:focus {{
          outline: none;
          border-color: {p['ACCENT']};
        }}
        
        .chart-container {{
          width: 100%;
          height: 100%;
          padding: 10px;
        }}
        
        /* Improvements for Plotly tooltip */
        .js-plotly-plot .plotly .hoverlayer .hovertext {{
          background: {'rgba(15,23,42,0.95)' if p['MODE']=='dark' else 'rgba(255,255,255,0.95)'} !important;
          border: 1px solid {p['BORDER']} !important;
          border-radius: 8px !important;
          box-shadow: 0 8px 25px rgba(0,0,0,.3) !important;
        }}
        
        .js-plotly-plot .plotly .hoverlayer .hovertext path {{
          fill: {p['ACCENT']} !important;
        }}
      </style>
      
      <div id="{controls_id}">
        <label for="topN-{instance_id}">Show:</label>
        <select id="topN-{instance_id}">
          <option value="5">Top 5</option>
          <option value="8" selected>Top 8</option>
          <option value="15">Top 15</option>
          <option value="20">Top 20</option>
          <option value="0">All</option>
        </select>
        
        <label for="sort-{instance_id}" style="margin-left: 8px;">Order:</label>
        <select id="sort-{instance_id}">
          <option value="cases" selected>By Cases</option>
          <option value="events">By Events</option>
          <option value="name">By Name</option>
        </select>
      </div>
      
      <div class="chart-container" id="{chart_id}"></div>
    </div>
    
    <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
    <script>
      (function(){{
        const plotData = {plot_json};
        const instanceId = '{instance_id}';
        const chartDiv = document.getElementById('{chart_id}');
        const topNSelect = document.getElementById('topN-{instance_id}');
        const sortSelect = document.getElementById('sort-{instance_id}');
        
        let currentChart = null;
        
        function renderChart(topN = 8, sortBy = 'cases') {{
          // Filtering and sorting logic would be implemented here
          // For now we use data directly
          if (currentChart) {{
            Plotly.purge(chartDiv);
          }}
          
          // Adjust dynamic height
          const dataLength = plotData.data[0].y.length;
          const dynamicHeight = Math.max(400, dataLength * 35);
          chartDiv.style.height = dynamicHeight + 'px';
          
          currentChart = Plotly.newPlot(chartDiv, plotData.data, plotData.layout, {{
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
            modeBarButtonsToAdd: [],
            responsive: true
          }});
          
          // Adjust parent container
          const wrap = document.getElementById('{wrap_id}');
          wrap.style.height = (dynamicHeight + 20) + 'px';
        }}
        
        // Event listeners for controls
        topNSelect.addEventListener('change', function(e) {{
          const topN = parseInt(e.target.value);
          const sortBy = sortSelect.value;
          renderChart(topN, sortBy);
        }});
        
        sortSelect.addEventListener('change', function(e) {{
          const topN = parseInt(topNSelect.value);
          const sortBy = e.target.value;
          renderChart(topN, sortBy);
        }});
        
        // Initial render
        renderChart(8, 'cases');
        
        // Handle resizing
        window.addEventListener('resize', function() {{
          if (currentChart) {{
            Plotly.Plots.resize(chartDiv);
          }}
        }});
        
        // Cleanup function
        window.activityChartCleanup_{instance_id} = function() {{
          if (currentChart) {{
            Plotly.purge(chartDiv);
          }}
        }};
      }})();
    </script>
    """
    
    try:
        key = f"activity-chart-{instance_id}"
        components.html(html, height=height + 20, scrolling=False, key=key)
    except TypeError:
        components.html(html, height=height + 20, scrolling=False)


# ------------------------------------------------------------
# Palettes (light/dark)
# ------------------------------------------------------------
def get_palette(mode: str) -> Dict[str, str]:
    mode = (mode or "dark").lower()
    if mode == "light":
        return dict(
            MODE="light",
            BG="#f7f9fc",
            SURFACE="#ffffff",
            CARD="#ffffff",
            BORDER="#e5e7eb",
            TEXT="#0f172a",
            SOFT="#475569",
            ACCENT="#2563eb",
            ACCENT2="#1d4ed8",
            # Petri
            PLACE_FILL="#fffbd4",
            TRANS_FILL="#eef2ff",
            START_FILL="#7de9a5",
            END_FILL="#ff8a8a",
            EDGE_COLOR="#3b82f6",
            START_TEXT="#065f46",
            END_TEXT="#7f1d1d",
            PLACE_TEXT="#0f172a",
        )
    # dark (default)
    return dict(
        MODE="dark",
        BG="#0b1220",
        SURFACE="#0f172a",
        CARD="#111827",
        BORDER="#293241",
        TEXT="#e7eef7",
        SOFT="#b6c2d4",
        ACCENT="#7cc4ff",
        ACCENT2="#3ea0ff",
        # Petri
        PLACE_FILL="#0f172a",
        TRANS_FILL="#192233",
        START_FILL="#22c55e",
        END_FILL="#ef4444",
        EDGE_COLOR="#89b4ff",
        START_TEXT="#06120a",
        END_TEXT="#140505",
        PLACE_TEXT="#e7eef7",
    )






# ------------------------------------------------------------
# Injectable CSS according to palette (careful: braces {{ }})
# ------------------------------------------------------------
def inject_css(p: Dict[str, str]) -> None:
    st.markdown(f"""
    <style>
    :root {{
      --bg: {p['BG']}; --surface: {p['SURFACE']}; --card: {p['CARD']};
      --border: {p['BORDER']}; --text: {p['TEXT']}; --soft: {p['SOFT']};
      --accent: {p['ACCENT']}; --accent2: {p['ACCENT2']};
    }}
    html, body, .stApp {{ background: var(--bg); color: var(--text); }}

    [data-testid="stHeader"] {{ background: var(--bg) !important; border-bottom: 1px solid {p['BORDER']}; }}
    [data-testid="stAppViewContainer"], .main, .block-container {{ background: var(--bg) !important; }}

    section[data-testid="stSidebar"] > div {{ background: var(--bg); }}
    div[data-testid="stSidebar"]::before {{
      content: "Interface"; position: sticky; top: 0; left: 0;
      background: var(--accent2); color: #001222; font-weight: 800;
      padding: .35rem .7rem; border-radius: .35rem; margin: .5rem;
      box-shadow: 0 2px 0 rgba(0,0,0,.35);
    }}

    h1, h2, h3, h4, h5 {{ color: var(--text); letter-spacing:.2px; }}

    .title-band {{
      background: linear-gradient(90deg, #0c1528, #0f1c33);
      padding:.9rem 1.2rem; border-radius:.7rem; font-weight:700; font-size:1.25rem;
      border:1px solid var(--border);
      {'' if p['MODE']=='dark' else 'background: linear-gradient(90deg, #f8fafc, #ffffff);'}
    }}

    .block {{
      background: var(--card); border:1px solid var(--border);
      border-radius:.9rem; padding:1rem; color:var(--text);
      box-shadow: 0 10px 26px rgba(0,0,0,.25);
      /* NEW: consistent vertical spacing between boxes */
      margin-bottom: 1rem;
    }}
    .dashed {{ border:1.5px dashed var(--border); border-radius:.8rem; padding:1rem; color:var(--soft); }}

    .stTabs [data-baseweb="tab"] {{ font-weight:700; color:var(--soft); }}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{ color:var(--text); border-bottom:2px solid var(--accent); }}
    .small {{ color:var(--soft); font-size:.9rem; }}

    a, a:visited {{ color: var(--accent); }}

    /* AgGrid */
    .ag-theme-streamlit, .ag-root-wrapper {{
      background: var(--card) !important; color: var(--text) !important;
      border:1px solid var(--border) !important; border-radius:.7rem;
    }}
    .ag-theme-streamlit .ag-header, .ag-theme-streamlit .ag-header-row {{
      background: {('#0e1626' if p['MODE']=='dark' else '#f1f5f9')} !important;
      color: var(--text) !important; border-color: var(--border) !important;
    }}
    .ag-theme-streamlit .ag-row {{ background: var(--card) !important; }}
    .ag-theme-streamlit .ag-row-odd {{ background: {('#0f1b2e' if p['MODE']=='dark' else '#f8fafc')} !important; }}
    .ag-theme-streamlit .ag-cell {{ color: var(--text) !important; border-color: var(--border) !important; }}

    ::-webkit-scrollbar {{ width:10px; height:10px; }}
    ::-webkit-scrollbar-track {{ background: var(--bg); }}
    ::-webkit-scrollbar-thumb {{ background: {('#1e2b42' if p['MODE']=='dark' else '#d1d5db')}; border-radius:10px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: {('#26344f' if p['MODE']=='dark' else '#bfc5ce')}; }}

    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] > div > p,
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
      color: var(--text) !important; opacity: .98 !important; font-weight: 600 !important;
    }}
    [data-testid="stFileUploaderDropzone"] p {{ color: var(--soft) !important; opacity: .95 !important; }}

    /* Metric cards */
    .metric-card {{ background: var(--card); border:1px solid var(--border);
                    border-radius: 14px; padding: 12px 14px; box-shadow: 0 8px 24px rgba(0,0,0,.18); }}
    .metric-name {{ font-weight:700; font-size:.95rem; color: var(--soft); }}
    .metric-value {{ font-weight:800; font-size:1.35rem; margin:.25rem 0 .5rem 0; color: var(--text); }}
    .metric-bar {{ width:100%; height:8px; background: rgba(127,127,127,.18);
                   border-radius: 999px; border:1px solid var(--border); }}
    .metric-bar > span {{ display:block; height:100%; background: var(--accent); border-radius: 999px; }}

    /* Header with help (to the right) */
    .metrics-head{{display:flex;align-items:center;gap:.5rem;margin:.25rem 0 .6rem}}
    .metrics-head .title{{font-weight:800;font-size:1.05rem}}
    .metrics-head .spacer{{flex:1}}
    .help-badge{{
      position:relative;display:inline-flex;align-items:center;justify-content:center;
      width:22px;height:22px;border-radius:50%;cursor:pointer;
      border:1px solid var(--border); background: var(--card); color: var(--accent);
      font-weight:900; line-height:22px; user-select:none;
    }}
    .help-badge:hover{{border-color:var(--accent)}}
    .help-badge .tip{{
      display:none; position:absolute; top:26px; right:0; z-index:50;
      width:min(560px, 78vw); background:var(--card); color:var(--text);
      border:1px solid var(--border); border-radius:12px; padding:12px 14px;
      box-shadow:0 12px 30px rgba(0,0,0,.35);
      text-align:left;
    }}
    .help-badge:hover .tip{{display:block}}
    .tip p{{margin:.2rem 0; color:var(--soft); font-size:.92rem}}
    .tip ul{{margin:.3rem 0 0 1rem}}
    .tip li{{margin:.25rem 0; color:var(--soft)}}

    /* S score table */
    .score-table table{{width:100%;border-collapse:separate;border-spacing:0;overflow:hidden;
                        border:1px solid var(--border);border-radius:12px;background:var(--card)}}
    .score-table thead th{{text-align:left;padding:.65rem .8rem;border-bottom:1px solid var(--border);color:var(--soft)}}
    .score-table tbody td{{padding:.70rem .8rem;border-bottom:1px solid {p['BORDER']};color:var(--text)}}
    .score-table tbody tr:last-child td{{border-bottom:none}}

    /* Subtitles generated from separators -> bold + slightly larger */
    .why-subtitle{{margin:.9rem 0 .45rem 0;font-weight:900;font-size:1.18rem;color:var(--text)}}

    /* Radio text visible in both modes */
    .stRadio [data-testid="stWidgetLabel"] p,
    .stRadio [data-testid="stWidgetLabel"] span,
    .stRadio label div p,
    .stRadio label div span {{
        color: {p['TEXT']} !important;
        font-weight: 600 !important;
        opacity: 0.95 !important;
    }}

    .stRadio [role="radiogroup"] label {{
        color: {p['TEXT']} !important;
        background: {p['CARD']} !important;
    }}

    .stRadio [role="radiogroup"] label:hover {{
        background: {p['ACCENT']}15 !important;
        border-color: {p['ACCENT']} !important;
    }}

    .stRadio [role="radiogroup"] label[data-baseweb="radio"] {{
        border: 1px solid {p['BORDER']} !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
        margin: 2px 0 !important;
    }}

    .stRadio [role="radiogroup"] label[data-baseweb="radio"]:hover {{
        border-color: {p['ACCENT']} !important;
        background: {p['ACCENT']}10 !important;
    }}

    /* Selected radio */
    .stRadio [role="radiogroup"] label[data-baseweb="radio"][class*="selected"] {{
        background: {p['ACCENT']}20 !important;
        border-color: {p['ACCENT']} !important;
        color: {p['TEXT']} !important;
    }}

    /* --- Radios without "box" (only for view selector) --- */
    .no-pill-radio [role="radiogroup"] > * {{
      background: transparent !important;
      border: 0 !important;
      box-shadow: none !important;
      padding: 2px 8px !important;
      margin: 0 6px 0 0 !important;   /* old margin (we override below with gap) */
      border-radius: 0 !important;
    }}
    .no-pill-radio [role="radiogroup"] [aria-checked="true"],
    .no-pill-radio [role="radiogroup"] [aria-checked="false"] {{
      background: transparent !important;
    }}

    /* 🔥 NEW: layout & real separation between radio options */
    .stRadio [role="radiogroup"] {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px 12px !important;   /* space between "pills"/options */
    }}
    /* Override margins so gap wins */
    .no-pill-radio [role="radiogroup"] > *,
    .stRadio [role="radiogroup"] > label,
    .stRadio [role="radiogroup"] > div {{
      margin: 0 !important;
    }}

    </style>
    """, unsafe_allow_html=True)


# ------------------------------------------------------------
# Safe pickle loading - CORRECTED VERSION
# ------------------------------------------------------------
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if name == 'Marking':
            return dict
        try:
            return super().find_class(module, name)
        except Exception:
            Dummy = type(name, (object,), {})
            Dummy.__init__ = lambda *a, **k: None
            return Dummy


# CORRECTED LOAD FUNCTIONS - with state_file_path as parameter to invalidate cache
@st.cache_data(show_spinner=False)
def load_best_model(state_file_path: str):
    if state_file_path and os.path.exists(state_file_path):
        with open(state_file_path, "rb") as f:
            state = CustomUnpickler(f).load()
        return state.get('agente1', {}).get('best_model', {})
    return {}

@st.cache_data(show_spinner=False)
def load_agent1(state_file_path: str):
    if state_file_path and os.path.exists(state_file_path):
        with open(state_file_path, "rb") as f:
            state = CustomUnpickler(f).load()
        return state.get("agente1", {})
    return {}

@st.cache_data(show_spinner=False)
def load_agent3(state_file_path: str):
    if state_file_path and os.path.exists(state_file_path):
        with open(state_file_path, "rb") as f:
            state = CustomUnpickler(f).load()
        return state.get("agente3", {}).get("result", {})
    return {}

@st.cache_data(show_spinner=False)
def load_agent4(state_file_path: str):
    if state_file_path and os.path.exists(state_file_path):
        with open(state_file_path, "rb") as f:
            state = CustomUnpickler(f).load()
        return state.get("agente4", {})
    return {}

@st.cache_data(show_spinner=False)
def load_agent8(state_file_path: str):
    if state_file_path and os.path.exists(state_file_path):
        with open(state_file_path, "rb") as f:
            state = CustomUnpickler(f).load()
        return state.get("agente8", [])
    return []

# WRAPPERS to maintain compatibility (optional, or update all calls)
def get_current_best_model():
    state_path = st.session_state.get('state_file_path', '')
    return load_best_model(state_path)

def get_current_agent1():
    state_path = st.session_state.get('state_file_path', '')
    return load_agent1(state_path)

def get_current_agent3():
    state_path = st.session_state.get('state_file_path', '')
    return load_agent3(state_path)

def get_current_agent4():
    state_path = st.session_state.get('state_file_path', '')
    return load_agent4(state_path)

def get_current_agent8():
    state_path = st.session_state.get('state_file_path', '')
    return load_agent8(state_path)

    


def get_log_summaries_strict() -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []

    # agente3
    try:
        a3 = get_current_agent3()  # Use corrected function
        s3 = a3.get("log_summary")
        if isinstance(s3, str) and s3.strip():
            out.append(("agente3", convert_separators_to_subtitles(s3.strip())))
    except Exception:
        pass

    return out

def get_agent8_problem_pairs() -> List[Dict[str, str]]:
    """
    Maps:
      - 'problem_description' (or 'problem_descripcion') → 'Problem'
      - 'recommendation' → 'Solution'
    and returns a list of {'Problem': ..., 'Solution': ...}
    """
    items = get_current_agent8()
    pairs: List[Dict[str, str]] = []
    for it in items:
            problem = (
                it.get("problem_description")
                or it.get("problem_descripcion") 
            )
            solution = it.get("recommendation")
            if problem and solution:
                pairs.append({
                    "Problem": str(problem).strip(),
                    "Solution": str(solution).strip()
                })
    return pairs


def detect_model_type(model_name: str) -> str:
    low = (model_name or "").lower()
    if "heuristic" in low: return "heuristic"
    if "inductive" in low: return "inductive"
    if "alpha" in low:     return "alpha"
    return "unknown"


# ------------------------------------------------------------
# Label translation
# ------------------------------------------------------------
FRIENDLY_ALIASES = {
    "source": "Start", "start": "Start", "start_event": "Start",
    "sink": "End", "end": "End", "end_event": "End",
    "tau": "Internal step (τ)", "hid": "Internal step (τ)",
}
def prettify_label(raw: str) -> str:
    if raw is None: return ""
    s = str(raw); low = s.lower()
    for k,v in FRIENDLY_ALIASES.items():
        if low == k or low.startswith(f"{k}_"): return v
    for pref in ("p_","t_","hid_","tau_","place_","trans_"):
        if low.startswith(pref): s = s[len(pref):]; low = s.lower(); break
    if low in ("hid","tau","silent","none",""): return "Internal step (τ)"
    s = " ".join(s.replace("_"," ").split()).strip()
    return s[:1].upper()+s[1:] if s else "Internal step (τ)"


# ------------------------------------------------------------
# Helpers: metrics
# ------------------------------------------------------------
_METRIC_ALIASES = {
    "fitness": "Fitness",
    "alignment_fitness": "Fitness (alignment)",
    "replay_fitness": "Fitness (replay)",
    "log_fitness": "Fitness (log)",
    "average_trace_fitness": "Fitness (average trace)",
    "perc_fit_traces": "Conforming traces (%)",
    "percentage_of_fitting_traces": "Conforming traces (%)",
    "precision": "Precision",
    "behavioral_precision": "Precision (behavioral)",
    "generalization": "Generalization",
    "simplicity": "Simplicity",
    "fscore": "F1-score",
    "f1": "F1-score",
    "soundness": "Soundness",
    "completeness": "Completeness",
    "coverage": "Coverage",
    "score": "Selection score",
}

def _flatten(d: Dict, prefix: str = "") -> Dict[str, float]:
    flat = {}
    if not isinstance(d, dict): return flat
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            flat.update(_flatten(v, key))
        elif isinstance(v, (list, tuple)) and len(v) == 2 and isinstance(v[1], (int, float)):
            flat[str(v[0])] = float(v[1])
        elif isinstance(v, (int, float)):
            flat[key] = float(v)
    return flat

def extract_conformance_metrics(best_model: Dict) -> Dict[str, float]:
    metrics_dict = None
    if isinstance(best_model.get("conf"), dict):
        metrics_dict = best_model["conf"]
    if metrics_dict is None:
        for c in ("metrics","conformance","conformance_metrics","quality_metrics","evaluation","eval","model_quality","scores"):
            if isinstance(best_model.get(c), dict):
                metrics_dict = best_model[c]; break
    if metrics_dict is None: return {}
    flat = _flatten(metrics_dict)
    cleaned = {}
    for k, v in flat.items():
        base = k.split(".")[-1].lower().strip()
        friendly = _METRIC_ALIASES.get(base, k.split(".")[-1].replace("_", " ").title())
        cleaned[friendly] = v
    if isinstance(best_model.get("score"), (int, float)):
        cleaned[_METRIC_ALIASES["score"]] = float(best_model["score"])
    return cleaned

def _value_to_percent(v: float) -> float:
    if v is None: return 0.0
    try: x = float(v)
    except Exception: return 0.0
    if 0.0 <= x <= 1.5: return max(0.0, min(1.0, x)) * 100.0
    if 1.5 < x <= 100.0: return max(0.0, min(100.0, x))
    return 100.0 if x > 100 else 0.0

def normalize_metrics_for_display(metrics: Dict[str, float]) -> Dict[str, float]:
    out = dict(metrics)
    # % -> 0-1
    for k in list(out.keys()):
        k_low = k.lower()
        if "traces" in k_low and "%" in k_low:
            try: out["Conforming traces (0-1)"] = float(out.pop(k)) / 100.0
            except Exception: out["Conforming traces (0-1)"] = out.pop(k)
    if "Fitness (log)" in out and "Fitness (log)" not in out:
        out["Fitness (log)"] = out.pop("Fitness (log)")
    if "Selection score" in out and "Selection score" not in out:
        out["Selection score"] = out.pop("Selection score")
    return out

def get_output_summary_from_state(best_model: Dict) -> str:
    a1 = get_current_agent1()
    summary = a1.get("output_summary")
    if not summary:
        summary = (best_model.get("output_summary") or best_model.get("summary") or
                   best_model.get("selection_summary") or best_model.get("justification") or
                   best_model.get("why") or "")
        if isinstance(summary, dict):
            for k in ("summary","output_summary","text","reason"):
                if isinstance(summary.get(k), str) and summary.get(k).strip():
                    summary = summary[k]; break
            else: summary = ""
    return str(summary).strip()


# ------------------------------------------------------------
# output_summary parser (table + subtitles + cleaning)
# ------------------------------------------------------------
_SCORE_ROW_RE = re.compile(
    r"""(?ix)
    (?P<name>[^\n:]+?)\s*:\s*
    S\s*[_\-\s]*initial\s*=\s*(?P<Sini>[\d\.]+)\s*,\s*
    Penalt\w*\s*=\s*(?P<PEN>[\d\.]+)\s*,\s*
    S\s*[_\-\s]*final\s*=\s*(?P<Sfin>[\d\.]+)
    """
)

def parse_score_rows(text: str) -> List[Tuple[str, float, float, float]]:
    rows = []
    for m in _SCORE_ROW_RE.finditer(text or ""):
        name = _clean_candidate_name(m.group("name"))
        try:
            rows.append((name, float(m.group("Sini")), float(m.group("PEN")), float(m.group("Sfin"))))
        except Exception:
            pass
    return rows


def build_scores_table_html(rows: List[Tuple[str,float,float,float]]) -> str:
    if not rows: return ""
    head = "<div class='score-table'><table><thead><tr><th>Model</th><th>S_initial</th><th>Penalty</th><th>S_final</th></tr></thead><tbody>"
    body = [
        f"<tr><td>{name}</td><td>{si:.3f}</td><td>{pe:.3f}</td><td>{sf:.3f}</td></tr>"
        for (name, si, pe, sf) in rows
    ]
    tail = "</tbody></table></div>"
    return head + "".join(body) + tail

def strip_json_output_block(txt: str) -> str:
    # remove JSON_OUTPUT in block or line
    txt = re.sub(r"(?is)\n?JSON[_\-\s]*OUTPUT\s*:.*?$", "", txt)
    txt = re.sub(r"(?is)```.*?JSON[_\-\s]*OUTPUT.*?```", "", txt)
    return txt.strip()

def remove_inline_score_lines(s: str) -> str:
    """
    Removes summary lines of type:
      'Heuristic Miner: S_initial=..., Penalty=..., S_final=...' (with or without '|')
    """
    out = []
    pat = re.compile(r"(S\s*[_\-\s]*initial|S\s*[_\-\s]*final|Penalty)\s*=", re.I)
    for line in s.splitlines():
        if pat.search(line) and ("|" in line or ":" in line):
            continue
        out.append(line)
    return "\n".join(out)

def _clean_candidate_name(name: str) -> str:
    # remove pipes, bullets and dashes left/right and normalize spaces
    name = re.sub(r'^[\s\|\•\·\-\–\—]+', '', str(name))
    name = re.sub(r'[\s\|\•\·\-\–\—]+$', '', name)
    name = re.sub(r'\s+', ' ', name)
    return name.strip()

def convert_separators_to_subtitles(md: str) -> str:
    """
    Converts separators to subtitles and removes lines with bars/dashes:
      - '______  Text'  -> <div class='why-subtitle'>Text</div>
      - Line of only dashes/underscores followed by text in next line
      - 'Title' followed in next line by '-----' -> subtitle
    """
    lines = md.splitlines()
    out = []
    i = 0
    sep_only = re.compile(r"^\s*([^\w\s])\1{5,}\s*$")            # -----, _____, ——…
    sep_with_text = re.compile(r"^\s*([^\w\s])\1{2,}\s*(.+?)\s*$")  # --- Title

    while i < len(lines):
        line = lines[i]

        m = sep_with_text.match(line)
        if m and m.group(2).strip():
            title = m.group(2).strip().rstrip(":")
            out.append(f"<div class='why-subtitle'>{title}</div>")
            i += 1
            continue

        if i + 1 < len(lines) and lines[i].strip() and sep_only.match(lines[i+1]):
            title = lines[i].strip().rstrip(":")
            out.append(f"<div class='why-subtitle'>{title}</div>")
            i += 2
            continue

        if sep_only.match(line):
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                title = lines[j].strip().rstrip(":")
                out.append(f"<div class='why-subtitle'>{title}</div>")
                i = j + 1
                continue
            i += 1
            continue

        out.append(line)
        i += 1

    return "\n".join(out)

# --- NEW: robust EN/ES anchoring for table ---
def inject_marker_at_table_anchor(text: str) -> Tuple[str, bool]:
    if not text:
        return text, False
    patterns = [
        r"(?i)\bbelow\b[^.\n]{0,200}\b(one[\-\s]*line|summary)?\b[^.\n]{0,200}\btable\b[^:\n]*:?",     # EN
        r"(?i)\ba\s+continuaci[oó]n\b[^.\n]{0,200}\btabla\b[^:\n]*:?",                                 # ES
        r"(?i)\b(se\s+mostrar[aá]|se\s+muestra|se\s+ver[aá])\b[^.\n]{0,200}\btabla\b[^:\n]*:?",        # ES variants
        r"(?i)\b(siguiente|sig\.)\s+tabla\b[^:\n]*:?",                                                 # ES 'siguiente tabla'
    ]
    lines = text.splitlines(True)
    for i, ln in enumerate(lines):
        if any(re.search(p, ln) for p in patterns):
            lines[i] = "[[SCORE_TABLE]]\n"
            return ("".join(lines), True)
    return ("".join(lines), False)

def format_summary_with_table(summary: str) -> Tuple[str, str]:
    """
    - Extracts S rows and builds HTML table
    - Removes JSON_OUTPUT
    - Inserts [[SCORE_TABLE]] at anchor line (EN/ES)
    - Removes inline 'summary line' with S_initial/S_final to avoid duplication
    - Converts separators to subtitles
    """
    if not summary:
        return "", ""

    rows = parse_score_rows(summary)
    table_html = build_scores_table_html(rows)

    s = strip_json_output_block(summary)

    # Place marker at anchor (EN/ES)
    s, placed = inject_marker_at_table_anchor(s)

    # Additional fallback in case exact EN text appears undetected
    if not placed:
        m = re.search(r"(?is)below\s+is\s+the\s+one[\-\u2013]?\s*line\s+table[^\n]*", s)
        if m:
            s = s[:m.start()] + "[[SCORE_TABLE]]" + s[m.end():]
            placed = True

    s = remove_inline_score_lines(s)
    s = convert_separators_to_subtitles(s)

    return s, table_html


# ------------------------------------------------------------
# Graphviz according to palette
# ------------------------------------------------------------
def build_petri_graph(net, im, fm, show_ids: bool, p: Dict[str,str]) -> graphviz.Digraph:
    dot = graphviz.Digraph("PetriNet", format="svg")
    dot.attr("graph", rankdir="LR", bgcolor=p["SURFACE"], pad="0.2",
             splines="spline", overlap="false", concentrate="true",
             nodesep="0.35", ranksep="0.55")
    dot.attr("node", fontname="Helvetica", fontsize="11", penwidth="1.2")
    dot.attr("edge", fontname="Helvetica", fontsize="10",
             color=p["EDGE_COLOR"], fontcolor=p["SOFT"], arrowsize="0.7")

    places = getattr(net, "_PetriNet__places", [])
    im_names = [getattr(x,"_Place__name","") for x in (im or {}).keys()] if isinstance(im,dict) else []
    fm_names = [getattr(x,"_Place__name","") for x in (fm or {}).keys()] if isinstance(fm,dict) else []

    for place in places:
        raw_id = getattr(place, "_Place__name","")
        if raw_id in im_names:
            friendly, fill, fcolor = "Start", p["START_FILL"], p["START_TEXT"]
        elif raw_id in fm_names:
            friendly, fill, fcolor = "End", p["END_FILL"], p["END_TEXT"]
        else:
            friendly, fill, fcolor = prettify_label(raw_id), p["PLACE_FILL"], p["PLACE_TEXT"]
        label = f"{friendly} ({raw_id})" if (show_ids and raw_id and friendly != raw_id) else friendly
        dot.node(raw_id, label=label, shape="circle", style="filled",
                 fillcolor=fill, color=p["BORDER"], fontcolor=fcolor)

    transitions = getattr(net, "_PetriNet__transitions", [])
    for trans in transitions:
        t_id = getattr(trans, "_Transition__name","")
        t_label = getattr(trans, "_Transition__label","")
        base = t_label if t_label not in (None,"","None") else ("τ" if (str(t_id).startswith("hid_") or t_id in ("", None)) else t_id)
        friendly = prettify_label(base)
        label = f"{friendly} ({t_id})" if (show_ids and t_id and friendly != t_id) else friendly
        dot.node(t_id, label=label, shape="box", style="rounded,filled",
                 height="0.28", width="0.9", margin="0.06,0.04",
                 fillcolor=p["TRANS_FILL"], color=p["ACCENT2"], fontcolor=p["TEXT"])

    for arc in getattr(net, "_PetriNet__arcs", []):
        s_obj = getattr(arc, "_Arc__source", None); t_obj = getattr(arc, "_Arc__target", None)
        s = getattr(s_obj, "_Place__name","") or getattr(s_obj, "_Transition__name","")
        t = getattr(t_obj, "_Place__name","") or getattr(t_obj, "_Transition__name","")
        if s and t:
            w = getattr(arc, "_Arc__weight", 1)
            attrs = {"color": p["EDGE_COLOR"], "fontcolor": p["SOFT"]}
            if isinstance(w, int) and w > 1: attrs["label"] = str(w)
            dot.edge(s, t, **attrs)
    return dot

def render_interactive_svg(dot: graphviz.Digraph, p: Dict[str, str],
                           height: int = 560, show_legend: bool = True, 
                           legend_collapsed: bool = True, instance_token: str = ""):
    import hashlib
    svg_text = dot.pipe(format="svg").decode("utf-8")
    
    # Unique ID that includes instance_token
    instance_id = hashlib.md5((svg_text + instance_token).encode()).hexdigest()[:10]
    
    wrap_id = f"petri-wrap-{instance_id}"
    svg_id = f"petri-svg-{instance_id}"
    legend_id = f"legend-{instance_id}"
    tools_id = f"tools-{instance_id}"
    toggle_id = f"legend-toggle-{instance_id}"

    svg_text = svg_text.replace('<svg', f'<svg id="{svg_id}" class="petri-svg" preserveAspectRatio="xMidYMid meet"', 1)

    legend_bg = 'rgba(15,23,42,0.94)' if p['MODE']=='dark' else 'rgba(255,255,255,0.96)'
    tools_bg  = legend_bg

    html = f"""
    <div id="{wrap_id}" style="position:relative;height:{height}px;background:{p['SURFACE']};
         border:1px solid {p['BORDER']}; border-radius:14px; overflow:hidden;">
      <style>
        #{wrap_id} {{ box-shadow: 0 10px 30px rgba(0,0,0,.25); border-color: {p['ACCENT']}; }}
        #{legend_id} {{ position:absolute; left:12px; top:12px; z-index:10; background:{legend_bg};
          border:1px solid {p['BORDER']}; color:{p['TEXT']}; border-radius:12px; padding:8px 10px; max-width:46%;
          box-shadow:0 10px 30px rgba(0,0,0,.45); user-select:none; }}
        #{legend_id} .hdr {{ display:flex; gap:.4rem; align-items:center; cursor:pointer; font-weight:700; color:{p['TEXT']}; }}
        #{legend_id} .hdr span:last-child {{ color: {p['ACCENT']}; }}
        #{legend_id}.{ 'collapsed' if legend_collapsed else '' } .body {{ display:none; }}
        #{legend_id} .row {{ display:flex; align-items:center; gap:.5rem; margin:.25rem 0; color:{p['SOFT']}; }}
        #{legend_id} .dot {{ width:14px; height:14px; border-radius:50%; border:1px solid {p['BORDER']}; display:inline-block; }}
        #{legend_id} .sw  {{ width:14px; height:14px; border-radius:4px; border:1px solid {p['ACCENT2']}; display:inline-block; }}
        #{legend_id} .tau {{ width:16px; height:16px; border:1px solid {p['SOFT']}; border-radius:4px;
                        display:inline-flex; align-items:center; justify-content:center; font-weight:700; }}
        #{tools_id} {{ position:absolute; right:12px; top:12px; z-index:10; background:{tools_bg};
          border:1px solid {p['BORDER']}; border-radius:12px; display:flex; gap:6px; padding:6px;
          box-shadow:0 10px 30px rgba(0,0,0,.45); user-select:none; }}
        #{tools_id} button {{ background:{('#0c1424' if p['MODE']=='dark' else '#f1f5f9')};
          color:{p['TEXT']}; border:1px solid {p['BORDER']}; border-radius:8px; padding:4px 8px; font-size:12px; cursor:pointer; }}
        #{tools_id} button:hover {{ border-color:{p['ACCENT']}; color:{p['ACCENT']}; }}
        #{svg_id} {{ width:100% !important; height:100% !important; display:block; }}
        
        /* Improved loading states */
        .petri-loading {{
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          background: {p['CARD']};
          padding: 12px 20px;
          border-radius: 10px;
          border: 1px solid {p['BORDER']};
          color: {p['TEXT']};
          z-index: 100;
          font-weight: 600;
          box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }}
      </style>

      <div id="{legend_id}" class="{'collapsed' if legend_collapsed else ''}">
        <div class="hdr" id="{toggle_id}"><span>🛈</span><span>Legend</span></div>
        <div class="body">
          <div class="row"><span class="dot" style="background:{p['START_FILL']}"></span> Start</div>
          <div class="row"><span class="dot" style="background:{p['END_FILL']}"></span> End</div>
          <div class="row"><span class="dot" style="background:{p['PLACE_FILL']}"></span> State</div>
          <div class="row"><span class="sw"  style="background:{p['TRANS_FILL']}"></span> Activity</div>
          <div class="row"><span class="tau">τ</span> Internal step (silent)</div>
        </div>
      </div>

      <div id="{tools_id}">
        <button data-btn="fit">Fit</button>
        <button data-btn="reset">Reset</button>
        <button data-btn="zin">＋</button>
        <button data-btn="zout">－</button>
      </div>

      {svg_text}
    </div>

    <script src="https://cdn.jsdelivr.net/npm/svg-pan-zoom@3.6.1/dist/svg-pan-zoom.min.js"></script>
    <script>
      (function(){{
        let panZoomInstance = null;
        let initializationAttempts = 0;
        const MAX_ATTEMPTS = 12;

        function showLoading() {{
          const wrapEl = document.getElementById('{wrap_id}');
          if (wrapEl && !wrapEl.querySelector('.petri-loading')) {{
            const loadingDiv = document.createElement('div');
            loadingDiv.className = 'petri-loading';
            loadingDiv.innerHTML = 'Loading visualization...';
            wrapEl.appendChild(loadingDiv);
          }}
        }}

        function hideLoading() {{
          const wrapEl = document.getElementById('{wrap_id}');
          if (wrapEl) {{
            const loadingEl = wrapEl.querySelector('.petri-loading');
            if (loadingEl) loadingEl.remove();
          }}
        }}

        function initializePetri() {{
          const wrapEl = document.getElementById('{wrap_id}');
          const svg = document.getElementById('{svg_id}');
          
          if (!svg || !wrapEl) {{
            initializationAttempts++;
            if (initializationAttempts < MAX_ATTEMPTS) {{
              setTimeout(initializePetri, 200);
            }} else {{
              hideLoading();
            }}
            return;
          }}

          // Check visibility with stricter criteria
          const rect = wrapEl.getBoundingClientRect();
          const style = window.getComputedStyle(wrapEl);
          const isVisible = rect.width > 50 && rect.height > 50 && 
                           style.display !== 'none' && 
                           style.visibility !== 'hidden' &&
                           style.opacity !== '0';
          
          if (!isVisible) {{
            initializationAttempts++;
            if (initializationAttempts < MAX_ATTEMPTS) {{
              setTimeout(initializePetri, 200);
            }} else {{
              hideLoading();
            }}
            return;
          }}

          showLoading();

          // Clean previous instance if exists
          if (panZoomInstance) {{
            try {{
              panZoomInstance.destroy();
            }} catch(e) {{
              console.log('Cleanup error:', e);
            }}
            panZoomInstance = null;
          }}

          // Prepare SVG with delay
          setTimeout(() => {{
            svg.removeAttribute('width');
            svg.removeAttribute('height');
            svg.style.width = '100%';
            svg.style.height = '100%';
            svg.style.display = 'block';

            try {{
              panZoomInstance = svgPanZoom(svg, {{
                zoomEnabled: true,
                controlIconsEnabled: false,
                fit: true,
                center: true,
                minZoom: 0.1,
                maxZoom: 15,
                beforeZoom: function() {{
                  return true;
                }}
              }});

              function refit() {{
                try {{
                  if (panZoomInstance) {{
                    panZoomInstance.updateBBox();
                    panZoomInstance.resize();
                    panZoomInstance.fit();
                    panZoomInstance.center();
                  }}
                }} catch(e) {{
                  console.log('Refit error:', e);
                }}
              }}

              // Configure buttons with staggered delays
              setTimeout(() => {{
                const fitBtn = wrapEl.querySelector('[data-btn="fit"]');
                const resetBtn = wrapEl.querySelector('[data-btn="reset"]');
                const zinBtn = wrapEl.querySelector('[data-btn="zin"]');
                const zoutBtn = wrapEl.querySelector('[data-btn="zout"]');

                if (fitBtn) fitBtn.onclick = refit;
                if (resetBtn) resetBtn.onclick = function() {{
                  panZoomInstance.resetZoom();
                  panZoomInstance.center();
                }};
                if (zinBtn) zinBtn.onclick = function() {{ panZoomInstance.zoomIn(); }};
                if (zoutBtn) zoutBtn.onclick = function() {{ panZoomInstance.zoomOut(); }};
              }}, 150);

              // Scheduled refits with increasing delays
              setTimeout(refit, 300);
              setTimeout(refit, 800);
              setTimeout(refit, 1500);
              setTimeout(refit, 2500);

              // Hide loading after everything is ready
              setTimeout(hideLoading, 1000);

            }} catch(error) {{
              console.error('PanZoom initialization error:', error);
              hideLoading();
            }}
          }}, 100);
        }}

        // Initialization strategy with multiple delays
        function startInitialization() {{
          // Initial delay to allow Streamlit to render
          setTimeout(() => {{
            initializePetri();
            
            // Try again after some time in case of render delays
            setTimeout(initializePetri, 1000);
            setTimeout(initializePetri, 3000);
          }}, 500);
        }}

        // Start when ready
        if (document.readyState === 'loading') {{
          document.addEventListener('DOMContentLoaded', startInitialization);
        }} else {{
          startInitialization();
        }}

        // Legend toggle
        document.addEventListener('click', function(e) {{
          if (e.target.id === '{toggle_id}' || e.target.closest('#{toggle_id}')) {{
            const legend = document.getElementById('{legend_id}');
            if (legend) legend.classList.toggle('collapsed');
          }}
        }});

        // Observe window resize changes
        window.addEventListener('resize', function() {{
          if (panZoomInstance) {{
            setTimeout(() => {{
              try {{
                panZoomInstance.resize();
                panZoomInstance.fit();
              }} catch(e) {{}}
            }}, 300);
          }}
        }});
      }})();
    </script>
    """

    try:
        key = f"petri-{instance_id}"
        components.html(html, height=height + 16, scrolling=False, key=key)
    except TypeError:
        components.html(html, height=height + 16, scrolling=False)

# ------------------------------------------------------------
# Fallback Matplotlib (not normally used)
# ------------------------------------------------------------
def build_petri_figure(net, im, fm, show_ids: bool, p: Dict[str,str]):
    from random import random
    from collections import defaultdict
    nodes=set(); adj=defaultdict(set)
    for x in getattr(net, "_PetriNet__places", []): nodes.add(getattr(x,"_Place__name",""))
    for x in getattr(net, "_PetriNet__transitions", []): nodes.add(getattr(x,"_Transition__name",""))
    for a in getattr(net, "_PetriNet__arcs", []):
        s = getattr(a,"_Arc__source"); t = getattr(a,"_Arc__target")
        s = getattr(s,"_Place__name","") or getattr(s,"_Transition__name",""); t = getattr(t,"_Place__name","") or getattr(t,"_Transition__name","")
        if s and t: adj[s].add(t); adj[t].add(s)
    pos = {n:(random(),random()) for n in nodes}
    xs=[p_[0] for p_ in pos.values()]; ys=[p_[1] for p_ in pos.values()]
    minx,maxx=min(xs),max(xs); miny,maxy=min(ys),max(ys)
    pos={n:((x-minx)/(maxx-minx+1e-6),(y-miny)/(maxy-miny+1e-6)) for n,(x,y) in pos.items()}
    fig, ax = plt.subplots(figsize=(10,8)); fig.patch.set_facecolor(p["BG"]); ax.set_facecolor(p["BG"]); ax.axis("off")
    place_names = {getattr(x,"_Place__name","") for x in getattr(net,"_PetriNet__places",[])}
    trans_labels={}
    for t in getattr(net,"_PetriNet__transitions",[]):
        n=getattr(t,"_Transition__name",""); lab=getattr(t,"_Transition__label","") or n
        if n.startswith("hid_") or n=="": lab="τ"
        trans_labels[n]=lab
    im_names=[getattr(x,"_Place__name","") for x in (im or {}).keys()] if isinstance(im,dict) else []
    fm_names=[getattr(x,"_Place__name","") for x in (fm or {}).keys()] if isinstance(fm,dict) else []
    for u in nodes:
        for v in adj[u]:
            if u < v:
                x1,y1=pos[u]; x2,y2=pos[v]
                ax.plot([x1,x2],[y1,y2],linewidth=.8,color=p["EDGE_COLOR"])
    for n in nodes:
        x,y=pos[n]
        if n in place_names:
            if n in im_names: show,fill,tcol = "Start", p["START_FILL"], p["START_TEXT"]
            elif n in fm_names: show,fill,tcol = "End", p["END_FILL"], p["END_TEXT"]
            else: show,fill,tcol = prettify_label(n), p["PLACE_FILL"], p["PLACE_TEXT"]
            txt=f"{show} ({n})" if show_ids and show!=n else show
            ax.add_patch(Circle((x,y), 0.017, facecolor=fill, edgecolor=p["BORDER"]))
            ax.text(x,y,txt,ha="center",va="center",fontsize=7.2,color=tcol)
        else:
            base=trans_labels.get(n,n); friendly=prettify_label(base)
            txt=f"{friendly} ({n})" if show_ids and friendly!=n else friendly
            ax.add_patch(Rectangle((x-0.022,y-0.018),0.044,0.036, facecolor=p["TRANS_FILL"], edgecolor=p["ACCENT2"]))
            ax.text(x,y,txt,ha="center",va="center",fontsize=7.2,color=p["TEXT"])
    return fig

# ------------------------------------------------------------
# DFG with pm4py + Graphviz (vertical, Celonis style)
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def read_event_log(uploaded) -> Tuple[object, pd.DataFrame]:
    """
    Reads CSV/XES/XES.GZ from st.file_uploader and returns (EventLog, sorted DataFrame).
    Requires standard CSV columns: case:concept:name, concept:name, time:timestamp
    """
    from pm4py.objects.conversion.log import converter as log_converter
    from pm4py.util import xes_constants as xes
    from pm4py.objects.log.importer.xes import importer as xes_importer
    import tempfile, os

    name = (uploaded.name or "").lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded)
        if "time:timestamp" in df.columns:
            df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], errors="coerce")
        # Basic ordering
        df = df.sort_values(by=["case:concept:name", "time:timestamp"], kind="mergesort")
        # Conversion to EventLog
        params = {
            log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: "case:concept:name"
        }
        evlog = log_converter.apply(df, variant=log_converter.Variants.TO_EVENT_LOG, parameters=params)
        return evlog, df

    elif name.endswith(".xes") or name.endswith(".xes.gz"):
        # Save to tmp and use importer
        suffix = ".xes.gz" if name.endswith(".xes.gz") else ".xes"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name
        evlog = xes_importer.apply(tmp_path)
        # Auxiliary DataFrame (for quick tables)
        records = []
        for trace in evlog:
            case_id = trace.attributes.get("concept:name")
            for ev in trace:
                records.append({
                    "case:concept:name": case_id,
                    "concept:name": ev.get("concept:name"),
                    "time:timestamp": ev.get("time:timestamp"),
                    "org:resource": ev.get("org:resource"),
                    "org:role": ev.get("org:role"),
                })
        df = pd.DataFrame(records)
        if "time:timestamp" in df:
            df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], errors="coerce")
        df = df.sort_values(by=["case:concept:name", "time:timestamp"], kind="mergesort")
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return evlog, df

    else:
        raise ValueError("Unsupported format. Upload CSV, XES or XES.GZ.")


def discover_dfg_with_pm4py(evlog):
    """
    Returns:
      - dfg: dict {(a,b): freq}
      - starts: dict {a: freq}
      - ends: dict {a: freq}
      - act_freq: dict {a: freq}
      - n_cases: int
    """
    from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
    from pm4py.statistics.start_activities.log import get as sa_get
    from pm4py.statistics.end_activities.log import get as ea_get
    from pm4py.statistics.attributes.log import get as attr_get

    dfg = dfg_discovery.apply(evlog, variant=dfg_discovery.Variants.FREQUENCY)
    starts = sa_get.get_start_activities(evlog)
    ends = ea_get.get_end_activities(evlog)
    act_freq = attr_get.get_attribute_values(evlog, attribute_key="concept:name")
    n_cases = len(evlog)
    return dfg, starts, ends, act_freq, n_cases

# --- Stats and order helpers ---
def _fmt_duration(seconds: float) -> str:
    try:
        s = float(seconds)
    except Exception:
        return "-"
    if s < 1:
        return f"{s*1000:.0f} ms"
    m, sec = divmod(int(round(s)), 60)
    h, m = divmod(m, 60)
    if h > 0: return f"{h}h {m}m"
    if m > 0: return f"{m}m {sec}s"
    return f"{sec}s"

def compute_activity_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Stats per activity:
      A) if there's start_timestamp -> dur = end - start
      B) if NOT -> dur = time until next event of SAME case
         (approx. sojourn/time between consecutive activities).
    """
    if df is None or df.empty or "concept:name" not in df.columns or "time:timestamp" not in df.columns:
        return {}

    tmp = df.copy()
    tmp["time:timestamp"] = pd.to_datetime(tmp["time:timestamp"], errors="coerce")

    use = pd.DataFrame(columns=["concept:name", "dur_s"])

    if "start_timestamp" in tmp.columns:
        tmp["start_timestamp"] = pd.to_datetime(tmp["start_timestamp"], errors="coerce")
        useA = tmp[["concept:name", "start_timestamp", "time:timestamp"]].dropna()
        if not useA.empty:
            useA = useA.assign(dur_s=(useA["time:timestamp"] - useA["start_timestamp"]).dt.total_seconds())
            useA = useA.query("dur_s >= 0")
            use = useA[["concept:name", "dur_s"]]

    # Fallback B if A didn't contribute (or if start_timestamp doesn't exist)
    if use.empty:
        tmp = tmp.sort_values(by=["case:concept:name", "time:timestamp"], kind="mergesort")
        tmp["next_ts"] = tmp.groupby("case:concept:name")["time:timestamp"].shift(-1)
        useB = tmp[["concept:name", "time:timestamp", "next_ts"]].dropna()
        if not useB.empty:
            useB = useB.assign(dur_s=(useB["next_ts"] - useB["time:timestamp"]).dt.total_seconds())
            useB = useB.query("dur_s >= 0")
            use = useB[["concept:name", "dur_s"]]

    if use.empty:
        return {}

    g = use.groupby("concept:name")["dur_s"]
    out: Dict[str, Dict[str, float]] = {}
    for act, s in g:
        s = s.dropna()
        if s.empty:
            continue
        out[act] = dict(
            mean=float(s.mean()),
            std=float(s.std(ddof=0) if s.size > 1 else 0.0),
            max=float(s.max()),
            min=float(s.min()),
        )
    return out


def compute_activity_order(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty: return {}
    d = df.sort_values(by=["case:concept:name", "time:timestamp"], kind="mergesort")
    positions, counts = {}, {}
    for _, g in d.groupby("case:concept:name"):
        seq = g["concept:name"].tolist()
        if not seq: continue
        compact = [seq[0]]
        for a in seq[1:]:
            if a != compact[-1]: compact.append(a)
        L = max(1, len(compact))
        for idx, act in enumerate(compact):
            pos = idx / (L - 1) if L > 1 else 0.0
            positions[act] = positions.get(act, 0.0) + pos
            counts[act]    = counts.get(act, 0) + 1
    return {a: positions[a] / counts[a] for a in positions}

def _shorten_txt(s: str, maxlen: int = 28) -> str:
    s = str(s or "")
    return (s[:maxlen-1] + "…") if len(s) > maxlen else s

def _top3_names(series: pd.Series) -> str:
    if series is None:
        return ""
    vc = series.dropna().astype(str).value_counts()
    if vc.empty:
        return ""
    tops = [_shorten_txt(x) for x in vc.head(3).index.tolist()]
    return ", ".join(tops)

def build_log_stats_html_from_df(df: pd.DataFrame) -> str:
    """
    Builds an HTML block with:
      - # of cases
      - # of activities + Top 3 activities
      - # of resources + Top 3 (if column exists)
      - # of roles + Top 3 (if column exists)
      - Timestamp range (min → max)
    """
    if df is None or df.empty:
        return '<div class="small" style="margin-top:.5rem;color:var(--soft)">No data to summarize.</div>'

    # Cases
    n_cases = df["case:concept:name"].nunique() if "case:concept:name" in df.columns else 0

    # Activities
    n_acts = df["concept:name"].nunique() if "concept:name" in df.columns else 0
    top_acts = _top3_names(df["concept:name"]) if "concept:name" in df.columns else ""

    # Resources (if they exist)
    has_res = ("org:resource" in df.columns) and df["org:resource"].notna().any()
    n_res = df["org:resource"].dropna().nunique() if has_res else 0
    top_res = _top3_names(df["org:resource"]) if has_res else ""

    # Roles (if they exist)
    has_role = ("org:role" in df.columns) and df["org:role"].notna().any()
    n_roles = df["org:role"].dropna().nunique() if has_role else 0
    top_roles = _top3_names(df["org:role"]) if has_role else ""

    # Timestamp range
    ts_min = ts_max = "-"
    if "time:timestamp" in df.columns:
        ts = pd.to_datetime(df["time:timestamp"], errors="coerce")
        if ts.notna().any():
            ts_min = ts.min()
            ts_max = ts.max()
            # Compact readable format
            ts_min = ts_min.strftime("%Y-%m-%d %H:%M")
            ts_max = ts_max.strftime("%Y-%m-%d %H:%M")


    role_html = f'<div><b>Roles:</b> {n_roles} <span class="small">(top 3: {top_roles})</span></div>' if has_role else ""

    return f"""
    <div style="margin-top:.6rem;padding:.65rem .8rem;border:1px dashed var(--border);
                border-radius:.7rem;background:var(--card);">
      <div style="display:flex;flex-wrap:wrap;gap:.6rem 1rem;align-items:center;">
        <div><b>Cases:</b> {n_cases} \n </div>
        <div><b>Time range:</b> {ts_min} → {ts_max}</div>
        <div><b>Activities:</b> {n_acts} <span class="small">(top 3: {top_acts})</span></div>
        <div><b>Resources:</b> {n_res} <span class="small">(top 3: {top_res})</span></div>
        {role_html}
      </div>
    </div>
    """


def _edge_label_text(cnt: int) -> str:
    return str(int(cnt))

def _attrs_to_text_md(x, level: int = 0) -> str:
    indent = "  " * level
    lines = []
    
    for k, v in x[0].items():
            if isinstance(v, (dict, list, tuple)):
                lines.append(f"- **{k['name']}:**")
                lines.append(_attrs_to_text_md(v['summary'], level + 1))
                lines.append("\n")
            else:
                lines.append(f"- **{k['name']}:** {v['summary']} \n")
    
    return "\n".join(lines)

@st.cache_data(show_spinner=False)
def get_agent3_attributes_md() -> str:
    """
    Extracts state['agente3']['result']['atributes'] and returns as Markdown (bullets).
    Without heuristics or alternative keys.
    """
    a3 = get_current_agent3()             # you already have this helper in your app
    attrs = a3["attributes"]        # fixed key
    return _attrs_to_text_md(attrs).strip()


def build_dfg_graphviz(
    dfg, starts, ends, act_freq, n_cases: int, p: Dict[str,str],
    min_freq: int = 1,
    act_order: Optional[Dict[str, float]] = None,
    act_stats: Optional[Dict[str, Dict[str, float]]] = None,
    show_stats: bool = False
) -> graphviz.Digraph:

    dot = graphviz.Digraph("DFG", format="svg")
    dot.attr("graph",
             rankdir="TB", bgcolor=p["SURFACE"], pad="0.2",
             splines="spline", overlap="false", concentrate="true",
             nodesep="0.38", ranksep="0.75", newrank="true")
    dot.attr("node", fontname="Helvetica", fontsize="11", penwidth="1.3")
    dot.attr("edge", fontname="Helvetica", fontsize="10",
             color=p["EDGE_COLOR"],  fontcolor=('#0b1220' if p['MODE']=='dark' else p['SOFT']), arrowsize="0.7")

    # Nodes to show, respecting threshold
    activities = set()
    for (a, b), c in dfg.items():
        if c >= min_freq:
            activities.add(a); activities.add(b)
    for a, c in starts.items():
        if c >= min_freq: activities.add(a)
    for a, c in ends.items():
        if c >= min_freq: activities.add(a)

    # Anchors
    dot.node("_START_", "Start", shape="hexagon", style="filled",
             fillcolor=p["START_FILL"], color=p["ACCENT"], fontcolor=p["START_TEXT"])
    dot.node("_END_", "End", shape="hexagon", style="filled",
             fillcolor=p["END_FILL"], color=p["ACCENT"], fontcolor=p["END_TEXT"])
    dot.body.append('{rank=source; "_START_";}')
    dot.body.append('{rank=sink; "_END_";}')

    act_order = act_order or {}
    def level_of(act: str) -> int:
        v = act_order.get(act, 0.5)
        return int(max(0, min(5, round(v * 5))))  # 6 levels (0..5)

    levels = {i: [] for i in range(6)}
    for a in activities:
        levels[level_of(a)].append(a)
    for i in range(6):
        if levels[i]:
            nodes = " ".join(f'"{a}"' for a in levels[i])
            dot.body.append(f'{{rank=same; {nodes};}}')

    def stats_suffix(a: str) -> str:
        if not (show_stats and act_stats and a in act_stats):
            return ""
        s = act_stats[a]
        return "\n" + (
            f"avg={_fmt_duration(s['mean'])}, "
            f"std={_fmt_duration(s['std'])}\n"
            f"max={_fmt_duration(s['max'])}, "
            f"min={_fmt_duration(s['min'])}"
        ) + "\n"

    for a in sorted(activities, key=lambda x: (level_of(x), x.lower())):
        label = prettify_label(a) + stats_suffix(a)
        dot.node(a, label=label, shape="box", style="rounded,filled",
                 height="0.28", width="0.99", margin="0.09,0.06",
                 fillcolor=p["CARD"], color=p["ACCENT2"], fontcolor=p["TEXT"])

    all_vals = list(dfg.values()) + list(starts.values()) + list(ends.values()) + [1]
    max_edge = max(all_vals)
    def _penwidth(cnt: int) -> str:
        return f"{1.4 + 4.6*(cnt/max_edge):.3f}"

    for a, c in starts.items():
        if a in activities and c >= min_freq:
            dot.edge("_START_", a, label=_edge_label_text(c), penwidth=_penwidth(c))

    for (a, b), c in dfg.items():
        if c >= min_freq and a in activities and b in activities:
            minlen = max(1, level_of(b) - level_of(a))
            dot.edge(a, b, label=_edge_label_text(c), penwidth=_penwidth(c), minlen=str(minlen))

    for a, c in ends.items():
        if a in activities and c >= min_freq:
            dot.edge(a, "_END_", label=_edge_label_text(c), penwidth=_penwidth(c))

    dot.node("_START_", xlabel=f"{n_cases} cases", fontcolor=p["SOFT"])
    dot.node("_END_",   xlabel=f"{n_cases} cases", fontcolor=p["SOFT"])
    return dot

def render_interactive_svg_dfg(dot: graphviz.Digraph, p: Dict[str, str],
                               height: int = 620, legend_collapsed: bool = True,
                               instance_token: str = ""):
    import hashlib
    svg_text = dot.pipe(format="svg").decode("utf-8")
    
    # Unique ID for this instance
    base_token = f"{instance_token}-{int(legend_collapsed)}-{p['MODE']}"
    instance_id = hashlib.md5((svg_text + base_token).encode()).hexdigest()[:8]
    
    wrap_id   = f"dfg-wrap-{instance_id}"
    svg_id    = f"dfg-svg-{instance_id}"
    legend_id = f"legend-dfg-{instance_id}"
    tools_id  = f"tools-dfg-{instance_id}"
    toggle_id = f"legend-dfg-toggle-{instance_id}"

    svg_text = svg_text.replace('<svg', f'<svg id="{svg_id}" class="dfg-svg" preserveAspectRatio="xMidYMid meet"', 1)

    legend_bg = 'rgba(15,23,42,0.94)' if p['MODE']=='dark' else 'rgba(255,255,255,0.96)'
    tools_bg  = legend_bg
    label_bg  = 'rgba(255,255,255,0.86)' if p['MODE']=='dark' else 'rgba(255,255,255,0.45)'
    hl_color  = '#facc15' if p['MODE']=='dark' else '#f97316'
    strong_label_bg = 'rgba(255,255,255,0.98)' if p['MODE']=='dark' else 'rgba(255,255,255,0.75)'
    strong_text_fill = '#0b1220' if p['MODE']=='dark' else p['TEXT']

    html = f"""
    <div id="{wrap_id}" style="position:relative;height:{height}px;background:{p['SURFACE']};
         border:1px solid {p['BORDER']}; border-radius:14px; overflow:hidden;">
      <style>
        #{wrap_id} {{ box-shadow: 0 10px 30px rgba(0,0,0,.25); border-color: {p['ACCENT']}; }}
        #{legend_id} {{ position:absolute; left:12px; top:12px; z-index:10; background:{legend_bg};
          border:1px solid {p['BORDER']}; color:{p['TEXT']}; border-radius:12px; padding:8px 10px; max-width:46%;
          box-shadow:0 10px 30px rgba(0,0,0,.45); user-select:none; }}
        #{legend_id} .hdr {{ display:flex; gap:.4rem; align-items:center; cursor:pointer; font-weight:700; color:{p['TEXT']}; }}
        #{legend_id} .hdr span:last-child {{ color: {p['ACCENT']}; }}
        #{legend_id}.{ 'collapsed' if legend_collapsed else '' } .body {{ display:none; }}
        #{legend_id} .row {{ display:flex; align-items:center; gap:.5rem; margin:.25rem 0; color:{p['SOFT']}; }}
        #{legend_id} .dot {{ width:14px; height:14px; border-radius:4px; border:1px solid {p['ACCENT2']}; display:inline-block; background:{p['CARD']}; }}
        #{legend_id} .hex {{ width:14px; height:14px; display:inline-block; border:1px solid {p['BORDER']}; }}

        #{tools_id} {{ position:absolute; right:12px; top:12px; z-index:10; background:{tools_bg};
          border:1px solid {p['BORDER']}; border-radius:12px; display:flex; gap:6px; padding:6px;
          box-shadow:0 10px 30px rgba(0,0,0,.45); user-select:none; }}
        #{tools_id} button {{ background:{('#0c1424' if p['MODE']=='dark' else '#f1f5f9')};
          color:{p['TEXT']}; border:1px solid {p['BORDER']}; border-radius:8px; padding:4px 8px; font-size:12px; cursor:pointer; }}
        #{tools_id} button:hover {{ border-color:{p['ACCENT']}; color:{p['ACCENT']}; }}

        #{svg_id} {{ width:100% !important; height:100% !important; display:block; }}

        #{wrap_id} g.edge {{ transition: opacity .12s ease, filter .12s ease; cursor:pointer; }}
        #{wrap_id} g.edge path, #{wrap_id} g.edge polygon {{ transition: all .12s ease; }}
        #{wrap_id} g.edge path {{ pointer-events: stroke; }}
        #{wrap_id} g.edge polygon {{ pointer-events: all; }}
        #{wrap_id} g.edge.hover path, #{wrap_id} g.edge.selected path {{ stroke: {hl_color} !important; stroke-width: 2.6px !important; }}
        #{wrap_id} g.edge.hover polygon, #{wrap_id} g.edge.selected polygon {{ fill: {hl_color} !important; stroke: {hl_color} !important; }}
        #{wrap_id} g.edge .__edgebg {{ fill: {label_bg}; }}
        #{wrap_id} g.edge.hover .__edgebg, #{wrap_id} g.edge.selected .__edgebg {{ fill: {strong_label_bg}; stroke: {p['ACCENT']}; stroke-width: .8; }}
        #{wrap_id} g.edge.hover text, #{wrap_id} g.edge.selected text {{ fill: {strong_text_fill}; }}
        #{wrap_id}.dim-others g.edge:not(.selected) {{ opacity: .28; filter: grayscale(20%); }}
      </style>

      <div id="{legend_id}" class="{'collapsed' if legend_collapsed else ''}">
        <div class="hdr" id="{toggle_id}"><span>🛈</span><span>Legend</span></div>
        <div class="body">
          <div class="row"><span class="hex" style="background:{p['START_FILL']}"></span> Start</div>
          <div class="row"><span class="hex" style="background:{p['END_FILL']}"></span> End</div>
          <div class="row"><span class="dot"></span> Activity</div>
          <div class="row">▸ Thickness = path frequency</div>
          <div class="row">▸ Highlight = hover / selection ({'yellow' if p['MODE']=='dark' else 'orange'})</div>
        </div>
      </div>

      <div id="{tools_id}">
        <button data-btn="fit">Fit</button>
        <button data-btn="reset">Reset</button>
        <button data-btn="zin">＋</button>
        <button data-btn="zout">－</button>
        <button data-btn="clear">Clear selection</button>
      </div>

      {svg_text}
    </div>

    <script src="https://cdn.jsdelivr.net/npm/svg-pan-zoom@3.6.1/dist/svg-pan-zoom.min.js"></script>
    <script>
      (function(){{
        // Global variables for this instance
        window.dfgInstances = window.dfgInstances || {{}};
        let panZoom = null;
        let resizeObserver = null;
        let isInitialized = false;

        function initializeDFG() {{
          const wrap = document.getElementById('{wrap_id}');
          const svg  = document.getElementById('{svg_id}');
          
          if (!svg || !wrap) {{
            setTimeout(initializeDFG, 100);
            return;
          }}

          // Check if already initialized
          if (isInitialized) {{
            return;
          }}

          const rect = wrap.getBoundingClientRect();
          if (rect.width === 0 || rect.height === 0) {{
            setTimeout(initializeDFG, 100);
            return;
          }}

          // Clean previous instance if exists
          if (window.dfgInstances['{instance_id}']) {{
            try {{
              window.dfgInstances['{instance_id}'].destroy();
            }} catch(e) {{}}
          }}

          svg.removeAttribute('width'); 
          svg.removeAttribute('height');
          svg.style.width = '100%'; 
          svg.style.height = '100%'; 
          svg.style.display = 'block';
          svg.style.filter = 'drop-shadow(0 12px 28px rgba(0,0,0,.35))';

          // Initialize pan-zoom
          panZoom = svgPanZoom(svg, {{ 
            zoomEnabled: true, 
            controlIconsEnabled: false, 
            fit: true, 
            center: true, 
            minZoom: 0.1, 
            maxZoom: 20,
            beforeZoom: function() {{
              return true;
            }}
          }});

          // Save instance
          window.dfgInstances['{instance_id}'] = panZoom;
          isInitialized = true;

          function refit() {{
            try {{
              if (panZoom && panZoom.getSvg) {{
                panZoom.updateBBox();
                panZoom.resize();
                panZoom.fit();
                panZoom.center();
              }}
            }} catch(e) {{ 
              console.log('DFG refit error:', e); 
            }}
          }}

          // Buttons
          wrap.querySelector('[data-btn="fit"]').addEventListener('click', refit);
          wrap.querySelector('[data-btn="reset"]').addEventListener('click', () => {{
            panZoom.resetZoom();
            panZoom.center();
          }});
          wrap.querySelector('[data-btn="zin"]').addEventListener('click', () => panZoom.zoomIn());
          wrap.querySelector('[data-btn="zout"]').addEventListener('click', () => panZoom.zoomOut());
          wrap.querySelector('[data-btn="clear"]').addEventListener('click', () => {{
            svg.querySelectorAll('g.edge.selected').forEach(g => g.classList.remove('selected'));
            wrap.classList.remove('dim-others');
          }});

          // Label backgrounds and z-order
          const pad = 4;
          svg.querySelectorAll('g.edge').forEach(g => g.parentNode.appendChild(g));
          svg.querySelectorAll('g.edge text').forEach(t => {{
            if (t.getAttribute('data-boxed') === '1') return;
            const bb = t.getBBox();
            const r  = document.createElementNS(svg.namespaceURI, 'rect');
            r.setAttribute('x', bb.x - pad);
            r.setAttribute('y', bb.y - pad);
            r.setAttribute('width',  bb.width + pad*2);
            r.setAttribute('height', bb.height + pad*2);
            r.setAttribute('rx','4'); r.setAttribute('ry','4');
            r.setAttribute('fill', '{label_bg}');
            r.setAttribute('class','__edgebg');
            t.parentNode.insertBefore(r, t);
            t.style.userSelect = 'text'; 
            t.style.webkitUserSelect = 'text';
            t.setAttribute('data-boxed','1');
          }});

          function updateDimState() {{
            const anySel = svg.querySelector('g.edge.selected');
            if (anySel) wrap.classList.add('dim-others'); 
            else wrap.classList.remove('dim-others');
          }}

          svg.querySelectorAll('g.edge').forEach(g => {{
            g.addEventListener('mouseenter', () => g.classList.add('hover'));
            g.addEventListener('mouseleave', () => g.classList.remove('hover'));
            g.addEventListener('click', (ev) => {{
              if (window.getSelection) window.getSelection().removeAllRanges();
              g.classList.toggle('selected');
              updateDimState();
              ev.stopPropagation();
            }});
          }});

          svg.addEventListener('click', (ev) => {{
            if (!ev.target.closest('g.edge')) {{
              svg.querySelectorAll('g.edge.selected').forEach(g => g.classList.remove('selected'));
              updateDimState();
            }}
          }});

          // Resize change observer
          if (window.ResizeObserver) {{
            resizeObserver = new ResizeObserver(() => {{
              setTimeout(refit, 50);
            }});
            resizeObserver.observe(wrap);
          }}

          // Initial refits
          setTimeout(refit, 100);
          setTimeout(refit, 500);
          
          // Handle tab visibility
          const handleVisibility = () => {{
            if (document.visibilityState === 'visible') {{
              setTimeout(refit, 300);
            }}
          }};
          document.addEventListener('visibilitychange', handleVisibility);

          // Cleanup function
          window.dfgCleanup_{instance_id} = function() {{
            if (resizeObserver) {{
              resizeObserver.disconnect();
            }}
            document.removeEventListener('visibilitychange', handleVisibility);
            if (panZoom) {{
              try {{
                panZoom.destroy();
              }} catch(e) {{}}
            }}
            delete window.dfgInstances['{instance_id}'];
          }};
        }}

        // Delayed initialization
        function startInitialization() {{
          // Wait for Streamlit to render tabs
          setTimeout(() => {{
            const checkVisibility = setInterval(() => {{
              const wrap = document.getElementById('{wrap_id}');
              if (wrap) {{
                const style = window.getComputedStyle(wrap);
                if (style.display !== 'none') {{
                  clearInterval(checkVisibility);
                  initializeDFG();
                }}
              }}
            }}, 100);
            
            // Safety timeout
            setTimeout(() => clearInterval(checkVisibility), 5000);
          }}, 100);
        }}

        if (document.readyState === 'loading') {{
          document.addEventListener('DOMContentLoaded', startInitialization);
        }} else {{
          startInitialization();
        }}

        // Legend toggle
        document.addEventListener('click', function(e) {{
          if (e.target.id === '{toggle_id}' || e.target.closest('#{toggle_id}')) {{
            const legend = document.getElementById('{legend_id}');
            if (legend) legend.classList.toggle('collapsed');
          }}
        }});
      }})();
    </script>
    """

    try:
        key = f"dfg-{instance_id}"
        components.html(html, height=height + 16, scrolling=False, key=key)
    except TypeError:
        components.html(html, height=height + 16, scrolling=False)


def quick_text_similarity(text1, text2):
    """Quick similarity between two texts using SequenceMatcher"""
    if not text1 or not text2:
        return 0.0
    
    # Basic cleaning for better matching
    def clean_text(text):
        text = str(text).lower()
        # Remove punctuation and extra spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    clean1 = clean_text(text1)
    clean2 = clean_text(text2)
    
    if not clean1 or not clean2:
        return 0.0
        
    return SequenceMatcher(None, clean1, clean2).ratio()

def fast_semantic_matching(agent8_problem, agent4_problems):
    """
    Fast matching that uses direct text similarity.
    Optimized for speed.
    """
    if not agent8_problem or not agent4_problems:
        return None, 0.0
    
    best_match = None
    best_score = 0.0
    
    for problem in agent4_problems:
        # Calculate similarity with justification (more detailed)
        justification_sim = quick_text_similarity(agent8_problem, problem.get('justification', ''))
        
        # Calculate similarity with description (more concise)
        description_sim = quick_text_similarity(agent8_problem, problem.get('description', ''))
        
        # Use the highest similarity
        current_score = max(justification_sim, description_sim)
        
        if current_score > best_score:
            best_score = current_score
            best_match = problem
    
    return best_match, best_score


def match_problems_with_solutions():
    """
    Improved matching between agent4 problems and agent8 solutions.
    """
    agent4_data = get_current_agent4()
    agent8_items = get_current_agent8()
    
    matched_pairs = []
    
    # Extract problems from agent4
    agent4_problems = []
    if 'conformance_issues' in agent4_data and 'issues' in agent4_data['conformance_issues']:
        for issue in agent4_data['conformance_issues']['issues']:
            problem_data = {
                'description': issue.get('description', ''),
                'justification': issue.get('justification', ''),
                'type': issue.get('type', ''),
                'element': issue.get('element', ''),
                'process_flows': issue.get('process_flows', [])
            }
            agent4_problems.append(problem_data)
    
    # For debugging
    st.sidebar.write(f"🔍 Found {len(agent4_problems)} problems in agent4")
    st.sidebar.write(f"🔍 Found {len(agent8_items)} problems in agent8")
    
    # Improved matching strategy
    for i, agent8_item in enumerate(agent8_items):
        problem = agent8_item.get("problem_description") or agent8_item.get("problem_descripcion", "")
        solution = agent8_item.get("recommendation", "")
        
        if problem and solution:
            # First try index matching if same quantity
            matching_problem = None
            problem_justification = ""
            
            if i < len(agent4_problems):
                # Direct index matching
                matching_problem = agent4_problems[i]
                problem_justification = matching_problem.get('justification', '')
                st.sidebar.write(f"✅ Match by index: Problem {i+1}")
            else:
                # Find best match by similarity
                best_score = 0
                best_match = None
                
                for agent4_problem in agent4_problems:
                    # Calculate similarity with description and justification
                    desc_sim = quick_text_similarity(problem, agent4_problem.get('description', ''))
                    just_sim = quick_text_similarity(problem, agent4_problem.get('justification', ''))
                    current_score = max(desc_sim, just_sim)
                    
                    if current_score > best_score and current_score > 0.3:
                        best_score = current_score
                        best_match = agent4_problem
                
                if best_match:
                    matching_problem = best_match
                    problem_justification = best_match.get('justification', '')
                    st.sidebar.write(f"✅ Match by similarity ({best_score:.2f}): Problem {i+1}")
                else:
                    st.sidebar.write(f"❌ No match: Problem {i+1}")
            
            matched_pairs.append({
                "Problem": str(problem).strip(),
                "Solution": str(solution).strip(),
                "_justification": problem_justification  # Hidden field
            })
    
    return matched_pairs


def get_agent8_problem_pairs() -> List[Dict[str, str]]:
    """
    Maps problems with solutions INCLUDING hidden justification.
    """
    return match_problems_with_solutions()


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
with st.sidebar:
    default_dark = st.session_state.get("theme", "dark") == "dark"
    dark_on = st.toggle("🌙 Dark mode", value=default_dark)
    st.session_state["theme"] = "dark" if dark_on else "light"

    # === DOMAIN SELECTOR ===
    st.markdown("---")
    st.markdown("### 🏷️ Process Domain")

    domain_options = ["Manufacturing", "Banking", "IT Support"]

    if 'selected_domain' not in st.session_state:
        st.session_state.selected_domain = domain_options[0]
    
    selected_domain = st.selectbox(
        "Select your process domain:",
        options=domain_options,
        index=domain_options.index(st.session_state.selected_domain),
        label_visibility="collapsed"
    )
    
    if selected_domain != st.session_state.selected_domain:
        st.session_state.selected_domain = selected_domain
        st.rerun()
    
    st.markdown(f"**Current domain:** `{st.session_state.selected_domain}`")
    
   
    # === ADVANCED OPTIONS ===
    with st.expander("⚙️ Advanced options", expanded=False):
        # ✅ NEW: Checkbox for Preprocessed State
        use_preprocessed_state = st.checkbox(
            "Preprocessed state", 
            value=True,
            help="Use a preprocessed state to speed up processing. Only available for predefined domains."
        )
        
        upsert_qdrant = st.checkbox(
            "📊 Index to Qdrant (recommended)", 
            value=True,
            help="Store results for semantic searches and session management. Data expires automatically."
        )
        
        ttl_hours = st.slider(
            "⏰ Data lifetime (hours)",
            min_value=1,
            max_value=24,
            value=3,
            help="Data will be automatically deleted after this time"
        )
        ttl_seconds = ttl_hours * 3600

    
    st.markdown("---")

    st.markdown("### 📤 Upload Process File")
    uploaded = st.file_uploader("Upload your event log", type=["csv","xes","xes.gz"], label_visibility="collapsed")
    
    # Processing button - KEEP
    if uploaded is not None:
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🚀 Process", use_container_width=True):
                success = process_file_backend(
                    uploaded,
                    st.session_state.selected_domain,
                    upsert_qdrant=upsert_qdrant,
                    ttl_seconds=ttl_seconds,
                    use_preprocessed_state=use_preprocessed_state 
                )
                if success:
                    st.rerun()
        with col2:
            if st.button("🔄 Clear", use_container_width=True):
                if 'state_file_path' in st.session_state:
                    try: os.remove(st.session_state['state_file_path'])
                    except: pass
                    del st.session_state['state_file_path']
                st.rerun()

    
    st.markdown("---")
    
    # Current state - SIMPLIFIED
    if 'state_file_path' in st.session_state:
        st.success("✅ Status: Processed")
    else:
        st.info("⏳ Status: Waiting for file")
    
    st.caption(f"⏱️ {datetime.now().strftime('%Y-%m-%d %H:%M')}")


# ------------------------------------------------------------
# Apply chosen palette
# ------------------------------------------------------------
PALETTE = get_palette(st.session_state.get("theme", "dark"))
inject_css(PALETTE)



# Replace current script with this (after inject_css())
st.markdown("""
<script>
// Universal management system for SVG components
window.svgGlobalManager = {
    components: new Map(),
    
    register: function(id, type) {
        this.components.set(id, {
            type: type,
            lastActive: Date.now(),
            initialized: false
        });
    },
    
    refreshAll: function() {
        console.log('Refreshing all SVG components...');
        this.components.forEach((data, id) => {
            const element = document.getElementById(id);
            if (element) {
                const svg = element.querySelector('svg');
                if (svg && window.svgPanZoom) {
                    try {
                        const instance = window.svgPanZoom(svg);
                        if (instance) {
                            setTimeout(() => {
                                instance.resize();
                                instance.fit();
                                instance.center();
                            }, 300);
                        }
                    } catch(e) {
                        console.log('Refresh error for', id, e);
                    }
                }
            }
        });
    },
    
    cleanup: function() {
        const now = Date.now();
        const MAX_AGE = 120000; // 2 minutes
        
        this.components.forEach((data, id) => {
            if (now - data.lastActive > MAX_AGE) {
                const element = document.getElementById(id);
                if (element) {
                    const svg = element.querySelector('svg');
                    if (svg && window.svgPanZoom) {
                        try {
                            const instance = window.svgPanZoom(svg);
                            if (instance) instance.destroy();
                        } catch(e) {}
                    }
                }
                this.components.delete(id);
            }
        });
    }
};

// Run periodic cleanup
setInterval(() => {
    window.svgGlobalManager.cleanup();
}, 30000);

// Observe tab changes for automatic reinitialization
const tabObserver = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
        if (mutation.type === 'attributes' && mutation.attributeName === 'class') {
            const target = mutation.target;
            if (target.classList.contains('.stTab') && target.getAttribute('aria-selected') === 'true') {
                setTimeout(() => {
                    window.svgGlobalManager.refreshAll();
                }, 500);
            }
        }
    });
});

// Start observation when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        const tabs = document.querySelectorAll('.stTab');
        tabs.forEach(tab => {
            tabObserver.observe(tab, { attributes: true });
        });
    });
} else {
    const tabs = document.querySelectorAll('.stTab');
    tabs.forEach(tab => {
        tabObserver.observe(tab, { attributes: true });
    });
}
</script>
""", unsafe_allow_html=True)

st.markdown(f"""
<style>
.minimal-hero {{
    text-align: center;
    padding: 2rem 0 1rem 0;
    border-bottom: 3px solid {PALETTE['ACCENT']};
    margin-bottom: 2rem;
}}
.minimal-title {{
    font-size: 2.4rem;
    font-weight: 800;
    color: {PALETTE['TEXT']};
    letter-spacing: -0.5px;
    margin-bottom: 0.5rem;
}}
.minimal-sub {{
    font-size: 1.1rem;
    color: {PALETTE['SOFT']};
    font-weight: 400;
}}
</style>

<div class="minimal-hero">
    <div class="minimal-title">
        🚀 Intelligent Process Analyzer
    </div>
    <div class="minimal-sub">
        Discover insights, optimize flows and continuously improve
    </div>
</div>
""", unsafe_allow_html=True)




# ------------------------------------------------------------
# Event log summary (only shown when file is uploaded)
# ------------------------------------------------------------
if is_backend_ready():
    # Read log to build highlights from DF (cached)
    try:
        _evlog_preview, _df_preview = read_event_log(uploaded)
    except Exception as e:
        _evlog_preview, _df_preview = None, None
        st.warning(f"Could not prepare summary from DF: {e}")

    with st.container():
        summaries = get_log_summaries_strict()
        # Base block: if there's agent summary, we show it
        if summaries:
            src, html_txt = summaries[0]
            # Add DF highlights to end of summary
            stats_html = build_log_stats_html_from_df(_df_preview)
            st.markdown(
                f'<div class="block"><b>Event log summary:</b><br>{html_txt}{stats_html}</div>',
                unsafe_allow_html=True
            )
        else:
            # No agent summary: show only DF highlights
            stats_html = build_log_stats_html_from_df(_df_preview)
            st.markdown(
                f'<div class="block"><b>Event log summary:</b><br>{stats_html}</div>',
                unsafe_allow_html=True
            )
    st.markdown("")
else:
    st.markdown(
        '<div class="block dashed">⏳ <b>Event log summary:</b><br>Upload a CSV or XES file to see the log summary.</div>',
        unsafe_allow_html=True
    )
    st.markdown("")





st.markdown("")

if not is_backend_ready():
    # === Only show Chat until CSV/XES is uploaded ===
    (chat_tab,) = st.tabs(["Chat"])

    with chat_tab:
        st.markdown("### Chat")
        
        
        st.markdown(
            '<div class="block" style="margin-bottom: 3rem;">This chat will save process information in a shared memory between agents, '
            'to use as <b>global context</b> for analysis.'
            '<br><b>To begin, upload your CSV or XES file.</b> When loaded, the other tabs will be enabled and '
            'I can respond using that context.</div>',
            unsafe_allow_html=True
        )
        if "msgs" not in st.session_state:
            st.session_state.msgs = []
        for role, text in st.session_state.msgs:
            with st.chat_message(role):
                st.write(text)
        prompt = st.chat_input("Write here (demo, no backend)")
        if prompt:
            st.session_state.msgs.append(("user", prompt))
            st.session_state.msgs.append(("assistant", "Before continuing: please upload your CSV or XES to load the process context."))
            st.rerun()

    st.markdown("")
    st.caption("When you upload a file the other tabs with visualizations and metrics will be unlocked.")

else:
    # === Unlock ALL tabs when there's a file ===
    tabs = st.tabs(["Petri net + BPMN", "DFG + Attributes", "Problem and recommendation report", "Chat"])

    # ============================================================
    # TAB: Petri net
    # ============================================================
    with tabs[0]:
        # --- State for Petri refresh (if it doesn't exist) ---
        if 'petri_refresh_counter' not in st.session_state:
            st.session_state.petri_refresh_counter = 0
        if 'petri_show_ids' not in st.session_state:
            st.session_state.petri_show_ids = False

        # --- Header with refresh button aligned to the right ---
        h1, h2 = st.columns([0.92, 0.08])
        with h1:
            st.markdown("### Petri net")
        with h2:
            if st.button("🔄", key=f"petri_refresh_top_{st.session_state.petri_refresh_counter}",
                        help="Refresh Petri net", use_container_width=True):
                st.session_state.petri_refresh_counter += 1
                st.rerun()

        # (rest of your logic as is)
        best_model = get_current_best_model()
        model_name = best_model.get("name","Unnamed model")
        model_type = detect_model_type(model_name)
        st.markdown(f"<b>Discovered model:</b> {model_type.capitalize()} miner", unsafe_allow_html=True)
        st.write("")

        net = best_model.get("net"); im = best_model.get("im"); fm = best_model.get("fm")

        # --- Controls (WITHOUT 🔄 button here) ---
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            show_ids = st.toggle(
                "Show technical ID next to name",
                value=st.session_state.petri_show_ids,
                key=f"petri_toggle_main_{st.session_state.petri_refresh_counter}"
            )
            if show_ids != st.session_state.petri_show_ids:
                st.session_state.petri_show_ids = show_ids
                st.session_state.petri_refresh_counter += 1
                st.rerun()

        if net:
            try:
                dot = build_petri_graph(net, im, fm, show_ids, PALETTE)
                instance_token = f"petri-{show_ids}-{PALETTE['MODE']}-{st.session_state.petri_refresh_counter}-{st.session_state.global_refresh_counter}"
                render_interactive_svg(
                    dot, PALETTE, height=410, show_legend=True,
                    legend_collapsed=True, instance_token=instance_token
                )
            except Exception as e:
                st.error(f"Could not visualize Petri net: {e}")
        else:
            st.warning("Petri net not found in loaded model.")


    
        st.write("")

        # ======== METRICS ========
        raw_metrics = extract_conformance_metrics(best_model)
        metrics = normalize_metrics_for_display(raw_metrics)

        st.markdown("""
        <div class="metrics-head">
        <div class="title">Conformance checking metrics</div>
        <div class="spacer"></div>
        <div class="help-badge">?
            <div class="tip">
            <p><b>Definitions:</b></p>
            <ul>
                <li><b>Precision</b>: measures how much <i>extra behavior</i> the model allows compared to the log. <code>1: nothing extra</code></li>
                <li><b>Fitness (log)</b>: global fit to log (token replay at log level). <code>1: perfect fit</code></li>
                <li><b>Fitness (average trace)</b>: average fit per case/trace. <code>1: perfect fit</code></li>
                <li><b>Generalization</b>: ability to explain valid variants without overfitting. <code>higher is better</code></li>
                <li><b>Simplicity</b>: preference for smaller/clearer models. <code>higher is better</code></li>
                <li><b>Conforming traces (0–1)</b>: fraction of cases that fully comply with model (previously in %, here normalized).</li>
            </ul>
            <p class="small">Calculation (PM4Py): <code>token_replay</code>, <code>precision.ETCONFORMANCE_TOKEN</code>, <code>generalization</code>, <code>simplicity</code>.</p>
            </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

        def render_metric_card(name: str, value):
            percent = _value_to_percent(value)
            try:
                label_val = f"{float(value):.3f}"
            except Exception:
                label_val = str(value)
            st.markdown(
                f"""
                <div class="metric-card">
                <div class="metric-name">{name}</div>
                <div class="metric-value">{label_val}</div>
                <div class="metric-bar"><span style="width:{percent:.2f}%"></span></div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.write("")

        desired = [
            "Precision",
            "Fitness (log)",
            "Fitness (average trace)",
            "Generalization",
            "Simplicity",
            "Conforming traces (0-1)",
        ]
        present = [k for k in desired if k in metrics]

        if present:
            row1 = st.columns(3, gap="small")
            for i, name in enumerate(present[:3]):
                with row1[i]:
                    render_metric_card(name, metrics[name])

            row2 = st.columns(3, gap="small")
            for i, name in enumerate(present[3:6]):
                with row2[i]:
                    render_metric_card(name, metrics[name])
        else:
            st.markdown('<div class="dashed">No conformance metrics found in model.</div>', unsafe_allow_html=True)

        st.write("")

        # -------- Why was this model chosen? (with formula help) --------
        st.markdown("""
        <div class="metrics-head">
        <div class="title">Why was this model chosen?</div>
        <div class="spacer"></div>
        <div class="help-badge">?
            <div class="tip">
            <p><b>Decision rule (S):</b></p>
            <ul>
                <li><code>S = 0.5*(Fitness + Precision) + 0.3*Generalization + 0.2*Simplicity</code></li>
                <li>Penalty for underfitting/permissiveness: if <code>(Fitness - Precision) &gt; 0.30</code>, subtract <code>0.15*((Fitness - Precision) - 0.30)</code></li>
            </ul>
            </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

        summary_text = get_output_summary_from_state(best_model)
        summary_fmt, table_html = format_summary_with_table(summary_text)

        if "[[SCORE_TABLE]]" in summary_fmt and table_html:
            before, after = summary_fmt.split("[[SCORE_TABLE]]", 1)
            if before.strip():
                st.markdown(before, unsafe_allow_html=True)
            st.markdown(table_html, unsafe_allow_html=True)
            if after.strip():
                st.markdown(after, unsafe_allow_html=True)
        else:
            if table_html:
                st.markdown(table_html, unsafe_allow_html=True)
            if summary_fmt:
                st.markdown(summary_fmt, unsafe_allow_html=True)
            else:
                st.markdown("_(No summary available in state)_", unsafe_allow_html=True)
     
    with tabs[1]:
        import plotly.graph_objects as go  # in case it wasn't imported above

        # --- Fallback to avoid KeyError ---
        if 'global_refresh_counter' not in st.session_state:
            st.session_state.global_refresh_counter = 0

        # --- Unified state for DFG + Chart refresh ---
        st.session_state.setdefault('dfg_local_refresh', 0)
        st.session_state.setdefault('dfg_min_freq', 1)
        st.session_state.setdefault('dfg_show_stats', False)

        # --- Header with ONE refresh button ---
        h1, h2 = st.columns([0.92, 0.08])
        with h1:
            st.markdown("### DFG")
        with h2:
            if st.button("🔄", key=f"dfg_refresh_top_{st.session_state.dfg_local_refresh}",
                        help="Refresh DFG and panel charts", use_container_width=True):
                st.session_state.dfg_local_refresh += 1
                st.rerun()

        st.write("")

        # Uploaded file
        _uploaded = globals().get("uploaded", st.session_state.get("uploaded", None))

        # ============================== DFG (Graphviz) ==============================
        try:
            evlog, df_log = read_event_log(_uploaded)
            dfg, starts, ends, act_freq, n_cases = discover_dfg_with_pm4py(evlog)

            max_f = max(list(dfg.values()) + list(starts.values()) + list(ends.values()) + [1])
            new_min_freq = st.slider(
                "Minimum edge frequency",
                1, int(max_f),
                value=st.session_state.dfg_min_freq,
                help="Filters infrequent paths and reduces crossings",
                key=f"dfg_min_freq_{st.session_state.global_refresh_counter}"
            )
            if new_min_freq != st.session_state.dfg_min_freq:
                st.session_state.dfg_min_freq = new_min_freq
                st.session_state.dfg_local_refresh += 1
                st.rerun()

            new_show_stats = st.toggle(
                "Show statistics per activity (avg, std, max, min)",
                value=st.session_state.dfg_show_stats,
                key=f"dfg_show_stats_{st.session_state.global_refresh_counter}"
            )
            if new_show_stats != st.session_state.dfg_show_stats:
                st.session_state.dfg_show_stats = new_show_stats
                st.session_state.dfg_local_refresh += 1
                st.rerun()

            act_order = compute_activity_order(df_log)
            act_stats = compute_activity_stats(df_log) if st.session_state.dfg_show_stats else {}

            dot_dfg = build_dfg_graphviz(
                dfg, starts, ends, act_freq, n_cases, PALETTE,
                min_freq=st.session_state.dfg_min_freq,
                act_order=act_order,
                act_stats=act_stats,
                show_stats=st.session_state.dfg_show_stats
            )

            token = (
                f"dfg-{st.session_state.dfg_show_stats}-"
                f"{st.session_state.dfg_min_freq}-"
                f"{PALETTE['MODE']}-{n_cases}-"
                f"{st.session_state.dfg_local_refresh}-"
                f"{st.session_state.global_refresh_counter}"
            )
            render_interactive_svg_dfg(
                dot_dfg, PALETTE, height=400, legend_collapsed=True,
                instance_token=token
            )
        except Exception as e:
            st.error(f"Could not build DFG: {e}")
            df_log = None


        st.markdown("##### 📈 Temporal Trend")

        try:
                if df_log is not None and 'time:timestamp' in df_log.columns:
                    
                    st.markdown('<div class="no-pill-radio">', unsafe_allow_html=True)
                    view_type = st.radio(
                        "Select view:",
                        ("Monthly", "Semester: Jan-Jun", "Semester: Jul-Dec"),
                        horizontal=True,
                        key=f"trend_view_{st.session_state.dfg_local_refresh}",
                    )
                    st.markdown('</div></div>', unsafe_allow_html=True)  # close radio and box

                    # Monthly view
                    if view_type == "Monthly":
                        available_months = get_available_months(df_log)
                        month_display_names = get_month_display_names(available_months)

                        if available_months:
                            ctrl = st.container()
                            with ctrl:
                                c1, c2 = st.columns([1, 1])
                                with c1:
                                    selected_month_display = st.selectbox(
                                        "Select Month:",
                                        options=month_display_names,
                                        index=0,
                                        key=f"month_selector_{st.session_state.dfg_local_refresh}",
                                        help="Daily trend of selected month"
                                    )
                                with c2:
                                    show_monthly_summary = st.toggle(
                                        "Show monthly summary",
                                        value=True,
                                        key=f"show_monthly_summary_{st.session_state.dfg_local_refresh}"
                                    )

                            selected_month_value = available_months[month_display_names.index(selected_month_display)]
                            trend_fig = create_monthly_trend_chart(
                                df_log, PALETTE, selected_month_value, show_summary=show_monthly_summary
                            )

                            st.plotly_chart(
                                trend_fig,
                                use_container_width=True,
                                config={
                                    'displayModeBar': True,
                                    'displaylogo': False,
                                    'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d', 'autoScale2d'],
                                    'scrollZoom': True,
                                    'responsive': True
                                },
                                key=f"monthly_trend_chart_{st.session_state.dfg_local_refresh}"
                            )
                        else:
                            st.warning("No months with valid timestamps found")

                    # Semester view
                    else:
                        semester = "January-June" if "Jan-Jun" in view_type else "July-December"
                        available_years = get_available_years(df_log)

                        if available_years:
                            c1, c2 = st.columns([1, 1])
                            with c1:
                                selected_year = st.selectbox(
                                    "Select Year:",
                                    options=available_years,
                                    index=0,
                                    key=f"year_selector_{st.session_state.dfg_local_refresh}",
                                    help="Semester analysis by year"
                                )
                            with c2:
                                show_semester_summary = st.toggle(
                                    "Show semester summary",
                                    value=True,
                                    key=f"show_semester_summary_{st.session_state.dfg_local_refresh}"
                                )
                        else:
                            selected_year = datetime.now().year
                            show_semester_summary = True
                            st.info("No specific years found; using current year")

                        semester_fig = create_semester_trend_chart(
                            df_log, PALETTE, semester, selected_year, show_summary=show_semester_summary
                        )
                        st.plotly_chart(
                            semester_fig,
                            use_container_width=True,
                            config={
                                'displayModeBar': True,
                                'displaylogo': False,
                                'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d', 'autoScale2d'],
                                'scrollZoom': True,
                                'responsive': True
                            },
                            key=f"semester_trend_chart_{st.session_state.dfg_local_refresh}"
                        )
                else:
                    st.warning("No data available for temporal trend")
        except Exception as e:
                st.error(f"Error generating temporal trend: {str(e)}")
     
        st.markdown("#### Panel: Activities, Resources/roles and Attributes")
        left_col, right_col = st.columns([0.52, 0.48], gap="large")

        # ----------------------------- LEFT: Activity Frequency ----------------
        with left_col:
            st.markdown("##### 📊 Activity Frequency")

            c1, c2 = st.columns([1, 1])
            with c1:
                chart_top_n = st.selectbox(
                    "Activities to show",
                    options=[5, 8, 15, 20],
                    index=1,
                    key=f"chart_top_n_{st.session_state.dfg_local_refresh}",
                )
            with c2:
                chart_show_all = st.checkbox(
                    "Show all activities",
                    value=False,
                    key=f"chart_show_all_{st.session_state.dfg_local_refresh}"
                )

            try:
                if df_log is not None and not df_log.empty:
                    ACTIVITY_COL = 'concept:name'
                    CASE_COL = 'case:concept:name'

                    if ACTIVITY_COL not in df_log.columns or CASE_COL not in df_log.columns:
                        st.warning("Log does not contain required columns 'concept:name' and 'case:concept:name'.")
                    else:
                        # Unique cases per activity
                        case_freq = (
                            df_log.groupby(ACTIVITY_COL)[CASE_COL]
                            .nunique()
                            .reset_index(name='Number of cases')
                            .rename(columns={ACTIVITY_COL: 'Activity'})
                        )

                        # Events per activity
                        event_freq = (
                            df_log[ACTIVITY_COL]
                            .value_counts()
                            .reset_index(name='Number of events')
                        )
                        event_freq.columns = ['Activity', 'Number of events']

                        # Merge and clean
                        chart_df = (
                            pd.merge(case_freq, event_freq, on='Activity', how='outer')
                            .fillna(0)
                        )
                        chart_df['Number of cases'] = chart_df['Number of cases'].astype(int)
                        chart_df['Number of events'] = chart_df['Number of events'].astype(int)

                        # Order and trim
                        chart_df = chart_df.sort_values('Number of cases', ascending=False)
                        if not chart_show_all and chart_top_n > 0:
                            chart_df = chart_df.head(chart_top_n)
                        chart_df = chart_df.sort_values('Number of cases', ascending=True)

                        total_cases = df_log[CASE_COL].nunique()
                        total_events = len(df_log)

                        # Compact header
                        st.markdown(f"""
                        <div style="border: 1px solid {PALETTE['BORDER']}; border-radius: 12px; padding: 0px; 
                                    background: {PALETTE['SURFACE']}; margin-bottom: 8px; overflow: hidden;">
                            <div style="background: {'rgba(15,23,42,0.94)' if PALETTE['MODE']=='dark' else 'rgba(255,255,255,0.96)'}; 
                                        padding: 10px 12px; border-bottom: 1px solid {PALETTE['BORDER']};
                                        display:flex; align-items:center; gap:8px;">
                                <span style="font-weight:700; color:{PALETTE['TEXT']}; font-size:13px;">
                                    Activity Frequency
                                </span>
                                <span style="flex:1;"></span>
                                <span style="color:{PALETTE['SOFT']}; font-size:12px;">
                                    {total_cases:,.0f} cases • {total_events:,.0f} events
                                </span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Chart
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            y=chart_df['Activity'],
                            x=chart_df['Number of cases'],
                            orientation='h',
                            marker=dict(
                                color=PALETTE['ACCENT'],
                                line=dict(color=PALETTE['ACCENT'], width=1)
                            ),
                            hovertemplate=(
                                "<b>%{y}</b><br>"
                                "Cases with this activity: %{x:,.0f}<br>"
                                "Total events: %{customdata:,.0f}<br>"
                                "<extra></extra>"
                            ),
                            customdata=chart_df['Number of events'],
                            name=''
                        ))
                        fig.update_layout(
                            height=400,
                            plot_bgcolor=PALETTE['SURFACE'],
                            paper_bgcolor=PALETTE['SURFACE'],
                            font=dict(color=PALETTE['TEXT'], size=12),
                            margin=dict(l=10, r=10, t=12, b=10),
                            showlegend=False,
                            xaxis=dict(
                                title='Cases containing this activity',
                                gridcolor=PALETTE['BORDER'],
                                gridwidth=1,
                                showline=True,
                                linecolor=PALETTE['BORDER'],
                                tickformat=',.0f'
                            ),
                            yaxis=dict(
                                title='',
                                gridcolor=PALETTE['BORDER'],
                                showline=True,
                                linecolor=PALETTE['BORDER'],
                                tickfont=dict(size=11)
                            ),
                            hoverlabel=dict(
                                bgcolor='rgba(15,23,42,0.9)' if PALETTE['MODE']=='dark' else 'rgba(255,255,255,0.95)',
                                font_size=12,
                                font_color=PALETTE['TEXT']
                            )
                        )
                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                            config={
                                'displayModeBar': True,
                                'displaylogo': False,
                                'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                                'scrollZoom': True
                            },
                            key=f"activity_chart_main_{st.session_state.dfg_local_refresh}"
                        )

                        # ---------------- Activity × Resource Heatmap ----------------
                        st.markdown("##### 🔥 Activity × Resource Heatmap")

                        try:
                            if (
                                df_log is not None and not df_log.empty and
                                'concept:name' in df_log.columns and
                                'org:resource' in df_log.columns
                            ):
                                ACTIVITY_COL = 'concept:name'
                                RESOURCE_COL = 'org:resource'
                                CASE_COL = 'case:concept:name'

                                # ---------------- Controls ----------------
                                c1, c2, c3 = st.columns([1, 1, 1])
                                with c1:
                                    hm_metric = st.selectbox(
                                        "Metric",
                                        options=["Events", "Cases"],
                                        index=0,
                                        key=f"hm_metric_{st.session_state.dfg_local_refresh}"
                                    )
                                with c2:
                                    hm_scale = st.selectbox(
                                        "Scale",
                                        options=["Counts", "% by resource"],   # << before was "per activity"
                                    
                                        index=1,                        # default: % by resource (your request)
                                        key=f"hm_scale_{st.session_state.dfg_local_refresh}"
                                    )
                                with c3:
                                    hm_show_zeros = st.checkbox(
                                        "Label zeros",
                                        value=False,
                                        key=f"hm_showzeros_{st.session_state.dfg_local_refresh}"
                                    )

                                # --- Base table (same as before) ---
                                if hm_metric == "Events":
                                    base = df_log.groupby([RESOURCE_COL, ACTIVITY_COL]).size().reset_index(name='value')
                                else:
                                    base = df_log.groupby([RESOURCE_COL, ACTIVITY_COL])[CASE_COL].nunique().reset_index(name='value')

                                all_resources = sorted(df_log[RESOURCE_COL].dropna().unique().tolist())
                                all_activities = sorted(df_log[ACTIVITY_COL].dropna().unique().tolist())

                                pivot = pd.DataFrame(0, index=all_resources, columns=all_activities, dtype=float)
                                if not base.empty:
                                    for _, r in base.iterrows():
                                        pivot.at[r[RESOURCE_COL], r[ACTIVITY_COL]] = float(r['value'])

                                # --- Normalization: % by RESOURCE (row) ---
                                cs = 'Blues' if PALETTE['MODE'] == 'light' else 'Viridis'
                                if hm_scale == "% by resource":
                                    row_sums = pivot.sum(axis=1).replace(0.0, 1.0)
                                    heat = (pivot.div(row_sums, axis=0)) * 100.0
                                    text_df = heat.round(0).astype(int).astype(str) + '%'
                                    cbar_title = "% within resource"
                                    zmax = 100
                                    hover_val = "%{z:.0f}%"
                                else:
                                    heat = pivot.copy()
                                    text_df = pivot.round(0).astype(int).astype(str)
                                    cbar_title = "Count"
                                    zmax = None
                                    hover_val = "%{z:.0f}"

                                # --- Figure (only changes colorscale and texts) ---
                                fig_hm = go.Figure(data=go.Heatmap(
                                    z=heat.values.tolist(),
                                    x=all_activities,
                                    y=all_resources,
                                    colorscale=cs,                     # << light/dark
                                    zmin=0,
                                    zmax=zmax,
                                    colorbar=dict(title=cbar_title),
                                    hovertemplate="Resource: %{y}<br>Activity: %{x}<br>Value: " + hover_val + "<extra></extra>",
                                    xgap=1, ygap=1,
                                    text=text_df.values,
                                    texttemplate="%{text}",
                                    textfont={"size": 10},
                                    showscale=True
                                ))

                                if not hm_show_zeros:
                                    fig_hm.data[0].text = text_df.mask(heat == 0.0, "").values


                                # ---------------- Aesthetics and fixed height ----------------
                                HEATMAP_HEIGHT = 400
                                def _clamp(v, lo, hi): 
                                    return max(lo, min(hi, v))
                                n_res = len(all_resources)
                                n_act = len(all_activities)
                                y_tick_size = _clamp(int(14 - 0.08 * n_res), 8, 12)
                                x_tick_size = _clamp(int(14 - 0.05 * n_act), 8, 12)

                                fig_hm.update_layout(
                                    height=HEATMAP_HEIGHT,
                                    margin=dict(l=6, r=6, t=4, b=4),
                                    plot_bgcolor=PALETTE['SURFACE'],
                                    paper_bgcolor=PALETTE['SURFACE'],
                                    font=dict(color=PALETTE['TEXT'], size=12),
                                )
                                fig_hm.update_xaxes(
                                    tickangle=45,
                                    tickfont=dict(size=x_tick_size),
                                    showgrid=False,
                                    automargin=True,
                                    linecolor=PALETTE['BORDER']
                                )
                                fig_hm.update_yaxes(
                                    autorange="reversed",
                                    tickfont=dict(size=y_tick_size),
                                    showgrid=False,
                                    automargin=True,
                                    linecolor=PALETTE['BORDER']
                                )

                                st.plotly_chart(
                                    fig_hm,
                                    use_container_width=True,
                                    config={
                                        'displayModeBar': True,
                                        'displaylogo': False,
                                        'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d', 'autoScale2d'],
                                        'scrollZoom': True,
                                        'responsive': True
                                    },
                                    key=f"hm_activity_resource_{st.session_state.dfg_local_refresh}"
                                )
                            else:
                                st.info("Heatmap requires columns 'concept:name' and 'org:resource' (and optionally 'case:concept:name').")
                        except Exception as e:
                            st.error(f"Could not build Activity×Resource heatmap: {e}")

                else:
                    st.warning("No data available for activity chart")
            except Exception as e:
                st.error(f"Could not create activity analysis: {e}")

            # In the "👥 Resources × Roles" section within the right column:

            try:
                # CHECK IF org:role COLUMN EXISTS BEFORE SHOWING HEATMAP
                if df_log is not None and 'org:role' in df_log.columns and 'org:resource' in df_log.columns:
                    st.markdown("##### 👥 Resources × Roles Heatmap")  # <-- THIS INSIDE THE IF

                    RESOURCE_COL = 'org:resource'
                    ROLE_COL = 'org:role'
                    CASE_COL = 'case:concept:name'

                    # Controls for heatmap
                    c1, c2, c3 = st.columns([1, 1, 1])
                    with c1:
                        rr_metric = st.selectbox(
                            "Metric",
                            options=["Events", "Cases"],
                            index=0,
                            key=f"rr_metric_{st.session_state.dfg_local_refresh}"
                        )
                    with c2:
                        rr_norm = st.selectbox(
                            "Normalization",
                            options=["% by role", "% by resource", "Counts"],
                            index=0,
                            key=f"rr_norm_{st.session_state.dfg_local_refresh}"
                        )
                    with c3:
                        rr_show_zeros = st.checkbox(
                            "Label zeros",
                            value=False,
                            key=f"rr_zeros_{st.session_state.dfg_local_refresh}"
                        )

                    # Logic for cases vs events metric
                    if rr_metric == "Cases" and CASE_COL not in df_log.columns:
                        st.info("'case:concept:name' not found. Will use 'Events' metric.")
                        rr_metric = "Events"

                    if rr_metric == "Events":
                        base = df_log.groupby([RESOURCE_COL, ROLE_COL]).size().reset_index(name='value')
                    else:
                        base = df_log.groupby([RESOURCE_COL, ROLE_COL])[CASE_COL].nunique().reset_index(name='value')

                    all_resources = sorted(df_log[RESOURCE_COL].dropna().unique().tolist())
                    all_roles = sorted(df_log[ROLE_COL].dropna().unique().tolist())

                    # Create matrix with ROLES in rows (Y) and RESOURCES in columns (X)
                    pivot = pd.DataFrame(0, index=all_roles, columns=all_resources, dtype=float)
                    if not base.empty:
                        for _, r in base.iterrows():
                            pivot.at[r[ROLE_COL], r[RESOURCE_COL]] = float(r['value'])

                    # Apply normalization according to selection
                    cs = 'Blues' if PALETTE['MODE'] == 'light' else 'Viridis'
                    heat = pivot.copy()
                    cbar_title = "Count"
                    hover_val = "%{z:.0f}"
                    zmax = None

                    if rr_norm == "% by role":  # Normalize by rows (roles)
                        row_sums = heat.sum(axis=1).replace(0.0, 1.0)
                        heat = (heat.div(row_sums, axis=0)) * 100.0
                        cbar_title = "% within role"
                        hover_val = "%{z:.0f}%"
                        zmax = 100
                    elif rr_norm == "% by resource":  # Normalize by columns (resources)
                        col_sums = heat.sum(axis=0).replace(0.0, 1.0)
                        heat = (heat.div(col_sums, axis=1)) * 100.0
                        cbar_title = "% within resource"
                        hover_val = "%{z:.0f}%"
                        zmax = 100

                    # Prepare text for display
                    text_df = heat.round(0).astype(int).astype(str) + ('' if zmax is None else '%')

                    # Create heatmap with ROLES in Y and RESOURCES in X
                    fig_rr = go.Figure(data=go.Heatmap(
                        z=heat.values.tolist(),
                        x=all_resources,  # X axis: Resources
                        y=all_roles,      # Y axis: Roles (left side)
                        colorscale=cs,
                        zmin=0,
                        zmax=zmax,
                        colorbar=dict(title=cbar_title),
                        hovertemplate="Role: %{y}<br>Resource: %{x}<br>Value: " + hover_val + "<extra></extra>",
                        xgap=1, ygap=1,
                        text=text_df.values,
                        texttemplate="%{text}",
                        textfont={"size": 10},
                        showscale=True
                    ))

                    # Hide text in cells with value 0 if not requested to show zeros
                    if not rr_show_zeros:
                        fig_rr.data[0].text = text_df.mask(heat == 0.0, "").values

                    # Adjust font sizes according to number of elements
                    def _clamp(v, lo, hi): 
                        return max(lo, min(hi, v))
                    
                    n_res = len(all_resources)
                    n_roles = len(all_roles)
                    y_tick_size = _clamp(int(14 - 0.08 * n_roles), 8, 12)
                    x_tick_size = _clamp(int(14 - 0.05 * n_res), 8, 12)

                    # Configure layout
                    fig_rr.update_layout(
                        height=400,
                        margin=dict(l=6, r=6, t=4, b=4),
                        plot_bgcolor=PALETTE['SURFACE'],
                        paper_bgcolor=PALETTE['SURFACE'],
                        font=dict(color=PALETTE['TEXT'], size=12),
                    )
                    fig_rr.update_xaxes(
                        tickangle=45,
                        tickfont=dict(size=x_tick_size),
                        showgrid=False,
                        automargin=True,
                        linecolor=PALETTE['BORDER'],
                        title="Resource"
                    )
                    fig_rr.update_yaxes(
                        autorange="reversed",
                        tickfont=dict(size=y_tick_size),
                        showgrid=False,
                        automargin=True,
                        linecolor=PALETTE['BORDER'],
                        title="Role"
                    )

                    # Show chart
                    st.plotly_chart(
                        fig_rr,
                        use_container_width=True,
                        config={
                            "displayModeBar": True,
                            "displaylogo": False,
                            "modeBarButtonsToRemove": ["pan2d", "lasso2d", "select2d", "autoScale2d"],
                            "scrollZoom": True,
                            "responsive": True,
                        },
                        key=f"rr_heatmap_{st.session_state.dfg_local_refresh}"
                    )
                else:
                    # Only show message if org:role column doesn't exist
                    if df_log is not None and 'org:role' not in df_log.columns:
                        st.info("Column 'org:role' not found in data. Resources × Roles heatmap not available.")
                    elif df_log is not None and 'org:resource' not in df_log.columns:
                        st.info("Column 'org:resource' not found in data. Resources × Roles heatmap not available.")
                        
            except Exception as e:
                st.error(f"Could not build Resources×Roles heatmap: {e}")

        # ----------------------------- RIGHT: Temporal Trend --------------------------
        with right_col:
            st.markdown("### 📊 Attributes Summary")

            try:
                a3 = get_current_agent3()
                attrs = a3.get("attributes", [])
                
                if attrs:
                    # Create container with better formatting
                    st.markdown("""
                    <div style="background: var(--card); border: 1px solid var(--border); 
                                border-radius: 12px; padding: 1rem; margin: 0.5rem 0;">
                    """, unsafe_allow_html=True)
                    
                    for attr in attrs:
                        name = attr.get('name', '')
                        summary = attr.get('summary', '')
                        
                        if name and summary:
                            # Improve formatting of each attribute
                            st.markdown(f"""
                            <div style="margin-bottom: 1rem; padding-bottom: 0.8rem; 
                                        border-bottom: 1px solid var(--border);">
                                <div style="font-weight: 700; color: var(--accent); 
                                            font-size: 1.05rem; margin-bottom: 0.3rem;">
                                    🏷️ {name}
                                </div>
                                <div style="color: var(--text); line-height: 1.4; 
                                            font-size: 0.95rem; padding-left: 0.5rem;">
                                    {summary}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Add quick statistics
                    total_attrs = len(attrs)
                    st.caption(f"📋 {total_attrs} process attributes analyzed")
                    
                else:
                    st.markdown("""
                    <div class="block dashed">
                        No attributes found to display in analysis.
                    </div>
                    """, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Error loading attributes: {str(e)}")
                st.markdown("""
                <div class="block dashed">
                    Could not load attribute summary from analysis.
                </div>
                """, unsafe_allow_html=True)
            

    with tabs[2]:
        st.markdown("### Problem and recommendation report")
        st.info("👉 Click any cell in the row to see complete problem and solution details.")

        pairs = get_agent8_problem_pairs()

        if not pairs:
            st.warning("No Problem–Solution pairs found. Check state or format.")
        else:
            # Create DataFrame only with Problem and Solution (justification is hidden)
            df_tbl = pd.DataFrame(pairs, columns=["Problem", "Solution"])
            
            # Get complete agent8 data for details
            agent8_items = get_current_agent8()

            # Click handler for any cell in the row
            on_cell_clicked = JsCode(
                """function(params){ 
                    try{ 
                        // Mark this row as selected
                        params.node.setDataValue('_selected', true);
                        // Clear other rows
                        params.api.forEachNode(function(node) {
                            if (node !== params.node) {
                                node.setDataValue('_selected', false);
                            }
                        });
                    } catch(err){} 
                }"""
            )

            gb = GridOptionsBuilder.from_dataframe(df_tbl)

            # Main configuration
            gb.configure_pagination(enabled=False)
            
            gb.configure_grid_options(
                domLayout="normal",
                onCellClicked=on_cell_clicked,
                suppressScrollOnNewData=True,
                headerHeight=48,
                getRowHeight=JsCode("""
                    function(params) {
                        const MIN_ROW_HEIGHT = 80;
                        const LINE_HEIGHT = 18;
                        const HORIZONTAL_PADDING = 16;
                        
                        if (!params.data) return MIN_ROW_HEIGHT;
                        
                        let availableWidth = 800;
                        try {
                            if (params.api && params.api.gridCore && params.api.gridCore.gridOptionsWrapper) {
                                const gridElement = params.api.gridCore.gridOptionsWrapper.getGui();
                                if (gridElement && gridElement.clientWidth) {
                                    availableWidth = gridElement.clientWidth;
                                }
                            }
                        } catch (err) {
                            console.log('Could not get grid width, using default value');
                        }
                        
                        const columnWidth = (availableWidth - HORIZONTAL_PADDING * 2) / 2;
                        
                        const probText = params.data['Problem'] ? String(params.data['Problem']) : '';
                        const solText = params.data['Solution'] ? String(params.data['Solution']) : '';
                        
                        function calculateLines(text, availableWidth) {
                            if (!text) return 1;
                            const avgCharWidth = 8.5;
                            const charsPerLine = Math.max(1, Math.floor(availableWidth / avgCharWidth));
                            const lines = Math.ceil(text.length / charsPerLine);
                            return Math.max(1, lines);
                        }
                        
                        const probLines = calculateLines(probText, columnWidth);
                        const solLines = calculateLines(solText, columnWidth);
                        
                        const maxLines = Math.max(probLines, solLines);
                        const calculatedHeight = Math.max(MIN_ROW_HEIGHT, (maxLines * LINE_HEIGHT) + 24);
                        
                        return calculatedHeight;
                    }
                """),
                onGridSizeChanged=JsCode("""
                    function(params) {
                        if (params.api) {
                            setTimeout(() => {
                                params.api.resetRowHeights();
                                params.api.sizeColumnsToFit();
                            }, 50);
                        }
                    }
                """),
                onFirstDataRendered=JsCode("""
                    function(params) {
                        setTimeout(() => {
                            params.api.resetRowHeights();
                            params.api.sizeColumnsToFit();
                        }, 200);
                    }
                """)
            )

            # Column configuration
            gb.configure_default_column(
                resizable=True,
                wrapText=True,
                autoHeight=False,
                cellStyle={
                    'white-space': 'normal',
                    'line-height': '1.4',
                    'padding': '10px',
                    'fontSize': '15px',
                    'word-wrap': 'break-word'
                }
            )

            gb.configure_column("Problem", 
                            header_name="Problem", 
                            minWidth=300,
                            flex=1.5,
                            autoHeight=True,
                            tooltipField="Problem",
                            cellStyle=JsCode("""
                                function(params) {
                                    if (params.data._selected) {
                                        return {
                                            'white-space': 'normal',
                                            'line-height': '1.4',
                                            'padding': '10px',
                                            'fontSize': '15px',
                                            'backgroundColor': '#3b82f6',
                                            'color': 'white',
                                            'fontWeight': 'bold',
                                            'word-wrap': 'break-word'
                                        };
                                    }
                                    return {
                                        'white-space': 'normal',
                                        'line-height': '1.4',
                                        'padding': '10px',
                                        'fontSize': '15px',
                                        'word-wrap': 'break-word'
                                    };
                                }
                            """))

            gb.configure_column("Solution", 
                            header_name="Solution", 
                            minWidth=300,
                            flex=1.5,
                            autoHeight=True,
                            tooltipField="Solution",
                            cellStyle=JsCode("""
                                function(params) {
                                    if (params.data._selected) {
                                        return {
                                            'white-space': 'normal',
                                            'line-height': '1.4',
                                            'padding': '10px',
                                            'fontSize': '15px',
                                            'backgroundColor': '#3b82f6',
                                            'color': 'white',
                                            'fontWeight': 'bold',
                                            'word-wrap': 'break-word'
                                        };
                                    }
                                    return {
                                        'white-space': 'normal',
                                        'line-height': '1.4',
                                        'padding': '10px',
                                        'fontSize': '15px',
                                        'word-wrap': 'break-word'
                                    };
                                }
                            """))

            # Add hidden column for selection state
            gb.configure_column("_selected", 
                            hide=True, 
                            editable=True)

            # Theme palettes
            LIGHT_PALETTE = {
                'BACKGROUND': '#ffffff',
                'HEADER_BG': 'rgba(37,99,235,0.08)',
                'HEADER_TEXT': '#1f2937',
                'BORDER': '#e5e7eb',
                'ACCENT': '#2563eb',
                'CELL_BG': '#ffffff',
                'CELL_TEXT': '#374151',
                'HOVER_BG': 'rgba(37,99,235,0.06)',
                'ROW_ALT_BG': '#f9fafb'
            }

            DARK_PALETTE = {
                'BACKGROUND': '#1a1a1a',
                'HEADER_BG': 'rgba(59, 130, 246, 0.25)',
                'HEADER_TEXT': '#ffffff',
                'BORDER': '#374151',
                'ACCENT': '#3b82f6',
                'CELL_BG': '#1f2937',
                'CELL_TEXT': '#f8fafc',
                'HOVER_BG': 'rgba(59, 130, 246, 0.15)',
                'ROW_ALT_BG': '#111827'
            }

            THEME_PALETTE = DARK_PALETTE if PALETTE["MODE"] == "dark" else LIGHT_PALETTE

            custom_css = {
                ".ag-root-wrapper": {
                    "border": f"1px solid {THEME_PALETTE['BORDER']}",
                    "border-radius": "12px",
                    "width": "100%",
                    "background-color": THEME_PALETTE['BACKGROUND'],
                    "color": THEME_PALETTE['CELL_TEXT'],
                    "overflow": "hidden",
                },
                ".ag-header": {
                    "background-color": THEME_PALETTE['HEADER_BG'],
                    "border-bottom": f"2px solid {THEME_PALETTE['ACCENT']}",
                    "color": THEME_PALETTE['HEADER_TEXT']
                },
                ".ag-header-cell-text": {
                    "font-weight": "700",
                    "font-size": "15.5px",
                    "color": THEME_PALETTE['HEADER_TEXT']
                },
                ".ag-header-viewport": { 
                    "border-left": "0",
                },
                ".ag-header-cell": { 
                    "border-left": "0",
                    "color": THEME_PALETTE['HEADER_TEXT']
                },
                ".ag-row": {
                    "background-color": THEME_PALETTE['CELL_BG'],
                    "color": THEME_PALETTE['CELL_TEXT'],
                    "border-bottom": f"1px solid {THEME_PALETTE['BORDER']}",
                    "cursor": "pointer"
                },
                ".ag-row-odd": {
                    "background-color": THEME_PALETTE['ROW_ALT_BG'],
                    "color": THEME_PALETTE['CELL_TEXT']
                },
                ".ag-row-hover": {
                    "background-color": f"{THEME_PALETTE['HOVER_BG']} !important"
                },
                ".ag-cell": {
                    "font-size": "0.95rem",
                    "white-space": "normal !important",
                    "line-height": "1.5 !important",
                    "padding": "10px 7px !important",
                    "border-left": "0 !important",
                    "overflow-wrap": "break-word !important",
                    "word-wrap": "break-word !important",
                    "color": f"{THEME_PALETTE['CELL_TEXT']} !important",
                    "background-color": "transparent !important",
                    "border-right": f"1px solid {THEME_PALETTE['BORDER']} !important",
                    "border-bottom": f"1px solid {THEME_PALETTE['BORDER']} !important"
                },
                ".ag-row .ag-cell:first-child": { 
                    "border-left": "0 !important" 
                },
                ".ag-body-viewport": {
                    "overflow-y": "auto !important",
                    "overflow-x": "hidden !important",
                    "border-left": "0",
                    "background-color": THEME_PALETTE['BACKGROUND']
                },
                ".ag-body-horizontal-scroll": {
                    "display": "none !important"
                },
                ".ag-center-cols-viewport": {
                    "overflow-x": "hidden !important",
                    "width": "100% !important",
                    "background-color": THEME_PALETTE['BACKGROUND']
                },
                ".ag-header-row": {
                    "color": f"{THEME_PALETTE['HEADER_TEXT']} !important"
                },
                ".ag-cell-value": {
                    "color": f"{THEME_PALETTE['CELL_TEXT']} !important"
                }
            }

            # Add _selected column to DataFrame
            df_tbl_with_selection = df_tbl.copy()
            df_tbl_with_selection['_selected'] = False

            # DEFINE grid here
            grid = AgGrid(
                df_tbl_with_selection,
                gridOptions=gb.build(),
                theme="streamlit",
                height=520,
                fit_columns_on_grid_load=True,
                allow_unsafe_jscode=True,
                update_mode=GridUpdateMode.VALUE_CHANGED,
                custom_css=custom_css,
                reload_data=True
            )

            # Show details when row is selected
            df_after = pd.DataFrame(grid["data"])
            selected_rows = df_after[df_after['_selected'] == True]

            if not selected_rows.empty:
                selected_index = selected_rows.index[0]
                
                try:
                    selected_index_int = int(selected_index)
                except (ValueError, TypeError):
                    selected_index_int = -1
                
                if 0 <= selected_index_int < len(pairs):
                    selected_pair = pairs[selected_index_int]
                    
                    # Show problem justification
                    st.markdown("#### 📋 Detailed Problem Justification")
                    problem_justification = selected_pair.get('_justification', 'Not available')
                    
                    if problem_justification and problem_justification != 'Not available':
                        if PALETTE["MODE"] == "dark":
                            st.markdown(f"""
                            <div style="
                                background: #78350f; 
                                color: #fef3c7; 
                                padding: 1rem; 
                                border-radius: 0.5rem; 
                                border-left: 4px solid #f59e0b;
                                margin: 0.5rem 0;
                            ">
                                {problem_justification}
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.warning(problem_justification)
                    else:
                        st.info("No detailed justification found for this problem.")
                    
                    # Find corresponding item in agent8_items
                    if selected_index_int < len(agent8_items):
                        selected_item = agent8_items[selected_index_int]
                        
                        if selected_item:
                            # Show solution
                            st.markdown("#### 💡 Solution")
                            solution_text = selected_item.get('recommendation', 'Not available')
                            
                            if PALETTE["MODE"] == "dark":
                                st.markdown(f"""
                                <div style="
                                    background: #1e3a8a; 
                                    color: #e1e7f0; 
                                    padding: 1rem; 
                                    border-radius: 0.5rem; 
                                    border-left: 4px solid #3b82f6;
                                    margin: 0.5rem 0;
                                ">
                                    {solution_text}
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.info(solution_text)
                            
                            # Show solution justification if exists
                            solution_justification = selected_item.get('justification_of_recommendation')
                            if solution_justification:
                                st.markdown("#### ❓ Why is this a solution?")
                                if PALETTE["MODE"] == "dark":
                                    st.markdown(f"""
                                    <div style="
                                        background: #065f46; 
                                        color: #d1fae5; 
                                        padding: 1rem; 
                                        border-radius: 0.5rem; 
                                        border-left: 4px solid #10b981;
                                        margin: 0.5rem 0;
                                    ">
                                        {solution_justification}
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.success(solution_justification)
                        
                     
                        # Handle scenario_analysis safely
                        scenario_analysis = selected_item.get('scenario_analysis') if selected_item else {}
                        if scenario_analysis is None:
                            scenario_analysis = {}
                        
                        # Show qualitative analysis if exists
                        qualitative_analysis = scenario_analysis.get('qualitative_analysis')
                        if qualitative_analysis:
                            st.markdown("#### 📊 Qualitative Analysis")
                            if PALETTE["MODE"] == "dark":
                                st.markdown(f"""
                                <div style="
                                    background: #78350f; 
                                    color: #fef3c7; 
                                    padding: 1rem; 
                                    border-radius: 0.5rem; 
                                    border-left: 4px solid #f59e0b;
                                    margin: 0.5rem 0;
                                ">
                                    {qualitative_analysis}
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.warning(qualitative_analysis)
                        
                        # Show quantitative analysis if exists
                        quantitative_analysis = scenario_analysis.get('quantitative_analysis')
                        if quantitative_analysis:
                            st.markdown("#### 📈 Quantitative Analysis")
                            if PALETTE["MODE"] == "dark":
                                st.markdown(f"""
                                <div style="
                                    background: #1e3a8a; 
                                    color: #e1e7f0; 
                                    padding: 1rem; 
                                    border-radius: 0.5rem; 
                                    border-left: 4px solid #3b82f6;
                                    margin: 0.5rem 0;
                                ">
                                    {quantitative_analysis}
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.info(quantitative_analysis)
                        
                        
                        # Show alternative solutions if they exist
                        alternative_solutions = scenario_analysis.get('alternative_solutions')
                        if alternative_solutions:
                            st.markdown("#### 🔄 Alternative Solutions")
                            for i, alt_solution in enumerate(alternative_solutions, 1):
                                if PALETTE["MODE"] == "dark":
                                    st.markdown(f"""
                                    <div style="
                                        background: #374151; 
                                        color: #f3f4f6; 
                                        padding: 0.75rem; 
                                        border-radius: 0.375rem; 
                                        margin: 0.25rem 0;
                                        border-left: 3px solid #6b7280;
                                    ">
                                        {i}. {alt_solution}
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.write(f"{i}. {alt_solution}")
                        
                else:
                    st.warning("Solution index out of range.")

    
    with tabs[3]:
        st.markdown("### 💬 Intelligent Chat with Context")

        # State
        if "enhanced_chat_messages" not in st.session_state:
            st.session_state.enhanced_chat_messages = []
        if "conversation_id" not in st.session_state:
            st.session_state.conversation_id = str(uuid.uuid4())

        # === CHAT FEED (top) ===
        feed = st.container()        # Fixed container for entire history
        with feed:
            # Existing history
            for msg in st.session_state.enhanced_chat_messages:
                with st.chat_message(msg["role"]):
                    if PALETTE["MODE"] == "dark":
                        st.markdown(f'<div style="color: white;">{msg["content"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(msg["content"])
                    
            # Slot where "thinking..." and this turn's response will be drawn
            live_slot = st.empty()

        # === INPUT (always bottom) ===
        prompt = st.chat_input("Ask a question about the presented report ...")

        if prompt:
            # Add user message to feed (top)
            st.session_state.enhanced_chat_messages.append({"role": "user", "content": prompt})

            with feed:
                with st.chat_message("user"):
                    st.markdown(f'<div style="color:{PALETTE["TEXT"]};">{prompt}</div>', unsafe_allow_html=True)
                with st.chat_message("assistant"):
                    spinner_placeholder = st.empty()
                    spinner_placeholder.markdown("🟦 Searching in books, case studies and your process...")

            # === Backend call ===
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/chat/query-enhanced",
                    json={
                        "message": prompt,
                        "domain": st.session_state.selected_domain,
                        "client_id": CLIENT_ID,
                        "conversation_id": st.session_state.conversation_id,
                        "request_id": st.session_state.get("request_id", "default"),
                    },
                    timeout=60,
                )
                if resp.status_code != 200:
                    raise RuntimeError(f"Server error: {resp.status_code}")

                data = resp.json()
                raw_response = data.get("response", "")
                
                # SPECIFIC HANDLING FOR LIST OF DICTIONARIES FORMAT
                answer = ""
                try:
                    # Case 1: If it's a list of dictionaries (like [{'type': 'text', 'text': '...'}])
                    if isinstance(raw_response, list):
                        # Extract all texts from dictionaries in the list
                        text_parts = []
                        for item in raw_response:
                            if isinstance(item, dict) and item.get('type') == 'text' and 'text' in item:
                                text_parts.append(item['text'])
                            elif isinstance(item, dict) and 'text' in item:
                                text_parts.append(item['text'])
                            elif isinstance(item, str):
                                text_parts.append(item)
                        
                        if text_parts:
                            answer = '\n'.join(text_parts)
                        else:
                            answer = str(raw_response)
                    
                    # Case 2: If it's a string, try to parse as JSON
                    elif isinstance(raw_response, str):
                        try:
                            parsed_json = json.loads(raw_response)
                            if isinstance(parsed_json, list):
                                # Handle same list case
                                text_parts = []
                                for item in parsed_json:
                                    if isinstance(item, dict) and item.get('type') == 'text' and 'text' in item:
                                        text_parts.append(item['text'])
                                    elif isinstance(item, dict) and 'text' in item:
                                        text_parts.append(item['text'])
                                    elif isinstance(item, str):
                                        text_parts.append(item)
                                
                                if text_parts:
                                    answer = '\n'.join(text_parts)
                                else:
                                    answer = str(parsed_json)
                            elif isinstance(parsed_json, dict):
                                answer = parsed_json.get("text", raw_response)
                            else:
                                answer = str(parsed_json)
                        except (json.JSONDecodeError, AttributeError):
                            # If JSON parsing fails, use string directly
                            answer = raw_response
                    
                    # Case 3: If already a dictionary
                    elif isinstance(raw_response, dict):
                        answer = raw_response.get("text", str(raw_response))
                    
                    # Case 4: Any other type
                    else:
                        answer = str(raw_response) if raw_response else "Could not get a response."
                        
                except Exception as parse_error:
                    # Safe fallback if everything fails
                    answer = f"Response received: {str(raw_response)[:500]}..." if raw_response else "Could not get a response."
                    
                evidence = data.get("evidence", [])
                context_info = data.get("context_used", {})

                # Paint response IN THE FEED, using live slot
                with feed:
                    if PALETTE["MODE"] == "dark":
                        spinner_placeholder.markdown(f'<div style="color: white;">{answer}</div>', unsafe_allow_html=True)
                    else:
                        spinner_placeholder.markdown(answer)
                        
                    if context_info:
                        st.caption(f"📚 Search: {context_info.get('collections_with_results',0)}/3 collections, {context_info.get('total_evidence_found',0)} evidences")

                # Persist in history
                st.session_state.enhanced_chat_messages.append({
                    "role": "assistant",
                    "content": answer,
                    "evidence": evidence,
                    "context_info": context_info,
                })

            except Exception as e:
                err = f"🔌 Error: {e}"
                with feed:
                    spinner_placeholder.markdown(err)
                st.session_state.enhanced_chat_messages.append({"role": "assistant", "content": err})

            # Rerender so complete feed stays top and input bottom
            st.rerun()
  
    st.markdown(f"""
    <style>
    /* ====== Main container: occupies viewport and scrolls ====== */
    [data-testid="stAppViewContainer"] {{
        height: 100vh;
        overflow-y: auto;
        overscroll-behavior: contain;
    }}

    /* Prevents last message from being covered by input */
    .main .block-container {{
        padding-bottom: 7.5rem; /* adjust if you change input height */
    }}

    /* ====== ChatInput stuck bottom, no line or shadow ====== */
    [data-testid="stChatInput"] {{
        position: sticky;
        bottom: 0;
        background: {PALETTE['BG']};
        padding: .75rem 0;
        margin: 0;
        border-top: none !important;   /* remove line */
        box-shadow: none !important;   /* in case theme adds shadow */
        z-index: 100;
    }}
    /* In case Streamlit inserts <hr> or Divider inside input */
    [data-testid="stChatInput"] hr,
    [data-testid="stChatInput"] [data-testid="stDivider"] {{
        display: none !important;
    }}

    /* ====== (Optional) If you wrap messages in #chat-body ======
    Only feed scrolls and input stays fixed */
    #chat-body {{
        max-height: calc(100vh - 8.5rem); /* header + input approx */
        overflow-y: auto;
        padding-right: .25rem;
    }}

    /* Chat messages: spacing and consistent color */
    .stChatMessage {{
        margin-bottom: 1rem;
    }}
    .stChatMessage div {{
        color: {PALETTE['TEXT']} !important;
    }}

    /* Expanders and auxiliary texts readable in both themes */
    .streamlit-expanderContent {{
        color: {PALETTE['TEXT']} !important;
    }}
    .stCaption {{
        color: {PALETTE['SOFT']} !important;
    }}

    /* Smooth scroll adjustment */
    html {{
        scroll-behavior: smooth;
    }}
    </style>
    """, unsafe_allow_html=True)
