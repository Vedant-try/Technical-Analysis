# TA1.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from datetime import date, timedelta

st.set_page_config(layout="wide")
st.title("📊 Simple Technical Analysis App (pandas-based, stable)")

# Sidebar controls
st.sidebar.header("Indicator & Range Options")
show_sma = st.sidebar.checkbox("Show SMA (window)", True)
sma_window = st.sidebar.number_input("SMA window", min_value=2, max_value=200, value=20, step=1)

show_ema = st.sidebar.checkbox("Show EMA (span)", True)
ema_span = st.sidebar.number_input("EMA span", min_value=2, max_value=200, value=20, step=1)

show_boll = st.sidebar.checkbox("Show Bollinger Bands", True)
bb_window = st.sidebar.number_input("BB window", min_value=2, max_value=200, value=20, step=1)
bb_stddev = st.sidebar.number_input("BB std devs", min_value=1.0, max_value=5.0, value=2.0, step=0.1)

show_rsi = st.sidebar.checkbox("Show RSI", True)
rsi_length = st.sidebar.number_input("RSI length", min_value=2, max_value=200, value=14, step=1)

show_macd = st.sidebar.checkbox("Show MACD", True)
macd_fast = st.sidebar.number_input("MACD fast EMA", min_value=2, max_value=50, value=12, step=1)
macd_slow = st.sidebar.number_input("MACD slow EMA", min_value=3, max_value=200, value=26, step=1)
macd_signal = st.sidebar.number_input("MACD signal EMA", min_value=1, max_value=50, value=9, step=1)

# Date range
today = date.today()
default_start = today - timedelta(days=180)
start_date = st.sidebar.date_input("Start date", default_start)
end_date = st.sidebar.date_input("End date", today)
if start_date >= end_date:
    st.sidebar.error("Start date must be before end date")

# Ticker input
ticker = st.text_input("Enter Stock Ticker (e.g. AAPL, TSLA, RELIANCE.BO):", "AAPL").strip().upper()

def safe_series(df, col):
    """Return a 1-D Series with no NaNs for plotting, or None if not available."""
    if col in df.columns:
        s = df[col].dropna()
        if not s.empty:
            return s
    return None

if ticker and start_date < end_date:
    with st.spinner(f"Downloading {ticker} data..."):
        try:
            # yfinance: end is exclusive, add one day to include end_date
            yf_start = start_date.strftime("%Y-%m-%d")
            yf_end = (end_date + timedelta(days=1)).strftime("%Y-%m-%d")
            data = yf.download(ticker, start=yf_start, end=yf_end, progress=False)
        except Exception as e:
            st.error(f"Error downloading data: {e}")
            st.stop()

    if data is None or data.empty:
        st.error("No data found for that ticker / date range.")
        st.stop()

    data.index = pd.to_datetime(data.index)
    st.write(f"### Showing latest rows for {ticker}")
    st.dataframe(data.tail())

    # Create a working copy
    df = data.copy()

    # --- Indicators computed with pandas/numpy (robust, returns 1-D arrays) ---
    # SMA
    if show_sma:
        if len(df) >= sma_window:
            df[f"SMA_{sma_window}"] = df["Close"].rolling(window=sma_window, min_periods=1).mean()
        else:
            st.warning(f"Not enough rows to compute SMA({sma_window}). Need at least {sma_window} rows.")

    # EMA (using pandas ewm)
    if show_ema:
        if len(df) >= ema_span:
            df[f"EMA_{ema_span}"] = df["Close"].ewm(span=ema_span, adjust=False).mean()
        else:
            st.warning(f"Not enough rows to compute EMA({ema_span}). Need at least {ema_span} rows.")

    # Bollinger Bands
    if show_boll:
        if len(df) >= bb_window:
            middle = df["Close"].rolling(window=bb_window, min_periods=1).mean()
            std = df["Close"].rolling(window=bb_window, min_periods=1).std(ddof=0)
            df[f"BB_upper_{bb_window}"] = middle + bb_stddev * std
            df[f"BB_lower_{bb_window}"] = middle - bb_stddev * std
        else:
            st.warning(f"Not enough rows to compute Bollinger Bands (window {bb_window}).")

    # RSI (Wilder smoothing via ewm)
    if show_rsi:
        if len(df) >= rsi_length + 1:
            delta = df["Close"].diff()
            gain = delta.clip(lower=0.0)
            loss = -delta.clip(upper=0.0)
            # Wilder smoothing using alpha = 1/length (adjust=False)
            avg_gain = gain.ewm(alpha=1.0 / rsi_length, adjust=False, min_periods=rsi_length).mean()
            avg_loss = loss.ewm(alpha=1.0 / rsi_length, adjust=False, min_periods=rsi_length).mean()
            rs = avg_gain / avg_loss
            df[f"RSI_{rsi_length}"] = 100 - (100 / (1 + rs))
        else:
            st.warning(f"Not enough rows to compute RSI({rsi_length}).")

    # MACD (EMAs) and signal/hist
    if show_macd:
        if len(df) >= macd_slow:
            ema_fast = df["Close"].ewm(span=macd_fast, adjust=False).mean()
            ema_slow = df["Close"].ewm(span=macd_slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=macd_signal, adjust=False).mean()
            hist = macd_line - signal_line
            df[f"MACD_{macd_fast}_{macd_slow}"] = macd_line
            df[f"Signal_{macd_fast}_{macd_slow}_{macd_signal}"] = signal_line
            df[f"Hist_{macd_fast}_{macd_slow}_{macd_signal}"] = hist
        else:
            st.warning(f"Not enough rows to compute MACD (slow={macd_slow}).")

    # --- Safe plotting helper: use only non-NaN points for each trace ---
    def add_line_trace(fig, series, name, line_kwargs=None):
        if series is None:
            return
        line_kwargs = line_kwargs or {}
        fig.add_trace(go.Scatter(x=series.index, y=series.values, name=name, **line_kwargs))

    # Prepare base candlestick figure (use full valid OHLC rows)
    candlestick_df = df.dropna(subset=["Open", "High", "Low", "Close"])
    fig = go.Figure(data=[go.Candlestick(
        x=candlestick_df.index,
        open=candlestick_df["Open"],
        high=candlestick_df["High"],
        low=candlestick_df["Low"],
        close=candlestick_df["Close"],
        name="Price"
    )])

    # Add SMA/EMA/Bands safely
    if show_sma:
        col = f"SMA_{sma_window}"
        s = safe_series(df, col)
        add_line_trace(fig, s, col, {"line": {"color": "blue", "width": 1.5}})

    if show_ema:
        col = f"EMA_{ema_span}"
        s = safe_series(df, col)
        add_line_trace(fig, s, col, {"line": {"color": "orange", "width": 1.5}})

    if show_boll:
        col_up = f"BB_upper_{bb_window}"
        col_lo = f"BB_lower_{bb_window}"
        s_up = safe_series(df, col_up)
        s_lo = safe_series(df, col_lo)
        add_line_trace(fig, s_up, col_up, {"line": {"color": "green", "dash": "dot"}})
        add_line_trace(fig, s_lo, col_lo, {"line": {"color": "red", "dash": "dot"}})

    fig.update_layout(title=f"{ticker} Price Chart", xaxis_rangeslider_visible=False, height=600)
    st.plotly_chart(fig, use_container_width=True)

    # RSI panel
    if show_rsi:
        col = f"RSI_{rsi_length}"
        s = safe_series(df, col)
        if s is not None:
            st.subheader(f"RSI ({rsi_length})")
            st.line_chart(s)
            st.caption("RSI: values above 70 often considered overbought, below 30 oversold (convention).")
        else:
            st.info("RSI not available for the chosen range / length.")

    # MACD panel
    if show_macd:
        macd_col = f"MACD_{macd_fast}_{macd_slow}"
        sig_col = f"Signal_{macd_fast}_{macd_slow}_{macd_signal}"
        hist_col = f"Hist_{macd_fast}_{macd_slow}_{macd_signal}"
        s_macd = safe_series(df, macd_col)
        s_sig = safe_series(df, sig_col)
        s_hist = safe_series(df, hist_col)
        if s_macd is not None and s_sig is not None and s_hist is not None:
            st.subheader("MACD")
            fig_m = go.Figure()
            fig_m.add_trace(go.Scatter(x=s_macd.index, y=s_macd.values, name="MACD", line=dict(width=1.5)))
            fig_m.add_trace(go.Scatter(x=s_sig.index, y=s_sig.values, name="Signal", line=dict(width=1.0)))
            fig_m.add_trace(go.Bar(x=s_hist.index, y=s_hist.values, name="Histogram"))
            fig_m.update_layout(height=300, showlegend=True)
            st.plotly_chart(fig_m, use_container_width=True)
        else:
            st.info("MACD not available for the chosen range / parameters.")

    # Show tail of computed indicator columns for debugging/verification
    st.subheader("Computed indicator snapshot")
    indicator_cols = [c for c in df.columns if any(prefix in c for prefix in ("SMA_", "EMA_", "BB_", "RSI_", "MACD_", "Signal_", "Hist_"))]
    if indicator_cols:
        st.dataframe(df[indicator_cols].tail(10))
    else:
        st.info("No indicators computed (toggle options on the left).")

else:
    st.info("Enter a ticker and valid date range to begin.")
