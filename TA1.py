import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import ta

# App title
st.title("📊 Simple Technical Analysis App (with Sidebar Controls)")

# Sidebar controls
st.sidebar.header("Indicator Options")
show_sma = st.sidebar.checkbox("Show SMA (20)", True)
show_ema = st.sidebar.checkbox("Show EMA (20)", True)
show_boll = st.sidebar.checkbox("Show Bollinger Bands", True)
show_rsi = st.sidebar.checkbox("Show RSI", True)
show_macd = st.sidebar.checkbox("Show MACD", True)

# User input
ticker = st.text_input("Enter Stock Ticker (e.g. AAPL, TSLA, RELIANCE.BO):", "AAPL")

if ticker:
    # Download stock data
    data = yf.download(ticker, period="6mo", interval="1d")

    if not data.empty:
        st.write(f"### Showing data for {ticker}")
        st.dataframe(data.tail())

        # -------------------------------
        # Technical Indicators
        # -------------------------------
        if show_sma:
            data["SMA20"] = data["Close"].rolling(window=20).mean()

        if show_ema:
            ema = ta.trend.EMAIndicator(close=data["Close"], window=20)
            data["EMA20"] = ema.ema_indicator().squeeze()

        if show_rsi:
            rsi = ta.momentum.RSIIndicator(close=data["Close"], window=14)
            data["RSI"] = rsi.rsi().squeeze()

        if show_macd:
            macd = ta.trend.MACD(close=data["Close"])
            data["MACD"] = macd.macd().squeeze()
            data["Signal"] = macd.macd_signal().squeeze()
            data["Hist"] = macd.macd_diff().squeeze()

        if show_boll:
            boll = ta.volatility.BollingerBands(close=data["Close"], window=20, window_dev=2)
            data["BB_upper"] = boll.bollinger_hband().squeeze()
            data["BB_lower"] = boll.bollinger_lband().squeeze()

        # -------------------------------
        # Clean up NaNs before plotting
        # -------------------------------
        plot_cols = ["SMA20", "EMA20", "BB_upper", "BB_lower", "RSI", "MACD", "Signal", "Hist"]
        existing_cols = [col for col in plot_cols if col in data.columns]
        plot_data = data.dropna(subset=existing_cols)

        # -------------------------------
        # Price Chart
        # -------------------------------
        fig = go.Figure(data=[go.Candlestick(
            x=plot_data.index,
            open=plot_data['Open'], high=plot_data['High'],
            low=plot_data['Low'], close=plot_data['Close'],
            name="Candlestick"
        )])

        if show_sma:
            fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data["SMA20"], line=dict(color='blue'), name="SMA20"))

        if show_ema:
            fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data["EMA20"], line=dict(color='orange'), name="EMA20"))

        if show_boll:
            fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data["BB_upper"], line=dict(color='green', dash="dot"), name="BB Upper"))
            fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data["BB_lower"], line=dict(color='red', dash="dot"), name="BB Lower"))

        fig.update_layout(title=f"{ticker} Price Chart", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # -------------------------------
        # RSI
        # -------------------------------
        if show_rsi:
            st.subheader("Relative Strength Index (RSI)")
            st.line_chart(plot_data["RSI"], height=200, use_container_width=True)

        # -------------------------------
        # MACD
        # -------------------------------
        if show_macd:
            st.subheader("MACD")
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=plot_data.index, y=plot_data["MACD"], name="MACD", line=dict(color='blue')))
            fig_macd.add_trace(go.Scatter(x=plot_data.index, y=plot_data["Signal"], name="Signal", line=dict(color='orange')))
            fig_macd.add_trace(go.Bar(x=plot_data.index, y=plot_data["Hist"], name="Histogram", marker_color='gray'))
            st.plotly_chart(fig_macd, use_container_width=True)

    else:
        st.error("Ticker not found. Try again.")
