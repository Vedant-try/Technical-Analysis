import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import ta

# App title
st.title("📊 Simple Technical Analysis App (using ta library)")

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
        # SMA (pandas)
        data["SMA20"] = data["Close"].rolling(window=20).mean()

        # EMA
        ema = ta.trend.EMAIndicator(close=data["Close"], window=20)
        data["EMA20"] = ema.ema_indicator()

        # RSI
        rsi = ta.momentum.RSIIndicator(close=data["Close"], window=14)
        data["RSI"] = rsi.rsi()

        # MACD
        macd = ta.trend.MACD(close=data["Close"])
        data["MACD"] = macd.macd()
        data["Signal"] = macd.macd_signal()
        data["Hist"] = macd.macd_diff()

        # Bollinger Bands
        boll = ta.volatility.BollingerBands(close=data["Close"], window=20, window_dev=2)
        data["BB_upper"] = boll.bollinger_hband()
        data["BB_lower"] = boll.bollinger_lband()

        # -------------------------------
        # Plot candlestick + SMA/EMA + Bollinger
        # -------------------------------
        fig = go.Figure(data=[go.Candlestick(
            x=data.index,
            open=data['Open'], high=data['High'],
            low=data['Low'], close=data['Close'],
            name="Candlestick"
        )])

        fig.add_trace(go.Scatter(x=data.index, y=data["SMA20"], line=dict(color='blue'), name="SMA20"))
        fig.add_trace(go.Scatter(x=data.index, y=data["EMA20"], line=dict(color='orange'), name="EMA20"))
        fig.add_trace(go.Scatter(x=data.index, y=data["BB_upper"], line=dict(color='green', dash="dot"), name="BB Upper"))
        fig.add_trace(go.Scatter(x=data.index, y=data["BB_lower"], line=dict(color='red', dash="dot"), name="BB Lower"))

        fig.update_layout(title=f"{ticker} Price Chart", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # -------------------------------
        # RSI Plot
        # -------------------------------
        st.subheader("Relative Strength Index (RSI)")
        st.line_chart(data["RSI"], height=200, use_container_width=True)

        # -------------------------------
        # MACD Plot
        # -------------------------------
        st.subheader("MACD")
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=data.index, y=data["MACD"], name="MACD", line=dict(color='blue')))
        fig_macd.add_trace(go.Scatter(x=data.index, y=data["Signal"], name="Signal", line=dict(color='orange')))
        fig_macd.add_trace(go.Bar(x=data.index, y=data["Hist"], name="Histogram", marker_color='gray'))
        st.plotly_chart(fig_macd, use_container_width=True)

    else:
        st.error("Ticker not found. Try again.")
