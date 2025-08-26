# ta_app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import pandas_ta as ta


# App title
st.title("📊 Simple Technical Analysis App (with pandas_ta)")

# User input
ticker = st.text_input("Enter Stock Ticker (e.g. AAPL, TSLA, RELIANCE.BO):", "AAPL")

if ticker:
    # Download stock data
    data = yf.download(ticker, period="6mo", interval="1d")

    if not data.empty:
        st.write(f"### Showing data for {ticker}")
        st.dataframe(data.tail())

        # Add Technical Indicators
        data["SMA20"] = ta.sma(data["Close"], length=20)
        data["EMA20"] = ta.ema(data["Close"], length=20)
        data["RSI"] = ta.rsi(data["Close"], length=14)
        macd = ta.macd(data["Close"], fast=12, slow=26, signal=9)
        data = pd.concat([data, macd], axis=1)

        # Plot candlestick chart with SMA and EMA
        fig = go.Figure(data=[go.Candlestick(
            x=data.index,
            open=data['Open'], high=data['High'],
            low=data['Low'], close=data['Close'],
            name="Candlestick"
        )])
        fig.add_trace(go.Scatter(x=data.index, y=data["SMA20"], line=dict(color='blue'), name="SMA20"))
        fig.add_trace(go.Scatter(x=data.index, y=data["EMA20"], line=dict(color='orange'), name="EMA20"))
        fig.update_layout(title=f"{ticker} Price with SMA & EMA", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # RSI Plot
        st.subheader("Relative Strength Index (RSI)")
        st.line_chart(data["RSI"], height=200, use_container_width=True)

        # MACD Plot
        st.subheader("MACD")
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=data.index, y=data["MACD_12_26_9"], name="MACD", line=dict(color='blue')))
        fig_macd.add_trace(go.Scatter(x=data.index, y=data["MACDs_12_26_9"], name="Signal", line=dict(color='orange')))
        fig_macd.add_trace(go.Bar(x=data.index, y=data["MACDh_12_26_9"], name="Histogram", marker_color='gray'))
        st.plotly_chart(fig_macd, use_container_width=True)

    else:
        st.error("Ticker not found. Try again.")


