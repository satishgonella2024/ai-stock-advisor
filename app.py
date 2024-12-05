import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import json
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh

# Set page configuration
st.set_page_config(
    page_title="Real-Time AI Stock Advisor",
    layout="wide",
)

# Global variables to store rolling data and indicators
if 'rolling_window' not in st.session_state:
    st.session_state['rolling_window'] = pd.DataFrame()
if 'dow_rolling_window' not in st.session_state:
    st.session_state['dow_rolling_window'] = pd.DataFrame()
if 'daily_high' not in st.session_state:
    st.session_state['daily_high'] = float('-inf')
if 'daily_low' not in st.session_state:
    st.session_state['daily_low'] = float('inf')
if 'buying_momentum' not in st.session_state:
    st.session_state['buying_momentum'] = 0.0
if 'selling_momentum' not in st.session_state:
    st.session_state['selling_momentum'] = 0.0

# Fetch historical data for Apple (AAPL) and Dow Jones (DJI)
@st.cache_data(ttl=60)
def fetch_stock_data():
    stock = yf.Ticker("AAPL")
    dow_jones = yf.Ticker("^DJI")

    # Get data for the current day at 1-minute intervals
    data = stock.history(period="1d", interval="1m")
    dow_data = dow_jones.history(period="1d", interval="1m")

    return data, dow_data

# Function to process a new stock update every minute
def process_stock_update(data, dow_data):
    if not data.empty and not dow_data.empty:
        # Simulate receiving a new data point for AAPL and Dow Jones
        update = data.iloc[0].to_frame().T
        dow_update = dow_data.iloc[0].to_frame().T
        data.drop(data.index[0], inplace=True)
        dow_data.drop(dow_data.index[0], inplace=True)

        # Append the new data points to the rolling windows
        st.session_state['rolling_window'] = pd.concat([st.session_state['rolling_window'], update])
        st.session_state['dow_rolling_window'] = pd.concat([st.session_state['dow_rolling_window'], dow_update])

        # Update daily high and low
        st.session_state['daily_high'] = max(st.session_state['daily_high'], update['Close'].values[0])
        st.session_state['daily_low'] = min(st.session_state['daily_low'], update['Close'].values[0])

        # Calculate momentum
        if len(st.session_state['rolling_window']) >= 2:
            price_change = update['Close'].values[0] - st.session_state['rolling_window']['Close'].iloc[-2]
            if price_change > 0:
                st.session_state['buying_momentum'] += price_change
            else:
                st.session_state['selling_momentum'] += abs(price_change)

# Function to calculate technical indicators and generate insights
def calculate_insights():
    if len(st.session_state['rolling_window']) >= 5:
        # Calculate technical indicators for AAPL
        rolling_avg = st.session_state['rolling_window']['Close'].rolling(window=5).mean().iloc[-1]
        ema = st.session_state['rolling_window']['Close'].ewm(span=5, adjust=False).mean().iloc[-1]
        std = st.session_state['rolling_window']['Close'].rolling(window=5).std().iloc[-1]
        bollinger_upper = rolling_avg + (2 * std)
        bollinger_lower = rolling_avg - (2 * std)

        # RSI calculation
        delta = st.session_state['rolling_window']['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -1 * delta.clip(upper=0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean().iloc[-1]
        avg_loss = loss.rolling(window=14, min_periods=1).mean().iloc[-1]
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs)) if avg_loss != 0 else 100

        # Dow Jones rolling average
        dow_rolling_avg = st.session_state['dow_rolling_window']['Close'].rolling(window=5).mean().iloc[-1]

        # Generate natural language insights
        get_natural_language_insights(
            rolling_avg, ema, rsi, bollinger_upper, bollinger_lower,
            st.session_state['daily_high'], st.session_state['daily_low'],
            st.session_state['buying_momentum'], st.session_state['selling_momentum'],
            dow_rolling_avg
        )

# Function to generate natural language insights using Ollama API
def get_natural_language_insights(
    rolling_avg, ema, rsi, bollinger_upper, bollinger_lower,
    daily_high, daily_low, buying_momentum, selling_momentum,
    dow_rolling_avg
):
    prompt = f"""
As a professional stock analyst, provide insights based on the following data:
- Apple's 5-minute rolling average: {rolling_avg:.2f}
- EMA: {ema:.2f}
- RSI: {rsi:.2f}
- Bollinger Bands Upper: {bollinger_upper:.2f}
- Bollinger Bands Lower: {bollinger_lower:.2f}
- Daily High: {daily_high:.2f}
- Daily Low: {daily_low:.2f}
- Buying Momentum: {buying_momentum:.2f}
- Selling Momentum: {selling_momentum:.2f}
- Dow Jones 5-minute rolling average: {dow_rolling_avg:.2f}
Provide a concise analysis of the current stock trend and market sentiment. Limit your response to 100 words.
"""

    # Use the Ollama API to generate the response
    try:
        response = requests.post(
            'http://localhost:11434/generate',
            json={
                'model': 'llama2',
                'prompt': prompt
            },
            stream=True  # Stream the response
        )

        response_text = ''
        for line in response.iter_lines():
            if line:
                # Each line is a JSON object
                line_json = json.loads(line.decode('utf-8'))
                if 'done' in line_json and line_json['done']:
                    break
                if 'response' in line_json:
                    response_text += line_json['response']

    except Exception as e:
        response_text = f"Error generating insight: {e}"

    # Display the insight
    st.markdown("### Natural Language Insight")
    st.write(response_text)

# Main function to run the Streamlit app
def main():
    # Set up the page
    st.title("Real-Time AI Stock Advisor")
    st.write("This application provides real-time stock analysis and insights using Llama 2 and Streamlit.")

    # Auto-refresh every minute
    count = st_autorefresh(interval=60 * 1000, key="datarefresh")

    # Fetch the data
    data, dow_data = fetch_stock_data()

    # Process stock update and calculate insights
    process_stock_update(data, dow_data)
    calculate_insights()

    # Display stock data and technical indicators
    if not st.session_state['rolling_window'].empty:
        st.markdown("### Latest Stock Data (AAPL)")
        st.dataframe(st.session_state['rolling_window'].tail())

    if not st.session_state['dow_rolling_window'].empty:
        st.markdown("### Latest Dow Jones Data (DJI)")
        st.dataframe(st.session_state['dow_rolling_window'].tail())

    # Display technical indicators
    if len(st.session_state['rolling_window']) >= 5:
        st.markdown("### Technical Indicators (AAPL)")
        rolling_avg = st.session_state['rolling_window']['Close'].rolling(window=5).mean().iloc[-1]
        ema = st.session_state['rolling_window']['Close'].ewm(span=5, adjust=False).mean().iloc[-1]
        std = st.session_state['rolling_window']['Close'].rolling(window=5).std().iloc[-1]
        bollinger_upper = rolling_avg + (2 * std)
        bollinger_lower = rolling_avg - (2 * std)

        delta = st.session_state['rolling_window']['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -1 * delta.clip(upper=0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean().iloc[-1]
        avg_loss = loss.rolling(window=14, min_periods=1).mean().iloc[-1]
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs)) if avg_loss != 0 else 100

        st.write(f"**5-Minute Rolling Average:** {rolling_avg:.2f}")
        st.write(f"**Exponential Moving Average (EMA):** {ema:.2f}")
        st.write(f"**Bollinger Upper Band:** {bollinger_upper:.2f}")
        st.write(f"**Bollinger Lower Band:** {bollinger_lower:.2f}")
        st.write(f"**Relative Strength Index (RSI):** {rsi:.2f}")

    # Display charts
    if len(st.session_state['rolling_window']) > 0:
        st.markdown("### Stock Price Chart (AAPL)")
        st.line_chart(st.session_state['rolling_window']['Close'])

    if len(st.session_state['dow_rolling_window']) > 0:
        st.markdown("### Dow Jones Price Chart (DJI)")
        st.line_chart(st.session_state['dow_rolling_window']['Close'])

if __name__ == "__main__":
    main()