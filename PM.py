import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Title of the app
st.title('Stock Price Search')

# Input for ticker symbol
ticker = st.text_input('Enter a stock ticker symbol (e.g., AAPL):')

# Initialize session state to store historical data
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = {}

# Function to filter data to only include 'Close' price and Date
def filter_data(data):
    return data[['Close']]

# Function to format the date and close price
def format_data(data):
    data.index = data.index.strftime('%d-%b').str.upper()
    data['Close'] = data['Close'].round(2)
    return data

# Function to get historical data
def get_historical_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y", interval="1mo")
    filtered_data = filter_data(hist)
    return format_data(filtered_data)

# Display the data
if ticker:
    data = get_historical_data(ticker)
    if not data.empty:
        st.session_state.historical_data[ticker] = data
        combined_data = pd.concat(st.session_state.historical_data.values(), axis=1, keys=st.session_state.historical_data.keys())
        st.write(f"Historical data for {', '.join(st.session_state.historical_data.keys())}:")
        st.dataframe(combined_data)
    else:
        st.write("No data found for the given ticker symbol.")

# Button to generate percentage change table
if st.button('Generate?'):
    if st.session_state.historical_data:
        combined_data = pd.concat(st.session_state.historical_data.values(), axis=1, keys=st.session_state.historical_data.keys())
        pct_change_data = combined_data.pct_change().dropna() * 100
        pct_change_data = pct_change_data.round(2)
        st.write("Percentage change from one month to the next:")
        st.dataframe(pct_change_data)
    else:
        st.write("No data available to generate percentage change table.")

# Button to generate average percentage change table
if st.button('Generate Average?'):
    if st.session_state.historical_data:
        combined_data = pd.concat(st.session_state.historical_data.values(), axis=1, keys=st.session_state.historical_data.keys())
        pct_change_data = combined_data.pct_change().dropna() * 100
        pct_change_data = pct_change_data.round(2)
        
        # Calculate the average percentage change for each ticker
        avg_pct_change = pct_change_data.mean()
        avg_pct_change_df = avg_pct_change.to_frame(name='Return').T
        st.write("Average percentage change for each ticker:")
        st.dataframe(avg_pct_change_df)
        
        # Store avg_pct_change in session state for later use
        st.session_state.avg_pct_change = avg_pct_change
        st.session_state.pct_change_data = pct_change_data
    else:
        st.write("No data available to generate average percentage change table.")

# Section to input portfolio weights
st.write("### Portfolio Weights")
if st.session_state.historical_data:
    tickers = list(st.session_state.historical_data.keys())
    weights = {}
    for ticker in tickers:
        weight = st.number_input(f"Weight for {ticker}:", min_value=0.0, max_value=1.0, step=0.01)
        weights[ticker] = weight

    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    if total_weight > 0:
        normalized_weights = {ticker: weight / total_weight for ticker, weight in weights.items()}
        weights_df = pd.DataFrame.from_dict(normalized_weights, orient='index', columns=['Weight'])
        st.write("Portfolio Weights Matrix:")
        st.dataframe(weights_df)
        
        # Calculate the weighted average return
        if 'avg_pct_change' in st.session_state:
            avg_pct_change = st.session_state.avg_pct_change
            weighted_avg_return = sum(normalized_weights[ticker] * avg_pct_change[ticker] for ticker in tickers)
            weighted_avg_return = float(weighted_avg_return)  # Convert to float
            st.write(f"Weighted Average Return: {weighted_avg_return:.2f}%")
            
            # Calculate the portfolio risk
            pct_change_data = st.session_state.pct_change_data
            weights_array = np.array(list(normalized_weights.values()))
            cov_matrix = pct_change_data.cov()
            portfolio_variance = np.dot(weights_array.T, np.dot(cov_matrix, weights_array))
            portfolio_risk = np.sqrt(portfolio_variance)
            st.write(f"Portfolio Risk: {portfolio_risk:.2f}%")
        else:
            st.write("Please generate the average percentage change table first.")
    else:
        st.write("Please enter weights for the tickers.")
else:
    st.write("No tickers available to generate portfolio weights matrix.")