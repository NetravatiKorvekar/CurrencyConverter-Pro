import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import json
import os
from PIL import Image
import base64
import io

# Set page config
st.set_page_config(
    page_title="CurrencyConverter Pro",
    page_icon="💱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create data directory if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

# Apply custom CSS
def local_css():
    st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
        color: #212529;
    }
    .dark-mode .main {
        background-color: #212529;
        color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
        border: none;
    }
    .dark-mode .stButton>button {
        background-color: #6c757d;
    }
    .stSelectbox>div>div {
        background-color: white;
        border-radius: 5px;
    }
    .dark-mode .stSelectbox>div>div {
        background-color: #343a40;
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #e9ecef;
    }
    .dark-mode .sidebar .sidebar-content {
        background-color: #343a40;
    }
    </style>
    """, unsafe_allow_html=True)

# Functions for API handling
def get_exchange_rates(base_currency='USD'):
    """Fetch exchange rates from API"""
    try:
        # Using Exchange Rate API (free tier)
        url = f"https://api.exchangerate-api.com/v4/latest/{base_currency}"
        response = requests.get(url)
        data = response.json()
        
        # Save the data locally for offline use
        with open(f'data/exchange_rates_{base_currency}.json', 'w') as f:
            json.dump(data, f)
        
        return data['rates']
    except Exception as e:
        st.error(f"Error fetching exchange rates: {e}")
        # Try to load from local file if available
        try:
            with open(f'data/exchange_rates_{base_currency}.json', 'r') as f:
                data = json.load(f)
            st.warning("Using cached exchange rates (offline mode)")
            return data['rates']
        except:
            st.error("Could not load cached data. Please check your internet connection.")
            return {}

def get_crypto_rates():
    """Fetch cryptocurrency rates"""
    try:
        # Using CoinGecko API (free tier)
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            'ids': 'bitcoin,ethereum,ripple,litecoin,cardano,dogecoin',
            'vs_currencies': 'usd,eur,gbp,jpy'
        }
        response = requests.get(url, params=params)
        data = response.json()
        
        # Save the data locally for offline use
        with open('data/crypto_rates.json', 'w') as f:
            json.dump(data, f)
        
        # Format data to match exchange rates structure
        crypto_rates = {}
        for crypto, rates in data.items():
            crypto_key = crypto.upper()
            for currency, rate in rates.items():
                if currency.upper() == 'USD':
                    crypto_rates[crypto_key] = 1/rate  # Inverted to match exchange rate format
        
        return crypto_rates
    except Exception as e:
        st.error(f"Error fetching crypto rates: {e}")
        # Try to load from local file if available
        try:
            with open('data/crypto_rates.json', 'r') as f:
                data = json.load(f)
            st.warning("Using cached crypto rates (offline mode)")
            
            # Format data to match exchange rates structure
            crypto_rates = {}
            for crypto, rates in data.items():
                crypto_key = crypto.upper()
                for currency, rate in rates.items():
                    if currency.upper() == 'USD':
                        crypto_rates[crypto_key] = 1/rate
            
            return crypto_rates
        except:
            return {}

def get_historical_rates(base_currency, target_currency, days=7):
    """Simulate historical data for demonstration purposes"""
    # In a real app, you'd use a proper historical data API
    
    # Check if historical data exists locally
    filename = f'data/historical_{base_currency}_{target_currency}.json'
    
    if os.path.exists(filename):
        # Load existing data and check if it's still recent
        with open(filename, 'r') as f:
            historical_data = json.load(f)
        
        last_date = datetime.strptime(list(historical_data.keys())[-1], "%Y-%m-%d").date()
        today = datetime.now().date()
        
        if (today - last_date).days <= 1:
            return historical_data
    
    # Generate simulated historical data
    dates = [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]
    
    # For demonstration, generate slightly random rates based on current rate
    current_rate = 1.0
    if base_currency != target_currency:
        try:
            rates = get_exchange_rates(base_currency)
            current_rate = rates.get(target_currency, 1.0)
        except:
            current_rate = 1.0
    
    # Generate random-ish variations around the current rate
    np.random.seed(hash(f"{base_currency}{target_currency}") % 10000)
    variations = np.random.normal(0, 0.01, days)
    
    historical_rates = {}
    for i, date in enumerate(dates):
        # Make the simulated data somewhat realistic with a slight trend
        adjustment = variations[i] + (i/days) * 0.02  # Add a slight trend
        historical_rates[date] = round(current_rate * (1 + adjustment), 4)
    
    # Save data locally
    with open(filename, 'w') as f:
        json.dump(historical_rates, f)
    
    return historical_rates

def save_conversion_history(from_currency, to_currency, amount, result, timestamp=None):
    """Save conversion history to file"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    history_file = 'data/conversion_history.csv'
    
    # Create history file if it doesn't exist
    if not os.path.exists(history_file):
        pd.DataFrame(columns=['timestamp', 'from_currency', 'to_currency', 'amount', 'result']).to_csv(history_file, index=False)
    
    # Read existing history
    history_df = pd.read_csv(history_file)
    
    # Add new entry
    new_entry = pd.DataFrame({
        'timestamp': [timestamp],
        'from_currency': [from_currency],
        'to_currency': [to_currency],
        'amount': [amount],
        'result': [result]
    })
    
    # Append to history and save - fixing the FutureWarning
    if history_df.empty:
        updated_df = new_entry
    else:
        updated_df = pd.concat([history_df, new_entry], ignore_index=True)
    
    updated_df.to_csv(history_file, index=False)

def load_conversion_history(limit=10):
    """Load conversion history from file"""
    history_file = 'data/conversion_history.csv'
    
    if not os.path.exists(history_file):
        return pd.DataFrame(columns=['timestamp', 'from_currency', 'to_currency', 'amount', 'result'])
    
    history_df = pd.read_csv(history_file)
    return history_df.tail(limit)

# Initialize session state for app settings
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

if 'favorite_currencies' not in st.session_state:
    st.session_state.favorite_currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CAD']

if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now()

if 'rates' not in st.session_state:
    st.session_state.rates = {}

if 'crypto_rates' not in st.session_state:
    st.session_state.crypto_rates = {}

if 'alerts' not in st.session_state:
    st.session_state.alerts = []

# Apply dark mode if enabled
if st.session_state.dark_mode:
    st.markdown('<style>body {color: white; background-color: #121212;}</style>', unsafe_allow_html=True)
    st.markdown('<div class="dark-mode">', unsafe_allow_html=True)

# App Title and Introduction
st.title('💱 CurrencyConverter Pro')
st.markdown('Real-time currency conversion with historical data and more!')

# Sidebar settings and options
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Dark/Light mode toggle
    dark_mode = st.checkbox("Dark Mode", value=st.session_state.dark_mode)
    if dark_mode != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_mode
        st.rerun()  # Changed from experimental_rerun
    
    # Auto-refresh settings
    st.subheader("Auto-Refresh")
    refresh_interval = st.selectbox(
        "Refresh Interval",
        ["Manual", "1 minute", "5 minutes", "15 minutes", "1 hour"],
        index=0
    )
    
    # Check if it's time to refresh based on interval
    current_time = datetime.now()
    if refresh_interval == "1 minute" and (current_time - st.session_state.last_refresh).seconds >= 60:
        st.session_state.rates = get_exchange_rates()
        st.session_state.crypto_rates = get_crypto_rates()
        st.session_state.last_refresh = current_time
    elif refresh_interval == "5 minutes" and (current_time - st.session_state.last_refresh).seconds >= 300:
        st.session_state.rates = get_exchange_rates()
        st.session_state.crypto_rates = get_crypto_rates()
        st.session_state.last_refresh = current_time
    elif refresh_interval == "15 minutes" and (current_time - st.session_state.last_refresh).seconds >= 900:
        st.session_state.rates = get_exchange_rates()
        st.session_state.crypto_rates = get_crypto_rates()
        st.session_state.last_refresh = current_time
    elif refresh_interval == "1 hour" and (current_time - st.session_state.last_refresh).seconds >= 3600:
        st.session_state.rates = get_exchange_rates()
        st.session_state.crypto_rates = get_crypto_rates()
        st.session_state.last_refresh = current_time
    
    if st.button("🔄 Refresh Now"):
        with st.spinner("Fetching latest rates..."):
            st.session_state.rates = get_exchange_rates()
            st.session_state.crypto_rates = get_crypto_rates()
            st.session_state.last_refresh = current_time
            st.success("Rates updated!")
    
    st.caption(f"Last updated: {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Favorite currencies management
    st.subheader("Favorites")
    available_currencies = ["USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "CNY", "HKD", "NZD", 
                          "SEK", "KRW", "SGD", "NOK", "MXN", "INR", "RUB", "ZAR", "BRL", "TRY",
                          "BTC", "ETH", "XRP", "LTC", "ADA", "DOGE"]
    
    selected_favorites = st.multiselect(
        "Select your favorite currencies",
        available_currencies,
        st.session_state.favorite_currencies
    )
    
    if selected_favorites != st.session_state.favorite_currencies:
        st.session_state.favorite_currencies = selected_favorites
    
    # Currency alerts
    st.subheader("Currency Alerts")
    alert_base = st.selectbox("Base Currency", available_currencies, index=0, key="alert_base")
    alert_target = st.selectbox("Target Currency", available_currencies, index=1, key="alert_target")
    alert_threshold = st.number_input("Target Rate", min_value=0.001, value=1.0, step=0.001, key="alert_threshold")
    alert_condition = st.selectbox("Condition", ["Above", "Below"], key="alert_condition")
    
    if st.button("Add Alert"):
        st.session_state.alerts.append({
            "base": alert_base,
            "target": alert_target,
            "threshold": alert_threshold,
            "condition": alert_condition,
            "active": True
        })
        st.success(f"Alert added: {alert_base}/{alert_target} {alert_condition.lower()} {alert_threshold}")
    
    if st.session_state.alerts:
        st.subheader("Active Alerts")
        for i, alert in enumerate(st.session_state.alerts):
            if alert["active"]:
                st.write(f"{alert['base']}/{alert['target']} {alert['condition'].lower()} {alert['threshold']}")
                if st.button(f"Remove", key=f"remove_alert_{i}"):
                    st.session_state.alerts[i]["active"] = False
                    st.rerun()  # Changed from experimental_rerun

# Main area divided into tabs
tab1, tab2, tab3, tab4 = st.tabs(["Convert", "History", "Charts", "Multi-Compare"])

# Fetch rates if we don't have them yet
if not st.session_state.rates:
    with st.spinner("Fetching exchange rates..."):
        st.session_state.rates = get_exchange_rates()
        st.session_state.crypto_rates = get_crypto_rates()
        st.session_state.last_refresh = datetime.now()

# Combine regular currency rates and crypto rates
all_rates = {**st.session_state.rates, **st.session_state.crypto_rates}

# Tab 1: Currency Converter
with tab1:
    st.header("Currency Converter")
    
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        # Filter for favorite currencies if selected
        currency_options = list(all_rates.keys())
        currency_options = sorted([c for c in currency_options if c in st.session_state.favorite_currencies]) if st.session_state.favorite_currencies else sorted(currency_options)
        
        # Add USD as default if not in the list
        if "USD" not in currency_options:
            currency_options.insert(0, "USD")
        
        from_currency = st.selectbox("From Currency", currency_options, index=0)
        amount = st.number_input("Amount", min_value=0.01, value=1.0, step=0.01)
    
    with col2:
        st.write("")
        st.write("")
        if st.button("↔️ Swap"):
            # We'll handle the swap in col3
            pass
        
        # Calculator mode
        st.write("")
        st.write("Calculator Mode")
        calc_operation = st.selectbox("Operation", ["+", "-", "×", "÷", "%"], index=0)
        calc_value = st.number_input("Value", value=0.0, step=0.01)
    
    with col3:
        to_currency = st.selectbox("To Currency", currency_options, index=1 if len(currency_options) > 1 else 0)
        
        # Set up base rate for conversion
        base_rate = 1.0
        target_rate = 1.0
        
        if from_currency == "USD":
            target_rate = all_rates.get(to_currency, 1.0)
        elif to_currency == "USD":
            base_rate = all_rates.get(from_currency, 1.0)
        else:
            # Convert via USD as base
            base_rate = all_rates.get(from_currency, 1.0)
            target_rate = all_rates.get(to_currency, 1.0)
            
        # Calculate result
        if from_currency == "USD":
            result = amount * target_rate
        elif to_currency == "USD":
            result = amount / base_rate
        else:
            result = amount / base_rate * target_rate
        
        # Apply calculator operation if selected
        if calc_operation == "+":
            result += calc_value
        elif calc_operation == "-":
            result -= calc_value
        elif calc_operation == "×":
            result *= calc_value
        elif calc_operation == "÷":
            if calc_value != 0:
                result /= calc_value
        elif calc_operation == "%":
            result *= (1 + calc_value/100)
            
        # Display result
        st.markdown(f"### {result:.4f} {to_currency}")
        
        # Show exchange rate
        exchange_rate = target_rate / base_rate if from_currency != "USD" or to_currency != "USD" else target_rate
        st.caption(f"1 {from_currency} = {exchange_rate:.6f} {to_currency}")
        st.caption(f"1 {to_currency} = {(1/exchange_rate):.6f} {from_currency}")
    
    # Save conversion to history
    if st.button("Save Conversion"):
        save_conversion_history(from_currency, to_currency, amount, result)
        st.success("Conversion saved to history!")
    
    # Check alerts
    for alert in st.session_state.alerts:
        if not alert["active"]:
            continue
            
        if alert["base"] == from_currency and alert["target"] == to_currency:
            current_rate = exchange_rate
            if (alert["condition"] == "Above" and current_rate > alert["threshold"]) or \
               (alert["condition"] == "Below" and current_rate < alert["threshold"]):
                st.warning(f"🔔 Alert triggered: {from_currency}/{to_currency} is {current_rate:.4f}, which is {alert['condition'].lower()} your target of {alert['threshold']}")

# Tab 2: Conversion History
with tab2:
    st.header("Conversion History")
    
    history_df = load_conversion_history(20)
    
    if not history_df.empty:
        # Format the data for display
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
        history_df['formatted_time'] = history_df['timestamp'].dt.strftime("%Y-%m-%d %H:%M")
        history_df['conversion'] = history_df.apply(
            lambda x: f"{x['amount']} {x['from_currency']} → {x['result']:.2f} {x['to_currency']}", 
            axis=1
        )
        
        # Display in a nice table
        st.dataframe(
            history_df[['formatted_time', 'conversion']],
            column_config={
                "formatted_time": "Time",
                "conversion": "Conversion"
            },
            hide_index=True
        )
        
        if st.button("Clear History"):
            if os.path.exists('data/conversion_history.csv'):
                os.remove('data/conversion_history.csv')
                st.success("History cleared!")
                st.rerun()  # Changed from experimental_rerun
    else:
        st.info("No conversion history found. Save some conversions to see them here.")

# Tab 3: Charts
with tab3:
    st.header("Exchange Rate Charts")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        chart_base = st.selectbox("Base Currency", sorted(all_rates.keys()), index=0, key="chart_base")
    
    with chart_col2:
        chart_target = st.selectbox("Target Currency", sorted(all_rates.keys()), index=1 if len(all_rates) > 1 else 0, key="chart_target")
    
    chart_period = st.select_slider(
        "Time Period",
        options=["7 Days", "14 Days", "30 Days"],
        value="7 Days"
    )
    
    # Convert period to days
    if chart_period == "7 Days":
        days = 7
    elif chart_period == "14 Days":
        days = 14
    else:
        days = 30
    
    # Get historical data
    with st.spinner("Loading historical data..."):
        historical_data = get_historical_rates(chart_base, chart_target, days)
        
        # Prepare data for chart
        dates = list(historical_data.keys())
        rates = list(historical_data.values())
        
        # Create chart
        fig = px.line(
            x=dates, 
            y=rates,
            labels={"x": "Date", "y": f"Exchange Rate ({chart_base}/{chart_target})"},
            title=f"{chart_base}/{chart_target} Exchange Rate - Past {days} Days"
        )
        
        # Customize chart
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title=f"Exchange Rate ({chart_base}/{chart_target})",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add min/max/avg statistics
        st.subheader("Statistics")
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        
        with stats_col1:
            min_rate = min(rates)
            min_date = dates[rates.index(min_rate)]
            st.metric("Minimum Rate", f"{min_rate:.4f}", f"on {min_date}")
        
        with stats_col2:
            max_rate = max(rates)
            max_date = dates[rates.index(max_rate)]
            st.metric("Maximum Rate", f"{max_rate:.4f}", f"on {max_date}")
        
        with stats_col3:
            avg_rate = sum(rates) / len(rates)
            current_rate = rates[0]
            change = ((current_rate - avg_rate) / avg_rate) * 100
            st.metric("Average Rate", f"{avg_rate:.4f}", f"{change:+.2f}% vs current")

# Tab 4: Multi-Currency Comparison - MODIFIED SECTION WITH FIXES
with tab4:
    st.header("Multi-Currency Comparison")
    
    base_currency = st.selectbox("Base Currency", sorted(all_rates.keys()), index=0, key="multi_base")
    
    # Select currencies to compare
    compare_currencies = st.multiselect(
        "Select currencies to compare",
        [c for c in sorted(all_rates.keys()) if c != base_currency],
        default=st.session_state.favorite_currencies[:4] if len(st.session_state.favorite_currencies) >= 4 else sorted(all_rates.keys())[:5]
    )
    
    # Add option for chart display type
    chart_display_type = st.radio(
        "Chart Display Type",
        ["Absolute Values", "Normalized (Base Day = 100)"],
        index=1  # Default to normalized view for better visualization
    )
    
    if compare_currencies:
        # Create comparison table
        comparison_data = []
        
        for curr in compare_currencies:
            if base_currency == "USD":
                rate = all_rates.get(curr, 1.0)
            elif curr == "USD":
                rate = 1.0 / all_rates.get(base_currency, 1.0)
            else:
                # Convert via USD
                base_to_usd = 1.0 / all_rates.get(base_currency, 1.0) if base_currency != "USD" else 1.0
                usd_to_target = all_rates.get(curr, 1.0) if curr != "USD" else 1.0
                rate = base_to_usd * usd_to_target
            
            # Get some historical data for trend
            historical = get_historical_rates(base_currency, curr, 7)
            dates = list(historical.keys())
            rates = list(historical.values())
            
            # Calculate simple trend
            oldest_rate = rates[-1]
            change = ((rate - oldest_rate) / oldest_rate) * 100
            
            comparison_data.append({
                "Currency": curr,
                "Rate": rate,
                "7d_Change": change
            })
        
        # Create DataFrame for display
        comparison_df = pd.DataFrame(comparison_data)
        
        # Format for display
        st.subheader(f"Comparison to {base_currency}")
        
        # Display table with metrics
        for idx, row in comparison_df.iterrows():
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                st.write(f"### {row['Currency']}")
            
            with col2:
                st.metric(
                    f"1 {base_currency} = {row['Rate']:.4f} {row['Currency']}",
                    f"{1/row['Rate']:.4f} {row['Currency']} = 1 {base_currency}",
                    delta=f"{row['7d_Change']:+.2f}% (7d)",
                    delta_color="normal"
                )
            
            with col3:
                # Quick convert
                amount = st.number_input(f"Convert {base_currency}", 
                                         min_value=0.01, 
                                         value=1.0, 
                                         step=0.01, 
                                         key=f"multi_amount_{row['Currency']}")
                st.write(f"**{amount * row['Rate']:.2f} {row['Currency']}**")
        
        # Create comparative chart
        st.subheader("Comparative Performance (Last 7 Days)")
        
        # Get historical data for all selected currencies
        chart_data = {}
        for curr in compare_currencies:
            historical = get_historical_rates(base_currency, curr, 7)
            for date, rate in historical.items():
                if date not in chart_data:
                    chart_data[date] = {}
                chart_data[date][curr] = rate
        
        # Convert to DataFrame
        chart_df = pd.DataFrame.from_dict(chart_data, orient='index')
        chart_df.index.name = 'Date'
        chart_df.reset_index(inplace=True)
        
        # Create the chart with fixes
        fig = go.Figure()
        
        # Calculate y-axis range for absolute values
        if chart_display_type == "Absolute Values":
            min_val = float('inf')
            max_val = float('-inf')
            
            for curr in compare_currencies:
                if curr in chart_df.columns:
                    curr_min = chart_df[curr].min()
                    curr_max = chart_df[curr].max()
                    min_val = min(min_val, curr_min)
                    max_val = max(max_val, curr_max)
            
            # Add some padding (5%)
            y_range_min = min_val * 0.95
            y_range_max = max_val * 1.05
        
        for curr in compare_currencies:
            if curr in chart_df.columns:
                if chart_display_type == "Normalized (Base Day = 100)":
                    # Normalize each series to start at 100
                    first_val = chart_df[curr].iloc[-1]  # Last row is oldest date
                    y_values = chart_df[curr] / first_val * 100
                    y_axis_title = "Indexed Value (First day = 100)"
                else:
                    # Use absolute values
                    y_values = chart_df[curr]
                    y_axis_title = f"Exchange Rate (1 {base_currency} to Currency)"
                
                fig.add_trace(go.Scatter(
                    x=chart_df['Date'],
                    y=y_values,
                    mode='lines',
                    name=f"{base_currency}/{curr}"
                ))
        
        # Set chart title and format
        chart_title = f"Exchange Rates vs {base_currency} (Last 7 Days)"
        if chart_display_type == "Normalized (Base Day = 100)":
            chart_title += " - Normalized"
        
        fig.update_layout(
            title=chart_title,
            xaxis_title="Date",
            yaxis_title=y_axis_title,
            legend_title="Currency Pair",
            hovermode="x unified"
        )
        
        # Set y-axis range for absolute values
        if chart_display_type == "Absolute Values":
            fig.update_layout(yaxis=dict(range=[y_range_min, y_range_max]))
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please select at least one currency to compare.")

# Footer
st.markdown("---")
col1, col2 = st.columns([3, 1])
with col1:
    st.caption("CurrencyConverter Pro - Real-time exchange rates and tools")
    st.caption("Data sources: Exchange Rate API (fictional implementation)")
with col2:
    st.caption(f"Last updated: {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}")
    if st.session_state.rates:
        online_status = "🟢 Online" if datetime.now().timestamp() - st.session_state.last_refresh.timestamp() < 3600 else "🟠 Data may be outdated"
        st.caption(online_status)
    else:
        st.caption("🔴 Offline - Using cached data")

# Notification system for alerts
for alert in st.session_state.alerts:
    if not alert["active"]:
        continue
        
    # Check if alert should be triggered
    base = alert["base"]
    target = alert["target"]
    threshold = alert["threshold"]
    condition = alert["condition"]
    
    # Get current rate
    if base == "USD":
        current_rate = all_rates.get(target, 1.0)
    elif target == "USD":
        current_rate = 1.0 / all_rates.get(base, 1.0)
    else:
        # Convert via USD
        base_to_usd = 1.0 / all_rates.get(base, 1.0) if base != "USD" else 1.0