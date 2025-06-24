import pandas as pd
import streamlit as st
import numpy as np
from itertools import combinations

import warnings
import time
import hashlib
import json
from datetime import datetime
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
warnings.filterwarnings('ignore')
import numpy as np

def linear_regression_fast(x, y):
    """Fast linear regression replacement for sklearn"""
    n = len(x)
    if n < 2:
        return 0.0, 0.0
    
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    xy_sum = np.sum((x - x_mean) * (y - y_mean))
    xx_sum = np.sum((x - x_mean) ** 2)
    yy_sum = np.sum((y - y_mean) ** 2)
    
    if xx_sum == 0.0 or yy_sum == 0.0:
        return 0.0, 0.0
    
    beta = xy_sum / xx_sum
    correlation = xy_sum / np.sqrt(xx_sum * yy_sum)
    r_squared = correlation ** 2
    
    return beta, r_squared
# Configure Streamlit page
st.set_page_config(
    page_title="Range Bound Strategies",
    page_icon="📊",
    layout="wide"
)
def apply_modern_dashboard_theme():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f29 50%, #0f1419 100%);
        color: #e8eaed !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f29 50%, #0f1419 100%);
        color: #e8eaed !important;
    }

    /* Headers with gradient text */
    h1 {
        background: linear-gradient(135deg, #4fc3f7 0%, #29b6f6 50%, #03a9f4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700 !important;
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem !important;
    }

    h2 {
        background: linear-gradient(135deg, #66bb6a 0%, #4caf50 50%, #388e3c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 600 !important;
        font-size: 1.8rem !important;
        margin: 1rem 0 0.5rem 0 !important;
    }

    h3 {
        background: linear-gradient(135deg, #ff7043 0%, #ff5722 50%, #d84315 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 600 !important;
        font-size: 1.4rem !important;
    }

    h4, h5, h6, label, p, span, div {
        color: #e8eaed !important;
        font-weight: 400 !important;
    }

    /* Sidebar with glassmorphism */
    section[data-testid="stSidebar"] {
        background: rgba(26, 31, 41, 0.8) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(79, 195, 247, 0.2) !important;
    }

    section[data-testid="stSidebar"] > div {
        background: transparent !important;
    }

    /* Enhanced Metrics Cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(79, 195, 247, 0.1) 0%, rgba(41, 182, 246, 0.05) 100%) !important;
        border: 1px solid rgba(79, 195, 247, 0.2) !important;
        border-radius: 12px !important;
        padding: 16px !important;
        backdrop-filter: blur(10px) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
        transition: all 0.3s ease !important;
    }

    [data-testid="metric-container"]:hover {
        border-color: rgba(79, 195, 247, 0.4) !important;
        box-shadow: 0 12px 40px rgba(79, 195, 247, 0.2) !important;
        transform: translateY(-2px) !important;
    }

    /* Modern Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #4fc3f7 0%, #29b6f6 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.5rem !important;
        box-shadow: 0 4px 12px rgba(79, 195, 247, 0.3) !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #29b6f6 0%, #03a9f4 100%) !important;
        box-shadow: 0 6px 20px rgba(79, 195, 247, 0.4) !important;
        transform: translateY(-1px) !important;
    }

    .stButton > button:active {
        transform: translateY(0px) !important;
        box-shadow: 0 2px 8px rgba(79, 195, 247, 0.3) !important;
    }

    /* Enhanced Form Controls */
    .stSelectbox > div > div {
        background: rgba(15, 20, 25, 0.9) !important;
        border: 1px solid rgba(79, 195, 247, 0.4) !important;
        border-radius: 8px !important;
        color: #ffffff !important;
    }

    .stSelectbox > div > div > div {
        color: #ffffff !important;
        background: rgba(15, 20, 25, 0.9) !important;
    }

    .stSelectbox label {
        color: #4fc3f7 !important;
        font-weight: 600 !important;
    }

    .stMultiSelect > div > div {
        background: rgba(15, 20, 25, 0.9) !important;
        border: 1px solid rgba(79, 195, 247, 0.4) !important;
        border-radius: 8px !important;
    }

    .stMultiSelect > div > div > div {
        color: #ffffff !important;
        background: rgba(15, 20, 25, 0.9) !important;
    }

    .stMultiSelect label {
        color: #4fc3f7 !important;
        font-weight: 600 !important;
    }

    /* MultiSelect dropdown options */
    .stMultiSelect > div > div > div > div {
        background: rgba(15, 20, 25, 0.95) !important;
        color: #ffffff !important;
        border: 1px solid rgba(79, 195, 247, 0.3) !important;
    }

    /* MultiSelect selected items */
    .stMultiSelect > div > div span[data-baseweb="tag"] {
        background: linear-gradient(135deg, #4fc3f7 0%, #29b6f6 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
    }

    .stNumberInput > div > div > input {
        background: rgba(15, 20, 25, 0.9) !important;
        border: 1px solid rgba(79, 195, 247, 0.4) !important;
        border-radius: 8px !important;
        color: #ffffff !important;
        font-weight: 500 !important;
    }

    .stNumberInput label {
        color: #4fc3f7 !important;
        font-weight: 600 !important;
    }

    /* Dropdown options styling */
    [data-baseweb="popover"] {
        background: rgba(15, 20, 25, 0.95) !important;
        border: 1px solid rgba(79, 195, 247, 0.3) !important;
        border-radius: 8px !important;
        backdrop-filter: blur(20px) !important;
    }

    [role="option"] {
        background: rgba(15, 20, 25, 0.9) !important;
        color: #ffffff !important;
        border-radius: 4px !important;
        margin: 2px !important;
    }

    [role="option"]:hover {
        background: rgba(79, 195, 247, 0.2) !important;
        color: #ffffff !important;
    }

    [aria-selected="true"][role="option"] {
        background: linear-gradient(135deg, #4fc3f7 0%, #29b6f6 100%) !important;
        color: #ffffff !important;
    }

    .stCheckbox > label {
        background: rgba(26, 31, 41, 0.5) !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
        border: 1px solid rgba(79, 195, 247, 0.2) !important;
    }

    /* Enhanced Expandable Sections */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(102, 187, 106, 0.15) 0%, rgba(76, 175, 80, 0.1) 100%) !important;
        border: 1px solid rgba(102, 187, 106, 0.3) !important;
        border-radius: 12px !important;
        color: #e8eaed !important;
        font-weight: 600 !important;
        backdrop-filter: blur(10px) !important;
    }

    .streamlit-expanderContent {
        background: rgba(26, 31, 41, 0.5) !important;
        border: 1px solid rgba(102, 187, 106, 0.2) !important;
        border-top: none !important;
        border-radius: 0 0 12px 12px !important;
        backdrop-filter: blur(10px) !important;
    }

    /* Enhanced Data Tables */
    .dataframe {
        background: rgba(26, 31, 41, 0.8) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(79, 195, 247, 0.2) !important;
        overflow: hidden !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
    }

    .dataframe th {
        background: linear-gradient(135deg, rgba(79, 195, 247, 0.2) 0%, rgba(41, 182, 246, 0.15) 100%) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border-bottom: 2px solid rgba(79, 195, 247, 0.3) !important;
    }

    .dataframe td {
        color: #e8eaed !important;
        border-bottom: 1px solid rgba(79, 195, 247, 0.1) !important;
    }

    .dataframe tr:hover {
        background: rgba(79, 195, 247, 0.05) !important;
    }

    /* Info/Warning/Success Boxes */
    .stInfo {
        background: linear-gradient(135deg, rgba(3, 169, 244, 0.1) 0%, rgba(3, 169, 244, 0.05) 100%) !important;
        border: 1px solid rgba(3, 169, 244, 0.3) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px) !important;
    }

    .stWarning {
        background: linear-gradient(135deg, rgba(255, 152, 0, 0.1) 0%, rgba(255, 152, 0, 0.05) 100%) !important;
        border: 1px solid rgba(255, 152, 0, 0.3) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px) !important;
    }

    .stSuccess {
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.1) 0%, rgba(76, 175, 80, 0.05) 100%) !important;
        border: 1px solid rgba(76, 175, 80, 0.3) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px) !important;
    }

    .stError {
        background: linear-gradient(135deg, rgba(244, 67, 54, 0.1) 0%, rgba(244, 67, 54, 0.05) 100%) !important;
        border: 1px solid rgba(244, 67, 54, 0.3) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px) !important;
    }

    /* Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(135deg, #4fc3f7 0%, #29b6f6 100%) !important;
        border-radius: 8px !important;
    }

    /* Scrollbars */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(26, 31, 41, 0.5);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #4fc3f7 0%, #29b6f6 100%);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #29b6f6 0%, #03a9f4 100%);
    }

    /* Custom spacing */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }

    /* Animated loading spinner */
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    .stSpinner > div {
        border-color: #4fc3f7 transparent #4fc3f7 transparent !important;
    }
    </style>
    """, unsafe_allow_html=True)
# Apply it
apply_modern_dashboard_theme()



# ==================== LOCAL RANGE TRADING FUNCTIONS ====================

def detect_local_ranges(historical_values, window_size=10, min_range_duration=5):
    """
    Detect local trading ranges using rolling statistics
    """
    if len(historical_values) < window_size * 2:
        return None
    
    # Calculate rolling statistics
    rolling_mean = pd.Series(historical_values).rolling(window=window_size, center=True).mean()
    rolling_std = pd.Series(historical_values).rolling(window=window_size, center=True).std()
    
    # Identify periods of low volatility (potential range-bound periods)
    volatility_threshold = rolling_std.quantile(0.3)  # Bottom 30% of volatility periods
    
    # Find recent local range (last 20 periods)
    recent_window = min(20, len(historical_values))
    recent_values = historical_values[-recent_window:]
    
    recent_min = np.min(recent_values)
    recent_max = np.max(recent_values)
    recent_mean = np.mean(recent_values)
    recent_range = recent_max - recent_min
    
    return {
        'recent_min': recent_min,
        'recent_max': recent_max,
        'recent_mean': recent_mean,
        'recent_range': recent_range,
        'recent_window_size': recent_window,
        'is_in_recent_range': recent_range > 0
    }

def detect_support_resistance_levels(historical_values, lookback_period=20):
    """
    Detect support and resistance levels using price clustering
    """
    if len(historical_values) < lookback_period:
        recent_values = historical_values
    else:
        recent_values = historical_values[-lookback_period:]
    
    try:
        # Find local peaks and troughs
        peaks, _ = find_peaks(recent_values, distance=3)
        resistance_levels = recent_values[peaks] if len(peaks) > 0 else []
        
        # Find support levels (troughs) - invert data to find peaks
        troughs, _ = find_peaks(-recent_values, distance=3)
        support_levels = recent_values[troughs] if len(troughs) > 0 else []
        
        # Cluster levels to find significant ones
        all_levels = np.concatenate([resistance_levels, support_levels]) if len(resistance_levels) > 0 and len(support_levels) > 0 else []
        
        significant_levels = []
        if len(all_levels) >= 2:
            try:
                # Use KMeans to cluster similar levels
                n_clusters = min(5, len(all_levels))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(all_levels.reshape(-1, 1))
                significant_levels = kmeans.cluster_centers_.flatten()
                significant_levels = sorted(significant_levels)
            except:
                # Fallback: use percentiles
                significant_levels = [
                    np.percentile(recent_values, 20),
                    np.percentile(recent_values, 50),
                    np.percentile(recent_values, 80)
                ]
        
        return {
            'support_levels': [level for level in significant_levels if level <= np.median(recent_values)],
            'resistance_levels': [level for level in significant_levels if level > np.median(recent_values)],
            'all_significant_levels': significant_levels,
            'recent_peaks': resistance_levels,
            'recent_troughs': support_levels
        }
    except:
        return {
            'support_levels': [],
            'resistance_levels': [],
            'all_significant_levels': [],
            'recent_peaks': [],
            'recent_troughs': []
        }

def calculate_local_range_signals(current_value, historical_values, local_range_mode='adaptive', breakout_threshold=0.20):
    """
    Calculate trading signals based on local ranges with breakout protection
    """
    results = {
        'signal': 'HOLD',
        'signal_strength': 0,
        'local_position': 0.5,
        'range_type': local_range_mode,
        'confidence': 0,
        'breakout_detected': False,
        'breakout_distance': 0.0
    }
    
    if local_range_mode == 'recent':
        # Use recent X periods as the trading range
        recent_periods = min(15, len(historical_values))
        recent_values = historical_values[-recent_periods:]
        
        local_min = np.min(recent_values)
        local_max = np.max(recent_values)
        local_range = local_max - local_min
        
        if local_range > 0:
            # Check for breakout before generating signals
            breakout_buffer = local_range * breakout_threshold
            extended_min = local_min - breakout_buffer
            extended_max = local_max + breakout_buffer
            
            # Calculate breakout distance
            if current_value < extended_min:
                results['breakout_detected'] = True
                results['breakout_distance'] = (extended_min - current_value) / local_range
                results['signal'] = 'HOLD'
                results['signal_strength'] = 0
                results['local_position'] = (current_value - local_min) / local_range
                results['confidence'] = 0
            elif current_value > extended_max:
                results['breakout_detected'] = True
                results['breakout_distance'] = (current_value - extended_max) / local_range
                results['signal'] = 'HOLD'
                results['signal_strength'] = 0
                results['local_position'] = (current_value - local_min) / local_range
                results['confidence'] = 0
            else:
                # Within acceptable range - generate normal signals
                position = (current_value - local_min) / local_range
                results['local_position'] = position
                
                # More aggressive thresholds for local ranges
                if position <= 0.2:  # Bottom 20% of recent range
                    results['signal'] = 'BUY'
                    results['signal_strength'] = (0.2 - position) * 500
                elif position >= 0.8:  # Top 20% of recent range
                    results['signal'] = 'SELL'
                    results['signal_strength'] = (position - 0.8) * 500
                    
                results['confidence'] = min(100, recent_periods / 15 * 100)
            
            results['local_min'] = local_min
            results['local_max'] = local_max
            results['local_range'] = local_range
            results['extended_min'] = extended_min
            results['extended_max'] = extended_max
    
    elif local_range_mode == 'adaptive':
        # Adaptive range based on recent volatility
        local_info = detect_local_ranges(historical_values)
        
        if local_info and local_info['is_in_recent_range']:
            local_min = local_info['recent_min']
            local_max = local_info['recent_max']
            local_range = local_info['recent_range']
            
            if local_range > 0:
                # Check for breakout
                breakout_buffer = local_range * breakout_threshold
                extended_min = local_min - breakout_buffer
                extended_max = local_max + breakout_buffer
                
                # Calculate breakout distance
                if current_value < extended_min:
                    results['breakout_detected'] = True
                    results['breakout_distance'] = (extended_min - current_value) / local_range
                    results['signal'] = 'HOLD'
                    results['signal_strength'] = 0
                    results['local_position'] = (current_value - local_min) / local_range
                    results['confidence'] = 0
                elif current_value > extended_max:
                    results['breakout_detected'] = True
                    results['breakout_distance'] = (current_value - extended_max) / local_range
                    results['signal'] = 'HOLD'
                    results['signal_strength'] = 0
                    results['local_position'] = (current_value - local_min) / local_range
                    results['confidence'] = 0
                else:
                    # Within acceptable range
                    position = (current_value - local_min) / local_range
                    results['local_position'] = position
                    
                    # Adaptive thresholds
                    buy_threshold = 0.25
                    sell_threshold = 0.75
                    
                    if position <= buy_threshold:
                        results['signal'] = 'BUY'
                        results['signal_strength'] = (buy_threshold - position) * 400
                    elif position >= sell_threshold:
                        results['signal'] = 'SELL'
                        results['signal_strength'] = (position - sell_threshold) * 400
                    
                    results['confidence'] = 75
                
                results['local_min'] = local_min
                results['local_max'] = local_max
                results['local_range'] = local_range
                results['extended_min'] = extended_min
                results['extended_max'] = extended_max
    
    elif local_range_mode == 'support_resistance':
        # Use support/resistance levels
        sr_info = detect_support_resistance_levels(historical_values)
        
        if sr_info['support_levels'] and sr_info['resistance_levels']:
            support_levels = sr_info['support_levels']
            resistance_levels = sr_info['resistance_levels']
            
            nearest_support = max([s for s in support_levels if s <= current_value], default=min(support_levels))
            nearest_resistance = min([r for r in resistance_levels if r >= current_value], default=max(resistance_levels))
            
            level_range = nearest_resistance - nearest_support
            if level_range > 0:
                # Check for breakout beyond support/resistance levels
                breakout_buffer = level_range * breakout_threshold
                extended_support = nearest_support - breakout_buffer
                extended_resistance = nearest_resistance + breakout_buffer
                
                # Calculate breakout distance
                if current_value < extended_support:
                    results['breakout_detected'] = True
                    results['breakout_distance'] = (extended_support - current_value) / level_range
                    results['signal'] = 'HOLD'
                    results['signal_strength'] = 0
                    results['local_position'] = (current_value - nearest_support) / level_range
                    results['confidence'] = 0
                elif current_value > extended_resistance:
                    results['breakout_detected'] = True
                    results['breakout_distance'] = (current_value - extended_resistance) / level_range
                    results['signal'] = 'HOLD'
                    results['signal_strength'] = 0
                    results['local_position'] = (current_value - nearest_support) / level_range
                    results['confidence'] = 0
                else:
                    # Within acceptable range
                    position = (current_value - nearest_support) / level_range
                    results['local_position'] = position
                    
                    # Signal based on proximity to levels
                    support_distance = abs(current_value - nearest_support) / level_range
                    resistance_distance = abs(current_value - nearest_resistance) / level_range
                    
                    if support_distance <= 0.1:  # Within 10% of support
                        results['signal'] = 'BUY'
                        results['signal_strength'] = (0.1 - support_distance) * 1000
                    elif resistance_distance <= 0.1:  # Within 10% of resistance
                        results['signal'] = 'SELL'
                        results['signal_strength'] = (0.1 - resistance_distance) * 1000
                    
                    results['confidence'] = min(100, len(sr_info['all_significant_levels']) * 20)
                
                results['nearest_support'] = nearest_support
                results['nearest_resistance'] = nearest_resistance
                results['support_levels'] = support_levels
                results['resistance_levels'] = resistance_levels
                results['extended_support'] = extended_support
                results['extended_resistance'] = extended_resistance
    
    return results

# ==================== INSTRUMENT PARSING FUNCTIONS ====================

def parse_instrument_components(instrument_name):
    """
    Parse instrument name into Market, Month, Year components
    Returns: (market, month, year) or (None, None, None) if parsing fails
    """
    try:
        instrument = instrument_name.upper()
        
        # Extract market (SRA, ER, CRA, SON)
        market = None
        if instrument.startswith('SRA'):
            market = 'SRA'
            remaining = instrument[3:]
        elif instrument.startswith('ER'):
            market = 'ER'
            remaining = instrument[2:]
        elif instrument.startswith('CRA'):
            market = 'CRA'
            remaining = instrument[3:]
        elif instrument.startswith('SON'):
            market = 'SON'
            remaining = instrument[3:]
        else:
            return None, None, None
        
        # Extract month (H, M, U, Z)
        month = None
        month_codes = ['H', 'M', 'U', 'Z']
        for i, char in enumerate(remaining):
            if char in month_codes:
                month = char
                year_part = remaining[i+1:]
                break
        
        if not month:
            return None, None, None
        
        # Extract year (25, 26, 27, etc.)
        year = None
        # Look for 2-digit year at the end (before suffixes)
        suffixes = ['3MS', '6MS', '12MS', '3MF', '6MF', '12MF']
        year_string = year_part
        
        # Remove suffix if present
        for suffix in suffixes:
            if year_string.endswith(suffix):
                year_string = year_string[:-len(suffix)]
                break
        
        # Extract 2-digit year
        if len(year_string) >= 2 and year_string[:2].isdigit():
            year_num = int(year_string[:2])
            if 20 <= year_num <= 35:  # Valid range
                year = str(year_num)
        
        return market, month, year
        
    except Exception:
        return None, None, None

def filter_strategies_by_strategy_type(strategies_df, exclude_types):
    """
    Filter out specific strategy types (3MS, 6MS, 12MS, 3MF, 6MF, 12MF)
    """
    if strategies_df.empty or not exclude_types:
        return strategies_df
    
    def strategy_contains_excluded_types(strategy_name):
        # Check if strategy contains any excluded types
        for exclude_type in exclude_types:
            if exclude_type in strategy_name:
                return True
        return False
    
    # Keep strategies that DON'T contain excluded types
    mask = ~strategies_df['Strategy'].apply(strategy_contains_excluded_types)
    return strategies_df[mask]

def filter_strategies_by_specific_instruments(strategies_df, include_instruments=None, exclude_instruments=None):
    """
    Filter strategies by specific instrument names
    """
    if strategies_df.empty:
        return strategies_df
    
    def strategy_contains_instruments(strategy_name, instrument_list):
        """Check if strategy contains any instruments from the list"""
        if not instrument_list:
            return False
        
        # Parse instruments from strategy name
        parts = strategy_name.split(' - ')
        if len(parts) < 2:
            return False
        
        inst1 = parts[0]
        inst2_part = parts[1].split('*')[-1] if '*' in parts[1] else parts[1]
        strategy_instruments = [inst1, inst2_part]
        
        # Check if any strategy instrument is in the filter list
        return any(inst in instrument_list for inst in strategy_instruments)
    
    # Apply inclusion filter
    if include_instruments:
        mask = strategies_df['Strategy'].apply(
            lambda x: strategy_contains_instruments(x, include_instruments)
        )
        strategies_df = strategies_df[mask]
    
    # Apply exclusion filter
    if exclude_instruments:
        mask = strategies_df['Strategy'].apply(
            lambda x: not strategy_contains_instruments(x, exclude_instruments)
        )
        strategies_df = strategies_df[mask]
    
    return strategies_df

def filter_strategies_by_components(strategies_df, selected_markets, selected_months, selected_years):
    """
    Filter strategies based on market, month, year selections
    """
    if strategies_df.empty:
        return strategies_df
    
    def strategy_matches_filters(strategy_name):
        # Parse both instruments in the strategy
        parts = strategy_name.split(' - ')
        if len(parts) < 2:
            return False
        
        inst1 = parts[0]
        inst2_part = parts[1].split('*')[-1] if '*' in parts[1] else parts[1]
        
        # Check both instruments
        for inst in [inst1, inst2_part]:
            market, month, year = parse_instrument_components(inst)
            
            # If any component matches the filters, include the strategy
            market_match = not selected_markets or market in selected_markets
            month_match = not selected_months or month in selected_months
            year_match = not selected_years or year in selected_years
            
            # All components must match if filters are set
            if market_match and month_match and year_match:
                return True
        
        return False
    
    if not selected_markets and not selected_months and not selected_years:
        return strategies_df
    
    mask = strategies_df['Strategy'].apply(strategy_matches_filters)
    return strategies_df[mask]

# ==================== ENHANCED SIGNAL CALCULATION ====================

def calculate_enhanced_signals_with_local_ranges(live_df, regression_cache, historical_stats, 
                                               use_local_ranges=True, local_mode='adaptive',
                                               combine_signals='strongest', breakout_threshold=0.20):
    """
    Enhanced signal calculation that includes both global and local range analysis with breakout protection
    """
    results = []
    
    for pair_key, regression_data in regression_cache.items():
        inst1, inst2 = pair_key.split('_')
        
        if pair_key not in historical_stats:
            continue
        
        hist_stats = historical_stats[pair_key]
        
        # Skip if not range-bound globally
        if hist_stats['min_tests'] < 2 or hist_stats['max_tests'] < 2:
            continue
        
        try:
            # Get live values
            live_value1 = float(live_df[inst1].iloc[0])
            live_value2 = float(live_df[inst2].iloc[0])
            
            # Calculate current regression-adjusted strategy value
            beta = regression_data['beta']
            current_strategy = live_value1 - (beta * live_value2)
            
            # Global range analysis (existing logic)
            range_val = hist_stats['range']
            global_position = (current_strategy - hist_stats['min']) / range_val if range_val > 0 else 0.5
            
            # Global signals - IMPORTANT: Keep correlation filtering!
            global_signal = "HOLD"
            global_strength = 0
            correlation = regression_data['correlation']
            coeff_var = hist_stats['coeff_var']
            
            # CRITICAL: Only generate signals for high correlation (>=90%) and low coeff variation (<=50%)
            if global_position <= 0.25 and abs(correlation) >= 0.90 and coeff_var <= 0.5:
                global_signal = "BUY"
                global_strength = (0.25 - global_position) * abs(correlation) * (1/coeff_var) * 100
            elif global_position >= 0.75 and abs(correlation) >= 0.90 and coeff_var <= 0.5:
                global_signal = "SELL"
                global_strength = (global_position - 0.75) * abs(correlation) * (1/coeff_var) * 100
            
            # Local range analysis with breakout protection
            local_results = None
            if use_local_ranges and 'historical_strategy' in hist_stats:
                local_results = calculate_local_range_signals(
                    current_strategy, 
                    hist_stats['historical_strategy'], 
                    local_mode,
                    breakout_threshold
                )
            
            # Combine signals based on strategy (with breakout consideration)
            final_signal = global_signal
            final_strength = global_strength
            signal_source = "Global"
            breakout_info = ""
            
            if local_results:
                if local_results['breakout_detected']:
                    # If breakout detected, suppress local signals but may still use global
                    breakout_distance = local_results['breakout_distance']
                    breakout_info = f"Local breakout detected ({breakout_distance:.1%} beyond range)"
                    
                    # Only use global signal if it exists
                    if global_signal == 'HOLD':
                        final_signal = 'HOLD'
                        final_strength = 0
                        signal_source = f"No signal - {breakout_info}"
                    else:
                        signal_source = f"Global only - {breakout_info}"
                        
                elif local_results['signal'] != 'HOLD':
                    # Normal local signal processing (no breakout)
                    if combine_signals == 'strongest':
                        if local_results['signal_strength'] > global_strength:
                            final_signal = local_results['signal']
                            final_strength = local_results['signal_strength']
                            signal_source = f"Local ({local_mode})"
                    elif combine_signals == 'local_priority':
                        final_signal = local_results['signal']
                        final_strength = local_results['signal_strength']
                        signal_source = f"Local ({local_mode})"
                    elif combine_signals == 'global_priority' and global_signal == 'HOLD':
                        final_signal = local_results['signal']
                        final_strength = local_results['signal_strength']
                        signal_source = f"Local ({local_mode})"
            
            # Only include if it generates a signal
            if final_signal in ["BUY", "SELL"]:
                result_dict = {
                    'Strategy': f"{inst1} - {beta:.3f}*{inst2}",
                    'Current Value': round(current_strategy, 6),
                    'Global Range': round(hist_stats['range'], 6),
                    'Global Position': round(global_position, 4),
                    'Global Min': round(hist_stats['min'], 6),
                    'Global Max': round(hist_stats['max'], 6),
                    'Mean': round(hist_stats['mean'], 6),
                    'Coeff of Variation': round(coeff_var, 4),
                    'Correlation': round(correlation, 4),
                    'Beta Ratio': round(beta, 3),
                    'R-Squared': round(regression_data['r_squared'], 3),
                    'Signal': final_signal,
                    'Signal Strength': round(final_strength, 2),
                    'Signal Source': signal_source,
                    'Global Signal': global_signal,
                    'Global Strength': round(global_strength, 2),
                    'Min Tests': hist_stats['min_tests'],
                    'Max Tests': hist_stats['max_tests'],
                    'Instrument1': inst1,
                    'Instrument2': inst2
                }
                
                # Add local range info if available
                if local_results:
                    result_dict.update({
                        'Local Signal': local_results['signal'],
                        'Local Strength': round(local_results['signal_strength'], 2),
                        'Local Position': round(local_results['local_position'], 4),
                        'Local Confidence': round(local_results['confidence'], 1),
                        'Range Type': local_results['range_type'],
                        'Breakout Detected': local_results['breakout_detected'],
                        'Breakout Distance': round(local_results['breakout_distance'], 3) if local_results['breakout_detected'] else 0
                    })
                    
                    # Add specific local range boundaries if available
                    if 'local_min' in local_results:
                        result_dict.update({
                            'Local Min': round(local_results['local_min'], 6),
                            'Local Max': round(local_results['local_max'], 6),
                            'Local Range': round(local_results['local_range'], 6)
                        })
                        
                        # Add extended boundaries for breakout visualization
                        if 'extended_min' in local_results:
                            result_dict.update({
                                'Extended Min': round(local_results['extended_min'], 6),
                                'Extended Max': round(local_results['extended_max'], 6)
                            })
                    
                    # Add support/resistance levels if available
                    if 'nearest_support' in local_results:
                        result_dict.update({
                            'Nearest Support': round(local_results['nearest_support'], 6),
                            'Nearest Resistance': round(local_results['nearest_resistance'], 6)
                        })
                        
                        # Add extended support/resistance for breakout visualization
                        if 'extended_support' in local_results:
                            result_dict.update({
                                'Extended Support': round(local_results['extended_support'], 6),
                                'Extended Resistance': round(local_results['extended_resistance'], 6)
                            })
                
                results.append(result_dict)
                
        except Exception:
            continue
    
    return pd.DataFrame(results)

# ==================== EXISTING FUNCTIONS ====================

def calculate_regression_ratio_original(hist_series1, hist_series2):
    """Calculate regression ratio without sklearn"""
    try:
        aligned_data = pd.DataFrame({
            'y': hist_series1,
            'x': hist_series2
        }).dropna()
        
        if len(aligned_data) < 10:
            return None, None
        
        y = aligned_data['y'].values
        x = aligned_data['x'].values
        
        beta, r_squared = linear_regression_fast(x, y)
        
        if beta == 0.0 and r_squared == 0.0:
            return None, None
            
        return beta, r_squared
        
    except Exception:
        return None, None

@st.cache_data(ttl=86400)
def calculate_regression_ratio_cached(hist1_hash, hist2_hash, hist_series1, hist_series2):
    """Cached version of regression calculation"""
    return calculate_regression_ratio_original(hist_series1, hist_series2)

@st.cache_data(ttl=86400)
def load_historical_data():
    """Load and cache historical data - updates once daily"""
    try:
        df = pd.read_excel('QH-Data.xlsm', sheet_name='IM')
        df_clean = df[(df.iloc[:, :70] != 0).all(axis=1)]
        df_recent = df_clean.head(60).copy()
        st.info(f"📅 Historical data loaded: {len(df_recent)} days from {len(df_clean)} total valid rows")
        return df_recent
    except Exception as e:
        st.error(f"Error loading historical data: {str(e)}")
        return None

@st.cache_data(ttl=300)
def load_live_data():
    """Load live data - updates every 5 minutes"""
    try:
        live = pd.read_excel('QH-Data.xlsm', sheet_name='Live')
        live_copy = live.copy()
        names = live_copy.iloc[0, 1:].values
        live_prices = live_copy.iloc[2, 1:].values
        live_df = pd.DataFrame([live_prices], columns=names)
        data_hash = hashlib.md5(str(live_prices).encode()).hexdigest()
        return live_df, data_hash
    except Exception as e:
        st.error(f"Error loading live data: {str(e)}")
        return None, None

@st.cache_data(ttl=86400)
def get_instrument_groups_daily(df_recent):
    """Cache instrument categorization - based on daily historical data"""
    instruments_3ms = []
    instruments_6ms = []
    instruments_12ms = []
    instruments_3mf = []
    instruments_6mf = []
    instruments_12mf = []
    
    for col in df_recent.columns:
        if isinstance(col, str):
            year_suffix = extract_year_suffix(col)
            if year_suffix is not None and year_suffix <= 28:
                if col.endswith('3MS'):
                    instruments_3ms.append(col)
                elif col.endswith('6MS'):
                    instruments_6ms.append(col)
                elif col.endswith('12MS'):
                    instruments_12ms.append(col)
                elif col.endswith('3MF'):
                    instruments_3mf.append(col)
                elif col.endswith('6MF'):
                    instruments_6mf.append(col)
                elif col.endswith('12MF'):
                    instruments_12mf.append(col)
    
    return {
        '3MS': instruments_3ms,
        '6MS': instruments_6ms,
        '12MS': instruments_12ms,
        '3MF': instruments_3mf,
        '6MF': instruments_6mf,
        '12MF': instruments_12mf
    }

@st.cache_data(ttl=86400)
def calculate_all_regressions_daily(df_recent, instrument_groups):
    """Pre-calculate all regression coefficients - updates once daily"""
    regression_cache = {}
    valid_pairs = []
    
    # Same-type pairs with distance constraints
    for group_name, instruments in instrument_groups.items():
        if len(instruments) >= 2:
            for inst1, inst2 in combinations(instruments, 2):
                if is_valid_strategy_pair(inst1, inst2, group_name):
                    if inst1 in df_recent.columns and inst2 in df_recent.columns:
                        valid_pairs.append((inst1, inst2, 'same_type'))
    
    # Cross-type pairs
    all_instruments = []
    for instruments in instrument_groups.values():
        all_instruments.extend(instruments)
    
    for inst1, inst2 in combinations(all_instruments, 2):
        inst1_type = get_instrument_type(inst1)
        inst2_type = get_instrument_type(inst2)
        
        if inst1_type != inst2_type:
            if inst1 in df_recent.columns and inst2 in df_recent.columns:
                valid_pairs.append((inst1, inst2, 'cross_type'))
    
    total_pairs = len(valid_pairs)
    
    if total_pairs == 0:
        st.warning("No valid instrument pairs found for regression calculation")
        return regression_cache
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, (inst1, inst2, pair_type) in enumerate(valid_pairs):
        if idx % 25 == 0 or idx == total_pairs - 1:
            progress_value = min(idx / total_pairs, 1.0)
            progress_bar.progress(progress_value)
            status_text.text(f"Pre-calculating regressions: {idx + 1}/{total_pairs} ({pair_type})")
        
        try:
            hist1 = df_recent[inst1].dropna()
            hist2 = df_recent[inst2].dropna()
            
            if len(hist1) >= 10 and len(hist2) >= 10:
                beta, r_squared = calculate_regression_ratio_original(hist1, hist2)
                if beta is not None and r_squared is not None:
                    regression_cache[f"{inst1}_{inst2}"] = {
                        'beta': beta,
                        'r_squared': r_squared,
                        'correlation': hist1.corr(hist2)
                    }
        except Exception as e:
            continue
    
    progress_bar.progress(1.0)
    status_text.text(f"Completed: {len(regression_cache)} regression relationships calculated")
    
    time.sleep(1)
    progress_bar.empty()
    status_text.empty()
    
    st.success(f"✅ Pre-calculated {len(regression_cache)} regression relationships (cached for 24 hours)")
    
    return regression_cache

@st.cache_data(ttl=86400)
def calculate_historical_statistics_daily(df_recent, regression_cache):
    """Pre-calculate all historical statistics - updates once daily"""
    historical_stats = {}
    
    for pair_key, regression_data in regression_cache.items():
        inst1, inst2 = pair_key.split('_')
        beta = regression_data['beta']
        
        if inst1 in df_recent.columns and inst2 in df_recent.columns:
            hist1 = df_recent[inst1].dropna()
            hist2 = df_recent[inst2].dropna()
            
            historical_strategy = hist1 - (beta * hist2)
            historical_strategy = historical_strategy.dropna()
            
            if len(historical_strategy) >= 10:
                mean_strategy = historical_strategy.mean()
                std_strategy = historical_strategy.std()
                min_strategy = historical_strategy.min()
                max_strategy = historical_strategy.max()
                range_strategy = max_strategy - min_strategy
                
                min_tests, max_tests = calculate_boundary_tests_vectorized(historical_strategy)
                
                coeff_var = std_strategy / abs(mean_strategy) if abs(mean_strategy) > 0.0001 else 999
                
                historical_stats[pair_key] = {
                    'mean': mean_strategy,
                    'std': std_strategy,
                    'min': min_strategy,
                    'max': max_strategy,
                    'range': range_strategy,
                    'min_tests': min_tests,
                    'max_tests': max_tests,
                    'coeff_var': coeff_var,
                    'historical_strategy': historical_strategy.values
                }
    
    st.success(f"✅ Pre-calculated historical statistics for {len(historical_stats)} strategies (cached for 24 hours)")
    
    return historical_stats

def calculate_boundary_tests_vectorized(historical_strategy):
    """Optimized boundary testing using vectorized operations"""
    try:
        min_val = historical_strategy.min()
        max_val = historical_strategy.max()
        range_val = max_val - min_val
        
        if range_val <= 0:
            return 0, 0
        
        min_threshold = min_val + (range_val * 0.05)
        max_threshold = max_val - (range_val * 0.05)
        
        min_tests = np.sum(historical_strategy <= min_threshold)
        max_tests = np.sum(historical_strategy >= max_threshold)
        
        return min_tests, max_tests
    except:
        return 0, 0

def extract_year_suffix(instrument_name):
    """Extract the year suffix from instrument name"""
    try:
        name_without_suffix = instrument_name
        for suffix in ['3MS', '6MS', '12MS', '3MF', '6MF', '12MF']:
            if name_without_suffix.endswith(suffix):
                name_without_suffix = name_without_suffix[:-len(suffix)]
                break
        
        if len(name_without_suffix) >= 2:
            potential_year = name_without_suffix[-2:]
            if potential_year.isdigit():
                return int(potential_year)
        
        for i in range(len(instrument_name) - 1):
            if instrument_name[i:i+2].isdigit():
                potential_year = int(instrument_name[i:i+2])
                if 20 <= potential_year <= 35:
                    return potential_year
        
        return None
    except:
        return None

def get_instrument_type(instrument):
    """Get the type suffix of an instrument"""
    if instrument.endswith('3MS'):
        return '3MS'
    elif instrument.endswith('6MS'):
        return '6MS'
    elif instrument.endswith('12MS'):
        return '12MS'
    elif instrument.endswith('3MF'):
        return '3MF'
    elif instrument.endswith('6MF'):
        return '6MF'
    elif instrument.endswith('12MF'):
        return '12MF'
    return 'Unknown'

def extract_contract_info(instrument_name):
    """Extract contract month and year from instrument name"""
    try:
        month_codes = ['H', 'M', 'U', 'Z']
        
        for i, char in enumerate(instrument_name):
            if char in month_codes:
                month = char
                if i + 1 < len(instrument_name) and i + 2 < len(instrument_name):
                    year_str = instrument_name[i+1:i+3]
                    if year_str.isdigit():
                        year = int(year_str)
                        if year <= 28:
                            return month, year
        
        return None, None
    except:
        return None, None

def calculate_contract_distance(month1, year1, month2, year2):
    """Calculate the distance between two contracts in months"""
    if month1 is None or month2 is None or year1 is None or year2 is None:
        return float('inf')
    
    month_map = {'H': 3, 'M': 6, 'U': 9, 'Z': 12}
    
    if month1 not in month_map or month2 not in month_map:
        return float('inf')
    
    total_months1 = year1 * 12 + month_map[month1]
    total_months2 = year2 * 12 + month_map[month2]
    
    return abs(total_months1 - total_months2) // 3

def is_valid_strategy_pair(inst1, inst2, strategy_type):
    """Check if two instruments can form a valid strategy based on contract distance"""
    month1, year1 = extract_contract_info(inst1)
    month2, year2 = extract_contract_info(inst2)
    
    contract_distance = calculate_contract_distance(month1, year1, month2, year2)
    
    max_distances = {
        '3MS': 1, '6MS': 2, '12MS': 4,
        '3MF': 1, '6MF': 2, '12MF': 4
    }
    
    max_allowed = max_distances.get(strategy_type, 1)
    
    return contract_distance <= max_allowed

# ==================== UI ENHANCEMENT FUNCTIONS ====================

def render_local_range_controls():
    """Render controls for local range trading options"""
    st.header("🎯 Range Trading Configuration")
    # Enhanced metrics with modern cards
    st.markdown("""
    <div style="margin: 2rem 0;">
    <h2 style="text-align: center; margin-bottom: 1.5rem;">📈 Trading Signals Overview</h2>
    </div>
    """, unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        use_local_ranges = st.checkbox(
            "🔍 Enable Local Range Analysis",
            value=True,
            help="Analyze shorter-term trading ranges in addition to global ranges"
        )
    
    with col2:
        local_mode = st.selectbox(
            "Local Range Method:",
            ['recent', 'adaptive', 'support_resistance'],
            index=1,
            help="""
            Recent: Use last 15 periods as trading range
            Adaptive: Dynamic range based on volatility  
            Support/Resistance: Key levels based on peaks/troughs
            """
        )
    
    with col3:
        combine_signals = st.selectbox(
            "Signal Combination:",
            ['strongest', 'local_priority', 'global_priority'],
            index=0,
            help="""
            Strongest: Use signal with highest strength
            Local Priority: Prefer local signals when available
            Global Priority: Prefer global signals when available
            """
        )
    
    with col4:
        breakout_threshold = st.slider(
            "Breakout Threshold (%):",
            min_value=10.0,
            max_value=50.0,
            value=20.0,
            step=5.0,
            help="Suppress local signals when price moves beyond local range by this percentage"
        ) / 100.0  # Convert to decimal
    
    with col5:
        show_both_signals = st.checkbox(
            "📊 Show Both Signal Types",
            value=True,
            help="Display both global and local signals in results table"
        )
    
    # Information about breakout protection
    if use_local_ranges:
        st.info(f"🛡️ **Breakout Protection Active**: Local signals suppressed when price moves >{breakout_threshold*100:.0f}% beyond local range boundaries")
    
    return use_local_ranges, local_mode, combine_signals, show_both_signals, breakout_threshold

def render_enhanced_batch_filters(strategies_df):
    """Render batch filter controls that don't auto-refresh"""
    st.header("🔍 Enhanced Strategy Filters")
    st.info("💡 **Set all filters below and click 'Apply Filters' to refresh results**")
    
    # Create filter sections
    with st.container():
        # Row 1: Core filters
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            correlation_threshold = st.number_input(
                "Minimum Correlation (%):",
                min_value=0.0,
                max_value=100.0,
                value=90.0,
                step=0.1,
                help="Higher correlation means more predictable relationships",
                key="correlation_filter"
            )
        
        with col2:
            # Enhanced range filtering
            if not strategies_df.empty:
                if 'Global Range' in strategies_df.columns:
                    min_range = float(strategies_df['Global Range'].min())
                    max_range = float(strategies_df['Global Range'].max())
                    median_range = float(strategies_df['Global Range'].median())
                else:
                    min_range = float(strategies_df['Range'].min()) if 'Range' in strategies_df.columns else 0.0
                    max_range = float(strategies_df['Range'].max()) if 'Range' in strategies_df.columns else 1.0
                    median_range = float(strategies_df['Range'].median()) if 'Range' in strategies_df.columns else 0.5
            else:
                min_range, max_range, median_range = 0.0, 1.0, 0.5
            
            range_min_threshold = st.number_input(
                "Minimum Range:",
                min_value=min_range,
                max_value=max_range,
                value=median_range * 0.5,
                step=(max_range - min_range) / 100 if max_range > min_range else 0.01,
                format="%.6f",
                help="Larger range means more profit potential",
                key="range_min_filter"
            )
        
        with col3:
            range_max_threshold = st.number_input(
                "Maximum Range:",
                min_value=min_range,
                max_value=max_range,
                value=median_range * 2.0,
                step=(max_range - min_range) / 100 if max_range > min_range else 0.01,
                format="%.6f",
                help="Lower maximum filters out overly volatile strategies",
                key="range_max_filter"
            )
        
        with col4:
            # Signal source filter if available
            if not strategies_df.empty and 'Signal Source' in strategies_df.columns:
                signal_source_filter = st.selectbox(
                    "Signal Source Filter:",
                    ['All', 'Global', 'Local (recent)', 'Local (adaptive)', 'Local (support_resistance)'],
                    index=0,
                    help="Filter by signal generation method",
                    key="signal_source_filter"
                )
            else:
                signal_source_filter = 'All'
        
        st.markdown("---")
        
        # Row 2: Instrument component filters
        st.subheader("🏛️ Instrument Component Filters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Markets:**")
            selected_markets = st.multiselect(
                "Select Markets:",
                options=['SRA', 'ER', 'CRA', 'SON'],
                default=[],
                help="SRA=SOFR, ER=Euribor, CRA=CORRA, SON=SONIA",
                key="market_filter"
            )
        
        with col2:
            st.write("**Months:**")
            selected_months = st.multiselect(
                "Select Months:",
                options=['H', 'M', 'U', 'Z'],
                default=[],
                help="H=March, M=June, U=September, Z=December",
                key="month_filter"
            )
        
        with col3:
            st.write("**Years:**")
            selected_years = st.multiselect(
                "Select Years:",
                options=['25', '26', '27', '28'],
                default=[],
                help="25=2025, 26=2026, 27=2027, 28=2028",
                key="year_filter"
            )
        
        st.markdown("---")
        
        # NEW SECTION: Specific Instrument Filtering
        st.subheader("🎯 Specific Instrument Filtering")
        
        # Get unique instrument names from all strategies
        all_instruments = set()
        if not strategies_df.empty:
            for strategy in strategies_df['Strategy']:
                parts = strategy.split(' - ')
                if len(parts) >= 2:
                    inst1 = parts[0]
                    inst2_part = parts[1].split('*')[-1] if '*' in parts[1] else parts[1]
                    all_instruments.add(inst1)
                    all_instruments.add(inst2_part)
        
        available_instruments = sorted(list(all_instruments))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Include Specific Instruments:**")
            include_specific_instruments = st.multiselect(
                "Select instruments to include (leave empty to include all):",
                options=available_instruments,
                default=[],
                help="Only strategies containing these instruments will be shown.",
                key="include_instruments_filter"
            )
            
            if include_specific_instruments:
                st.info(f"✅ Will show strategies with {len(include_specific_instruments)} selected instruments")
        
        with col2:
            st.write("**Exclude Specific Instruments:**")
            exclude_specific_instruments = st.multiselect(
                "Select instruments to exclude:",
                options=available_instruments,
                default=[],
                help="Strategies containing these instruments will be hidden.",
                key="exclude_instruments_filter"
            )
            
            if exclude_specific_instruments:
                st.warning(f"❌ Will exclude strategies with {len(exclude_specific_instruments)} selected instruments")
        
        # Show conflict warning
        if include_specific_instruments and exclude_specific_instruments:
            overlap = set(include_specific_instruments) & set(exclude_specific_instruments)
            if overlap:
                st.error(f"⚠️ **Conflict**: {len(overlap)} instruments in both lists. Exclude takes priority.")
        
        st.markdown("---")
        
        # Row 3: Additional filters (Strategy Types)
        
        # Row 3: Additional filters
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if not strategies_df.empty and 'Breakout Detected' in strategies_df.columns:
                exclude_breakouts = st.checkbox(
                    "🛡️ Exclude Breakout Strategies",
                    value=False,
                    help="Filter out strategies with detected local range breakouts",
                    key="breakout_filter"
                )
            else:
                exclude_breakouts = False
        
        with col2:
            st.write("**Exclude Strategy Types:**")
            exclude_strategy_types = st.multiselect(
                "Exclude Types:",
                options=['3MS', '6MS', '12MS', '3MF', '6MF', '12MF'],
                default=[],
                help="3MS/6MS/12MS=Calendar Spreads, 3MF/6MF/12MF=Butterfly Spreads",
                key="exclude_strategy_filter"
            )
        
        with col3:
            if not strategies_df.empty and 'Signal Strength' in strategies_df.columns:
                min_signal_strength = st.number_input(
                    "Minimum Signal Strength:",
                    min_value=0.0,
                    max_value=float(strategies_df['Signal Strength'].max()) if len(strategies_df) > 0 else 100.0,
                    value=0.0,
                    step=1.0,
                    help="Filter strategies with weak signals",
                    key="signal_strength_filter"
                )
            else:
                min_signal_strength = 0.0
        
        with col4:
            # Coefficient of variation filter
            if not strategies_df.empty and 'Coeff of Variation' in strategies_df.columns:
                max_coeff_var = st.number_input(
                    "Maximum Coeff of Variation:",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.01,
                    help="Lower values indicate more stable strategies",
                    key="coeff_var_filter"
                )
            else:
                max_coeff_var = 0.5
        
        # Apply filters button
        st.markdown("---")
        st.markdown("<br>", unsafe_allow_html=True)
        apply_filters = st.button("🚀 Apply Filters & Refresh Results", type="primary", use_container_width=True)
    
    # Return all filter values and apply button state
    return {
        'apply_filters': apply_filters,
        'correlation_threshold': correlation_threshold,
        'range_min_threshold': range_min_threshold,
        'range_max_threshold': range_max_threshold,
        'signal_source_filter': signal_source_filter,
        'selected_markets': selected_markets,
        'selected_months': selected_months,
        'selected_years': selected_years,
        'include_specific_instruments': include_specific_instruments,
        'exclude_specific_instruments': exclude_specific_instruments,
        'exclude_breakouts': exclude_breakouts,
        'exclude_strategy_types': exclude_strategy_types,
        'min_signal_strength': min_signal_strength,
        'max_coeff_var': max_coeff_var
    }

def apply_enhanced_filters(strategies_df, filter_params):
    """Apply all filters to the strategies dataframe"""
    if strategies_df.empty:
        return strategies_df
    
    # Base filtering
    correlation_decimal = filter_params['correlation_threshold'] / 100.0
    range_column = 'Global Range' if 'Global Range' in strategies_df.columns else 'Range'
    
    # Ensure min is not greater than max
    range_min = filter_params['range_min_threshold']
    range_max = filter_params['range_max_threshold']
    if range_min > range_max:
        range_min = range_max
    
    filtered_strategies = strategies_df[
        (abs(strategies_df['Correlation']) >= correlation_decimal) &
        (strategies_df[range_column] >= range_min) &
        (strategies_df[range_column] <= range_max) &
        (strategies_df['Signal'].isin(['BUY', 'SELL']))
    ]
    
    # Signal source filtering
    if filter_params['signal_source_filter'] != 'All' and 'Signal Source' in filtered_strategies.columns:
        if filter_params['signal_source_filter'] == 'Global':
            filtered_strategies = filtered_strategies[filtered_strategies['Signal Source'] == 'Global']
        elif filter_params['signal_source_filter'].startswith('Local'):
            method = filter_params['signal_source_filter'].split('(')[1].rstrip(')')
            filtered_strategies = filtered_strategies[filtered_strategies['Signal Source'].str.contains(method)]
    
    # Instrument component filtering
    filtered_strategies = filter_strategies_by_components(
        filtered_strategies, 
        filter_params['selected_markets'], 
        filter_params['selected_months'], 
        filter_params['selected_years']
    )
    
    # Strategy type exclusion filtering
    filtered_strategies = filter_strategies_by_specific_instruments(
        filtered_strategies,
        include_instruments=filter_params['include_specific_instruments'] if filter_params['include_specific_instruments'] else None,
        exclude_instruments=filter_params['exclude_specific_instruments'] if filter_params['exclude_specific_instruments'] else None
    )
    
    # Breakout filtering
    if filter_params['exclude_breakouts'] and 'Breakout Detected' in filtered_strategies.columns:
        filtered_strategies = filtered_strategies[~filtered_strategies['Breakout Detected']]
    
    # Signal strength filtering
    if 'Signal Strength' in filtered_strategies.columns:
        filtered_strategies = filtered_strategies[
            filtered_strategies['Signal Strength'] >= filter_params['min_signal_strength']
        ]
    
    # Coefficient of variation filtering
    if 'Coeff of Variation' in filtered_strategies.columns:
        filtered_strategies = filtered_strategies[
            filtered_strategies['Coeff of Variation'] <= filter_params['max_coeff_var']
        ]
    
    return filtered_strategies

def render_enhanced_results_table(strategies_df, show_both_signals=True):
    """Render results table with both global and local range columns including breakout info"""
    if strategies_df.empty:
        st.warning("No strategies meet the current criteria.")
        return
    
    if show_both_signals and 'Signal Source' in strategies_df.columns:
        # Enhanced table with local range info and breakout detection
        display_columns = [
            'Strategy', 'Signal', 'Signal Strength', 'Signal Source',
            'Current Value', 'Global Position', 'Local Position', 
            'Global Range', 'Local Range', 'Correlation', 'Beta Ratio',
            'Global Signal', 'Local Signal', 'Global Strength', 'Local Strength'
        ]
        
        # Add breakout columns if available
        if 'Breakout Detected' in strategies_df.columns:
            display_columns.extend(['Breakout Detected', 'Breakout Distance'])
        
        # Filter columns that actually exist
        available_columns = [col for col in display_columns if col in strategies_df.columns]
        
        st.dataframe(
            strategies_df[available_columns], 
            use_container_width=True,
            column_config={
                'Signal Source': st.column_config.TextColumn('Signal Source', width='medium'),
                'Global Position': st.column_config.NumberColumn('Global Pos', format='%.3f'),
                'Local Position': st.column_config.NumberColumn('Local Pos', format='%.3f'),
                'Signal Strength': st.column_config.NumberColumn('Strength', format='%.1f'),
                'Global Strength': st.column_config.NumberColumn('G-Strength', format='%.1f'),
                'Local Strength': st.column_config.NumberColumn('L-Strength', format='%.1f'),
                'Breakout Detected': st.column_config.CheckboxColumn('Breakout'),
                'Breakout Distance': st.column_config.NumberColumn('Breakout Dist', format='%.3f')
            }
        )
        
        # Show breakout summary if breakout data exists
        if 'Breakout Detected' in strategies_df.columns:
            breakout_count = strategies_df['Breakout Detected'].sum()
            total_count = len(strategies_df)
            if breakout_count > 0:
                st.warning(f"⚠️ **Breakout Alert**: {breakout_count}/{total_count} strategies have local range breakouts detected")
        
    else:
        # Standard table
        st.dataframe(strategies_df, use_container_width=True)

def calculate_live_signals_fast(live_df, regression_cache, historical_stats, instrument_groups):
    """Original fast calculation for compatibility when local ranges are disabled"""
    results = []
    
    for pair_key, regression_data in regression_cache.items():
        inst1, inst2 = pair_key.split('_')
        
        if pair_key not in historical_stats:
            continue
        
        hist_stats = historical_stats[pair_key]
        
        if hist_stats['min_tests'] < 2 or hist_stats['max_tests'] < 2:
            continue
        
        try:
            live_value1 = float(live_df[inst1].iloc[0])
            live_value2 = float(live_df[inst2].iloc[0])
            
            beta = regression_data['beta']
            current_strategy = live_value1 - (beta * live_value2)
            
            range_val = hist_stats['range']
            position_in_range = (current_strategy - hist_stats['min']) / range_val if range_val > 0 else 0.5
            
            signal = "HOLD"
            signal_strength = 0
            correlation = regression_data['correlation']
            coeff_var = hist_stats['coeff_var']
            
            # CRITICAL: Only generate signals for high correlation and low coeff variation
            if position_in_range <= 0.25 and abs(correlation) >= 0.90 and coeff_var <= 0.5:
                signal = "BUY"
                signal_strength = (0.25 - position_in_range) * abs(correlation) * (1/coeff_var) * 100
            elif position_in_range >= 0.75 and abs(correlation) >= 0.90 and coeff_var <= 0.5:
                signal = "SELL"
                signal_strength = (position_in_range - 0.75) * abs(correlation) * (1/coeff_var) * 100
            
            if signal in ["BUY", "SELL"]:
                results.append({
                    'Strategy': f"{inst1} - {beta:.3f}*{inst2}",
                    'Current Value': round(current_strategy, 6),
                    'Range': round(hist_stats['range'], 6),
                    'Min': round(hist_stats['min'], 6),
                    'Max': round(hist_stats['max'], 6),
                    'Mean': round(hist_stats['mean'], 6),
                    'Coeff of Variation': round(coeff_var, 4),
                    'Position in Range': round(position_in_range, 4),
                    'Correlation': round(correlation, 4),
                    'Beta Ratio': round(beta, 3),
                    'R-Squared': round(regression_data['r_squared'], 3),
                    'Signal': signal,
                    'Signal Strength': round(signal_strength, 2),
                    'Min Tests': hist_stats['min_tests'],
                    'Max Tests': hist_stats['max_tests'],
                    'Instrument1': inst1,
                    'Instrument2': inst2
                })
                
        except Exception:
            continue
    
    return pd.DataFrame(results)

# ==================== MAIN APPLICATION ====================

# Auto-refresh functionality
st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1 style="margin-bottom: 0;">📊 Enhanced Range Bound Strategies</h1>
    <p style="font-size: 1.2rem; color: #9e9e9e; margin-top: 0.5rem;">
        Live Dashboard with Local Range Analysis & Instrument Filtering
    </p>
</div>
""", unsafe_allow_html=True)

# Add auto-refresh control and performance settings
col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

with col1:
    auto_refresh = st.checkbox("🔄 Auto-refresh every 5 minutes", value=False)

with col2:
    if st.button("🔄 Refresh Now"):
        st.rerun()

with col3:
    use_cache = st.checkbox("⚡ Fast Mode", value=True, help="Use caching for better performance")

with col4:
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()
    
    last_refresh_str = time.strftime("%H:%M:%S", time.localtime(st.session_state.last_refresh))
    st.write(f"Last updated: {last_refresh_str}")

# Performance warning
if not use_cache:
    st.warning("⚠️ Fast Mode disabled - calculations may be slower but more current")

# Auto-refresh logic
if auto_refresh:
    if 'next_refresh' not in st.session_state:
        st.session_state.next_refresh = time.time() + 300
    
    if time.time() >= st.session_state.next_refresh:
        st.session_state.last_refresh = time.time()
        st.session_state.next_refresh = time.time() + 300
        st.rerun()
    else:
        time_left = int(st.session_state.next_refresh - time.time())
        st.sidebar.info(f"⏱️ Next auto-refresh in {time_left} seconds")

# Load data with optimized daily caching strategy
with st.spinner("🔄 Loading market data and initializing dashboard..."):
    df_recent = load_historical_data()
    live_df, data_hash = load_live_data()

if df_recent is not None and live_df is not None:
    
    # Get instrument groups
    instrument_groups = get_instrument_groups_daily(df_recent)
    
    # Display instrument counts with updated breakdown
    total_instruments = sum(len(group) for group in instrument_groups.values())
    st.write(f"**Total Instruments Found:** {total_instruments}")
    st.write(f"**Using 60 days of historical data**")
    
    # Show breakdown with new categorization
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Calendar Spreads:**")
        st.write(f"- 3MS: {len(instrument_groups['3MS'])}")
        st.write(f"- 6MS: {len(instrument_groups['6MS'])}")
        st.write(f"- 12MS: {len(instrument_groups['12MS'])}")
    
    with col2:
        st.write("**Butterfly Spreads:**")
        st.write(f"- 3MF: {len(instrument_groups['3MF'])}")
        st.write(f"- 6MF: {len(instrument_groups['6MF'])}")
        st.write(f"- 12MF: {len(instrument_groups['12MF'])}")
    
    # Enhanced instrument breakdown by market, month, year
    with st.expander("📊 Instrument Breakdown by Market, Month & Year"):
        # Collect all instruments from all groups
        all_instruments = []
        for group in instrument_groups.values():
            all_instruments.extend(group)
        
        if all_instruments:
            # Analyze by market, month, year
            market_counts = {'SRA': 0, 'ER': 0, 'CRA': 0, 'SON': 0, 'Other': 0}
            month_counts = {'H': 0, 'M': 0, 'U': 0, 'Z': 0, 'Other': 0}
            year_counts = {'25': 0, '26': 0, '27': 0, '28': 0, 'Other': 0}
            
            for inst in all_instruments:
                market, month, year = parse_instrument_components(inst)
                
                # Count markets
                if market in market_counts:
                    market_counts[market] += 1
                else:
                    market_counts['Other'] += 1
                
                # Count months
                if month in month_counts:
                    month_counts[month] += 1
                else:
                    month_counts['Other'] += 1
                
                # Count years
                if year in year_counts:
                    year_counts[year] += 1
                else:
                    year_counts['Other'] += 1
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Available Markets:**")
                market_names = {
                    'SRA': 'SOFR',
                    'ER': 'Euribor', 
                    'CRA': 'CORRA',
                    'SON': 'SONIA'
                }
                for market, count in market_counts.items():
                    if count > 0:
                        name = market_names.get(market, market)
                        st.write(f"- {market} ({name}): {count} instruments")
            
            with col2:
                st.write("**Available Months:**")
                month_names = {
                    'H': 'March',
                    'M': 'June',
                    'U': 'September',
                    'Z': 'December'
                }
                for month, count in month_counts.items():
                    if count > 0:
                        name = month_names.get(month, month)
                        st.write(f"- {month} ({name}): {count} instruments")
            
            with col3:
                st.write("**Available Years:**")
                for year, count in year_counts.items():
                    if count > 0:
                        display_year = f"20{year}" if year != 'Other' else year
                        st.write(f"- {display_year}: {count} instruments")
    
    if total_instruments >= 2:
        
        # Pre-calculate all regressions and historical stats
        regression_cache = calculate_all_regressions_daily(df_recent, instrument_groups)
        historical_stats = calculate_historical_statistics_daily(df_recent, regression_cache)
        
        # ==================== ENHANCED RANGE TRADING CONTROLS ====================
        
        st.markdown("---")
        use_local_ranges, local_mode, combine_signals, show_both_signals, breakout_threshold = render_local_range_controls()
        
        # Enhanced signal calculation with breakout protection
        with st.spinner("🧮 Processing strategies and calculating signals..."):
            start_time = time.time()
            
            if use_local_ranges:
                strategies_df = calculate_enhanced_signals_with_local_ranges(
                    live_df, regression_cache, historical_stats, 
                    use_local_ranges=True, local_mode=local_mode, 
                    combine_signals=combine_signals, breakout_threshold=breakout_threshold
                )
            else:
                # Fall back to original calculation if local ranges disabled
                strategies_df = calculate_live_signals_fast(live_df, regression_cache, historical_stats, instrument_groups)
            
            calculation_time = time.time() - start_time
            
            range_type = f"Enhanced ({local_mode}) with {breakout_threshold*100:.0f}% breakout protection" if use_local_ranges else "Global Only"
            st.success(f"✅ {range_type} signals calculated in {calculation_time:.2f} seconds")

        if not strategies_df.empty:
            
            # ==================== ENHANCED BATCH FILTERING SECTION ====================
            
            st.markdown("---")
            
            # Render batch filter controls
            filter_params = render_enhanced_batch_filters(strategies_df)
            
            # Initialize filtered strategies (show all initially or when filters are applied)
            if 'filtered_strategies_df' not in st.session_state or filter_params['apply_filters']:
                if filter_params['apply_filters']:
                    # Apply all filters
                    with st.spinner("🔄 Applying filters..."):
                        filtered_strategies = apply_enhanced_filters(strategies_df, filter_params)
                        st.session_state.filtered_strategies_df = filtered_strategies
                        st.session_state.last_filter_params = filter_params
                        st.success("✅ Filters applied successfully!")
                else:
                    # Show all strategies initially
                    st.session_state.filtered_strategies_df = strategies_df
                    st.session_state.last_filter_params = filter_params
            
            # Use the filtered strategies from session state
            filtered_strategies = st.session_state.filtered_strategies_df
            
            # Enhanced sorting
            strategies_sorted = filtered_strategies.copy()
            
            if not strategies_sorted.empty:
                # Create enhanced sort priority
                strategies_sorted['Sort_Priority'] = strategies_sorted['Signal'].map({'BUY': 1, 'SELL': 1, 'HOLD': 2})
                
                strategies_sorted = strategies_sorted.sort_values([
                    'Sort_Priority',
                    'Signal Strength',
                    'Coeff of Variation'
                ], ascending=[True, False, True])
                
                strategies_sorted = strategies_sorted.drop('Sort_Priority', axis=1)
            
            st.header("🎯 Enhanced Active Trading Signals")
            
            # Enhanced signal summary with breakout information
            buy_signals = len(strategies_sorted[strategies_sorted['Signal'] == 'BUY']) if not strategies_sorted.empty else 0
            sell_signals = len(strategies_sorted[strategies_sorted['Signal'] == 'SELL']) if not strategies_sorted.empty else 0
            total_signals = buy_signals + sell_signals
            
            # Count by signal source and breakout status if available
            signal_source_counts = {}
            breakout_stats = {}
            if not strategies_sorted.empty and 'Signal Source' in strategies_sorted.columns:
                signal_source_counts = strategies_sorted['Signal Source'].value_counts().to_dict()
            if not strategies_sorted.empty and 'Breakout Detected' in strategies_sorted.columns:
                breakout_stats = {
                    'total_breakouts': strategies_sorted['Breakout Detected'].sum(),
                    'total_strategies': len(strategies_sorted),
                    'breakout_percentage': (strategies_sorted['Breakout Detected'].sum() / len(strategies_sorted) * 100) if len(strategies_sorted) > 0 else 0
                }
            
            # Display enhanced signal summary
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("🟢 BUY Signals", buy_signals)
            with col2:
                st.metric("🔴 SELL Signals", sell_signals)
            with col3:
                st.metric("📊 Total Active Signals", total_signals)
            with col4:
                if signal_source_counts:
                    local_signals = sum(count for source, count in signal_source_counts.items() if 'Local' in source)
                    st.metric("🎯 Local Range Signals", local_signals)
                else:
                    st.metric("🎯 Local Range Signals", "N/A")
            with col5:
                if breakout_stats:
                    breakout_count = breakout_stats['total_breakouts']
                    if breakout_count > 0:
                        st.metric("⚠️ Breakout Alerts", f"{breakout_count}", delta=f"{breakout_stats['breakout_percentage']:.1f}%")
                    else:
                        st.metric("🛡️ Range Integrity", "✅ All Clear")
                else:
                    st.metric("🛡️ Breakout Protection", "Active")
            
            # Enhanced description with applied filters info
            if 'last_filter_params' in st.session_state:
                filter_info = st.session_state.last_filter_params
                
                st.write(f"**Showing {len(strategies_sorted)} filtered active trading signals:**")
                
                # Show applied filters
                filter_descriptions = []
                filter_descriptions.append(f"Correlation >= {filter_info['correlation_threshold']}%")
                filter_descriptions.append(f"Range: {filter_info['range_min_threshold']:.6f} - {filter_info['range_max_threshold']:.6f}")
                
                if filter_info['signal_source_filter'] != 'All':
                    filter_descriptions.append(f"Signal source: {filter_info['signal_source_filter']}")
                
                if filter_info['selected_markets']:
                    market_names = {
                        'SRA': 'SOFR', 'ER': 'Euribor', 'CRA': 'CORRA', 'SON': 'SONIA'
                    }
                    selected_names = [f"{m} ({market_names[m]})" for m in filter_info['selected_markets']]
                    filter_descriptions.append(f"Markets: {', '.join(selected_names)}")
                
                if filter_info['selected_months']:
                    month_names = {'H': 'March', 'M': 'June', 'U': 'September', 'Z': 'December'}
                    selected_names = [f"{m} ({month_names[m]})" for m in filter_info['selected_months']]
                    filter_descriptions.append(f"Months: {', '.join(selected_names)}")
                
                if filter_info['selected_years']:
                    year_display = [f"20{y}" for y in filter_info['selected_years']]
                    filter_descriptions.append(f"Years: {', '.join(year_display)}")
                
                if filter_info['exclude_breakouts']:
                    filter_descriptions.append("Breakout strategies excluded")
                # NEW: Show specific instrument filtering info
                if filter_info.get('include_specific_instruments'):
                    include_count = len(filter_info['include_specific_instruments'])
                    instruments_preview = ', '.join(filter_info['include_specific_instruments'][:3])
                    if len(filter_info['include_specific_instruments']) > 3:
                        instruments_preview += f"... (+{len(filter_info['include_specific_instruments'])-3} more)"
                    filter_descriptions.append(f"🎯 Including instruments: {instruments_preview}")
                
                if filter_info.get('exclude_specific_instruments'):
                    exclude_count = len(filter_info['exclude_specific_instruments'])
                    instruments_preview = ', '.join(filter_info['exclude_specific_instruments'][:3])
                    if len(filter_info['exclude_specific_instruments']) > 3:
                        instruments_preview += f"... (+{len(filter_info['exclude_specific_instruments'])-3} more)"
                    filter_descriptions.append(f"❌ Excluding instruments: {instruments_preview}")
                
                if filter_info.get('exclude_strategy_types'):
                    excluded_types = ', '.join(filter_info['exclude_strategy_types'])
                    filter_descriptions.append(f"Excluded strategy types: {excluded_types}")
                
                if filter_info['min_signal_strength'] > 0:
                    filter_descriptions.append(f"Min signal strength: {filter_info['min_signal_strength']}")
                
                if filter_info['max_coeff_var'] < 1.0:
                    filter_descriptions.append(f"Max coeff variation: {filter_info['max_coeff_var']}")
                
                # Display filter descriptions
                for desc in filter_descriptions:
                    st.write(f"- {desc}")
                
                st.write(f"- **Analysis mode:** {'Enhanced with Local Ranges + Breakout Protection' if use_local_ranges else 'Global Ranges Only'}")
                if use_local_ranges:
                    st.write(f"- **Breakout threshold:** {breakout_threshold*100:.0f}% beyond local range")
            
            # Breakout warning if significant breakouts detected
            if breakout_stats and breakout_stats['breakout_percentage'] > 25:
                st.warning(f"⚠️ **High Breakout Activity**: {breakout_stats['breakout_percentage']:.1f}% of strategies show local range breakouts - market may be trending")
                        
            if not strategies_sorted.empty:
                # Display enhanced results table
                render_enhanced_results_table(strategies_sorted, show_both_signals)
                
                # Enhanced strategy chart section
                st.header("📈 Enhanced Historical Chart")
                st.write("**Select a strategy to view its enhanced range analysis:**")
                
                selected_strategy = st.selectbox(
                    "Choose strategy:",
                    strategies_sorted['Strategy'].tolist(),
                    index=0
                )
                
                if selected_strategy:
                    # Basic chart rendering without all the enhanced features for simplicity
                    strategy_row = strategies_sorted[strategies_sorted['Strategy'] == selected_strategy].iloc[0]
                    inst1 = strategy_row['Instrument1']
                    inst2 = strategy_row['Instrument2']
                    beta = strategy_row['Beta Ratio']
                    
                    # Calculate regression-adjusted historical strategy values
                    hist1 = df_recent[inst1].dropna()
                    hist2 = df_recent[inst2].dropna()
                    historical_strategy = hist1 - (beta * hist2)
                    historical_strategy = historical_strategy.dropna()
                    
                    # Reverse for chronological order
                    historical_strategy_reversed = historical_strategy.iloc[::-1]
                    
                    # Create basic chart
                    import plotly.graph_objects as go
                    
                    fig = go.Figure()
                    
                    # Add main strategy line
                    fig.add_trace(go.Scatter(
                        x=list(range(len(historical_strategy_reversed))),
                        y=historical_strategy_reversed.values,
                        mode='lines+markers',
                        name=f'{selected_strategy}',
                        line=dict(width=3, color='blue'),
                        marker=dict(size=6),
                        hovertemplate='<b>Day %{x}:</b> %{y:.6f}<br><extra></extra>'
                    ))
                    
                    # Add current value
                    current_value = strategy_row['Current Value']
                    fig.add_hline(y=current_value, line_dash="dash", line_color="red", line_width=3,
                                 annotation_text=f"Current: {current_value:.6f}")
                    
                    # Add global range boundaries
                    global_min = strategy_row['Global Min']
                    global_max = strategy_row['Global Max']
                    mean_value = strategy_row['Mean']
                    
                    fig.add_hline(y=mean_value, line_dash="dot", line_color="green", line_width=2,
                                 annotation_text=f"Global Mean: {mean_value:.6f}")
                    fig.add_hline(y=global_min, line_dash="dashdot", line_color="orange", line_width=1,
                                 annotation_text=f"Global Min: {global_min:.6f}")
                    fig.add_hline(y=global_max, line_dash="dashdot", line_color="orange", line_width=1,
                                 annotation_text=f"Global Max: {global_max:.6f}")
                    
                    # Add global range shading
                    fig.add_hrect(y0=global_min, y1=global_max, fillcolor="lightgray", opacity=0.2,
                                 annotation_text="Global Range")
                    
                    # Add local range info if available
                    if 'Local Min' in strategy_row and pd.notna(strategy_row['Local Min']):
                        local_min = strategy_row['Local Min']
                        local_max = strategy_row['Local Max']
                        
                        fig.add_hline(y=local_min, line_dash="dash", line_color="purple", line_width=2,
                                     annotation_text=f"Local Min: {local_min:.6f}")
                        fig.add_hline(y=local_max, line_dash="dash", line_color="purple", line_width=2,
                                     annotation_text=f"Local Max: {local_max:.6f}")
                        
                        # Add local range shading
                        fig.add_hrect(y0=local_min, y1=local_max, fillcolor="lightblue", opacity=0.15,
                                     annotation_text="Local Range")
                    
                    fig.update_layout(
                        title=f"Enhanced Range Analysis: {selected_strategy}",
                        xaxis_title="Days (Oldest → Most Recent)",
                        yaxis_title="Strategy Value",
                        height=600,
                        hovermode='x unified',
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show strategy details
                    with st.expander("📊 Strategy Details"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write("**Global Range Info:**")
                            st.write(f"- Global Min: {global_min:.6f}")
                            st.write(f"- Global Max: {global_max:.6f}")
                            st.write(f"- Global Mean: {mean_value:.6f}")
                            st.write(f"- Global Position: {strategy_row['Global Position']:.3f}")
                        
                        with col2:
                            if 'Local Min' in strategy_row and pd.notna(strategy_row['Local Min']):
                                st.write("**Local Range Info:**")
                                st.write(f"- Local Min: {strategy_row['Local Min']:.6f}")
                                st.write(f"- Local Max: {strategy_row['Local Max']:.6f}")
                                st.write(f"- Local Position: {strategy_row['Local Position']:.3f}")
                                if 'Range Type' in strategy_row:
                                    st.write(f"- Range Type: {strategy_row['Range Type']}")
                        
                        with col3:
                            st.write("**Signal Info:**")
                            st.write(f"- Final Signal: {strategy_row['Signal']}")
                            st.write(f"- Signal Strength: {strategy_row['Signal Strength']:.2f}")
                            if 'Signal Source' in strategy_row:
                                st.write(f"- Signal Source: {strategy_row['Signal Source']}")
                            if 'Breakout Detected' in strategy_row and strategy_row['Breakout Detected']:
                                st.write(f"- ⚠️ Breakout Detected: {strategy_row['Breakout Distance']:.1%}")
                        
            else:
                st.warning("No strategy pairs meet the specified enhanced criteria.")
                st.write("Try adjusting the filters using the 'Apply Filters' button above:")
                
                suggestion_col1, suggestion_col2 = st.columns(2)
                with suggestion_col1:
                    st.write("**Try adjusting:**")
                    st.write("- Lower correlation threshold (try 80%)")
                    st.write("- Increase range thresholds")
                    st.write("- Clear market/month/year selections")
                    st.write("- Reduce minimum signal strength")
                with suggestion_col2:
                    st.write("**Or modify:**")
                    st.write("- Increase coefficient of variation limit")
                    st.write("- Enable/disable breakout exclusion")
                    st.write("- Try different local range methods")
                    st.write("- Increase breakout threshold to 30-40%")
        else:
            st.error("No valid strategy pairs could be calculated.")
            
            # Helpful suggestions for troubleshooting
            st.subheader("🔧 Troubleshooting Suggestions")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Data Issues:**")
                st.write("- Check if Excel file 'QH-Data.xlsm' exists")
                st.write("- Verify 'IM' and 'Live' sheets are present")
                st.write("- Ensure instruments have valid data")
                st.write("- Check for sufficient historical data (60+ days)")
            
            with col2:
                st.write("**Configuration Issues:**")
                st.write("- Lower correlation threshold (try 80%)")
                st.write("- Increase range thresholds")
                st.write("- Try different market/month/year combinations")
                st.write("- Available: SRA (SOFR), ER (Euribor), CRA (CORRA), SON (SONIA)")
                st.write("- Available months: H (Mar), M (Jun), U (Sep), Z (Dec)")
                st.write("- Available years: 2025-2028")
                st.write("- Disable local range analysis temporarily")
                st.write("- Increase breakout threshold to 30-40%")
        
        st.markdown("---")
        
        # Add filter reset option
        if st.button("🔄 Reset All Filters", help="Clear all filters and show all strategies"):
            if 'filtered_strategies_df' in st.session_state:
                del st.session_state.filtered_strategies_df
            if 'last_filter_params' in st.session_state:
                del st.session_state.last_filter_params
            st.rerun()
        
    else:
        st.error(f"Not enough instruments available. Found {total_instruments} instruments, need at least 2.")
        
else:
    st.error("Unable to load data. Please check that 'QH-Data.xlsm' exists and contains 'IM' and 'Live' sheets.")

# ==================== SIDEBAR INFORMATION ====================

with st.sidebar:
    st.markdown("""
<div style="text-align: center; padding: 1rem 0;">
    <h2 style="background: linear-gradient(135deg, #4fc3f7, #29b6f6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 1.5rem;">
        📖 Dashboard Guide
    </h2>
</div>
""", unsafe_allow_html=True)
    
    st.subheader("🏛️ Market Types")
    st.write("- **SRA**: SOFR (Secured Overnight Financing Rate)")
    st.write("- **ER**: Euribor (Euro Interbank Offered Rate)")
    st.write("- **CRA**: CORRA (Canadian Overnight Repo Rate Average)")
    st.write("- **SON**: SONIA (Sterling Overnight Index Average)")
    
    st.subheader("📅 Contract Months")
    st.write("- **H**: March")
    st.write("- **M**: June") 
    st.write("- **U**: September")
    st.write("- **Z**: December")
    
    st.subheader("🎯 Strategy Types")
    st.write("- **3MS/6MS/12MS**: Calendar Spreads")
    st.write("- **3MF/6MF/12MF**: Butterfly Spreads")
    
    st.subheader("🔍 Local Range Methods")
    st.write("- **Recent**: Last 15 periods as range")
    st.write("- **Adaptive**: Volatility-based range")
    st.write("- **Support/Resistance**: Key price levels")

    st.subheader("🎯 Instrument Filtering")
    st.write("- **Include Instruments**: Show strategies with selected instruments")
    st.write("- **Exclude Instruments**: Hide strategies with selected instruments")
    st.write("- **Smart Parsing**: Automatically detects instruments in strategies")
    st.write("- **Conflict Detection**: Warns about overlapping selections")
    
    st.subheader("⚡ Quick Tips")
    st.write("- Use 'Apply Filters' to avoid constant refreshes")
    st.write("- Higher correlation = more predictable")
    st.write("- Lower coeff variation = more stable")
    st.write("- Enable breakout protection for trending markets")
    
    if 'last_filter_params' in st.session_state:
        st.subheader("🎛️ Current Filter Status")
        params = st.session_state.last_filter_params
        st.write(f"- Markets: {len(params.get('selected_markets', []))}")
        st.write(f"- Months: {len(params.get('selected_months', []))}")
        st.write(f"- Years: {len(params.get('selected_years', []))}")
        st.write(f"- Correlation: {params.get('correlation_threshold', 90)}%")
        if params.get('exclude_breakouts', False):
            st.write("- ⚠️ Breakouts excluded")
        # NEW: Show specific instrument filter status
        include_inst_count = len(params.get('include_specific_instruments', []))
        exclude_inst_count = len(params.get('exclude_specific_instruments', []))
        if include_inst_count > 0:
            st.write(f"- 🎯 Including: {include_inst_count} instruments")
        if exclude_inst_count > 0:
            st.write(f"- ❌ Excluding: {exclude_inst_count} instruments")

# ==================== PERFORMANCE METRICS ====================

if 'calculation_time' in locals():
    with st.expander("⚡ Performance Metrics"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Calculation Time", f"{calculation_time:.2f}s")
        
        with col2:
            if 'regression_cache' in locals():
                st.metric("Regression Pairs", len(regression_cache))
        
        with col3:
            if 'historical_stats' in locals():
                st.metric("Historical Stats", len(historical_stats))
        
        # Memory usage info
        import sys
        memory_mb = sys.getsizeof(locals()) / 1024 / 1024
        st.write(f"**Memory Usage**: ~{memory_mb:.1f} MB")
        
        # Cache status
        cache_info = []
        if use_cache:
            cache_info.append("✅ Historical data cached (24h)")
            cache_info.append("✅ Live data cached (5min)")
            cache_info.append("✅ Regressions cached (24h)")
            cache_info.append("✅ Statistics cached (24h)")
        else:
            cache_info.append("⚠️ Fast mode disabled")
        
        for info in cache_info:
            st.write(info)

# Add footer with version info
st.markdown("---")
st.markdown("""
<div style="margin-top: 4rem; padding: 2rem 0; text-align: center;">
    <div style="height: 1px; background: linear-gradient(90deg, transparent, #4fc3f7, transparent); margin: 2rem 0;"></div>
    <p style="color: #9e9e9e; font-size: 0.9rem;">
        Enhanced Range Bound Strategies Dashboard v2.1 - With Modern UI & Instrument Filtering
    </p>
    <p style="color: #616161; font-size: 0.8rem; margin-top: 0.5rem;">
        Powered by Advanced Analytics & Real-time Market Data
    </p>
</div>
""", unsafe_allow_html=True)
