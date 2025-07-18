import streamlit as st
import pandas as pd
import numpy as np
from fredapi import Fred
import pandas_datareader as pdr
import wbdata
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import plotly.express as px
import re
import smtplib
from email.mime.text import MIMEText

# Get keys from st.secrets (Cloud) or fallback to env (local)
fred_api_key = st.secrets.get('FRED_API_KEY')
fred = Fred(api_key=fred_api_key) if fred_api_key else None

# Indicators list with fetch functions
indicators = {
    # Risks proxies
    'Volatility Risk': {'func': lambda: yf.Ticker('^VIX').info.get('regularMarketPrice', np.nan), 'thresh': 20, 'desc': 'swings fog crash heart race scare'},
    'Credit Risk': {'func': lambda: scrape_moodys_defaults(), 'thresh': 5, 'desc': 'defaults drought crop loans fail'},
    'Interest Rate Risk': {'func': lambda: fred.get_series('FEDFUNDS')[-1] if fred else scrape_fed_rates(), 'thresh': 5, 'desc': 'rises hurt bonds brakes slow car'},
    'TED Spread': {'func': lambda: fred.get_series('TEDRATE')[-1] if fred else np.nan, 'thresh': 0.5, 'desc': 'liquidity jam'},
    'CPI Inflation': {'func': lambda: fred.get_series('CPIAUCSL')[-1] if fred else np.nan, 'thresh': 3, 'desc': 'fire burn cash'},

    # Short-term debt leading/coincident
    'Yield Curve': {'func': lambda: fred.get_series('T10Y2Y')[-1] if fred else np.nan, 'thresh': 0, 'desc': 'river spread bank profit growth'},
    'GDP': {'func': lambda: fred.get_series('GDP')[-1] if fred else np.nan, 'thresh': np.nan, 'desc': 'harvest output jobs'},
    'Capacity Utilization': {'func': lambda: fred.get_series('CAPUTLB50001S')[-1] if fred else np.nan, 'thresh': np.nan, 'desc': 'factory hum no smoke'},
    'PCE Inflation': {'func': lambda: fred.get_series('PCEPI')[-1] if fred else np.nan, 'thresh': np.nan, 'desc': 'heat balloon'},
    'Payrolls': {'func': lambda: fred.get_series('PAYEMS')[-1] if fred else np.nan, 'thresh': np.nan, 'desc': 'job adds team build'},
    'Unemployment': {'func': lambda: fred.get_series('UNRATE')[-1] if fred else np.nan, 'thresh': np.nan, 'desc': 'idle rust'},
    'P/E Ratio': {'func': lambda: scrape_multpl_pe(), 'thresh': 25, 'desc': 'overprice house'},
    'Fed Funds Rate': {'func': lambda: fred.get_series('FEDFUNDS')[-1] if fred else np.nan, 'thresh': np.nan, 'desc': 'borrow speed limit'},
    'Industrial Production': {'func': lambda: fred.get_series('INDPRO')[-1] if fred else np.nan, 'thresh': np.nan, 'desc': 'factory make'},

    # Long-term debt leading/coincident
    'Productivity': {'func': lambda: fred.get_series('OPHNFB')[-1] if fred else np.nan, 'thresh': np.nan, 'desc': 'output/input tools boost'},
    'Debt/GDP': {'func': lambda: (fred.get_series('GFDEBTN')[-1] / fred.get_series('GDP')[-1]) * 100 if fred else np.nan, 'thresh': np.nan, 'desc': 'borrow max cards burden'},
    'M2 Money Supply': {'func': lambda: fred.get_series('M2SL')[-1] if fred else np.nan, 'thresh': np.nan, 'desc': 'printing rain money'},
    'Asset Returns S&P': {'func': lambda: yf.Ticker('^GSPC').history(period='1d')['Close'][0], 'thresh': np.nan, 'desc': 'gains steady heal'},

    # Geo cycles
    'GDP per Capita': {'func': lambda: wbdata.get_series('NY.GDP.PCAP.CD')['USA'], 'thresh': np.nan, 'desc': 'income head living rise'},  # Example for USA
    'Military Spend': {'func': lambda: scrape_sipri_military(), 'thresh': np.nan, 'desc': 'arms costly drain'},
    'Corruption Index': {'func': lambda: scrape_transparency_cpi(), 'thresh': np.nan, 'desc': 'graft erode buy rot'},
    'Power Index': {'func': lambda: scrape_globalfirepower_index(), 'thresh': np.nan, 'desc': 'composite strength peak/drop'},
}

# Updated scrape functions as of July 19, 2025
def scrape_moodys_defaults():
    try:
        r = requests.get('https://www.moodys.com/web/en/us/insights/data-stories/us-corporate-default-risk-in-2025.html')
        soup = BeautifulSoup(r.text, 'html.parser')
        rate_text = soup.find(string=re.compile(r'high-yield.*default rate.*\d+\.\d+%.*\d+\.\d+%', re.I))
        if rate_text:
            rates = re.findall(r'\d+\.\d+', rate_text)
            return float(sum(map(float, rates)) / len(rates)) if rates else np.nan  # Average of 2.8-3.4% range ~3.1
        return np.nan
    except:
        return np.nan

def scrape_fed_rates():
    try:
        r = requests.get('https://www.federalreserve.gov/releases/h15/')
        soup = BeautifulSoup(r.text, 'html.parser')
        row = soup.find('th', string=re.compile(r'Federal funds \(effective\)', re.I)).parent if soup.find('th') else None
        if row:
            tds = row.find_all('td')
            latest = tds[-1].text.strip() if tds else np.nan  # Latest ~4.33% as of July 2025
            if latest == 'n.a.':
                treasury_row = soup.find('th', string=re.compile(r'10-year', re.I)).parent
                latest = treasury_row.find_all('td')[-1].text.strip() if treasury_row else np.nan  # Proxy ~4.44%
            return float(latest) if latest != 'n.a.' else np.nan
        return np.nan
    except:
        return np.nan

def scrape_multpl_pe():
    try:
        r = requests.get('https://www.multpl.com/s-p-500-pe-ratio')
        soup = BeautifulSoup(r.text, 'html.parser')
        return float(soup.find(id='current').text)  # Current ~30.03 as of July 18, 2025
    except:
        return np.nan

def scrape_sipri_military():
    try:
        url = 'https://sipri.org/sites/default/files/2025-04/2504_milex_data_sheet_2024.xlsx'  # Latest 2024 data (2025 not yet released)
        df = pd.read_excel(url, sheet_name='Share of GDP', skiprows=5)
        return df.loc[df['Country'] == 'USA', df.columns[-1]].values[0]
    except:
        return np.nan

def scrape_transparency_cpi():
    try:
        url = 'https://images.transparencycdn.org/images/CPI2024_FullDataSet.xlsx'  # Latest 2024 data (2025 not yet released, US score 65)
        df = pd.read_excel(url, skiprows=2)
        return df.loc[df['Country / Territory'] == 'United States', 'CPI score 2024'].values[0]
    except:
        return np.nan

def scrape_globalfirepower_index():
    try:
        r = requests.get('https://www.globalfirepower.com/countries-listing.php')
        soup = BeautifulSoup(r.text, 'html.parser')
        us_row = soup.find('div', string='United States').parent.parent
        return float(us_row.find('span', class_='powerIndex').text) if us_row else np.nan  # US ~0.0696 for 2025
    except:
        return np.nan

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_all():
    data = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {name: executor.submit(ind['func']) for name, ind in indicators.items()}
        for name, future in futures.items():
            try:
                data[name] = future.result()
            except Exception as e:
                data[name] = np.nan  # Silent fail to NaN
    return data

conn = sqlite3.connect('econ.db')

st.title('Econ Mirror Dashboard - July 19 2025')  # Updated date

if st.button('Refresh Now'):
    data = fetch_all()
    df = pd.DataFrame(list(data.items()), columns=['Indicator', 'Value'])
    df.to_sql('data', conn, if_exists='replace')
else:
    df = pd.read_sql('SELECT * FROM data', conn)

col1, col2 = st.columns(2)

with col1:
    st.subheader('Risks/Cycles Viz')
    for _, row in df.iterrows():
        thresh = indicators.get(row['Indicator'], {}).get('thresh', np.nan)
        value = row['Value'] if isinstance(row['Value'], (int, float)) else np.nan
        if np.isnan(value):
            continue  # Skip errors/NaN
        color = 'red' if not np.isnan(thresh) and value > thresh else 'green'
        fig = px.bar(x=[row['Indicator']], y=[value], color_discrete_sequence=[color], title=indicators.get(row['Indicator'], {}).get('desc', ''))
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.dataframe(df)

# Alerts
breaches = df[df.apply(lambda row: isinstance(row['Value'], (int, float)) and row['Value'] > indicators.get(row['Indicator'], {}).get('thresh', np.nan), axis=1)]
if not breaches.empty:
    try:
        msg = MIMEText(f'Flood: {breaches.to_string()}')
        msg['Subject'] = 'Econ Breach Alert'
        msg['From'] = st.secrets['EMAIL_USER']
        msg['To'] = st.secrets['EMAIL_TO']
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(st.secrets['EMAIL_USER'], st.secrets['EMAIL_PASS'])
            server.send_message(msg)
        st.warning('Alert emailed!')
    except Exception as e:
        st.error(f'Email failed: {e}')
