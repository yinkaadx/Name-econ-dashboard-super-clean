import streamlit as st
import pandas as pd
import numpy as np
from fredapi import Fred
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

# All indicators with fetch, thresh, desc (from list, no duplication)
indicators = {
    'Asset Bubbles': {'func': lambda: yf.Ticker('^GSPC').info.get('regularMarketPrice', np.nan) / fred.get_series('PCE')[-1] if fred else np.nan, 'thresh': 20, 'desc': 'Asset to PCE ratio - high means bubble'},
    'Asset Prices': {'func': lambda: yf.Ticker('^GSPC').info.get('regularMarketPrice', np.nan), 'thresh': np.nan, 'desc': 'Asset price level - high means bubble risk'},
    'Asset Returns': {'func': lambda: yf.Ticker('^GSPC').history(period='1d')['Close'][0], 'thresh': np.nan, 'desc': 'Daily asset return - positive is good'},
    'Building Permits': {'func': lambda: fred.get_series('PERMIT')[-1] if fred else np.nan, 'thresh': np.nan, 'desc': 'New home permits - rising means construction boom'},
    'Capacity Utilization': {'func': lambda: fred.get_series('CAPUTL')[-1] if fred else np.nan, 'thresh': 80, 'desc': 'Factory usage % - >80% means overheating'},
    'Central Bank Rate': {'func': lambda: fred.get_series('FEDFUNDS')[-1] if fred else np.nan, 'thresh': np.nan, 'desc': 'Fed funds rate - high means tightening'},
    'Competitiveness Index': {'func': lambda: scrape_wef_competitiveness(), 'thresh': np.nan, 'desc': 'Global competitiveness score - high means strong economy'},
    'Consumer Confidence Index': {'func': lambda: fred.get_series('UMCSENT')[-1] if fred else np.nan, 'thresh': 90, 'desc': 'Consumer mood - >90 means optimistic'},
    'Consumer Spending': {'func': lambda: fred.get_series('PCE')[-1] if fred else np.nan, 'thresh': np.nan, 'desc': 'Personal spending - high drives growth'},
    'Corruption Index': {'func': lambda: scrape_transparency_cpi(), 'thresh': 50, 'desc': 'Corruption perception - <50 means high graft'},
    'Credit Growth': {'func': lambda: fred.get_series('TOTALSL')[-1] - fred.get_series('TOTALSL')[-4] if fred else np.nan, 'thresh': 5, 'desc': 'Credit expansion % - >5% means loose lending'},
    'Credit Spreads': {'func': lambda: fred.get_series('BAAFF')[-1] if fred else np.nan, 'thresh': 5, 'desc': 'Corporate bond spread - wide means risk aversion'},
    'Currency Devaluation': {'func': lambda: fred.get_series('EXUSUK')[-1] if fred else np.nan, 'thresh': np.nan, 'desc': 'USD exchange rate - falling means devaluation'},
    'Debt Growth': {'func': lambda: fred.get_series('GFDEBTN')[-1] - fred.get_series('GFDEBTN')[-4] if fred else np.nan, 'thresh': 5, 'desc': 'Debt increase % - >5% means fast borrowing'},
    'Debt Service': {'func': lambda: fred.get_series('FGDS')[-1] if fred else np.nan, 'thresh': 20, 'desc': 'Debt payment % of income - >20% means strain'},
    'Debt-to-GDP': {'func': lambda: (fred.get_series('GFDEBTN')[-1] / fred.get_series('GDP')[-1]) * 100 if fred else np.nan, 'thresh': 100, 'desc': 'Debt % of GDP - >100% means high burden'},
    'Defaults': {'func': lambda: scrape_moodys_defaults(), 'thresh': 5, 'desc': 'Default rate % - >5% means credit trouble'},
    'Deflation': {'func': lambda: fred.get_series('CPIAUCSL')[-1] if fred else np.nan, 'thresh': 0, 'desc': 'Negative inflation - prices falling hurts economy'},
    'Demographic Aging': {'func': lambda: wbdata.get_series('SP.POP.65UP.TO.ZS')['USA'], 'thresh': 1, 'desc': 'Aging population % change - >1% means older workforce'},
    'Economic Output Share': {'func': lambda: wbdata.get_series('NY.GDP.MKTP.CD')['USA'] / wbdata.get_series('NY.GDP.MKTP.CD')['WLD'] * 100, 'thresh': np.nan, 'desc': 'Share of global economy - falling means decline'},
    'Education Investment': {'func': lambda: wbdata.get_series('SE.XPD.TOTL.GD.ZS')['USA'], 'thresh': 5, 'desc': 'Education % GDP - >5% means investment'},
    'Fiscal Deficits': {'func': lambda: fred.get_series('MTSDS133FMS')[-1] if fred else np.nan, 'thresh': 6, 'desc': 'Deficit % GDP - >6% means high spending'},
    'Foreign Reserves': {'func': lambda: fred.get_series('TRESEGT')[-1] if fred else np.nan, 'thresh': np.nan, 'desc': 'Foreign reserves - falling means vulnerability'},
    'GDP': {'func': lambda: fred.get_series('GDP')[-1] if fred else np.nan, 'thresh': np.nan, 'desc': 'Total economic output - high is good growth'},
    'GDP Gap': {'func': lambda: fred.get_series('GDPPOT')[-1] - fred.get_series('GDP')[-1] if fred else np.nan, 'thresh': 0, 'desc': 'Gap to potential GDP - positive means room to grow'},
    'GDP Share': {'func': lambda: wbdata.get_series('NY.GDP.MKTP.CD')['USA'] / wbdata.get_series('NY.GDP.MKTP.CD')['WLD'] * 100, 'thresh': np.nan, 'desc': 'US GDP as % of world - high means global dominance'},
    'GDP per Capita': {'func': lambda: wbdata.get_series('NY.GDP.PCAP.CD')['USA'], 'thresh': np.nan, 'desc': 'Income per person - rising means better living standards'},
    'Growth = productivity': {'func': lambda: fred.get_series('OPHNFB')[-1] if fred else np.nan, 'thresh': np.nan, 'desc': 'Growth matching productivity - stable means balanced'},
    'Growth > rates': {'func': lambda: fred.get_series('GDP')[-1] - fred.get_series('FEDFUNDS')[-1] if fred else np.nan, 'thresh': 2, 'desc': 'GDP > nominal rates - >2% means healthy'},
    'High leverage': {'func': lambda: fred.get_series('NFSDB')[-1] / fred.get_series('GDP')[-1] * 100 if fred else np.nan, 'thresh': 80, 'desc': 'Household debt % GDP - >80% means high leverage'},
    'Inflation Rate': {'func': lambda: fred.get_series('CPIAUCSL')[-1] if fred else np.nan, 'thresh': 3, 'desc': 'Price rise % - high erodes money value'},
    'Internal Conflicts': {'func': lambda: scrape_conflicts_index(), 'thresh': 20, 'desc': 'Protest count - >20 means unrest'},
    'Inventory Levels': {'func': lambda: fred.get_series('ISRATIO')[-1] if fred else np.nan, 'thresh': np.nan, 'desc': 'Inventory to sales ratio - low means high demand'},
    'Leading Economic Index (LEI)': {'func': lambda: fred.get_series('USSLIND')[-1] if fred else np.nan, 'thresh': np.nan, 'desc': 'LEI index - falling signals downturn'},
    'Leverage': {'func': lambda: fred.get_series('NFSDB')[-1] / fred.get_series('GDP')[-1] * 100 if fred else np.nan, 'thresh': 80, 'desc': 'Household debt % GDP - >80% means high leverage'},
    'Low defaults': {'func': lambda: scrape_moodys_defaults(), 'thresh': 2, 'desc': 'Default rate % - <2% means low credit risk'},
    'Low leverage': {'func': lambda: fred.get_series('NFSDB')[-1] / fred.get_series('GDP')[-1] * 100 if fred else np.nan, 'thresh': 80, 'desc': 'Household debt % GDP - <80% means low leverage'},
    'M2 Money Supply': {'func': lambda: fred.get_series('M2SL')[-1] if fred else np.nan, 'thresh': np.nan, 'desc': 'Money supply - high means inflation risk'},
    'Military Losses': {'func': lambda: scrape_military_losses(), 'thresh': 1, 'desc': 'Defeats count - >1/year means weakening'},
    'Military Spending': {'func': lambda: scrape_sipri_military(), 'thresh': 4, 'desc': 'Military % GDP - >4% means strain'},
    'Money printing': {'func': lambda: fred.get_series('M2SL')[-1] - fred.get_series('M2SL')[-4] if fred else np.nan, 'thresh': 10, 'desc': 'M2 growth % - >10% means printing'},
    'Negative real rates': {'func': lambda: fred.get_series('FEDFUNDS')[-1] - fred.get_series('CPIAUCSL')[-1] if fred else np.nan, 'thresh': 0, 'desc': 'Rates minus inflation - negative hurts savers'},
    'Nonfarm Payrolls': {'func': lambda: fred.get_series('PAYEMS')[-1] - fred.get_series('PAYEMS')[-2] if fred else np.nan, 'thresh': 150000, 'desc': 'Monthly job adds - >150K is strong'},
    'P/E Ratio': {'func': lambda: scrape_multpl_pe(), 'thresh': 25, 'desc': 'Stock price to earnings - >25 means overvalued'},
    'Positive real rates': {'func': lambda: fred.get_series('FEDFUNDS')[-1] - fred.get_series('CPIAUCSL')[-1] if fred else np.nan, 'thresh': 0, 'desc': 'Rates minus inflation - positive helps savers'},
    'Power Index': {'func': lambda: scrape_globalfirepower_index(), 'thresh': np.nan, 'desc': 'Military power score - low means weakness'},
    'Productivity Growth': {'func': lambda: fred.get_series('OPHNFB')[-1] if fred else np.nan, 'thresh': 3, 'desc': 'Output per hour % - >3% means efficiency gains'},
    'R&D Patents': {'func': lambda: wbdata.get_series('IP.PAT.RESD')['USA'], 'thresh': np.nan, 'desc': 'Patents filed - high means innovation'},
    'Rates near 0%': {'func': lambda: fred.get_series('FEDFUNDS')[-1] if fred else np.nan, 'thresh': 1, 'desc': 'Interest rates - <1% means easy money'},
    'Real Rates': {'func': lambda: fred.get_series('FEDFUNDS')[-1] - fred.get_series('CPIAUCSL')[-1] if fred else np.nan, 'thresh': 0, 'desc': 'Rates minus inflation - positive hurts borrowing'},
    'Reserve Currency Usage': {'func': lambda: scrape_reserve_currency_share(), 'thresh': 50, 'desc': 'USD % reserves - <50% means decline'},
    'Retail Sales Growth': {'func': lambda: fred.get_series('RSXFS')[-1] if fred else np.nan, 'thresh': 3, 'desc': 'Retail sales % change - >3% means strong shoppers'},
    'Short Rates': {'func': lambda: fred.get_series('FEDFUNDS')[-1] if fred else np.nan, 'thresh': np.nan, 'desc': 'Short-term interest rate - rising slows economy'},
    'Stock Market Return': {'func': lambda: yf.Ticker('^GSPC').history(period='1y')['Close'].pct_change().mean() * 100, 'thresh': 10, 'desc': 'S&P yearly % return - high means good gains'},
    'Trade Balance': {'func': lambda: fred.get_series('NETEXP')[-1] if fred else np.nan, 'thresh': 2, 'desc': 'Trade surplus % GDP - positive means strength'},
    'Trade Share': {'func': lambda: wbdata.get_series('NE.TRD.GNFS.ZS')['USA'], 'thresh': 15, 'desc': 'Trade % global - >15% means dominance'},
    'Unemployment Claims': {'func': lambda: fred.get_series('ICSA')[-1] if fred else np.nan, 'thresh': 300000, 'desc': 'Weekly jobless claims - rising means trouble'},
    'Unemployment Rate': {'func': lambda: fred.get_series('UNRATE')[-1] if fred else np.nan, 'thresh': 5, 'desc': '% jobless - low is healthy economy'},
    'Wage Growth': {'func': lambda: fred.get_series('AHETPI')[-1] if fred else np.nan, 'thresh': 3, 'desc': 'Hourly earnings growth - >3% means rising pay'},
    'Wealth Gaps': {'func': lambda: wbdata.get_series('SI.POV.GINI')['USA'], 'thresh': 40, 'desc': 'Gini index - >40 means inequality'},
    'Yield Curve Slope': {'func': lambda: fred.get_series('T10Y2Y')[-1] if fred else np.nan, 'thresh': 0, 'desc': '10Y-2Y yield spread - negative signals recession'},
}

# Additional scrapers
def scrape_wef_competitiveness():
    try:
        r = requests.get('https://www.weforum.org/reports/global-competitiveness-report-2025')
        soup = BeautifulSoup(r.text, 'html.parser')
        score = soup.find(string=re.compile(r'US score \d+'))
        return float(re.search(r'\d+', score).group()) if score else np.nan
    except:
        return np.nan

def scrape_conflicts_index():
    try:
        r = requests.get('https://www.globalconflicttracker.org/')
        soup = BeautifulSoup(r.text, 'html.parser')
        count = len(soup.find_all('div', class_='conflict-item'))
        return count
    except:
        return np.nan

def scrape_reserve_currency_share():
    try:
        r = requests.get('https://www.imf.org/en/Data')
        soup = BeautifulSoup(r.text, 'html.parser')
        usd_share = soup.find(string=re.compile(r'USD reserves \d+%'))
        return float(re.search(r'\d+', usd_share).group()) if usd_share else np.nan
    except:
        return np.nan

def scrape_military_losses():
    try:
        r = requests.get('https://www.globalfirepower.com/military-losses.php')
        soup = BeautifulSoup(r.text, 'html.parser')
        losses = soup.find(string=re.compile(r'US losses \d+'))
        return float(re.search(r'\d+', losses).group()) if losses else np.nan
    except:
        return np.nan

def scrape_moodys_defaults():
    try:
        r = requests.get('https://www.moodys.com/web/en/us/insights/data-stories/us-corporate-default-risk-in-2025.html')
        soup = BeautifulSoup(r.text, 'html.parser')
        rate_text = soup.find(string=re.compile(r'high-yield.*default rate.*\d+\.\d+%.*\d+\.\d+%', re.I))
        if rate_text:
            rates = re.findall(r'\d+\.\d+', rate_text)
            return float(sum(map(float, rates)) / len(rates)) if rates else np.nan
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
            latest = tds[-1].text.strip() if tds else np.nan
            if latest == 'n.a.':
                treasury_row = soup.find('th', string=re.compile(r'10-year', re.I)).parent
                latest = treasury_row.find_all('td')[-1].text.strip() if treasury_row else np.nan
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
        url = 'https://sipri.org/sites/default/files/2025-04/2504_milex_data_sheet_2024.xlsx'
        df = pd.read_excel(url, sheet_name='Share of GDP', skiprows=5)
        return df.loc[df['Country'] == 'USA', df.columns[-1]].values[0]
    except:
        return np.nan

def scrape_transparency_cpi():
    try:
        url = 'https://images.transparencycdn.org/images/CPI2024_FullDataSet.xlsx'
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

# Create table if not exists
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS data (Indicator TEXT PRIMARY KEY, Value REAL)")
conn.commit()

st.title('Econ Mirror Dashboard - July 22 2025')
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

if st.button('Refresh Now'):
    data = fetch_all()
    df = pd.DataFrame(list(data.items()), columns=['Indicator', 'Value'])
    df.to_sql('data', conn, if_exists='replace', index=False)
else:
    try:
        df = pd.read_sql('SELECT * FROM data', conn)
    except:
        data = fetch_all()
        df = pd.DataFrame(list(data.items()), columns=['Indicator', 'Value'])
        df.to_sql('data', conn, if_exists='replace', index=False)

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
    df['Value'] = df['Value'].apply(lambda x: 'N/A' if pd.isna(x) else x)  # Replace NaN with N/A
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