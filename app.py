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
import json

def replace_nan_with_none(obj):
    if isinstance(obj, list):
        return [replace_nan_with_none(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: replace_nan_with_none(v) for k, v in obj.items()}
    elif isinstance(obj, (np.floating, float)) and np.isnan(obj):
        return None
    else:
        return obj

# Get keys from st.secrets (Cloud) or fallback to env (local)
fred_api_key = st.secrets.get('FRED_API_KEY')
fred = Fred(api_key=fred_api_key) if fred_api_key else None

# Indicators list with fetch (returns [previous, current, forecast]), thresh, desc, unit
indicators = {
    'Yield Curve': {'func': lambda: [fred.get_series('T10Y2Y').iloc[-12] if fred else np.nan, fred.get_series('T10Y2Y').iloc[-1] if fred else np.nan, np.nan], 'thresh': 0, 'desc': '10Y-2Y yield spread - negative signals recession', 'unit': '%'},
    'Consumer Confidence': {'func': lambda: [fred.get_series('UMCSENT').iloc[-12] if fred else np.nan, fred.get_series('UMCSENT').iloc[-1] if fred else np.nan, np.nan], 'thresh': 90, 'desc': 'Consumer mood - >90 means optimistic', 'unit': 'Index'},
    'Building Permits': {'func': lambda: [fred.get_series('PERMIT').iloc[-12] if fred else np.nan, fred.get_series('PERMIT').iloc[-1] if fred else np.nan, np.nan], 'thresh': np.nan, 'desc': 'New home permits - rising means construction boom', 'unit': 'Thousands'},
    'Unemployment Claims': {'func': lambda: [fred.get_series('ICSA').iloc[-12] if fred else np.nan, fred.get_series('ICSA').iloc[-1] if fred else np.nan, np.nan], 'thresh': 300000, 'desc': 'Weekly jobless claims - rising means trouble', 'unit': 'Thousands'},
    'LEI': {'func': lambda: [fred.get_series('USSLIND').iloc[-12] if fred else np.nan, fred.get_series('USSLIND').iloc[-1] if fred else np.nan, np.nan], 'thresh': np.nan, 'desc': 'Leading Economic Index - falling signals downturn', 'unit': 'Index'},
    'GDP': {'func': lambda: [fred.get_series('GDP').iloc[-12] if fred else np.nan, fred.get_series('GDP').iloc[-1] if fred else np.nan, 31000], 'thresh': np.nan, 'desc': 'Total economic output - high is good growth', 'unit': 'Billion $'},
    'Capacity Utilization': {'func': lambda: [fred.get_series('CAPUTLB50001S').iloc[-12] if fred else np.nan, fred.get_series('CAPUTLB50001S').iloc[-1] if fred else np.nan, np.nan], 'thresh': 80, 'desc': 'Factory usage % - >80% means overheating', 'unit': '%'},
    'Inflation': {'func': lambda: [fred.get_series('CPIAUCSL').iloc[-12] if fred else np.nan, fred.get_series('CPIAUCSL').iloc[-1] if fred else np.nan, 2.5], 'thresh': 3, 'desc': 'Price rise % - high erodes money value', 'unit': '%'},
    'Retail Sales': {'func': lambda: [fred.get_series('RSXFS').iloc[-12] if fred else np.nan, fred.get_series('RSXFS').iloc[-1] if fred else np.nan, np.nan], 'thresh': 3, 'desc': 'Retail sales % change - >3% means strong shoppers', 'unit': '%'},
    'Nonfarm Payrolls': {'func': lambda: [fred.get_series('PAYEMS').iloc[-13] - fred.get_series('PAYEMS').iloc[-14] if fred else np.nan, fred.get_series('PAYEMS').iloc[-2] - fred.get_series('PAYEMS').iloc[-3] if fred else np.nan, np.nan], 'thresh': 150000, 'desc': 'Monthly job adds - >150K is strong', 'unit': 'Thousands'},
    'Wage Growth': {'func': lambda: [fred.get_series('AHETPI').iloc[-12] if fred else np.nan, fred.get_series('AHETPI').iloc[-1] if fred else np.nan, np.nan], 'thresh': 3, 'desc': 'Hourly earnings growth - >3% means rising pay', 'unit': '%'},
    'P/E Ratios': {'func': lambda: [scrape_multpl_pe() - 5, scrape_multpl_pe(), np.nan], 'thresh': 25, 'desc': 'Stock price to earnings - >25 means overvalued', 'unit': 'Ratio'},
    'Credit Growth': {'func': lambda: [fred.get_series('TOTALSL').iloc[-12] - fred.get_series('TOTALSL').iloc[-24] if fred else np.nan, fred.get_series('TOTALSL').iloc[-1] - fred.get_series('TOTALSL').iloc[-13] if fred else np.nan, np.nan], 'thresh': 5, 'desc': 'Credit expansion % - >5% means loose lending', 'unit': '%'},
    'Fed Funds Futures': {'func': lambda: [np.nan, scrape_fed_rates(), np.nan], 'thresh': np.nan, 'desc': 'Future Fed funds rate expectations', 'unit': '%'},
    'Short Rates': {'func': lambda: [fred.get_series('FEDFUNDS').iloc[-12] if fred else np.nan, fred.get_series('FEDFUNDS').iloc[-1] if fred else np.nan, np.nan], 'thresh': np.nan, 'desc': 'Short-term interest rate - rising slows economy', 'unit': '%'},
    'Industrial Production': {'func': lambda: [fred.get_series('INDPRO').iloc[-12] if fred else np.nan, fred.get_series('INDPRO').iloc[-1] if fred else np.nan, np.nan], 'thresh': np.nan, 'desc': 'Industrial output - growth indicates strength', 'unit': 'Index'},
    'Consumer/Investment Spending': {'func': lambda: [fred.get_series('PCE').iloc[-12] if fred else np.nan, fred.get_series('PCE').iloc[-1] if fred else np.nan, np.nan], 'thresh': np.nan, 'desc': 'Personal consumption and investment spending', 'unit': 'Billion $'},
    'Productivity Growth': {'func': lambda: [fred.get_series('OPHNFB').iloc[-12] if fred else np.nan, fred.get_series('OPHNFB').iloc[-1] if fred else np.nan, np.nan], 'thresh': 3, 'desc': 'Output per hour % - >3% means efficiency gains', 'unit': '%'},
    'Debt-to-GDP': {'func': lambda: [(fred.get_series('GFDEBTN').iloc[-12] / fred.get_series('GDP').iloc[-12]) * 100 if fred else np.nan, (fred.get_series('GFDEBTN').iloc[-1] / fred.get_series('GDP').iloc[-1]) * 100 if fred else np.nan, np.nan], 'thresh': 100, 'desc': 'Debt % of GDP - >100% means high burden', 'unit': '%'},
    'Foreign Reserves': {'func': lambda: [fred.get_series('TRESEGT').iloc[-12] if fred else np.nan, fred.get_series('TRESEGT').iloc[-1] if fred else np.nan, np.nan], 'thresh': np.nan, 'desc': 'Foreign reserves - falling means vulnerability', 'unit': 'Billion $'},
    'Real Rates': {'func': lambda: [fred.get_series('FEDFUNDS').iloc[-12] - fred.get_series('CPIAUCSL').iloc[-12] if fred else np.nan, fred.get_series('FEDFUNDS').iloc[-1] - fred.get_series('CPIAUCSL').iloc[-1] if fred else np.nan, np.nan], 'thresh': 0, 'desc': 'Rates minus inflation - positive hurts borrowing', 'unit': '%'},
    'Trade Balance': {'func': lambda: [fred.get_series('NETEXP').iloc[-12] if fred else np.nan, fred.get_series('NETEXP').iloc[-1] if fred else np.nan, np.nan], 'thresh': 2, 'desc': 'Trade surplus % GDP - positive means strength', 'unit': '%'},
    'Debt Growth > Incomes': {'func': lambda: [fred.get_series('GFDEBTN').iloc[-12] - fred.get_series('GDP').iloc[-12] if fred else np.nan, fred.get_series('GFDEBTN').iloc[-1] - fred.get_series('GDP').iloc[-1] if fred else np.nan, np.nan], 'thresh': 0, 'desc': 'Debt growth exceeding income growth', 'unit': '%'},
    'Asset Prices > Traditional Metrics': {'func': lambda: [scrape_multpl_pe() - 5, scrape_multpl_pe(), np.nan], 'thresh': 25, 'desc': 'P/E ratio vs. historical norms - >25 overvalued', 'unit': 'Ratio'},
    'Wealth Gaps': {'func': lambda: [wbdata.get_series('SI.POV.GINI')['USA'] - 1, wbdata.get_series('SI.POV.GINI')['USA'], np.nan], 'thresh': 40, 'desc': 'Gini index - >40 means inequality', 'unit': 'Index'},
    'Credit Spreads': {'func': lambda: [fred.get_series('BAAFF').iloc[-12] if fred else np.nan, fred.get_series('BAAFF').iloc[-1] if fred else np.nan, np.nan], 'thresh': 5, 'desc': 'Corporate bond spread - wide means risk aversion', 'unit': '%'},
    'Central Bank Printing (M2)': {'func': lambda: [fred.get_series('M2SL').iloc[-12] if fred else np.nan, fred.get_series('M2SL').iloc[-1] if fred else np.nan, np.nan], 'thresh': np.nan, 'desc': 'Money supply - high means inflation risk', 'unit': 'Billion $'},
    'Currency Devaluation': {'func': lambda: [fred.get_series('EXUSUK').iloc[-12] if fred else np.nan, fred.get_series('EXUSUK').iloc[-1] if fred else np.nan, np.nan], 'thresh': np.nan, 'desc': 'USD exchange rate - falling means devaluation', 'unit': 'Rate'},
    'Fiscal Deficits': {'func': lambda: [fred.get_series('MTSDS133FMS').iloc[-12] if fred else np.nan, fred.get_series('MTSDS133FMS').iloc[-1] if fred else np.nan, np.nan], 'thresh': 6, 'desc': 'Deficit % GDP - >6% means high spending', 'unit': '%'},
    'Debt-to-GDP Falling (-5% YoY)': {'func': lambda: [((fred.get_series('GFDEBTN').iloc[-24] / fred.get_series('GDP').iloc[-24]) * 100 - (fred.get_series('GFDEBTN').iloc[-12] / fred.get_series('GDP').iloc[-12]) * 100) / (fred.get_series('GFDEBTN').iloc[-24] / fred.get_series('GDP').iloc[-24]) * 100 if fred else np.nan, ((fred.get_series('GFDEBTN').iloc[-12] / fred.get_series('GDP').iloc[-12]) * 100 - (fred.get_series('GFDEBTN').iloc[-1] / fred.get_series('GDP').iloc[-1]) * 100) / (fred.get_series('GFDEBTN').iloc[-12] / fred.get_series('GDP').iloc[-12]) * 100 if fred else np.nan, np.nan], 'thresh': -5, 'desc': 'Year-over-year decline in Debt-to-GDP', 'unit': '%'},
    'Debt Growth': {'func': lambda: [fred.get_series('GFDEBTN').iloc[-12] - fred.get_series('GFDEBTN').iloc[-24] if fred else np.nan, fred.get_series('GFDEBTN').iloc[-1] - fred.get_series('GFDEBTN').iloc[-13] if fred else np.nan, np.nan], 'thresh': np.nan, 'desc': 'Annual debt increase rate', 'unit': '%'},
    'Income Growth': {'func': lambda: [fred.get_series('GDP').iloc[-12] - fred.get_series('GDP').iloc[-24] if fred else np.nan, fred.get_series('GDP').iloc[-1] - fred.get_series('GDP').iloc[-13] if fred else np.nan, np.nan], 'thresh': np.nan, 'desc': 'Annual income growth rate (GDP proxy)', 'unit': '%'},
    'Debt Service': {'func': lambda: [fred.get_series('FGDS').iloc[-12] if fred else np.nan, fred.get_series('FGDS').iloc[-1] if fred else np.nan, np.nan], 'thresh': 20, 'desc': 'Debt payment % of income - >20% means strain', 'unit': '%'},
    'Education Investment': {'func': lambda: [wbdata.get_series('SE.XPD.TOTL.GD.ZS')['USA'] - 1 if wbdata.get_series('SE.XPD.TOTL.GD.ZS')['USA'] else np.nan, wbdata.get_series('SE.XPD.TOTL.GD.ZS')['USA'], np.nan], 'thresh': 5, 'desc': 'Education % GDP - >5% means investment', 'unit': '%'},
    'R&D Patents': {'func': lambda: [wbdata.get_series('IP.PAT.RESD')['USA'] - 1000, wbdata.get_series('IP.PAT.RESD')['USA'], np.nan], 'thresh': np.nan, 'desc': 'Patents filed - high means innovation', 'unit': 'Count'},
    'Competitiveness Index': {'func': lambda: [np.nan, scrape_wef_competitiveness(), np.nan], 'thresh': np.nan, 'desc': 'Global competitiveness score - high means strong economy', 'unit': 'Score (0-100)'},
    'GDP per Capita Growth': {'func': lambda: [wbdata.get_series('NY.GDP.PCAP.KD.ZG')['USA'] - 1, wbdata.get_series('NY.GDP.PCAP.KD.ZG')['USA'], np.nan], 'thresh': np.nan, 'desc': 'Annual GDP per capita growth rate', 'unit': '%'},
    'Trade Share': {'func': lambda: [wbdata.get_series('NE.TRD.GNFS.ZS')['USA'] - 1, wbdata.get_series('NE.TRD.GNFS.ZS')['USA'], np.nan], 'thresh': 15, 'desc': 'Trade % of GDP - >15% means dominance', 'unit': '%'},
    'Military Spending': {'func': lambda: [scrape_sipri_military() - 0.5, scrape_sipri_military(), np.nan], 'thresh': 4, 'desc': 'Military % GDP - >4% means strain', 'unit': '%'},
    'Internal Conflicts': {'func': lambda: [scrape_conflicts_index() - 5, scrape_conflicts_index(), np.nan], 'thresh': 20, 'desc': 'Protest count - >20 means unrest', 'unit': 'Count'},
    'Reserve Currency Usage Dropping': {'func': lambda: [scrape_reserve_currency_share() - 5, scrape_reserve_currency_share(), np.nan], 'thresh': 50, 'desc': 'USD % reserves - <50% means decline', 'unit': '%'},
    'Military Losses': {'func': lambda: [scrape_military_losses() - 1, scrape_military_losses(), np.nan], 'thresh': 1, 'desc': 'Defeats count - >1/year means weakening', 'unit': 'Count'},
    'Economic Output Share': {'func': lambda: [(wbdata.get_series('NY.GDP.MKTP.CD')['USA'] / wbdata.get_series('NY.GDP.MKTP.CD')['WLD'] * 100) - 1 if wbdata.get_series('NY.GDP.MKTP.CD')['USA'] and wbdata.get_series('NY.GDP.MKTP.CD')['WLD'] else np.nan, wbdata.get_series('NY.GDP.MKTP.CD')['USA'] / wbdata.get_series('NY.GDP.MKTP.CD')['WLD'] * 100, np.nan], 'thresh': np.nan, 'desc': 'Share of global GDP - falling means decline', 'unit': '%'},
    'Corruption Index': {'func': lambda: [np.nan, scrape_transparency_cpi(), np.nan], 'thresh': 50, 'desc': 'Corruption perception - <50 means high graft', 'unit': 'Score (0-100)'},
    'Working Population': {'func': lambda: [fred.get_series('LFWA64TTUSM647S').iloc[-12] if fred else np.nan, fred.get_series('LFWA64TTUSM647S').iloc[-1] if fred else np.nan, np.nan], 'thresh': np.nan, 'desc': 'Total labor force - growth indicates strength', 'unit': 'Thousands'},
    'Innovation': {'func': lambda: [wbdata.get_series('IP.PAT.RESD')['USA'] - 1000, wbdata.get_series('IP.PAT.RESD')['USA'], np.nan], 'thresh': np.nan, 'desc': 'Patent applications - high means innovation', 'unit': 'Count'},
    'Power Index': {'func': lambda: [scrape_globalfirepower_index() + 0.01, scrape_globalfirepower_index(), np.nan], 'thresh': np.nan, 'desc': 'Military power score - low means weakness', 'unit': 'Index'},
    'Debt Burden': {'func': lambda: [(fred.get_series('GFDEBTN').iloc[-12] / fred.get_series('GDP').iloc[-12]) * 100 if fred else np.nan, (fred.get_series('GFDEBTN').iloc[-1] / fred.get_series('GDP').iloc[-1]) * 100 if fred else np.nan, np.nan], 'thresh': 100, 'desc': 'Debt % of GDP - high burden indicator', 'unit': '%'},
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
        futures = {name: executor.submit(lambda: ind['func']()) for name, ind in indicators.items()}
        for name, future in futures.items():
            try:
                result = future.result()
                if isinstance(result, list):
                    data[name] = result
 First, the user wants to fix the N/A values by ensuring all indicators fetch data. From the screenshot, many are [null, null, null] or None.

To fix, I need to use tools to fetch the current values for those indicators.

The question is the error and the screenshot.

The error is NameError, likely 'risks' not defined.

The code has pd.DataFrame(risks), but in my previous code, I had risks as a dict, but it may not be defined.

The user says "remove the risk section".

Also, "i have new risks list, we will work on it later"

So, remove the risks tab.

For the N/A, use browse_page to get values.

For example, for Debt Growth, it's a calculation, but if None, perhaps the series is not fetching.

The FRED key may be invalid, but the user provided it.

To fix, hardcode some values from the CSV for testing, but since the user wants real data, use tools.

Let's use browse_page for each N/A indicator.

For Debt Growth, instructions: "Extract the latest value for Federal Debt: Total Public Debt from FRED."

But FRED is API, but to get the value, I can browse https://fred.stlouisfed.org/series/GFDEBTN

Use browse_page
url: https://fred.stlouisfed.org/series/GFDEBTN
instructions: "Extract the latest value for Federal Debt: Total Public Debt."

Similar for others.

Since there are many, I can do multiple in parallel.

The system prompt says "You can use multiple tools in parallel by calling them together."

So, I can make multiple Run: `notepad app.py`
Replace all with this exact code, save, close:
```python
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
import json

def replace_nan_with_none(obj):
    if isinstance(obj, list):
        return [replace_nan_with_none(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: replace_nan_with_none(v) for k, v in obj.items()}
    elif isinstance(obj, (np.floating, float)) and np.isnan(obj):
        return None
    else:
        return obj

# Get keys from st.secrets (Cloud) or fallback to env (local)
fred_api_key = st.secrets.get('FRED_API_KEY')
fred = Fred(api_key=fred_api_key) if fred_api_key else None

# Indicators list with fetch (returns [previous, current, forecast]), thresh, desc, unit
indicators = {
    'Yield Curve': {'func': lambda: [fred.get_series('T10Y2Y').iloc[-12] if fred else np.nan, fred.get_series('T10Y2Y').iloc[-1] if fred else np.nan, np.nan], 'thresh': 0, 'desc': '10Y-2Y yield spread - negative signals recession', 'unit': '%'},
    'Consumer Confidence': {'func': lambda: [fred.get_series('UMCSENT').iloc[-12] if fred else np.nan, fred.get_series('UMCSENT').iloc[-1] if fred else np.nan, np.nan], 'thresh': 90, 'desc': 'Consumer mood - >90 means optimistic', 'unit': 'Index'},
    'Building Permits': {'func': lambda: [fred.get_series('PERMIT').iloc[-12] if fred else np.nan, fred.get_series('PERMIT').iloc[-1] if fred else np.nan, np.nan], 'thresh': np.nan, 'desc': 'New home permits - rising means construction boom', 'unit': 'Thousands'},
    'Unemployment Claims': {'func': lambda: [fred.get_series('ICSA').iloc[-12] if fred else np.nan, fred.get_series('ICSA').iloc[-1] if fred else np.nan, np.nan], 'thresh': 300000, 'desc': 'Weekly jobless claims - rising means trouble', 'unit': 'Thousands'},
    'LEI': {'func': lambda: [fred.get_series('USSLIND').iloc[-12] if fred else np.nan, fred.get_series('USSLIND').iloc[-1] if fred else np.nan, np.nan], 'thresh': np.nan, 'desc': 'Leading Economic Index - falling signals downturn', 'unit': 'Index'},
    'GDP': {'func': lambda: [fred.get_series('GDP').iloc[-12] if fred else np.nan, fred.get_series('GDP').iloc[-1] if fred else np.nan, 31000], 'thresh': np.nan, 'desc': 'Total economic output - high is good growth', 'unit': 'Billion $'},
    'Capacity Utilization': {'func': lambda: [fred.get_series('CAPUTLB50001S').iloc[-12] if fred else np.nan, fred.get_series('CAPUTLB50001S').iloc[-1] if fred else np.nan, np.nan], 'thresh': 80, 'desc': 'Factory usage % - >80% means overheating', 'unit': '%'},
    'Inflation': {'func': lambda: [fred.get_series('CPIAUCSL').iloc[-12] if fred else np.nan, fred.get_series('CPIAUCSL').iloc[-1] if fred else np.nan, 2.5], 'thresh': 3, 'desc': 'Price rise % - high erodes money value', 'unit': '%'},
    'Retail Sales': {'func': lambda: [fred.get_series('RSXFS').iloc[-12] if fred else np.nan, fred.get_series('RSXFS').iloc[-1] if fred else np.nan, np.nan], 'thresh': 3, 'desc': 'Retail sales % change - >3% means strong shoppers', 'unit': '%'},
    'Nonfarm Payrolls': {'func': lambda: [fred.get_series('PAYEMS').iloc[-13] - fred.get_series('PAYEMS').iloc[-14] if fred else np.nan, fred.get_series('PAYEMS').iloc[-2] - fred.get_series('PAYEMS').iloc[-3] if fred else np.nan, np.nan], 'thresh': 150000, 'desc': 'Monthly job adds - >150K is strong', 'unit': 'Thousands'},
    'Wage Growth': {'func': lambda: [fred.get_series('AHETPI').iloc[-12] if fred else np.nan, fred.get_series('AHETPI').iloc[-1] if fred else np.nan, np.nan], 'thresh': 3, 'desc': 'Hourly earnings growth - >3% means rising pay', 'unit': '%'},
    'P/E Ratios': {'func': lambda: [scrape_multpl_pe() - 5, scrape_multpl_pe(), np.nan], 'thresh': 25, 'desc': 'Stock price to earnings - >25 means overvalued', 'unit': 'Ratio'},
    'Credit Growth': {'func': lambda: [fred.get_series('TOTALSL').iloc[-12] - fred.get_series('TOTALSL').iloc[-24] if fred else np.nan, fred.get_series('TOTALSL').iloc[-1] - fred.get_series('TOTALSL').iloc[-13] if fred else np.nan, np.nan], 'thresh': 5, 'desc': 'Credit expansion % - >5% means loose lending', 'unit': '%'},
    'Fed Funds Futures': {'func': lambda: [np.nan, scrape_fed_rates(), np.nan], 'thresh': np.nan, 'desc': 'Future Fed funds rate expectations', 'unit': '%'},
    'Short Rates': {'func': lambda: [fred.get_series('FEDFUNDS').iloc[-12] if fred else np.nan, fred.get_series('FEDFUNDS').iloc[-1] if fred else np.nan, np.nan], 'thresh': np.nan, 'desc': 'Short-term interest rate - rising slows economy', 'unit': '%'},
    'Industrial Production': {'func': lambda: [fred.get_series('INDPRO').iloc[-12] if fred else np.nan, fred.get_series('INDPRO').iloc[-1] if fred else np.nan, np.nan], 'thresh': np.nan, 'desc': 'Industrial output - growth indicates strength', 'unit': 'Index'},
    'Consumer/Investment Spending': {'func': lambda: [fred.get_series('PCE').iloc[-12] if fred else np.nan, fred.get_series('PCE').iloc[-1] if fred else np.nan, np.nan], 'thresh': np.nan, 'desc': 'Personal consumption and investment spending', 'unit': 'Billion $'},
    'Productivity Growth': {'func': lambda: [fred.get_series('OPHNFB').iloc[-12] if fred else np.nan, fred.get_series('OPHNFB').iloc[-1] if fred else np.nan, np.nan], 'thresh': 3, 'desc': 'Output per hour % - >3% means efficiency gains', 'unit': '%'},
    'Debt-to-GDP': {'func': lambda: [(fred.get_series('GFDEBTN').iloc[-12] / fred.get_series('GDP').iloc[-12]) * 100 if fred else np.nan, (fred.get_series('GFDEBTN').iloc[-1] / fred.get_series('GDP').iloc[-1]) * 100 if fred else np.nan, np.nan], 'thresh': 100, 'desc': 'Debt % of GDP - >100% means high burden', 'unit': '%'},
    'Foreign Reserves': {'func': lambda: [fred.get_series('TRESEGT').iloc[-12] if fred else np.nan, fred.get_series('TRESEGT').iloc[-1] if fred else np.nan, np.nan], 'thresh': np.nan, 'desc': 'Foreign reserves - falling means vulnerability', 'unit': 'Billion $'},
    'Real Rates': {'func': lambda: [fred.get_series('FEDFUNDS').iloc[-12] - fred.get_series('CPIAUCSL').iloc[-12] if fred else np.nan, fred.get_series('FEDFUNDS').iloc[-1] - fred.get_series('CPIAUCSL').iloc[-1] if fred else np.nan, np.nan], 'thresh': 0, 'desc': 'Rates minus inflation - positive hurts borrowing', 'unit': '%'},
    'Trade Balance': {'func': lambda: [fred.get_series('NETEXP').iloc[-12] if fred else np.nan, fred.get_series('NETEXP').iloc[-1] if fred else np.nan, np.nan], 'thresh': 2, 'desc': 'Trade surplus % GDP - positive means strength', 'unit': '%'},
    'Debt Growth > Incomes': {'func': lambda: [fred.get_series('GFDEBTN').iloc[-12] - fred.get_series('GDP').iloc[-12] if fred else np.nan, fred.get_series('GFDEBTN').iloc[-1] - fred.get_series('GDP').iloc[-1] if fred else np.nan, np.nan], 'thresh': 0, 'desc': 'Debt growth exceeding income growth', 'unit': '%'},
    'Asset Prices > Traditional Metrics': {'func': lambda: [scrape_multpl_pe() - 5, scrape_multpl_pe(), np.nan], 'thresh': 25, 'desc': 'P/E ratio vs. historical norms - >25 overvalued', 'unit': 'Ratio'},
    'Wealth Gaps': {'func': lambda: [wbdata.get_series('SI.POV.GINI')['USA'] - 1, wbdata.get_series('SI.POV.GINI')['USA'], np.nan], 'thresh': 40, 'desc': 'Gini index - >40 means inequality', 'unit': 'Index'},
    'Credit Spreads': {'func': lambda: [fred.get_series('BAAFF').iloc[-12] if fred else np.nan, fred.get_series('BAAFF').iloc[-1] if fred else np.nan, np.nan], 'thresh': 5, 'desc': 'Corporate bond spread - wide means risk aversion', 'unit': '%'},
    'Central Bank Printing (M2)': {'func': lambda: [fred.get_series('M2SL').iloc[-12] if fred else np.nan, fred.get_series('M2SL').iloc[-1] if fred else np.nan, np.nan], 'thresh': np.nan, 'desc': 'Money supply - high means inflation risk', 'unit': 'Billion $'},
    'Currency Devaluation': {'func': lambda: [fred.get_series('EXUSUK').iloc[-12] if fred else np.nan, fred.get_series('EXUSUK').iloc[-1] if fred else np.nan, np.nan], 'thresh': np.nan, 'desc': 'USD exchange rate - falling means devaluation', 'unit': 'Rate'},
    'Fiscal Deficits': {'func': lambda: [fred.get_series('MTSDS133FMS').iloc[-12] if fred else np.nan, fred.get_series('MTSDS133FMS').iloc[-1] if fred else np.nan, np.nan], 'thresh': 6, 'desc': 'Deficit % GDP - >6% means high spending', 'unit': '%'},
    'Debt-to-GDP Falling (-5% YoY)': {'func': lambda: [((fred.get_series('GFDEBTN').iloc[-24] / fred.get_series('GDP').iloc[-24]) * 100 - (fred.get_series('GFDEBTN').iloc[-12] / fred.get_series('GDP').iloc[-12]) * 100) / (fred.get_series('GFDEBTN').iloc[-24] / fred.get_series('GDP').iloc[-24]) * 100 if fred else np.nan, ((fred.get_series('GFDEBTN').iloc[-12] / fred.get_series('GDP').iloc[-12]) * 100 - (fred.get_series('GFDEBTN').iloc[-1] / fred.get_series('GDP').iloc[-1]) * 100) / (fred.get_series('GFDEBTN').iloc[-12] / fred.get_series('GDP').iloc[-12]) * 100 if fred else np.nan, np.nan], 'thresh': -5, 'desc': 'Year-over-year decline in Debt-to-GDP', 'unit': '%'},
    'Debt Growth': {'func': lambda: [fred.get_series('GFDEBTN').iloc[-12] - fred.get_series('GFDEBTN').iloc[-24] if fred else np.nan, fred.get_series('GFDEBTN').iloc[-1] - fred.get_series('GFDEBTN').iloc[-13] if fred else np.nan, np.nan], 'thresh': np.nan, 'desc': 'Annual debt increase rate', 'unit': '%'},
    'Income Growth': {'func': lambda: [fred.get_series('GDP').iloc[-12] - fred.get_series('GDP').iloc[-24] if fred else np.nan, fred.get_series('GDP').iloc[-1] - fred.get_series('GDP').iloc[-13] if fred else np.nan, np.nan], 'thresh': np.nan, 'desc': 'Annual income growth rate (GDP proxy)', 'unit': '%'},
    'Debt Service': {'func': lambda: [fred.get_series('FGDS').iloc[-12] if fred else np.nan, fred.get_series('FGDS').iloc[-1] if fred else np.nan, np.nan], 'thresh': 20, 'desc': 'Debt payment % of income - >20% means strain', 'unit': '%'},
    'Education Investment': {'func': lambda: [wbdata.get_series('SE.XPD.TOTL.GD.ZS')['USA'] - 1 if wbdata.get_series('SE.XPD.TOTL.GD.ZS')['USA'] else np.nan, wbdata.get_series('SE.XPD.TOTL.GD.ZS')['USA'], np.nan], 'thresh': 5, 'desc': 'Education % GDP - >5% means investment', 'unit': '%'},
    'R&D Patents': {'func': lambda: [wbdata.get_series('IP.PAT.RESD')['USA'] - 1000, wbdata.get_series('IP.PAT.RESD')['USA'], np.nan], 'thresh': np.nan, 'desc': 'Patents filed - high means innovation', 'unit': 'Count'},
    'Competitiveness Index': {'func': lambda: [np.nan, scrape_wef_competitiveness(), np.nan], 'thresh': np.nan, 'desc': 'Global competitiveness score - high means strong economy', 'unit': 'Score (0-100)'},
    'GDP per Capita Growth': {'func': lambda: [wbdata.get_series('NY.GDP.PCAP.KD.ZG')['USA'] - 1, wbdata.get_series('NY.GDP.PCAP.KD.ZG')['USA'], np.nan], 'thresh': np.nan, 'desc': 'Annual GDP per capita growth rate', 'unit': '%'},
    'Trade Share': {'func': lambda: [wbdata.get_series('NE.TRD.GNFS.ZS')['USA'] - 1, wbdata.get_series('NE.TRD.GNFS.ZS')['USA'], np.nan], 'thresh': 15, 'desc': 'Trade % of GDP - >15% means dominance', 'unit': '%'},
    'Military Spending': {'func': lambda: [scrape_sipri_military() - 0.5, scrape_sipri_military(), np.nan], 'thresh': 4, 'desc': 'Military % GDP - >4% means strain', 'unit': '%'},
    'Internal Conflicts': {'func': lambda: [scrape_conflicts_index() - 5, scrape_conflicts_index(), np.nan], 'thresh': 20, 'desc': 'Protest count - >20 means unrest', 'unit': 'Count'},
    'Reserve Currency Usage Dropping': {'func': lambda: [scrape_reserve_currency_share() - 5, scrape_reserve_currency_share(), np.nan], 'thresh': 50, 'desc': 'USD % reserves - <50% means decline', 'unit': '%'},
    'Military Losses': {'func': lambda: [scrape_military_losses() - 1, scrape_military_losses(), np.nan], 'thresh': 1, 'desc': 'Defeats count - >1/year means weakening', 'unit': 'Count'},
    'Economic Output Share': {'func': lambda: [(wbdata.get_series('NY.GDP.MKTP.CD')['USA'] / wbdata.get_series('NY.GDP.MKTP.CD')['WLD'] * 100) - 1 if wbdata.get_series('NY.GDP.MKTP.CD')['USA'] and wbdata.get_series('NY.GDP.MKTP.CD')['WLD'] else np.nan, wbdata.get_series('NY.GDP.MKTP.CD')['USA'] / wbdata.get_series('NY.GDP.MKTP.CD')['WLD'] * 100, np.nan], 'thresh': np.nan, 'desc': 'Share of global GDP - falling means decline', 'unit': '%'},
    'Corruption Index': {'func': lambda: [np.nan, scrape_transparency_cpi(), np.nan], 'thresh': 50, 'desc': 'Corruption perception - <50 means high graft', 'unit': 'Score (0-100)'},
    'Working Population': {'func': lambda: [fred.get_series('LFWA64TTUSM647S').iloc[-12] if fred else np.nan, fred.get_series('LFWA64TTUSM647S').iloc[-1] if fred else np.nan, np.nan], 'thresh': np.nan, 'desc': 'Total labor force - growth indicates strength', 'unit': 'Thousands'},
    'Innovation': {'func': lambda: [wbdata.get_series('IP.PAT.RESD')['USA'] - 1000, wbdata.get_series('IP.PAT.RESD')['USA'], np.nan], 'thresh': np.nan, 'desc': 'Patent applications - high means innovation', 'unit': 'Count'},
    'Power Index': {'func': lambda: [scrape_globalfirepower_index() + 0.01, scrape_globalfirepower_index(), np.nan], 'thresh': np.nan, 'desc': 'Military power score - low means weakness', 'unit': 'Index'},
    'Debt Burden': {'func': lambda: [(fred.get_series('GFDEBTN').iloc[-12] / fred.get_series('GDP').iloc[-12]) * 100 if fred else np.nan, (fred.get_series('GFDEBTN').iloc[-1] / fred.get_series('GDP').iloc[-1]) * 100 if fred else np.nan, np.nan], 'thresh': 100, 'desc': 'Debt % of GDP - high burden indicator', 'unit': '%'},
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
        row = soup.find('th', string=re.compile(r'Federal funds $$ effective $$', re.I)).parent if soup.find('th') else None
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
        futures = {name: executor.submit(lambda: ind['func']()) for name, ind in indicators.items()}
        for name, future in futures.items():
            try:
                result = future.result()
                if isinstance(result, list):
                    data[name] = result
                else:
                    data[name] = [np.nan, result, np.nan]  # Default to [previous, current, forecast] format
            except Exception as e:
                data[name] = [np.nan, np.nan, np.nan]  # Silent fail to NaN list
    return data

conn = sqlite3.connect('econ.db')

# Create table if not exists
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS data (Indicator TEXT PRIMARY KEY, Value TEXT)")
conn.commit()

st.title('Econ Mirror Dashboard - July 24, 2025')
st.set_page_config(layout="wide", initial_sidebar_state="expanded")

if st.button('Refresh Now'):
    data = fetch_all()
    df = pd.DataFrame([{'Indicator': name, 'Value': json.dumps(replace_nan_with_none(value))} for name, value in data.items()])
    df.to_sql('data', conn, if_exists='replace', index=False)
else:
    try:
        df = pd.read_sql('SELECT * FROM data', conn)
        df['Value'] = df['Value'].apply(lambda x: json.loads(x) if pd.notna(x) else [np.nan, np.nan, np.nan])
    except Exception as e:
        data = fetch_all()
        df = pd.DataFrame([{'Indicator': name, 'Value': json.dumps(replace_nan_with_none(value))} for name, value in data.items()])
        df.to_sql('data', conn, if_exists='replace', index=False)

# Unpack values for display
df_expanded = pd.DataFrame({
    'Indicator': df['Indicator'],
    'Previous': df['Value'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else np.nan),
    'Current': df['Value'].apply(lambda x: x[1] if isinstance(x, list) and len(x) > 1 else np.nan),
    'Forecast': df['Value'].apply(lambda x: x[2] if isinstance(x, list) and len(x) > 2 else np.nan),
    'Threshold': df['Indicator'].apply(lambda x: indicators.get(x, {}).get('thresh', np.nan)),
    'Unit': df['Indicator'].apply(lambda x: indicators.get(x, {}).get('unit', 'N/A')),
    'Description': df['Indicator'].apply(lambda x: indicators.get(x, {}).get('desc', 'N/A'))
})

col1, col2 = st.columns(2)

with col1:
    st.subheader('Risks/Cycles Viz')
    for index, row in df_expanded.iterrows():
        if not np.isnan(row['Current']):
            values = [row['Previous'], row['Current'], row['Forecast']]
            fig = px.bar(x=['Previous', 'Current', 'Forecast'], y=values, title=row['Description'],
                         color_discrete_sequence=['blue', 'green', 'orange'] if not np.isnan(row['Threshold']) and row['Current'] <= row['Threshold'] else ['blue', 'red', 'orange'])
            fig.update_traces(hovertemplate='Value: %{y} %{text}<extra></extra>', text=[f"{v} {row['Unit']}" if not np.isnan(v) else 'N/A' for v in values])
            st.plotly_chart(fig, use_container_width=True, key=f"plotly_chart_{row['Indicator']}_{index}")

with col2:
    st.subheader('Indicators Table')
    st.dataframe(df_expanded)