#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 08:13:36 2026

@author: kshitizbhandari

Fama-French 3-Factor Replication
Point-in-Time S&P 500 Universe

==================
Framework
==================

    Constructs empirical monthly-rebalanced proxy Fama-French factors
    using a point-in-time S&P 500 universe.
    
    The objective is to evaluate how closely an index-constrained,
    large-cap implementation tracks the official Fama-French factors.

==================
Pipeline:
================== 

    - Downloads historical S&P 500 constituent changes.
    - Retrieves adjusted daily prices from Yahoo Finance.
    - Retrieves book value of equity and shares outstanding from SEC EDGAR.
    - Builds local Parquet caches.
    - Forms six value-weighted portfolios using 2x3 Size-Book-to-Market sorts.
    - Computes proxy Mkt-RF, SMB, and HML factors.
    - Compares proxy factors with the official Ken French Data Library.
    - Evaluates tracking performance using closed-form OLS regression.
    - Produces cumulative return, regression, and rolling-correlation diagnostics.
    
==================
Known Limitations
==================
 
   - Open-source pricing sources do not provide complete coverage of
     historical delisted securities, introducing survivorship bias.
        Exact numbers:
            Total Tickers in the S&P 500 (2015-2025)        : 754
            Tickers unavailable in price data (yfinance)    : 139
            Tickers unavailable on fundamentals (SEC EDGAR) :  77
    - Portfolio formation is performed monthly rather than annually.
      Accordingly, this project should be interpreted as a proxy
      implementation rather than an exact replication of the original
      Fama-French methodology.
      
"""

import os
import time
import logging
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path


#####################
# 0. Global Setup
#####################

# trading days
TRADING_DAYS = 252

# Secure User-Agent identity required by SEC Edgar API
SEC_HEADERS = {'User-Agent': 'YourName email@domain.com'}

# initialize project root as directory containing this script
PROJECT_ROOT = Path(__file__).resolve().parent

# walk up until repo marker
# assumes repo-root contains 'data/' directory
while not (PROJECT_ROOT / 'data').exists():
    if PROJECT_ROOT.parent == PROJECT_ROOT:
        raise FileNotFoundError('Could not find "data/" directory.'
                                'Please ensure "data/" exists at project root'
                                'and populate it with "DGS1MO.csv from FRED')
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_DIR = PROJECT_ROOT / 'data'
CACHE_DIR = PROJECT_ROOT / 'factor-cache'
PLOTS_DIR = PROJECT_ROOT / 'plots'
os.makedirs(CACHE_DIR, exist_ok = True)
os.makedirs(PLOTS_DIR, exist_ok = True)

START_YEAR = 2015
START_DATE = '01-01'
END_YEAR = 2025
END_DATE = '12-31'

# setup paths
DGS1MO_PATH = DATA_DIR / 'DGS1MO.csv'
PRICE_CACHE_PATH = CACHE_DIR / f'sp500_prices_{START_YEAR}_{END_YEAR}.parquet'
FUNDAMENTALS_CACHE_PATH = CACHE_DIR / f'sp500_fundamentals_{START_YEAR}_{END_YEAR}.parquet'
UNIVERSE_CACHE_PATH = CACHE_DIR / f'sp500_index_changes_{START_YEAR}_{END_YEAR}.parquet'

# suppress noisy yfinance warnings (e.g., delisted tickers)
logging.getLogger('yfinance').setLevel(logging.CRITICAL)


# === Global SEC CIK map initialization ===
# previously was running a slow O(N) linear search loop for each ticker
def download_sec_cik_map() -> dict:
    """
    Downloads the SEC's complete CIK (Central Index Key) matrix
    Normalizes mixed-source ticker to remove punctuation
    Returns 10-character (pre-padded) CIK hash table.
    """
    try:
        url = 'https://www.sec.gov/files/company_tickers.json'
        res = requests.get(url, headers = SEC_HEADERS, timeout = 10).json()
        
        # clean both sides of comparison to capture dual-class variations
        # (BRK-B, AMH-PG, etc.) removing hyphens and periods
        return {
            row['ticker'].replace('-', '').replace('.','').upper():
                # enforce 10 character string to be compatible with SEC
                str(row['cik_str']).zfill(10) 
                for row in res.values()
                }
    except Exception as e:
        print(f'Warning: Failed to build global SEC CIK map: {e}')
        return {}

# save the compiled dictionary once globally (reducing run-time)
SEC_CIK_MAP = download_sec_cik_map()



#####################
# 1. Data Pipeline
#####################

def load_dgs1mo_data(csv_path = DGS1MO_PATH):
    """ 
    Loads DGS1MO (1M US Treasury Constant Maturity Rate) from a local csv,
    transforms annual yields into daily decimals, and forward-fills
    holidays.
    
    Inputs:
        csv_path (str or Path): Local system path to the FRED csv download.
    
    Returns:
        pd.Series - Chronologically sorted daily risk-free rates indexed by
        date
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f'DGS1MO.csv not found at {csv_path.resolve()}'
                                'Please source from FRED.')
    
    # read csv file with two columns 'observation_date' and 'DGS1MO'
    df = pd.read_csv(csv_path, parse_dates = ['observation_date'])
    # standardize column layout
    df = df.rename(columns = {'observation_date': 'date', 'DGS1MO': 'rf'})
    
    # ensure values are numeric
    df['rf'] = pd.to_numeric(df['rf'], errors = 'coerce')
    # set date as index and sort
    df = df.set_index('date').sort_index()
    
    # forward fill missing values (holidays)
    df = df.ffill()
    
    # strip weekends: Saturdays (5) and Sundays (6)
    df = df[df.index.dayofweek < 5]
    
    # geometric compounding for a year with TRADING_DAYS
    df['rf'] = (1 + (df['rf']/100.0)) ** (1.0 / TRADING_DAYS) - 1.0
    
    # return risk-free rate as series
    return df['rf']


def get_point_in_time_universe_data() -> pd.DataFrame:
    """
    Fetches the open-source Point-in-Time S&P 500 index changes tracking sheet.
    
    Inputs:
        None
    Returns:
        pd.DataFrame:
    """
    
    url = 'https://raw.githubusercontent.com/fja05680/sp500/master/S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv'
      
    if UNIVERSE_CACHE_PATH.exists():
        file_age_days = (time.time() - UNIVERSE_CACHE_PATH.stat().st_mtime) / 86400
        # implement 30-day window cache to bypass redundant network use
        # user fja05680 highlights he updates the file every couple of months
        if file_age_days < 30:
            return pd.read_parquet(UNIVERSE_CACHE_PATH).sort_values('date')
        
    try:
        # parse using 'date' column in the file
        df_changes = pd.read_csv(url, parse_dates = ['date'])
        df_changes.to_parquet(UNIVERSE_CACHE_PATH, index = False)
    except Exception:
        # Fallback to the locally cached file if the remote source is unavailable
        df_changes = pd.read_parquet(UNIVERSE_CACHE_PATH)
    
    return df_changes.sort_values('date')


def get_constituents_on_date(target_date: pd.Timestamp, 
                             df_changes: pd.DataFrame) -> list:
    """
    Extracts the pool of active tickers present in S&P 500 on a specific
    historical trading date.
    
    Inputs:
        target_date (pd.TimeStamp): historical day to query
        df_changes (pd.DataFrame): point-in-time historical adjustment log
        
    Returns:
        list: list of individual ticker strings active on that day
    """
    target_dt = pd.to_datetime(target_date)
    # extract historical S&P tickers up to the target date
    historical_logs = df_changes [df_changes['date'] <= target_dt]
    
    if historical_logs.empty:
        return []
    
    # isolate the most recent index structure on or before the target date
    investable_on_date = historical_logs.iloc[-1]
    
    # format for yfinance compatibility (BRK.B -> BRK-B) and strip whitespace
    investable_list = [
        ticker.strip() for ticker in investable_on_date['tickers'].replace('.','-').split(',')
        ]
    
    return investable_list


def get_all_historical_tickers(df_changes: pd.DataFrame,
                               start_year: int,
                               end_year: int) -> list:
    """
    Scans the point-in-time index histories across target date limits
    Outputs all stock tickers ever listed on the S&P 500 in that time window
    
    Inputs:
        df_changes (pd.DataFrame): point-in-time historical adjustment log
        start_year (int): beginning of analysis boundary
        end_year (int): end of analysis boundary
    
    Returns:
        list -> complete sorted list of unique corporate tickers
    """
    # note: years are inputs but month/date is globalized
    start_date = pd.to_datetime(f'{start_year}-{START_DATE}')
    end_date = pd.to_datetime(f'{end_year}-{END_DATE}')
    
    # Initialize empty set for tickers
    all_tickers = set()

    # S&P change log before the start date 
    # -> need to extract universe on first day
    prior_logs = df_changes[df_changes['date'] <= start_date]    
    if not prior_logs.empty:
        # replace '.' with '-' for yfinance compatibility and add ticker to set
        for t in prior_logs.iloc[-1]['tickers'].replace('.','-').split(','):
            all_tickers.add(t.strip())
        
    # S&P change log within the time window
    relevant_logs = df_changes[ 
        (df_changes['date'] >= start_date) & (df_changes['date'] <= end_date)
        ]
    # extract any new tickers added/present within the time window
    for tickers_string in relevant_logs['tickers']:
        for ticker in tickers_string.replace('.','-').split(','):
            all_tickers.add(ticker.strip())
    
    return sorted(list(all_tickers))


def fetch_edgar_fundamentals(ticker: str,
                             start_year: int):
    """
    Queries SEC EDGAR Company Facts API using a global CIK map lookup
    Uses multiple XBRL taxonomy fallbacks.

    Inputs:
    ticker (str): stock ticker
    start_year: first year to gather filing date
    
    Returns:
        list: a list of dictionaries containing historical book value
        and shares outstanding mapped by filing dates
    """
    records = []
    try:
        # stripping punctuations to match with global map
        clean_target = ticker.replace('-', '').replace('.', '').upper()
        # load CIK value from global map
        cik = SEC_CIK_MAP.get(clean_target)
        
        if not cik:
            # empty list
            return records
        
        # extract point-in-time company facts data
        # contains complete structured XBRL corporate history
        facts_url = f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json'
        facts = requests.get(facts_url, headers = SEC_HEADERS, timeout = 10).json()
        us_gaap = facts.get('facts', {}).get('us-gaap', {})
        
        # accounting taxonomy keys sourced from open-source SEC EDGAR documentation
        # targets standard corporate, MLP, and trust balance sheet representation
        equity_keys = [
            'StockholdersEquity', 
            'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest', 
            'CommonStockSharesOutstandingWithAdditionalPaidInCapital',
            'PartnersCapital'
        ]
        equity_units = None
        
        for k in equity_keys:
            if k in us_gaap:
                equity_units = us_gaap[k].get('units', {}).get('USD', [])
                # break when USD unit is found for book value
                if equity_units:
                    break
        
        # Iterate through alternative taxonomy keys to capture
        # varying accounting representations for shares outstanding
        shares_keys = [
            'CommonStockSharesOutstanding', 
            'EntityCommonStockSharesOutstanding',
            'WeightedAverageNumberOfSharesOutstandingBasic',
            'CommonStockSharesIssued'
            ]
        shares_units = None
        
        for k in shares_keys:
            if k in us_gaap:
                shares_units = us_gaap[k].get('units', {}).get('shares', [])
                # break loop when shares unit is found for number of shares
                if shares_units:
                    break
        
        if not equity_units or not shares_units:
            # empty list
            return records
        
        df_equity = pd.DataFrame(equity_units)
        df_shares = pd.DataFrame(shares_units)
        
        # filter exclusively for audited 10-K records
        # guarantees structural consistency and drops messy 10-Q noise
        if 'form' in df_equity.columns:
            df_equity = df_equity[df_equity['form'] == '10-K']
            df_shares = df_shares[df_shares['form'] == '10-K']
        
        if df_equity.empty or df_shares.empty:
            return records
        
        df_equity['fy_year'] = df_equity['fy'].astype(str)
        df_shares['fy_year'] = df_shares['fy'].astype(str)

        # merge disclosures on shared fiscal year metrics
        merged = pd.merge(df_equity, df_shares, on = 'fy_year', suffixes = ('_eq', '_sh'))
        # filter out filings since the prior year of analysis start date
        merged = merged[merged['fy_year'].astype(float) >= (start_year - 1)]
        
        for _, row in merged.iterrows():
            records.append({
                'ticker': ticker,
                'filing_date': pd.to_datetime(row['filed_eq']),
                'book_value': float(row['val_eq']),
                'shares_outstanding': float(row['val_sh'])
                })
            
        
    except Exception:
        pass
    return records



#####################
# 2. Data Caching
#####################

def build_and_cache_data(start_year: int, end_year: int,
                         force_redownload: bool = False,
                         verbose = False):
    """
    Constructs local parquet cache files for
    - stock prices by querying Yahoo Finance API.
    - fundamentals by querying SEC EDGAR Company Facts API
    
    Uses throttled requests to not hit rate limits.

    Inputs:
        start_year (int): Starting year for data downloads
        end_year (int): Ending year for data download
        force_redownload (bool): toggle to overwrite existing local cache files
        verbose (bool): toggle to whether or not to show progress

    Returns:
        None (saves data structures directly onto cache files)
    """
    # point-in-time S&P 500 index changes tracking log
    df_changes = get_point_in_time_universe_data()
    # all relevant stocks for the time window
    all_tickers = get_all_historical_tickers(df_changes, start_year, end_year)
    
    # changed to one month prior to generate the January proxy factor
    start_str = (
        pd.Timestamp(f'{start_year}-{START_DATE}')
        - pd.DateOffset(months = 1)
        ).strftime('%Y-%m-%d')
    
    
    # yfinance was treating end date as exclusive (dropping 12-31 previously)
    end_str = (
        pd.Timestamp(f'{end_year}-{END_DATE}')
        + pd.Timedelta(days = 1)
        ).strftime('%Y-%m-%d')
    
    ## === Price caching ===
    if not PRICE_CACHE_PATH.exists() or force_redownload:
        if verbose:    
            print('Downloading historical daily prices for '
                  f'{len(all_tickers)} tickers...')
        
        raw_prices = yf.download(all_tickers,
                                 start = start_str,
                                 end = end_str,
                                 auto_adjust = True,
                                 progress = False
                                 )['Close']
        # drop tickers that are completely empty and save to cache
        raw_prices = raw_prices.dropna(axis = 1, how = 'all')
        raw_prices.to_parquet(PRICE_CACHE_PATH)
        
    else:
        if verbose:
            print('Price cache validated locally. Proceeding.\n')
        raw_prices = pd.read_parquet(PRICE_CACHE_PATH)
    
    # Extract tickers that actually have historical pricing data
    # saves time on SEC queries - no point in querying stocks with no price data
    if isinstance(raw_prices.columns, pd.MultiIndex):
        active_price_tickers = list(raw_prices.columns.get_level_values(-1).unique())
    else:
        active_price_tickers = list(raw_prices.columns)
        
    if verbose:
        # for debugging
        dropped_price_total = len(all_tickers) - len(active_price_tickers) 
        print(f'Total tickers before pricing section: {len(all_tickers)}')
        print(f'Tickers dropped by yfinance (pricing): {dropped_price_total}')
    
    ## === Fundamentals Caching ===
    if not FUNDAMENTALS_CACHE_PATH.exists() or force_redownload:
        if verbose:
            print('Extracting annual balance sheet values (throttled loop)...')
        
        fundamental_records = []
        
        # replaced fundamentals API scraping from yfinance to EDGAR API 
        for i, ticker in enumerate(active_price_tickers):
            # throttle requests to honor SEC compliance rate thresholds
            if i % 10 == 0 and i > 0:
                time.sleep(0.11)
                
            edgar_records = fetch_edgar_fundamentals(ticker, start_year)
            if edgar_records:
                # extend the records list
                # fetch_edgar_fundamentals() returns a list of dictionary
                fundamental_records.extend(edgar_records)
        
        df_fund = pd.DataFrame(fundamental_records)
        df_fund.to_parquet(FUNDAMENTALS_CACHE_PATH)
     
        if verbose:
            print('Fundamentals downloaded and stored to local Parquet cache.\n')
    
    else:
        if verbose:
            print('Fundamentals cache validated locally. Proceeding.\n')
            df_fund = pd.read_parquet(FUNDAMENTALS_CACHE_PATH)
    
    if verbose:
        successful_fundamental_tickers = set(df_fund['ticker'].unique())
        skipped_tickers = [t for t in active_price_tickers if t not in successful_fundamental_tickers]
        
        print(f'Recovered historical metrics via SEC EDGAR for {len(successful_fundamental_tickers)} stocks')
        print(f'Skipped tickers on fundamentals portion: {len(skipped_tickers)}')

    return None



#####################
# 3. Fama-French Proxy factors
#####################

def compute_ff_proxy_factors(start_year: int,
                             end_year: int) -> pd.DataFrame:
    """
    Executes the Fama-French 3-factor replication mechanism
    but with monthly rebalancing.
    
    Sorts the point-in-time investable universe into 2x3 value-weighted portfolios
    based on Size (Market Cap) and Value/Growth (Book-to-Market ratio).
    
    Computes monthly factor returns (Mkt-RF, SMB, HML) using monthly rebalancing.
    
    Inputs:
        start_year (int): starting year for the factor analysis window
        end_year (int): ending year for the factor analysis window
        
    Returns:
        pd.DataFrame: chronologically indexed DataFrame by last trading day
        of the month consisting of columns:
            - mkt-rf
            - smb
            - hml
            - rf_monthly
    """
    # load foundational data structures and cache
    df_changes = get_point_in_time_universe_data()    
    price_matrix = pd.read_parquet(PRICE_CACHE_PATH)
    df_fundamentals = pd.read_parquet(FUNDAMENTALS_CACHE_PATH)
    rf_series = load_dgs1mo_data()
    
    # all active trading days
    df_dates = pd.Series(price_matrix.index, index = price_matrix.index)
    # extract actual last trading day for each month
    month_end_dates = df_dates.resample('ME').max().values
    
    factor_results = []
    
    # iterate through rebalancing dates (month-end)
    # len - 1 bound -> final available date is ultimate liquidation
    # no new position can be formed at the last trading day
    for month_index in range(len(month_end_dates) - 1):
        rebalance_date = month_end_dates[month_index]
        next_month_end = month_end_dates[month_index + 1]
        
        # investable universe active in the index on formation day
        active_tickers = get_constituents_on_date(target_date = rebalance_date,
                                                  df_changes = df_changes)
        # active tickers with actual prices available
        valid_tickers = [ticker for ticker in active_tickers if ticker in price_matrix.columns]
        
        if not valid_tickers:
            logging.warning(f'No valid tickers to invest on {rebalance_date}')
            continue
        
        # price trajectory for the upcoming month
        # (starting from last trading day close this month) 
        month_prices = price_matrix.loc[rebalance_date:next_month_end, valid_tickers]
        if len(month_prices) < 2:
            continue
        
        # Return calculation
        # forward filled -> in case of delisting/acquisition (ensures 0% return post-event)
        month_returns = month_prices.pct_change().ffill().dropna(how = 'all')
        # cumulative 
        cum_month_returns = (1 + month_returns).prod() - 1
        
        cross_section_data = []
        
        # construct cross-sectional firm characteristics
        for ticker in valid_tickers:
            # closing price on rebalancing day to calculate size and ratios
            close_price = price_matrix.loc[rebalance_date, ticker]
            if pd.isna(close_price) or close_price <= 0:
                continue
            
            # filter fundamental filings published strictly on or before rebalance date
            # mimics real-world information availability and eliminates look-ahead bias
            ticker_fundamentals = df_fundamentals[
                (df_fundamentals['ticker'] == ticker) &
                (df_fundamentals['filing_date'] <= rebalance_date)
                ]
            
            if ticker_fundamentals.empty:
                continue
            
            # isolate most recent financial statement available on this date
            latest_filing = ticker_fundamentals.sort_values('filing_date').iloc[-1]
            
            # calculate market cap using active price and latest known share count
            market_cap = close_price * latest_filing['shares_outstanding']
            
            # validate market capitalization before division
            # was getting divide by zero errors in the check
            if pd.isna(market_cap) or market_cap <= 0:
                continue
            
            # book-to-market ratio
            # high value -> value stocks, low value -> growth stocks
            bm_ratio = latest_filing['book_value'] / market_cap
            
            # cumulative forward return for this ticker
            forward_return = cum_month_returns.get(ticker, np.nan)
            
            # skip tickers with missing structural metrics to prevent NaNs
            if pd.isna(forward_return) or pd.isna(bm_ratio):
                continue
            
            cross_section_data.append({
                'ticker': ticker,
                'market_cap': market_cap,
                'bm_ratio': bm_ratio,
                'forward_return': forward_return
                })
            
        df_cs = pd.DataFrame(cross_section_data)
        
        if df_cs.empty:
            continue
        
        # === 2x3 Cross-Categorized Sort Matrices ===
        
        # size divided by cross-sectional median market cap
        # Small portfolio elements < median, Big portfolio elements > median
        size_median = df_cs['market_cap'].median()
        
        # break points for HML portfolios:
        # top 30% (Value), middle 40% (Neutral), bottom 40% (Growth)
        bm_30 = df_cs['bm_ratio'].quantile(0.30)
        bm_70 = df_cs['bm_ratio'].quantile(0.70)
        
        # segment universe into Size and Value/Growth boolean masks
        small_mask = df_cs['market_cap'] <= size_median
        big_mask = df_cs['market_cap'] > size_median
        
        growth_mask = df_cs['bm_ratio'] <= bm_30
        neutral_mask = (df_cs['bm_ratio'] > bm_30) & (df_cs['bm_ratio'] < bm_70)
        value_mask = df_cs['bm_ratio'] >= bm_70
        
        # construct the 6 Size-Book-to-Market portfolios
        # (like Small-Growth, Big Growth, etc.)
        portfolios = {
            'SG': df_cs[small_mask & growth_mask],
            'SN': df_cs[small_mask & neutral_mask],
            'SV': df_cs[small_mask & value_mask],
            'BG': df_cs[big_mask & growth_mask],
            'BN': df_cs[big_mask & neutral_mask],
            'BV': df_cs[big_mask & value_mask]
            }
        
        # calculate value-weighted returns for each of the six portfolios
        portfolio_returns = {}
        for name, port_df in portfolios.items():
            if port_df.empty:
                portfolio_returns[name] = 0.0
                continue
            # market cap weights within this specific sub-portfolio
            weights = port_df['market_cap'] / port_df['market_cap'].sum()
            # value-weighted portfolio return
            portfolio_returns[name] = np.dot(port_df['forward_return'], weights)
        
        # === Factor equations ===
        
        # SMB (Small Minus Big) -> size premium
        # avg return of 3 small portfolios - avg return of 3 big portfolios
        smb = ((portfolio_returns['SG'] + portfolio_returns['SN'] + portfolio_returns['SV']) / 3.0) - \
              ((portfolio_returns['BG'] + portfolio_returns['BN'] + portfolio_returns['BV']) / 3.0)
        
        # HML (High Minus Low) -> value premium
        hml = ((portfolio_returns['SV'] + portfolio_returns['BV']) / 2.0) - \
              ((portfolio_returns['SG'] + portfolio_returns['BG']) / 2.0)
    
        # value-weighted broad market return across investable universe
        total_market_cap = df_cs['market_cap'].sum()
        market_weights = df_cs['market_cap'] / total_market_cap
        market_raw_return = np.dot(df_cs['forward_return'], market_weights)
        
        # extract daily risk-free rates recorded over monthly investment horizon
        rf_slice = rf_series.loc[rebalance_date:next_month_end]
        # the 1-month risk-free rate -> used to subtract from market return to get the market risk premium factor
        rf_monthly = 0.0 if rf_slice.empty else float((1 + rf_slice).prod() - 1)
        
        # Market Excess Return (Mkt - RF)
        market_excess = market_raw_return - rf_monthly
        
        factor_results.append({
            'month_end' : next_month_end,
            'mkt-rf': market_excess,
            'smb': smb,
            'hml': hml,
            'rf_monthly': rf_monthly
            })
        
    return pd.DataFrame(factor_results).set_index('month_end')
    


#####################
# 4. Fama-French Actual Factors
#####################

def fetch_official_ff_factors(start_year: int,
                              end_year: int) -> pd.DataFrame:
    """
    Downloads the official Fama-French Research Factors dataset.
    
    Drops annual observations, removes nonmonthly rows and metadata lines from dataset,
    and standardizes the operational dataset into a monthly PeriodIndex ('M')
    
    Inputs:
        start_year (int): starting year for the factor extraction window
        end_year (int): ending year for the factor extraction window

    Returns:
        pd.DataFrame: Chronologically sorted DataFrame containing the official monthly FF factors
    """
    import zipfile
    import io
    
    # official factors
    url = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip'
    
    # download and extract zip file
    response = requests.get(url, timeout = 10)
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        csv_filename = z.namelist()[0]
        with z.open(csv_filename) as f:
            # skip the first 3 rows (info about CRSP and 1-month T bill)
            df = pd.read_csv(f, skiprows = 3 )
    
    # rename 'Unnamed: 0' column to date
    df = df.rename(columns = {df.columns[0]: 'date'})
    
    # drop rows with empty dates 
    df = df.dropna(subset = ['date'])
    # convert to string and strip any whitespace for length checks
    df['date'] = df['date'].astype(str).str.strip()

    # Isolate monthly rows (eliminate annual rows and copyright lines)
        # Monthly rows have length 6 (e.g., '202401').
        # Annual rows have length 4 (e.g., '2024').
    df = df[ df['date'].str.len() == 6]
    
    # in case any header/notices passed str length == 6 test
    # keeping only numeric values for dates
    df = df[ df['date'].str.isnumeric() ]
    
    # parse YYYYMM strings into monthly PeriodIndex
    df['date'] = (
        pd.to_datetime(df['date'], format='%Y%m').dt.to_period('M')
        )
    df = df.set_index('date')
    
    # change the FF percentage values to decimals
    df = df.astype(float) / 100.0
    
    # filter to the operational time window
    start_dt = pd.Period(f"{start_year}-01", freq = 'M')
    end_dt = pd.Period(f"{end_year}-12", freq = 'M')
    
    return df.loc[start_dt:end_dt]


#####################
# 5. Helper functions
#####################

def align_to_monthly_period(df1: pd.DataFrame | pd.Series,
                            df2: pd.DataFrame | pd.Series,
                            how: str = 'inner') -> pd.DataFrame:
    """
    Standardizes two DataFrames or Series to a monthly PeriodIndex ('M')
    and merges them
    Accepts daily or monthly time series, strips duplicate intermediate entries,
    preserving only the last observation within each month, and
    joins on the combined index
    
    Input:
        df1 (pd.DataFrame): first financial dataset for alignment
        df2 (pd.DataFrame): second financial dataset for alignment
        how (str): structural join style passed to Pandas (e.g., 'inner', 'outer', 'left')
    
    Returns:
        pd.DataFrame -> a unified DataFrame indexed by a clean monthly PeriodIndex
    """
    # convert to DataFrame if Series, otherwise make a copy
    d1 = df1.to_frame() if isinstance(df1, pd.Series) else df1.copy()
    d2 = df2.to_frame() if isinstance(df2, pd.Series) else df2.copy()
    
    # standardize to monthly PeriodIndex
    d1.index = pd.to_datetime(d1.index.astype(str)).to_period('M')
    d2.index = pd.to_datetime(d2.index.astype(str)).to_period('M')
    
    # drop duplicate months if daily data was passed (by keeping last trading day)
    d1 = d1.loc[~d1.index.duplicated(keep = 'last')]
    d2 = d2.loc[~d2.index.duplicated(keep = 'last')]
    
    return d1.join(d2, how = how)



def calculate_ols_metrics(Y: np.ndarray | pd.Series,
                          X: np.ndarray | pd.Series):
    """
    Computes ordinary least squares (OLS) regression diagnostics between
    two aligned financial time series via closed-form matrix solution.
    
    Eliminates non-overlapping observation intervals, fits the linear model
            Y = alpha + beta * X + epsilon, and 
    estimates statistical parameters and associated uncertainty.

    Inputs:
        Y (np.ndarray | pd.Series): dependent variable (e.g., proxy factor returns)
        X (np.ndarray | pd.Series): independent variable (e.g., benchmark factor returns)
    
    Returns:
        dict: Diagnostic parameters containing:
            - 'correlation' (float): Pearson correlation coefficient
            - 'beta' (float): Estimated factor loading coefficient
            - 'beta_se' (float): Standard error of the beta coefficient
            - 'alpha' (float): Estimated intercept (idiosyncratic return)
            - 'alpha_se' (float): Standard error of the alpha coefficient
            - 'r_squared' (float): Coefficient of determination (goodness-of-fit)
            - 'rse' (float): Residual Standard Error of the regression
            - 'observations' (int): Total overlapping sample count (N)

    """
    # check for raw empty inputs
    if len(Y) == 0 or len(X) == 0:
        raise ValueError('Input vectors Y and X must not be empty.')
    
    # align both input vectors into a localized DataFrame to eliminate mutual missing index
    aligned_data = pd.DataFrame({'Y': Y, 'X': X}).dropna()
    
    Y_clean = aligned_data['Y'].values
    X_clean = aligned_data['X'].values
    
    N = len(Y_clean)
    
    if N == 0:
        raise ValueError('Vector alignment failed. Y and X have no overlapping indices.')
    if N <= 2:
        return {'error': 'Insufficient synchronized observations to calculate regression metrics'}
    
    # stacking a column of ones with independent variable
    # y_i = alpha + beta * x_i
    # => Y = [1  X][alpha beta]
    # Nx2 matrix
    design_matrix = np.column_stack([np.ones(N), X_clean])
    
    # To compute closed-form OLS parameters
        # closed-form OLS solution calculated by:
            # setting derivative with respect to beta = 0 (minimization for e^T e)
    # [alpha beta] = [X^T * X)^(-1) * (X^T * Y)
    XtX = design_matrix.T @ design_matrix
    XtY = design_matrix.T @ Y_clean
    
    try:
        # solved parameters 
        beta_hat = np.linalg.inv(XtX) @ XtY
    except np.linalg.LinAlgError:
        return {'error': 'Matrix inversion failed. Check variable X for zero variance (constant values)'}
    
    alpha = beta_hat[0]
    beta = beta_hat[1]
    
    # generate predicted values and extract residual tracking errors
    Y_hat = design_matrix @ beta_hat
    residuals = Y_clean - Y_hat
    
    # core sum of square metrics for variance division
    ss_residual = np.sum(residuals ** 2)
    ss_total = np.sum( (Y_clean - np.mean(Y_clean)) ** 2)
    
    # r^2 and baseline risk metrics
    r_squared = 1.0 - (ss_residual / ss_total)
    rse = np.sqrt(ss_residual / (N - 2))
    
    # Parameter Variance-Covariance matrix to extract parameter margins of error
    # RSE^2 * (X^T * X)^(-1)
    var_cov_matrix = (rse ** 2) * np.linalg.inv(XtX)
    
    alpha_se = np.sqrt(var_cov_matrix[0,0])
    beta_se = np.sqrt(var_cov_matrix[1, 1])
    
    correlation = np.corrcoef(X_clean, Y_clean)[0, 1]
    
    return {
        'correlation': correlation,
        'beta': beta,
        'beta_se': beta_se,
        'alpha': alpha,
        'alpha_se': alpha_se,
        'r_squared': r_squared,
        'rse': rse,
        'observations': N
        }


#####################
# 6. Visualization
#####################
    
def plot_factor_replication_summary(aligned_factors: pd.DataFrame):
    """
    Generates a 3-panel visual diagnostic dashboard comparing
    the empirical S&P 500 proxy factors against the official FF3 benchmarks.
    
    Construct:
        cumulative return trajectories for cumulative wealth growth ($1 initial),
        OLS scatter plots with exact trendline projections, and
        rolling 12-month correlation paths to evaluate structural stability.
    
    Input:
        aligned_factors (pd.DataFrame): Combined time series indexed monthly
        PeriodIndex ('M') containing matched proxy and benchmark factor streams.
        -> output from align_to_monthly_period() function.
    
    Returns:
        None (Saves a 300-DPI diagnostic layout to PLOTS_DIR)
    """
    # convert the already-aligned PeriodIndex to timestamps
    df_plot = aligned_factors.copy()
    if isinstance(df_plot.index, pd.PeriodIndex):
        df_plot.index = df_plot.index.to_timestamp(how='end')
    else:
        df_plot.index = pd.to_datetime(df_plot.index)
    
    # map column pairs (Proxy Column, Official Column, Descriptive Title)
    factor_mapping = [
        ('mkt-rf', 'Mkt-RF', 'Market Excess Return (Mkt-RF)'),
        ('smb', 'SMB', 'Size Premium (SMB)'),
        ('hml', 'HML', 'Value Premium (HML)')
    ]
    
    plt.style.use('ggplot')
    # Set figure and axes backgrounds to white
    plt.rcParams['axes.facecolor'] = 'white'      # Interior plot background to white
    plt.rcParams['axes.edgecolor'] = '#cbcbcb'    # Light gray bounding box around axes
    plt.rcParams['figure.facecolor'] = 'white'    # Exterior canvas background to white
    
    # 3x3 plots -> allocate additional width to cumulative return plots
    fig, axes = plt.subplots(3, 3, figsize=(18, 14), gridspec_kw={'width_ratios': [2, 1, 1.2]})
    fig.suptitle('Fama-French 3-Factor Replication Diagnostics'
                 '\nUniverse: S&P 500 PIT vs. Official CRSP Benchmark', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # p_col -> proxy column, o_col -> official column, title -> display title
    for i, (p_col, o_col, title) in enumerate(factor_mapping):
        p_series = df_plot[p_col]
        o_series = df_plot[o_col]
        
        # --- COLUMN 1: Cumulative Growth of $1 ---
        # cumulative return column: ith row, first column (0)
        ax_cum = axes[i, 0]
        cum_proxy = (1 + p_series).cumprod()
        cum_official = (1 + o_series).cumprod()
        
        ax_cum.plot(cum_proxy.index, cum_proxy,
                    label = 'Proxy (S&P 500)',
                    color = '#1f77b4', linewidth = 2)
        ax_cum.plot(cum_official.index, cum_official,
                    label = 'Official (FF)',
                    color = '#ff7f0e', linestyle = '--', linewidth = 1.8)
        
        ax_cum.set_title(f'Cumulative Growth of $1: {title}',
                         fontsize=11, fontweight='bold')
        ax_cum.set_ylabel('Portfolio Value ($)')
        
        if i == 0:
            ax_cum.legend(loc='upper left', frameon=True)
            
        # --- COLUMN 2: OLS Scatter & Custom Regression Line ---
        # regression column: ith row, second column (1)
        ax_reg = axes[i, 1]
        ax_reg.scatter(o_series, p_series, alpha = 0.5,
                       color='#2ca02c', edgecolors = 'none', s = 25)
        
        # regression using custom function
        metrics = calculate_ols_metrics(Y = p_series, X = o_series)
        # estimated regression coefficients
        alpha_val = metrics['alpha']
        beta_val = metrics['beta']
        
        # Regression line coordinates using extracted parameters
        # (y = alpha + beta * x)
        x_vals = np.array([o_series.min(), o_series.max()])
        y_vals = alpha_val + beta_val * x_vals  
        
        ax_reg.plot(x_vals, y_vals,
                    color = 'red', linestyle = '-', linewidth = 1.5, 
                    label = f'$\\beta$: {beta_val:.2f}\n$\\alpha$: {alpha_val:.2%}')
        ax_reg.axhline(0, color='black', linewidth=0.5, linestyle=':')
        ax_reg.axvline(0, color='black', linewidth=0.5, linestyle=':')
        ax_reg.set_title(f'OLS Fit: {o_col}', fontsize=11, fontweight='bold')
        ax_reg.set_xlabel('Official Factor Returns')
        ax_reg.set_ylabel('Proxy Factor Returns')
        ax_reg.legend(loc = 'lower right', frameon = True, handlelength = 0)

        # --- COLUMN 3: Rolling 12-Month Correlation ---
        # rolling correlation column: ith row, third column (2)
        ax_corr = axes[i, 2]
        rolling_corr = p_series.rolling(window = 12).corr(o_series)
        
        ax_corr.plot(rolling_corr.index, rolling_corr,
                     color = '#9467bd', linewidth = 1.5)
        # average rolling correlation marked as horizontal line
        ax_corr.axhline(rolling_corr.mean(), color = 'red', linestyle = ':',
                        label = f'Mean: {rolling_corr.mean():.2f}')
        ax_corr.set_title(f'Rolling 12M Correlation: {o_col}',
                          fontsize = 11, fontweight = 'bold')
        ax_corr.set_ylim(-0.2, 1.05)
        ax_corr.set_ylabel('Correlation Coefficient')
        ax_corr.legend(loc='lower left', frameon = True)

    # format timeline axes ticks uniformly - to prevent date crowding
    for ax in axes[:, 0].flatten():
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        
    for ax in axes[:, 2].flatten():
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))

    plt.tight_layout()
    
    # Save output image alongside your local cache directory structure
    try:
        output_path = PLOTS_DIR / 'factor_replication_diagnostics.png'
        plt.savefig(output_path, dpi = 300, bbox_inches = 'tight')
        print(f"\n[Success] Integrated OLS dashboard saved to: {output_path.resolve()}")
    except NameError:
        # Fallback if PLOTS_DIR path doesn't work
        plt.savefig('factor_replication_diagnostics.png', dpi = 300, bbox_inches = 'tight')
        print("\n[Success] Integrated OLS dashboard saved to working directory.")
        
    plt.show()

#####################
# -1. Execution
#####################

if __name__ == '__main__':
    print('='*61)
    print('Custom Fama-French 3-factor Replication')
    print('Universe: Point-in-Time S&P 500 | Monthly Rebalancing')
    print('='*61)
    
    print('\nPreparing local data cache...')
    build_and_cache_data(start_year = START_YEAR,
                         end_year = END_YEAR,
                         verbose = True)
    
    print('\nComputing proxy Fama-French factors...')
    proxy = compute_ff_proxy_factors(start_year = START_YEAR,
                                     end_year = END_YEAR)
    print('\nDownloading official Fama-French factors...')
    official = fetch_official_ff_factors(start_year = START_YEAR,
                                         end_year = END_YEAR)
    
    print('Aligning monthly factor series...')
    factors = align_to_monthly_period(proxy, official)
    print(f'Aligned factor observations: {len(factors)} months')
    
    print('\n' + '-'*50)
    print('Custom Factor Replication Results')
    print('-'*50)
    
    comparisons = [
        ('Market Excess Return', 'mkt-rf', 'Mkt-RF'),
        ('SMB', 'smb', 'SMB'),
        ('HML', 'hml', 'HML'),
        ]
    
    for name, proxy_col, official_col in comparisons:
        results = calculate_ols_metrics(
            Y = factors[proxy_col],
            X = factors[official_col]
            )
        
        print(f'\n{name}')
        print('-'*33)
        print(f'Correlation : {results["correlation"]:.4f}')
        print(f'R²          : {results["r_squared"]:.2%}')
        print(f'Beta        : {results["beta"]:.3f} ± {results["beta_se"]:.3f}')
        print(f'Alpha       : {results["alpha"]:.3%} ± {results["alpha_se"]:.3%}')
        print(f'RSE         : {results["rse"]:.2%}')
        print(f'Observations: {results["observations"]}')
        
    print('\nGenerating visual factor diagnostic dashboards...')
    plot_factor_replication_summary(aligned_factors = factors)
        
    print('\nProgram Terminated')