#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 08:13:36 2026

@author: kshitizbhandari

Fama-French factor replication on the S&P 500
    Done so far:
        - established global setup
        - implemented data pipeline for:
            - DGS1MO
            - point-in-time S&P tickers
        - implemented function for building and saving cache files
            - prices
            - book value data
        - implemented evaluation of Fama-French factors
            - by dividing the investable universe into 6 core portfolios
        - changed fundamentals data source from yfinance API to SEC EDGAR API
            - can extract data from further back unlike yfinance's 3-4 years only

    Issues so far:
        dropping:
            139-140 tickers on prices (delisted/acquired)
            77 on fundamental section
            - introduces survivorship bias
            
        -> tried using alternative datasets (like stooq) for delisted tickers
            -> still failed
        -> cannot find an open source database yet
            -> hence survivorship bias seems unavoidable without paid databases
    
    Things to do next:
        - regress proxy factors against the actual Fama-French factors
            Perfect correlation is not expected as S&P 500 is inherently a large-cap universe
        
"""

import os
import time
import logging
import requests
import numpy as np
import pandas as pd
import yfinance as yf
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
os.makedirs(CACHE_DIR, exist_ok = True)

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
        
        # clean both sides of comparision to capture dual-class variations
        # (BRK-B, AMH-PG, etc.) removing hypthens and dots
        return {
            row['ticker'].replace('-', '').replace('.','').upper():
                str(row['cik_str']).zfill(10) # enforce 10 character string to be compatible with SEC
                for row in res.values()
                }
    except Exception as e:
        print(f'Warning: Failed to compile global SEC CIK map: {e}')
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
        pd.Series - Chronologically sorted daily risk-free rates index-mapped by
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
        # localized recovery fallback if remote source is removed/moved
        df_changes = pd.read_parquet(UNIVERSE_CACHE_PATH)
    
    return df_changes.sort_values('date')


def get_constituents_on_date(target_date: pd.Timestamp, 
                             df_changes: pd.DataFrame) -> list:
    """
    Extracts the pool of active tickers present in S&P 500 on a specific
    historical trading date.
    
    Inputs:
        target_date (pd.TimeStamp): historical day for query
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
    
    # declare empty set for tickers
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
    Uses multi-tag institutional taxonomy fallbacks.

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
        
        # extract point-in-time disclosure dictionary 
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
        
        # cascade down alternative keys to capture varying accounting representations
        # for shares outstanding
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
    
    start_str = f'{start_year}-{START_DATE}'
    end_str = f'{end_year}-{END_DATE}'
    
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
        print(f'Skipped {len(skipped_tickers)} tickers (on fundamentals portion)')

    return None



#####################
# 3. Fama-French factors
#####################

def run_fama_french(start_year: int,
                    end_year: int) -> pd.DataFrame:
    """
    Executes the Fama-French 3-factor replication mechanism.
    
    Sorts the point-in-time investable universe into 2x3 value-weighted portfolios
    based on Size (Market Cap) and Value/Growth (Book-to-Market ratio).
    
    Evaluates monthly factor returns (Mkt-RF, SMB, HML) using monthly rebalancing.
    
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
        
        # construct fundamental financial metrics for all valid assets on the cross-sectional line
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
            
            # validating market cap before division
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
        
        # generate the 6 intersection portfolios (like Small-Growth, Big Growth, etc.)
        portfolios = {
            'SG': df_cs[small_mask & growth_mask],
            'SN': df_cs[small_mask & neutral_mask],
            'SV': df_cs[small_mask & value_mask],
            'BG': df_cs[big_mask & growth_mask],
            'BN': df_cs[big_mask & neutral_mask],
            'BV': df_cs[big_mask & value_mask]
            }
        
        # calculate value-weighted returns for each of 6 core portfolois
        portfolio_returns = {}
        for name, port_df in portfolios.items():
            if port_df.empty:
                portfolio_returns[name] = 0.0
                continue
            # market cap weights within this specific sub-portfolio
            weights = port_df['market_cap'] / port_df['market_cap'].sum()
            # dot product of weights and forward monthly returns
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
        
        # extract daily risk-free rates recorved over monthly investment horizon
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
# -1. Execution
#####################

if __name__ == '__main__':
    # testing for debugging
    build_and_cache_data(start_year = START_YEAR,
                     end_year = END_YEAR,
                     force_redownload = False,
                     verbose = True)
    
    print('\nBuilding proxy factors for S&P 500...')
    ff_factors_df = run_fama_french(start_year = START_YEAR,
                                    end_year = END_YEAR)
    
    print('\n=== Replicated S&P 500 FF factor proxies ===')
    print(ff_factors_df.map('{:.3%}'.format).to_string())
    
    print('\n=== Correlation matrix ===')
    print(ff_factors_df[['mkt-rf', 'smb', 'hml']].corr())

    