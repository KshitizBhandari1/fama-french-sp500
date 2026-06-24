#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 08:13:36 2026

@author: kshitizbhandari

Fama-French factor replication on the S&P 500
    Done so far:
        - establishhed global setup
        - implemented data pipeline for:
            - DGS1MO
            - point-in-time S&P tickers
        - implemented function for building and saving cache files
            - prices
            - book value data
    Issues so far:
        dropping 139-140 tickers on prices (delisted/acquired)
        and 19 on fundamental section
            - introduces survivorship bias
        also only recent 3-4 years of book value and share count data available
    
    Things to do next:
        - evaluate Fama-French factors by dividing the investable universe
        - plan is to implement monthly rebalancing
        - find a way to include delisted tickers
        - also find a way to extract book value older than 4 years
            (probably need a different data source)
        (last 2 will only be implemented if an open source data source is discovered)
"""

import os
import time
import logging
import pandas as pd
import yfinance as yf
from pathlib import Path


#####################
# 0. Global Setup
#####################

# trading days
TRADING_DAYS = 252

# initialize project root as directory containing this script
PROJECT_ROOT = Path(__file__).resolve().parent

# walk up until repo marker
# assumes rep-root contains 'data/' directory
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

# suppress noisy yfinance warniings (e.g., delisted tickers)
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

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
    df = df.rename(columns = {'observation_date': 'date', 'DGS1MO': 'r_f'})
    
    #ensure values are numeric
    df['r_f'] = pd.to_numeric(df['r_f'], errors = 'coerce')
    # set date as index and sort
    df = df.set_index('date').sort_index()
    
    # forward fill missing values (holidays)
    df = df.ffill()
    
    # strip weekends: Saturdays (5) and Sundays (6)
    df = df[df.index.dayofweek < 5]
    
    # geometric compounding for a year with TRADING_DAYS
    df['r_f'] = (1 + (df['r_f']/100.0)) ** (1.0 / TRADING_DAYS) - 1.0
    
    # return risk-free rate as series
    return df['r_f']


def get_point_in_time_universe_data() -> pd.DataFrame:
    """
    Fetches the open-source Point-in-Time S&P 500 index changes tracking sheet.
    
    Inputs:
        None
    Returns:
        pd.DataFrame:
    """
    
    url = 'https://raw.githubusercontent.com/fja05680/sp500/master/S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv'
    # parse using 'date' column in the file
    df_changes = pd.read_csv(url, parse_dates = ['date'])
    
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
    outputs all stock tickers ever listed on the S&P 500 in that time window
    
    Inputs:
        df_changes (pd.DataFrame): point-in-time historical adjustment log
        start_year (int): beginning of analysis boundary
        end_year (int): end of analysis boundary
    
    Returns:
        list -> complete sorted list of unique corporate tickers
    """
    # starting at beginning of year
    start_date = pd.to_datetime(f'{start_year}-{START_DATE}')
    # ending at end of year
    end_date = pd.to_datetime(f'{end_year}-{END_DATE}')
    
    # declare empty set
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


#####################
# 2. Data Caching
#####################

def build_and_cache_data(start_year: int, end_year: int,
                         force_redownload: bool = False,
                         verbose = False):
    """
    Constructs local parquet cache files for stock prices and fundamentals
    by querying Yahoo Finance API. Throttled requests to not hit rate limits.

    Inputs:
        start_year (int): Starting year for data downloads
        end_year (int): Ending year for data download
        force_redownload (bool): toggle to overwrite existing local cache files
        verbose (bool): toggle to whether or not show progress

    Returns:
        None (saves data structures directly onto cache files)

    """
    # point-in-time S&P 500 index changes tracking log
    df_changes = get_point_in_time_universe_data()
    # relevant stocks for the time window
    all_tickers = get_all_historical_tickers(df_changes, start_year, end_year)
    
    start_str = f'{start_year}-{START_DATE}'
    end_str = f'{end_year}-{END_DATE}'
    
    ## Price caching
    if not PRICE_CACHE_PATH.exists() or force_redownload:
        if verbose:    
            print(f'Downloading historical daily prices'
                  f'for {len(all_tickers)} tickers...')
        raw_prices = yf.download(all_tickers,
                                 start = start_str,
                                 end = end_str,
                                 auto_adjust = True, progress = False
                                 )['Close']
        # drop tickers that are completely empty
        raw_prices = raw_prices.dropna(axis = 1, how = 'all')
        raw_prices.to_parquet(PRICE_CACHE_PATH)
        
        if verbose:
            # for debugging
            if isinstance(raw_prices.columns, pd.MultiIndex):
                active_price_tickers = list(raw_prices.columns.get_level_values(-1).unique())
            else:
                active_price_tickers = list(raw_prices.columns)
            dropped_price_total = len(all_tickers) - len(active_price_tickers)
            
            print(f'Total tickers before pricing section: {len(all_tickers)}')
            print(f'Tickers dropped by yfinance (pricing): {dropped_price_total}')
        
    else:
        if verbose:
            print('Price cache validated locally. Proceeding.')
    
        
    ## Fundamentals Caching
    if not FUNDAMENTALS_CACHE_PATH.exists() or force_redownload:
        if verbose:
            print('Extracting annual balance sheet values (throttled loop)...')
        
        fundamental_records = []
        skipped_tickers = []        # for debugging
        
        for i, ticker in enumerate(active_price_tickers):
            if i % 20 == 0 and i > 0:
                time.sleep(0.1)    # added delay to avoid rate limits
            
            try:
                ticker_obj = yf.Ticker(ticker)
                bs = ticker_obj.balance_sheet
                
                # variations of string names for equity and number of shares
                equity_labels = ['Stockholders Equity',
                                 'Total Stockholders Equity',
                                 'Common Stock Equity']
                shares_labels = ['Ordinary Shares Number',
                                 'Shares Outstanding',
                                 'Implied Shares Outstanding']
                
                equity_row = next((bs.loc[label] for label in equity_labels if label in bs.index), None)
                shares_row = next((bs.loc[label] for label in shares_labels if label in bs.index), None)
                
                if equity_row is not None and shares_row is not None:
                    for date_index in bs.columns:
                        fundamental_records.append({
                            'ticker': ticker,
                            'filing_date': pd.to_datetime(date_index),
                            'book_value': float(equity_row[date_index]),
                            'shares_outstanding': float(shares_row[date_index])
                            })
                else:
                    skipped_tickers.append(ticker)
                
            except Exception as e:
                skipped_tickers.append(ticker)
                # for debugging
                if verbose:    
                    print(f'Failed parsing fundamentals for {ticker}: {str(e)}')
                continue
        
        df_fund = pd.DataFrame(fundamental_records)
        df_fund.to_parquet(FUNDAMENTALS_CACHE_PATH)
        if verbose:
            print('Fundamentals downloaded and stored to local Parquet cache.\n'
                  f'Successfully parsed {len(df_fund["ticker"].unique())} stocks')
            print(f'Skipped {len(skipped_tickers)} tickers (on fundamentals portion)')
    
    else:
        if verbose:
            print('Fundamentals cache validated locally. Proceeding.')

    return None


#####################
# -1. Execution
#####################
if __name__ == '__main__':
    # testing for debugging
        build_and_cache_data(start_year = START_YEAR,
                         end_year = END_YEAR,
                         force_redownload = True,
                         verbose = True)