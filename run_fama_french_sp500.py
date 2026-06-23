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
    Things to do next:
        - implement functions to cache prices and book value data
        - evaluate Fama-French factors by dividing the investable universe
        - plan is to implement monthly rebalancing
"""

import os
import pandas as pd
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
END_YEAR = 2025

# setup paths
DGS1MO_PATH = DATA_DIR / 'DGS1MO.csv'
PRICE_CACHE_PATH = CACHE_DIR / f'sp500_prices_{START_YEAR}_{END_YEAR}.parquet'
FUNDAMENTALS_CACHE_PATH = CACHE_DIR / f'sp500_fundamentals_{START_YEAR}_{END_YEAR}.parquet'


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
        pd.DataFrmae:
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
    start_date = pd.to_datetime(f'{start_year}-01-01')
    # ending at end of year
    end_date = pd.to_datetime(f'{end_year}-12-31')
    
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



    