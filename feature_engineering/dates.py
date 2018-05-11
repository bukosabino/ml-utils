# -*- coding: utf-8 -*-

"""Util function to do Feature Engineering with dates.
"""

from datetime import datetime

def timestamptodate(timestamp):
    return datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d %H:%M:%S')

def feature_engineering_dates(df, col_timestamp):
    df['Date'] = df['Timestamp'].apply(timestamptodate)
    df['Date'] = pd.to_datetime(df['Date'])

    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.weekofyear
    df['Weekday'] = df['Date'].dt.weekday
    df['Day'] = df['Date'].dt.day
    df['Hour'] = df['Date'].dt.hour
    df['Minute'] = df['Date'].dt.minute
    df['Second'] = df['Date'].dt.second

    # extra dates
    # df["yearmonth"] = df["Date"].dt.year*100 + df["Date"].dt.month
    # df["yearweek"] = df["Date"].dt.year*100 + df["Date"].dt.weekofyear
    # df["yearweekday"] = df["Date"].dt.year*10 + df["Date"].dt.weekday

    return df
