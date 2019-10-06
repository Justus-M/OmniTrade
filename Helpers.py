from itertools import islice
import os
import csv
import pandas as pd
import io
from datetime import datetime, timedelta


def reversed_lines(file):
    "Generate the lines of file in reverse order."
    part = ''
    for block in reversed_blocks(file):
        for c in reversed(block):
            if c == '\n' and part:
                yield part[::-1]
                part = ''
            part += c
    if part: yield part[::-1]


def reversed_blocks(file, blocksize=4096):
    "Generate blocks of file's contents in reverse order."
    file.seek(0, os.SEEK_END)
    here = file.tell()
    while 0 < here:
        delta = min(blocksize, here)
        here -= delta
        file.seek(here, os.SEEK_SET)
        yield file.read(delta)

def CsvEndReader(path, lines, Processed = False):
    out = []
    with open(path) as file:
        for line in islice(reversed_lines(file), lines):
            reader = csv.reader(io.StringIO(line), delimiter=',')
            out.append(next(reader))
    if Processed:
        out = pd.DataFrame(out).set_index(0)
    else:
        out = pd.DataFrame(out, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']).set_index('timestamp')

    return out[::-1]

def BuildTickerList(p):

    tickers = []
    Exclude = list(pd.read_csv('SyncLog.csv')['NoOverlap'])

    for ticker in os.listdir('/Data/Minute'):
        if 'csv' in ticker and ticker.replace('.csv', '') not in Exclude:
            tickers.append(ticker.replace('.csv', ''))

    startdate = []
    for ticker in tickers:
        row1 = pd.read_csv('/Data/Minute/%s.csv' % ticker, nrows=1)
        startdate.append([ticker, row1['timestamp'][0]])

    startdate = pd.DataFrame(startdate, columns=['ticker', 'startdate']).set_index('ticker')

    p['tickers'] = []
    lateststart = pd.to_datetime(datetime(2009, 1, 1))
    earliestend = datetime.today() - timedelta(days=7)
    for ticker in tickers:

        tickerstart = pd.to_datetime(startdate['startdate'].loc[ticker])
        tickerend = pd.to_datetime(CsvEndReader('/Data/Minute/%s.csv' % (ticker), 1).index.values[0])

        if tickerstart < lateststart and tickerend > earliestend:
            p['tickers'].append(ticker)

    return p

