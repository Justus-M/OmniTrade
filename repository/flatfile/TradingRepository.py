import pandas as pd
import os
import csv
import time
from random import random, getrandbits


class TradingRepository:
    def __init__(self):
        self.data_directory = r"/Users/philippwolff/Projects/OmniTrade/data"
        self.balance_directory = os.path.join(self.data_directory, "balance")
        self.balance_path = os.path.join(self.balance_directory, "balance.csv")
        if not os.path.isdir(self.balance_directory):
            os.mkdir(self.balance_directory)

        if not os.path.isfile(self.balance_path):
            with open(self.balance_path, 'a') as balance_file:
                wr = csv.writer(balance_file, delimiter =';')
                wr.writerow(['botId', 'ticker', 'quantity'])

        self.transaction_directory = os.path.join(self.data_directory, "transaction")
        self.transaction_path = os.path.join(self.transaction_directory, "transaction.csv")
        if not os.path.isdir(self.transaction_directory):
            os.mkdir(self.transaction_directory)

        if not os.path.isfile(self.transaction_path):
            with open(self.transaction_path, 'a') as transaction_file:
                wr = csv.writer(transaction_file, delimiter =';')
                wr.writerow(['botId', 'ticker', 'quantity', 'transaction_type', 'price', 'timestamp'])

    def get_balance(self, ticker, bot_id):
        df = pd.read_csv(self.balance_path, header=0, sep=';')
        result = df.loc[(df['ticker'] == ticker) & (df['botId'] == bot_id)]
        if len(result) == 0:
            return 0
        else:
            return result['quantity'].values[0]

    def make_buy_order(self, ticker, quantity, price, bot_id):
        df = pd.read_csv(self.balance_path, header=0, sep=';')
        result = df.loc[(df['ticker'] == ticker) & (df['botId'] == bot_id)]
        if len(result) == 0:
            new_quantity = quantity
            df.loc[df.index.max() + 1] = [bot_id, ticker, new_quantity]
        else:
            new_quantity = result.iloc[0]['quantity'] + quantity
            df.loc[(df['ticker'] == ticker) & (df['botId'] == bot_id)] = [ticker, new_quantity]
        df.to_csv(self.balance_path, sep=";", index=False)
        self.add_transaction(ticker, quantity, "buy", price, bot_id)

    def make_sell_order(self, ticker, quantity, price, bot_id):
        df = pd.read_csv(self.balance_path, header=0, sep=';')
        current_quantity = df.loc[(df['ticker'] == ticker) & (df['botId'] == bot_id), 'quantity'].tolist()[0]
        new_quantity = current_quantity - quantity
        if new_quantity == 0:
            df = df.loc[(df['ticker'] != ticker) & (df['botId'] != bot_id)]
        else:
            df.loc[(df['ticker'] == ticker) & (df['botId'] == bot_id)] = [bot_id, ticker, new_quantity]
        df.to_csv(self.balance_path, sep=";", index=False)
        self.add_transaction(ticker, quantity, "sell", price, bot_id)

    def add_transaction(self, ticker, quantity, transaction_type, price, bot_id):
        df = pd.read_csv(self.transaction_path, header=0, sep=';')
        df.loc[df.index.max() + 1] = [bot_id, ticker, quantity, transaction_type, price, int(time.time() * 1000)]
        df.to_csv(self.transaction_path, sep=";", index=False)

    def get_price(self, ticker):
        #df = pd.read_csv(self.predictions_path, header=0, sep=';')
        #return(df.loc[df['timestamp'].idxmax()])
        return random() + 10

    def get_predictions(self):
        #df = pd.read_csv(self.predictions_path, header=0, sep=';')
        return bool(getrandbits(1))
