import time
from repository.flatfile import TradingRepository
import math
import uuid


class TradeBot:
    def __init__(self, budget, config):
        self.budget = budget
        self.balance = budget
        self.config = config
        self.uuid = str(uuid.uuid4())
        self.TradingRepository = TradingRepository.TradingRepository()
        self.predictions = []

    def trade(self):
        while self.balance > 0:
            time.sleep(self.config['interval'])
            self.TradingRepository.get_
            for p in self.predictions:
                ticker = p['ticker']
                if p['prediction'] == 0:
                    self.sell_all(ticker)
                else:
                    current_price = self.TradingRepository.get_price()
                    quantity = math.floor(self.balance/current_price)
                    self.sell_all(ticker, quantity, current_price)

    def buy(self, ticker, quantity, price):
        cost = quantity * price
        if self.balance >= cost:
            self.balance = self.balance - cost
            self.TradingRepository.make_buy_order(ticker, quantity, price, self.uuid)

    def sell(self, ticker, quantity, price):
        self.balance = self.balance + (quantity * price)
        self.TradingRepository.make_sell_order(ticker, quantity, price, self.uuid)

    def sell_all(self, ticker, price):
        current_quantity = self.TradingRepository.get_balance(ticker, self.uuid)
        self.sell(ticker, current_quantity, price)
