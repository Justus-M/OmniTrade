import os

os.sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from business_rules import TradeBot

tb = TradeBot.TradeBot(100, {"interval": 5})
tb.buy("BTC", 10, price=10)
tb.buy("BTC", 10, price=10)
tb.sell_all("BTC", 11)
print(tb.balance)
