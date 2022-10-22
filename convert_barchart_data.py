import os
import pandas as pd
from datetime import timedelta

os.chdir('/Users/justusmulli/Projects/OmniTrade/Data')
ticker = 'vix'
all_data = pd.DataFrame()
for file in [f for f in os.listdir() if 'historical-data' in f and ticker in f]:
    data = pd.read_csv(file, skipfooter=1)
    all_data = all_data.append(data)

all_data['timestamp'] = pd.to_datetime(all_data.Time)
all_data.drop(columns=['Change', '%Chg', 'Volume'], inplace=True)
all_data.columns = list(map(lambda x: x.lower(), all_data.columns))
all_data = all_data[['timestamp', 'open', 'high', 'low', 'last']]
all_data = all_data.sort_values('timestamp')
all_data.timestamp = all_data.timestamp + timedelta(hours = 1)
all_data.drop_duplicates().to_csv(f'/Users/justusmulli/Projects/OmniTrade/Data/Minute/{ticker.upper()}.csv', index=None)
