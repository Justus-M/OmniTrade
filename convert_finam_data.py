import os
import pandas as pd

os.chdir('/Users/justusmulli/Projects/OmniTrade/Data')
all_data = pd.DataFrame()
for file in [f for f in os.listdir() if 'US1' in f]:
    print(file)
    data = pd.read_csv(file)
    data.columns = list(map(lambda x: x.replace('<', '').replace('>', ''), data.columns))
    data.TICKER = list(map(lambda x: x.replace('US1.', ''), data.TICKER))
    all_data = all_data.append(data)

all_data['timestamp'] = list(map(lambda x : str(x)[:4]+'-'+str(x)[4:6]+'-'+str(x)[6:],  all_data.DATE))
all_data.timestamp = all_data.timestamp + ' '
all_data.timestamp = all_data.timestamp + \
                     list(map(lambda x: str(x).replace('94500', '094500')[:2]+
                                        ':'+str(x).replace('94500', '094500')[2:4]+
                                        ':'+str(x).replace('94500', '094500')[4:], all_data.TIME))