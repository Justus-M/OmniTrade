def StockFilter(frame, tickercol, *keep):
	import pandas as pd
	singles = list(dict.fromkeys(frame[tickercol]))
	filtered = pd.DataFrame(index = frame[frame[tickercol] == singles[0]].index)
	for ticker in singles:
		for col in keep:
			filtered[ticker + " " + col] = frame[frame[tickercol] == ticker][col]
	return filtered
