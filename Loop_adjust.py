import pandas as pd

loop = pd.read_csv(r"./data/loop.csv")
print(loop)
print(loop.columns)

loop.drop(columns=['FTIME', 'TTIME', 'ARTH_SPD', 'HARM_SPD'], inplace=True)
loop.to_csv(r"./data/loop.csv", index=False)