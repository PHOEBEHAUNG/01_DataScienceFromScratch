import numpy as np
import pandas as pd
# from pandas import Series, DataFrame

print(f'The first numpy and pandas')
print(f'==========================================')

print(f'Demo Series')
print(f'==========================================')
obj = pd.Series([4, 7, -5, 3])
print(obj)
print(f'==========================================')
print(obj.array)
print(f'==========================================')
print(obj.index)
print(f'\n')

print(f'Demo Series wit Indexs')
print(f'==========================================')
obj2 = pd.Series([4, 7, -5, 3], index=["d", "b", "a", "c"])
print(obj2)
print(f'==========================================')
print(obj2.array)
print(f'==========================================')
print(obj2.index)
print(f'==========================================')
print(obj2['a'])
print(f'==========================================')
obj['b'] = 6
print(obj2[['c', 'a', 'd']])
print(f'==========================================')
print(obj2[obj2 > 0])
print(f'==========================================')
print(obj2 * 2)
print(f'==========================================')
print(np.exp(obj2))
print(f'==========================================')
sdata = {"Ohio": 35000, "Texas": 71000, "Oregon": 16000, "Utah": 5000}
obj3 = pd.Series(sdata)
obj3.to_dict()
print(f'==========================================')
states = ["California", "Ohio", "Oregon", "Texas", "Utah"]
obj4 = pd.Series(sdata, index=states)
print(obj4)
pd.isna(obj4)
pd.notna(obj4)
obj4.isna()
print(f'==========================================')
print(obj3 + obj4)


print('\n')
print('DataFrame')
print(f'==========================================')
data = {"state": ["Ohio", "Ohio", "Ohio", "Nevada", "Nevada", "Nevada"],
        "year": [2000, 2001, 2002, 2001, 2002, 2003],
        "pop": [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
df = pd.DataFrame(data)
print(df)

df2 = pd.DataFrame(data, columns=["year", "state", "pop", "debt"])
print(df2)

print(df2['state'])
# frame2.column works only when 
# the column name is a valid Python variable name and does not conflict with any of the method names in DataFrame
print(df2.year) # if a column’s name contains whitespace or symbols other than underscores, it cannot be accessed with the dot attribute method

# get data by row
print(df2.loc[1])
print(df2.iloc[2])

# set empty column 
df2['debt'] = 16.5

# delete column
df2["eastern"] = df2["state"] == "Ohio"
del df2["eastern"]
print(df2.columns)

print(f'==========================================')
populations = {"Ohio": {2000: 1.5, 2001: 1.7, 2002: 3.6}, "Nevada": {2001: 2.4, 2002: 2.9}}
df3 = pd.DataFrame(populations)
print(df3)
print(df3.T)

print(f'==========================================')
df4 = pd.DataFrame(populations, index=[2001, 2002, 2003])
print(df4)

print(f'==========================================')
pdata = {"Ohio": df3["Ohio"][:-1], "Nevada": df3["Nevada"][:2]}
df5 = pd.DataFrame(pdata)
print(df5)

df5.index.name = "year"
df5.columns.name = "state"
print(df5)

# Panda Dataframe to Numpy
print(f'==========================================')
print(df3.to_numpy())
print(df2.to_numpy())

# Panda DataFrame index
print(f'==========================================')
print(obj4.index)
print(obj4.index[1:])

labels = pd.Index(np.arange(3))
obj2 = pd.Series([1.5, -2.5, 0], index = labels)
print(labels)
print(obj2)

# Panda DataFrame columns
print(f'==========================================')
print(df3.columns)
print("Ohio" in df3.columns)
print(2003 in df3.index)
print(pd.Index(["foo", "foo", "bar", "bar"]))

# df.head()
# df.tail()
