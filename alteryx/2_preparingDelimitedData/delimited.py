from ayx import Alteryx
import pandas as pd

df = Alteryx.read('#1')
df

print(df['1'].head())

#df['1'].apply(lambda s: s.replace('"', "", regex=True))
df = df.replace('"', '', regex=True)
df = df.replace("'", "", regex=True)
print(df)

dfnew = df.rename(columns={"1":"Poem", "2":"Poem_ID", "3":"Poem_read_date" }, inplace=False)
                   
print(dfnew)