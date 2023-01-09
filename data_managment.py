import pandas as pd

df = pd.read_csv("data.txt", sep="\t")
df.drop(["Country Code", "Series Code"], axis=1)
df = df.iloc[:-5]

df["2016 [YR2016]"] = pd.to_numeric(df["2016 [YR2016]"], errors="coerce")
a = df.pivot_table(index="Series Name", columns="Country Name", values="2016 [YR2016]")


def del_many_na_country(df: pd.DataFrame, percent: float):
    res = a.isnull().sum(axis=0)
    for c in df:
        pass
    print(res[4])


del_many_na_country(a, 0.5)
