import pandas as pd


def build_adapted_df(df: pd.DataFrame):  # TODO: Try to not hardcod
    df.drop(["Country Code", "Series Code"], axis=1)
    df = df.iloc[:-5]
    val = pd.to_numeric(df.loc[:, ("2016 [YR2016]")], errors="coerce").copy()
    ndf = df.assign(**{"2016 [YR2016]": val})
    return ndf.pivot_table(
        index="Series Name", columns="Country Name", values="2016 [YR2016]"
    )


def del_many_na_country(
    df: pd.DataFrame, df_nb_null: pd.DataFrame, val_max_for_drop: float
):
    """Delete country depending of number of NaN authorized

    Args:
        df (pd.DataFrame): Orignal dataframe : serie by countries
        percent_auth (float): Percentage of NaN max that you authorize to a country to be accepted

    Returns:
        _type_: The new dataframe without contries with many NaN
    """
    too_much_null = df_nb_null[df_nb_null["NaN_count"] > val_max_for_drop]
    return df.drop(too_much_null.index, axis=1)


def compute_max_nan(df_nb_nan: pd.DataFrame, percent):
    return df_nb_nan.quantile(percent, axis=0).values[0]


def make_na_count(df: pd.DataFrame):
    df_nb_null = df.isnull().sum(axis=0)
    return df_nb_null.to_frame("NaN_count")


if __name__ == "__main__":
    originial_df = pd.read_csv("data.txt", sep="\t")
    df = build_adapted_df(originial_df)
    clean_df = del_many_na_country(df, 0.5, True)
    # print(clean_df)
