import pandas as pd
import seaborn


def build_adapted_df(df: pd.DataFrame):  # TODO: Try to not hardcod
    df.drop(["Country Code", "Series Code"], axis=1)
    df = df.iloc[:-5]
    val = pd.to_numeric(df.loc[:, ("2016 [YR2016]")], errors="coerce").copy()
    ndf = df.assign(**{"2016 [YR2016]": val})
    return ndf.pivot_table(
        index="Series Name", columns="Country Name", values="2016 [YR2016]"
    )


def del_many_na_country(
    df: pd.DataFrame, df_nb_null: pd.DataFrame, val_max_for_drop: float, is_colums=False
):
    """Delete country depending of number of NaN authorized

    Args:
        df (pd.DataFrame): Orignal dataframe : serie by countries
        percent_auth (float): Percentage of NaN max that you authorize to a country to be accepted

    Returns:
        _type_: The new dataframe without contries with many NaN
    """
    ax = 0 if is_colums else 1
    too_much_null = df_nb_null[df_nb_null["NaN_count"] > val_max_for_drop]
    return df.drop(too_much_null["Series"], axis=ax)


def del_many_na_series(
    df: pd.DataFrame, df_nb_null: pd.DataFrame, val_max_for_drop: float
):
    too_much_null = df_nb_null[df_nb_null["NaN_count"] > val_max_for_drop]
    return df.drop(too_much_null.index, axis=1)


def def_correled_series(df: pd.DataFrame, correlation_table: pd.DataFrame, limit: int):
    to_drop = []
    to_compare = list(correlation_table.index)
    for column in correlation_table.columns:
        to_compare.remove(column)
        for it in correlation_table[column].items():
            if abs(it[1]) > 1:
                to_drop.append(it[0])
    print(to_drop)
    return df.drop(to_drop, axis=0)
    # TODO: Faire un = TRUE quand supérieur, et regarder directement dans la table


def make_na_count(df: pd.DataFrame, is_colums=True):
    ax = 0 if is_colums else 1
    df_nb_null = df.isnull().sum(axis=ax)
    return df_nb_null.to_frame("NaN_count")


def corr_test_df(df: pd.DataFrame):
    df_T = df.T.corr("spearman")


if __name__ == "__main__":
    originial_df = pd.read_csv("data.txt", sep="\t")
    # df = build_adapted_df(originial_df)
    # df_nb_nan = make_na_count(df)
    # clean_df = del_many_na_country(df, df_nb_nan, 0.8)
    # df_corr = clean_df.T.corr("spearman")
    # print(seaborn.heatmap(df_corr))
    # df_step2_clean = def_correled_series(clean_df, df_corr, 1)
    # df_corr2 = df_step2_clean.T.corr("spearman")
    # print(seaborn.heatmap(df_corr2))
    # # print(clean_df)
