import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


def build_adapted_df(df: pd.DataFrame):  # TODO: Try to not hardcod
    df.drop(["Country Code", "Series Code"], axis=1)
    df = df.iloc[:-5]
    val = pd.to_numeric(df.loc[:, ("2018 [YR2018]")], errors="coerce").copy()
    ndf = df.assign(**{"2018 [YR2018]": val})
    return ndf.pivot_table(
        index="Series Name", columns="Country Name", values="2018 [YR2018]"
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
    too_much_null = df_nb_null[df_nb_null["NaN_count"] >= val_max_for_drop]
    new_df = df.drop(too_much_null["Series"], axis=0)
    return new_df


def del_many_na_series(
    df: pd.DataFrame, df_nb_null: pd.DataFrame, val_max_for_drop: float
):
    too_much_null = df_nb_null[df_nb_null["NaN_count"] > val_max_for_drop]
    new_df = df.drop(too_much_null.index, axis=1)
    return new_df


def del_correled_series(
    df: pd.DataFrame, correlation_table: pd.DataFrame, limit: float
):
    upper_tri = correlation_table.where(
        np.triu(np.ones(correlation_table.shape), k=1).astype(bool)
    )
    to_drop = [
        column for column in upper_tri.columns[::-1] if any(upper_tri[column] > limit)
    ]
    new_df = df.drop(to_drop, axis=0)
    return new_df


def make_na_count(df: pd.DataFrame, is_colums=True):
    ax = 0 if is_colums else 1
    df_nb_null = df.isnull().sum(axis=ax)
    return df_nb_null.to_frame("NaN_count")


def replace_nan_knn(df: pd.DataFrame):
    imputer = KNNImputer(missing_values=np.nan)
    imputed_DF = pd.DataFrame(imputer.fit_transform(df))
    imputed_DF.columns = df.columns
    imputed_DF.index = df.index
    return imputed_DF


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    zscore = StandardScaler().fit(df)
    norm_df = pd.DataFrame(zscore.transform(df), index=df.index, columns=df.columns)
    return norm_df


def export_clean_data(df: pd.DataFrame, file_name="clean_dataframe.csv"):
    df.to_csv(file_name)


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
