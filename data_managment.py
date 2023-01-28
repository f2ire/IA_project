import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


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
    """Delite series with too many NaN

    Args:
        df (pd.DataFrame): dataframe with series by columns
        df_nb_null (pd.DataFrame): dataframe with number of NaN by series
        val_max_for_drop (float): value max of NaN for a serie to be deleted

    Returns:
        _type_: dataframe without series with too many NaN
    """
    too_much_null = df_nb_null[df_nb_null["NaN_count"] > val_max_for_drop]
    new_df = df.drop(too_much_null.index, axis=1)
    return new_df


def del_correled_series(df: pd.DataFrame, correlation_table: pd.DataFrame, limit: float):
    """Delete series that are too correled

    Args:
        df (pd.DataFrame): dataframe with series by columns
        correlation_table (pd.DataFrame): correlation table between series
        limit (float): limit of correlation

    Returns:
        _type_: dataframe without series that are too correled
    """
    upper_tri = correlation_table.where(
        np.triu(np.ones(correlation_table.shape), k=1).astype(bool)
    )
    to_drop = [
        column for column in upper_tri.columns[::-1] if any(upper_tri[column] > limit)
    ]
    new_df = df.drop(to_drop, axis=1)
    return new_df


def make_na_count(df: pd.DataFrame, is_colums=True):
    """Make a dataframe with number of NaN by series

    Args:
        df (pd.DataFrame): dataframe with series by columns
        is_columns (bool, optional): Is it is columns. Defaults to True.

    Returns:
        _type_: dataframe with number of NaN by series
    """
    ax = 0 if is_colums else 1
    df_nb_null = df.isnull().sum(axis=ax)
    return df_nb_null.to_frame("NaN_count")


def replace_nan_knn(df: pd.DataFrame):
    """Replace NaN by KNN values

    Args:
        df (pd.DataFrame): dataframe with series by columns

    Returns:
        _type_: dataframe with NaN replaced by KNN values
    """
    imputer = KNNImputer(missing_values=np.nan)
    imputed_DF = pd.DataFrame(imputer.fit_transform(df))
    imputed_DF.columns = df.columns
    imputed_DF.index = df.index
    return imputed_DF


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dataframe"""
    zscore = StandardScaler().fit(df)
    norm_df = pd.DataFrame(zscore.transform(df), index=df.index, columns=df.columns)
    return norm_df


def export_clean_data(df: pd.DataFrame, file_name="clean_dataframe.csv"):
    """Export clean dataframe

    Args:
        df (pd.DataFrame): dataframe with series by columns
        file_name (str, optional): file name. Defaults to "clean_dataframe.csv".
    """
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
