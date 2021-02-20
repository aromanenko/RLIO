import scipy.stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import poisson
from scipy.stats import ttest_ind


def add_missing_dates(model_df, product_id, store_id):
    # Data as index and fill missing stock data
    new_df = model_df[
        (model_df["product_id"] == product_id)
        & (model_df["store_id"] == store_id)
    ].set_index("curr_date")[["stock"]]
    new_df = new_df.reindex(
        pd.date_range(np.min(new_df.index), np.max(new_df.index))
    ).fillna(method="ffill")

    # Add sales data to df with stock data
    _df_day_sales = model_df[(model_df["product_id"] == product_id)].set_index(
        "curr_date"
    )
    new_df = _df_day_sales[
        ["product_id", "store_id", "flg_spromo", "s_qty"]
    ].merge(new_df, how="right", left_index=True, right_index=True)

    return new_df


def calculate_demand(df, sku, store=4600):
    max_sales = df[
        (df["product_id"] == sku) & (df["store_id"] == store)
    ].s_qty.max()
    sales_oracle_day = df[
        (df["product_id"] == sku) & (df["store_id"] == store)
    ]["s_qty"]
    lambda_value = df[((df["product_id"] == sku) & (df["store_id"] == store))][
        "lambda"
    ]
    df.loc[
        ((df["product_id"] == sku) & (df["store_id"] == store)), "demand"
    ] = np.fmin(
        np.full((1, len(lambda_value)), max_sales),
        np.fmax(
            np.random.poisson(lambda_value, size=len(lambda_value)),
            sales_oracle_day,
        ),
    ).tolist()[
        0
    ]
    return df


def ttest_promo(df_promo, df_nopromo):
    # Check data for correctness
    if len(df_promo) > 0 and len(df_nopromo) > 0:
        _, p_value = ttest_ind(
            np.array(df_nopromo["s_qty"]), np.array(df_promo["s_qty"])
        )
        if p_value < 0.05:
            flag = True
            decision = "H1: different averages"
        else:
            flag = False
            decision = "H0: same averages"

    # Report if test is not available
    else:
        flag = False
        decision = "Not enough data"

    return flag, decision


def calculate_lambda_promo(
    df, product_id, store_id=4600, teta=1, enable_test=True
):
    # Choose data without promo for a given product
    df_nopromo = df.loc[
        (df["product_id"] == product_id)
        & (df["flg_spromo"] == 0)
        & (df["store_id"] == store_id)
    ]

    # Leave only correct data
    df_nopromo = df_nopromo.loc[
        (df_nopromo["stock"] > 0)
        & (df_nopromo["s_qty"] <= df_nopromo["stock"])
    ]

    # Count days, where sell all or part of product amount
    sales_part = len(df_nopromo.loc[df_nopromo["s_qty"] < df_nopromo["stock"]])
    sales_all = len(df_nopromo.loc[df_nopromo["s_qty"] == df_nopromo["stock"]])

    # Count lambda for poisson distribution for days without promo
    lambda_nopromo = df_nopromo["s_qty"].sum() / (
        sales_part + sales_all * teta
    )

    # Choose data with promo for a given product
    df_promo = df.loc[
        (df["product_id"] == product_id)
        & (df["store_id"] == store_id)
        & (df["flg_spromo"] == 1)
    ]

    # Leave only correct data
    df_promo = df_promo.loc[
        (df_promo["stock"] > 0) & (df_promo["s_qty"] <= df_promo["stock"])
    ]

    # Count days, where sell all or part of product amount
    sales_part = len(df_promo.loc[df_promo["s_qty"] < df_promo["stock"]])
    sales_all = len(df_promo.loc[df_promo["s_qty"] == df_promo["stock"]])

    # If required, conduct t_test and make a decision
    recount, decision = False, None
    if enable_test == True:
        recount, decision = ttest_promo(df_promo, df_nopromo)

    if enable_test == False or recount == True:
        # Count lambda for poisson distribution for days with promo
        if (sales_part + sales_all * teta) == 0:
            lambda_promo = 0
        else:
            lambda_promo = df_promo["s_qty"].sum() / (
                sales_part + sales_all * teta
            )
    else:
        lambda_promo = lambda_nopromo

    return lambda_nopromo, lambda_promo, decision


def add_lambda_window(df, product_id, store_id=4600, window=30, min_periods=7):
    model_df = df

    # Choose only needed data and perform rolling
    df = df.loc[
        (df["product_id"] == product_id) & (df["store_id"] == store_id)
    ].copy()
    df.loc[:, "lambda"] = (
        df["s_qty"]
        .rolling(center=True, window=window, min_periods=min_periods)
        .apply(np.nanmean)
    )

    # Fill empty data if it exists
    df["lambda"].fillna(method="ffill", inplace=True)
    df["lambda"].fillna(method="bfill", inplace=True)

    # Create lamda column
    model_df.loc[
        (
            (model_df["product_id"] == product_id)
            & (model_df["store_id"] == store_id)
        ),
        "lambda",
    ] = df["lambda"]

    return model_df


def add_lambda(df, product_id, store_id, lambda_nopromo, lambda_promo=None):
    # Choose all rows and add lambda
    df.loc[
        (df["product_id"] == product_id) & (df["store_id"] == store_id),
        ["lambda"],
    ] = lambda_nopromo

    # Choose all rows with promo and add lambda
    if lambda_promo != None:
        df.loc[
            (
                (df["product_id"] == product_id)
                & (df["store_id"] == store_id)
                & (df["flg_spromo"] == 1)
            ),
            ["lambda"],
        ] = lambda_promo

    return df


def restore_demand(df, product_id, store_id=4600, method="promo"):
    if method == "window":
        df = add_missing_dates(df, product_id, store_id)
        df = add_lambda_window(
            df, product_id, store_id=4600, window=30, min_periods=7
        )
        df = calculate_demand(df, product_id, store_id)

        return df

    if method == "promo":
        df = add_missing_dates(df, product_id, store_id)
        lambda_nopromo, lambda_promo, _ = calculate_lambda_promo(
            df,
            product_id=product_id,
            store_id=store_id,
            teta=1,
            enable_test=True,
        )
        df = add_lambda(df, product_id, store_id, lambda_nopromo, lambda_promo)
        df = calculate_demand(df, product_id, store_id)

        return df