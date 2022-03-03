from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import poisson, ttest_ind


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
    ].values
    lambda_value = lambda_value.astype(np.float64)
    sales_oracle_day = sales_oracle_day.fillna(0)
    sales_oracle_day = sales_oracle_day.values
    
    df.loc[
        ((df["product_id"] == sku) & (df["store_id"] == store)), "demand"
    ] = np.fmin(
        np.full((1, len(lambda_value)), max_sales),
        np.fmax(
            np.random.poisson(lambda_value, size=len(lambda_value)),
            sales_oracle_day, dtype=np.float64
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
    
    model_df["lambda"] = model_df["lambda"].fillna(0)
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
        
    df["lambda"] = df["lambda"].fillna(0)    
    return df


from itertools import product


def mix_features(
    df_model,
    product_id,
    min_periods=3,
    windows=[50],
    promo_filters=[0, 1, 2],
    deficit_filters=[0, 1, 2],
    target_var="Demand",
    store_id=4600,
):
    df = df_model.loc[
        (df_model["product_id"] == product_id)
        & (df_model["store_id"] == store_id)
    ].copy()

    df["Deficit"] = np.where((df.stock.isna() | df.stock == df.s_qty), 1, 0)

    # loop by filter variables and window
    for p, d, w in product(promo_filters, deficit_filters, windows):

        # define approproate dates for each SKU and Store pairs
        p_idx = d_idx = df.index
        if p < 2:
            p_idx = df.flg_spromo == p
        else:
            p_idx = df.flg_spromo == df.flg_spromo
        if d < 2:
            d_idx = df.Deficit == d
        else:
            d_idx = df.Deficit == df.Deficit

        # check whether filtered df in not empty
        if len(df[p_idx & d_idx].index) > 0:

            # lagged features calculation
            new_name = "amount_{0}prm_{1}dfc".format(p, d)
            df[new_name] = np.where((p_idx & d_idx), df.s_qty, np.nan)

            df = df.loc[
                (df["product_id"] == product_id) & (df["store_id"] == store_id)
            ].copy()
            df.loc[:, new_name] = (
                df[new_name]
                .rolling(center=True, window=w, min_periods=min_periods)
                .apply(np.nanmean)
            )
            df[new_name].fillna(method="ffill", inplace=True)
            df[new_name].fillna(method="bfill", inplace=True)
            df.loc[
                (
                    (df["product_id"] == product_id)
                    & (df["store_id"] == store_id)
                ),
                new_name,
            ] = df[new_name]

    df_model["lambda"] = df[new_name]
    del df

    return df_model


def restore_demand(df, product_id, store_id=4600, method="promo"):
    if method == "window":
        df = add_missing_dates(df, product_id, store_id)
        df = add_lambda_window(
            df, product_id, store_id=4600, window=30, min_periods=7
        )
        #df = calculate_demand(df, product_id, store_id)

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
        #df = calculate_demand(df, product_id, store_id)
        return df

    if method == "mix_features":
        df = add_missing_dates(df, product_id, store_id)
        df = mix_features(
            df_model=df, product_id=product_id, store_id=store_id
        )
        #df = calculate_demand(df, product_id, store_id)

        return df

    else:
        df = add_missing_dates(df, product_id, store_id)
        return df
