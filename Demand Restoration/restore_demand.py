# ----------------------------------------------------------------------
# ------------------------------- ИМПОРТЫ ------------------------------
# ----------------------------------------------------------------------

import numpy as np
import pandas as pd

from itertools import product
from scipy.stats import poisson, ttest_ind

# ----------------------------------------------------------------------
# -------------------- РАСЧЕТ LAMBDA ОКОННЫМ МЕТОДОМ -------------------
# ----------------------------------------------------------------------

def add_lambda_window(df, store_id, product_id, window=30, min_periods=7):
    """Добавление lambda, расчитанного оконным методом

    Args:
        df:
            [pandas.DataFrame] Исходный DataFrame
        store_id:
            [int] ID магазина
        product_id:
            [int] ID товара
        window:
            [int] Размер окна
        min_periods
            [int] Минимальное количество дней в интервале, для которых должны быть заполнены данные о продажах
                Если таких дней меньше - значение восстановденного спроса считается равным NaN,
                после чего в конце происходит заполенение таких дней последовательным применением методов ffill и bfill

    Returns:
        model_df
            [pandas.DataFrame] Исходный DataFrame + рассчитанная lambda
    """

    model_df = df.copy()

    # Choose only needed data and perform rolling
    df = df.loc[
        (df["product_id"] == product_id) &
        (df["store_id"] == store_id)
    ].copy()

    df.loc[:, "lambda_window"] = (
        df["s_qty"]
        .rolling(center=True, window=window, min_periods=min_periods)
        .apply(np.nanmean)
    )

    # Fill empty data if it exists
    df["lambda_window"].fillna(method="ffill", inplace=True)
    df["lambda_window"].fillna(method="bfill", inplace=True)

    # Create lamda column
    model_df.loc[
        (model_df["product_id"] == product_id) &
        (model_df["store_id"] == store_id),
        "lambda_window",
    ] = df["lambda_window"]

    model_df["lambda_window"] = model_df["lambda_window"].fillna(0)

    return model_df


# ----------------------------------------------------------------------
# ---------------- РАСЧЕТ LAMBDA С ПОМОЩЬЮ ФЛАГА ПРОМО -----------------
# ----------------------------------------------------------------------


def ttest_promo(df_promo, df_nopromo):
    """TBD

    Args:
        df_promo:
            [pandas.DataFrame] TBD
        df_nopromo:
            [pandas.DataFrame] TBD

    Returns:
        flag:
            [bool] TBD
        decision:
            [string] TBD
    """

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


def calculate_lambda_promo(df, store_id, product_id, teta=1, enable_test=True):
    """Расчет lambda на основе флага промо

    Args:
        df:
            [pandas.DataFrame] Исходный DataFrame
        store_id:
            [int] ID магазина
        product_id:
            [int] ID товара
        teta:
            [float] TBD
        enable_test:
            [bool] TBD

    Returns:
        lambda_nopromo:
            [float] - TBD
        lambda_promo:
            [float] - TBD
        decision:
            [sting] - TBD
    """

    # Choose data without promo for a given product
    df_nopromo = df.loc[
        (df["product_id"] == product_id) &
        (df["flg_spromo"] == 0) &
        (df["store_id"] == store_id)
    ]

    # Leave only correct data
    df_nopromo = df_nopromo.loc[
        (df_nopromo["stock"] > 0) &
        (df_nopromo["s_qty"] <= df_nopromo["stock"])
    ]

    # Count days, where sell all or part of product amount
    sales_part = len(df_nopromo.loc[df_nopromo["s_qty"] < df_nopromo["stock"]])
    sales_all = len(df_nopromo.loc[df_nopromo["s_qty"] == df_nopromo["stock"]])

    # Count lambda for poisson distribution for days without promo
    lambda_nopromo = df_nopromo["s_qty"].sum() / (sales_part + sales_all * teta)

    # Choose data with promo for a given product
    df_promo = df.loc[
        (df["product_id"] == product_id) &
        (df["store_id"] == store_id) &
        (df["flg_spromo"] == 1)
    ]

    # Leave only correct data
    df_promo = df_promo.loc[
        (df_promo["stock"] > 0) &
        (df_promo["s_qty"] <= df_promo["stock"])
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
            lambda_promo = df_promo["s_qty"].sum() / (sales_part + sales_all * teta)
    else:
        lambda_promo = lambda_nopromo

    return lambda_nopromo, lambda_promo, decision


def add_lambda_promo(df, store_id, product_id):
    """Расчет lambda методом промо и добавление к датафрейму

    Args:
        df:
            [pandas.DataFrame] Исходный DataFrame
        store_id:
            [int] ID магазина
        product_id:
            [int] ID товара

    Returns:
        df:
            [pandas.DataFrame] Исходный DataFrame + lambda_promo
    """

    lambda_nopromo, lambda_promo, _ = calculate_lambda_promo(
        df,
        product_id=product_id,
        store_id=store_id,
        teta=1,
        enable_test=True,
    )

    # Choose all rows and add lambda
    df.loc[
        (df["product_id"] == product_id) &
        (df["store_id"] == store_id),
        ["lambda_promo"]
    ] = lambda_nopromo

    # Choose all rows with promo and add lambda
    if lambda_promo != None:
        df.loc[
            (df["product_id"] == product_id) &
            (df["store_id"] == store_id) &
            (df["flg_spromo"] == 1),
            ["lambda_promo"]
        ] = lambda_promo

    df["lambda_promo"] = df["lambda_promo"].fillna(0)
    return df


# ----------------------------------------------------------------------
# ------------------------ СЭМПЛИРОВАНИЕ СПРОСА ------------------------
# ----------------------------------------------------------------------


def restore_demand(df, store_id, product_id, type='window'):
    """Сэмплирование спроса из распределения Пуассона

    Args:
        df:
            [pandas.DataFrame] Исходный DataFrame
        store_id:
            [int] ID магазина
        product_id:
            [int] ID товара
        type:
            [string - 'promo' or 'window'] Тип lambda для восстановления спроса

    Returns:
        df:
            [pandas.DataFrame] Исходный DataFrame + demand
    """

    max_sales = df[
        (df["product_id"] == product_id) & (df["store_id"] == store_id)
    ].s_qty.max()

    sales_oracle_day = df[
        (df["product_id"] == product_id) & (df["store_id"] == store_id)
    ]["s_qty"]

    if type == 'window':
        lambda_value = df[
            (df["product_id"] == product_id) & (df["store_id"] == store_id)
        ]["lambda_window"].values
    elif type == 'promo':
        lambda_value = df[
            (df["product_id"] == product_id) & (df["store_id"] == store_id)
        ]["lambda_promo"].values

    lambda_value = lambda_value.astype(np.float64)
    sales_oracle_day = sales_oracle_day.fillna(0)
    sales_oracle_day = sales_oracle_day.values

    df.loc[
        (df["product_id"] == product_id) & (df["store_id"] == store_id),
        "demand"
    ] = np.fmin(
        np.full((1, len(lambda_value)), max_sales),
        np.fmax(
            np.random.poisson(lambda_value, size=len(lambda_value)),
            sales_oracle_day, dtype=np.float64
        ),
    ).tolist()[0]

    return df
