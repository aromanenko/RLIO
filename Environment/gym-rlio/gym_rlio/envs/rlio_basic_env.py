############## ENV IMPORTS ##############
import os
import numpy as np
import pandas as pd
from math import ceil
from tqdm import tqdm
from itertools import product
########### DEMAND RESTORATION ##########
import sys
sys.path.append('/Users/mgcrp/Documents/GitHub/RLIO/Demand Restoration')
from restore_demand import add_lambda_window, add_lambda_promo, restore_demand
############## GYM IMPORTS ##############
import gym
from gym.utils import seeding
from gym import error, spaces, utils
################ SETTINGS ################
ECHELON_DATA_PATH = "/Users/mgcrp/Documents/GitHub/RLIO/Environment/echelon_processed_data"
STORE_RAW_DATA_PATH = "/Users/mgcrp/Documents/GitHub/RLIO/Environment/store_raw_data"
STORE_PROCESSED_DATA_PATH = "/Users/mgcrp/Documents/GitHub/RLIO/Environment/store_processed_data"
############ DISABLE WARNINGS ############
import warnings
warnings.filterwarnings('ignore')
##########################################


def _dummy_demand_restoration(df_input):
    """Заглушка для функции восстановления спроса

    Возвращает demand = sales

    Args:
        df_input:
            Pandas DataFrame of columns	[product_id, store_id, curr_date, s_qty, flg_spromo, stock, mply_qty,
            lead_time, batch_size, service_level]
            Входные данные о продажах

    Returns:
        df_output:
            Pandas DataFrame of columns	[product_id, store_id, curr_date, s_qty, flg_spromo, stock, mply_qty,
            lead_time, batch_size, service_level, demand]
            Входные данные о продажах + восстановленный спрос
    """

    df_output = df_input.copy(deep=True)
    df_output["demand"] = df_output["s_qty"].astype('int')
    return df_output


def _store_data_preprocessing(input_file, output_file):
    """Функция для нормализации данных о продажах магазина

    Делает:
        * Загружает исходный csv из директории STORE_RAW_DATA_PATH в pandas.DataFrame
        * Заполняет пропущенные даты между первым и последним появлением товара в магазине
        * Добавляет метаданные из предварительно нормализованных файлов о сети из директории ECHELON_DATA_PATH:
            * mply qty
            * lead time
            * batch size
            * service level
        * Заполняет NULL
            * stock и s_qty - 0.0
            * mply_qty, lead_time, batch_size, service_level - ffill + bfill
        * Рассчитывает lamda двумя методами:
            * promo
            * window
        * Сохраняет pandas.DataFrame с нормализованными данными в файл в директории STORE_PROCESSED_DATA_PATH

    Args:
        input_file:
            [String] Путь к сырому файлу
        output_file:
            [String] Путь для записи обработанного файла

    Returns:
        None
    """

    print('1 - Загрузка сырых данных')

    df = pd.read_csv(input_file, sep=';', decimal='.')
    df.curr_date = pd.to_datetime(df.curr_date, format='%d%b%Y')

    print('2 - Заполнение пропусков в датах')

    store = int( input_file[input_file.rfind('_')+1:input_file.rfind('.csv')] )

    tmp = []

    for sku in tqdm( df.product_id.unique() ):
        if len( pd.date_range(
            start = df[df.product_id == sku].curr_date.min(),
            end = df[df.product_id == sku].curr_date.max()
        ).difference( df[df.product_id == sku].curr_date) ):
            for date in pd.date_range(
                start = df[df.product_id == sku].curr_date.min(),
                end = df[df.product_id == sku].curr_date.max()
            ).difference( df[df.product_id == sku].curr_date ):
                tmp.append(
                    {
                        'product_id': sku,
                        'store_id': store,
                        'curr_date': date,
                        's_qty': 0,
                        'flg_spromo': 0,
                        'stock': np.nan
                    }
                )

    df = df.append(pd.DataFrame(tmp), ignore_index=True)

    print('3 - Добавление метаданных')

    print('3.1 - mply_qty')
    df_meta = pd.read_csv(os.path.join(ECHELON_DATA_PATH, "mply_qty_normalized.csv"), sep=';')
    df_meta.curr_date = pd.to_datetime(df_meta.curr_date)

    df = pd.merge(
        how='left',
        left=df,
        left_on=['product_id', 'store_id', 'curr_date'],
        right=df_meta,
        right_on=['product_id', 'store_id', 'curr_date']
    )

    print('3.2 - lead time')
    df_meta = pd.read_csv(os.path.join(ECHELON_DATA_PATH, "lead_time_normalized.csv"), sep=';')
    df_meta.curr_date = pd.to_datetime(df_meta.curr_date)

    df = pd.merge(
        how='left',
        left=df,
        left_on=['product_id', 'store_id', 'curr_date'],
        right=df_meta,
        right_on=['product_id', 'store_id', 'curr_date']
    )

    print('3.3 - batch_size')
    df_meta = pd.read_csv(os.path.join(ECHELON_DATA_PATH, "batch_size_normalized.csv"), sep=';')
    df_meta.curr_date = pd.to_datetime(df_meta.curr_date)

    df = pd.merge(
        how='left',
        left=df,
        left_on=['product_id', 'store_id', 'curr_date'],
        right=df_meta,
        right_on=['product_id', 'store_id', 'curr_date']
    )

    print('3.4 - service_level')
    df_meta = pd.read_csv(os.path.join(ECHELON_DATA_PATH, "service_level_normalized.csv"), sep=';')
    df_meta.curr_date = pd.to_datetime(df_meta.curr_date)

    df = pd.merge(
        how='left',
        left=df,
        left_on=['product_id', 'store_id', 'curr_date'],
        right=df_meta,
        right_on=['product_id', 'store_id', 'curr_date']
    )

    print("4 - Заполенение NULL'ов")
    df.s_qty.fillna(0, inplace=True)
    df.stock.fillna(0, inplace=True)

    df.mply_qty.fillna(method='ffill', inplace=True)
    df.lead_time.fillna(method='ffill', inplace=True)
    df.batch_size.fillna(method='ffill', inplace=True)
    df.service_level.fillna(method='ffill', inplace=True)

    df.mply_qty.fillna(method='bfill', inplace=True)
    df.lead_time.fillna(method='bfill', inplace=True)
    df.batch_size.fillna(method='bfill', inplace=True)
    df.service_level.fillna(method='bfill', inplace=True)

    print('5 - Расчет lambda')

    for store_id in df.store_id.unique():
        for product_id in tqdm( df[df.store_id == store_id].product_id.unique(), total=df[df.store_id == store_id].product_id.nunique() ):
            df = add_lambda_promo(df, store_id=store_id, product_id=product_id)
            df = add_lambda_window(df, store_id=store_id, product_id=product_id)

    print('6 - Запись в файл')
    df.to_csv(output_file, index=False)


def _dummy_apply_reward_calculation(row):
    """Базовый вариант расчета reward
    Только для применения в apply к pandas.DataFrame

    Args:
        row:
            [pandas.Series] Одна строка из pandas.DataFrame

    Returns:
        reward:
            [float] Рассчитанное значение reward
    """

    return row.sales - row.stock * ( (1 - row.service_level) / (row.service_level) )


def _dummy_apply_reward_calculation_v2(row, r=0.2, k=0.05):
    """Усложненный вариант расчета reward.
    К формуле, использованной в статье SMART, добавлены:
        • - ( r * fact_order ) - штраф за заказ
        • - ( k * (demand - sales) ) - штраф за неудовлетворенный спрос
    В данном случае r и k подобраны экспертно и зафиксированы как гиперпараметры
    Только для применения в apply к pandas.DataFrame

    Args:
        row:
            [pandas.Series] Одна строка из pandas.DataFrame

    Returns:
        reward:
            [float] Рассчитанное значение reward
    """

    return row.sales - r * row.order - k * (row.demand - row.sales) - row.stock * ( (1 - row.service_level) / (row.service_level) )


def _dummy_apply_reward_calculation_baseline(row):
    """Базовый вариант расчета reward
    Необходима для расчета reward по данным из реальной ритейл сети
    Только для применения в apply к pandas.DataFrame

    Args:
        row:
            [pandas.Series] Одна строка из pandas.DataFrame

    Returns:
        reward:
            [float] Рассчитанное значение reward
    """

    return row.s_qty - row.stock * ( (1 - row.service_level) / (row.service_level) )


def _dummy_apply_reward_calculation_baseline_v2(row, r=0.2, k=0.05):
    """Усложненный вариант расчета reward.
    К формуле, использованной в статье SMART, добавлены:
        • - ( r * fact_order ) - штраф за заказ
        • - ( k * (demand - sales) ) - штраф за неудовлетворенный спрос
    В данном случае r и k подобраны экспертно и зафиксированы как гиперпараметры
    Необходима для расчета reward по данным из реальной ритейл сети
    Только для применения в apply к pandas.DataFrame

    Args:
        row:
            [pandas.Series] Одна строка из pandas.DataFrame

    Returns:
        reward:
            [float] Рассчитанное значение reward
    """

    # return row.s_qty - r * row.order - k * (row.demand - row.sales) - row.stock * ( (1 - row.service_level) / (row.service_level) )
    return row.s_qty - k * (row.demand - row.s_qty) - row.stock * ( (1 - row.service_level) / (row.service_level) )


def _dummy_apply_order_calculation(row):
    """Заглушка для расчета order
    Сюда можно придумать функцию, вносящую хаос в объем заказа
    Только для применения в apply к pandas.DataFrame

    Args:
        row:
            [pandas.Series] Одна строка из pandas.DataFrame

    Returns:
        fact_order:
            [int] Фактическое значение order
    """

    return row.recommended_order


def _dummy_apply_order_calculation_v2(row):
    """Заглушка для расчета order
    Учитывает projected_stock
    Сюда можно придумать функцию, вносящую хаос в объем заказа
    Только для применения в apply к pandas.DataFrame

    Args:
        row:
            [pandas.Series] Одна строка из pandas.DataFrame

    Returns:
        fact_order:
            [int] Фактическое значение order
    """

    return max(0, row.recommended_order - row.projected_stock)


def _dummy_apply_order_calculation_with_batch_size(row):
    """Заглушка для расчета order
    Рассчитывает фактический объем заказа в зависимости от batch_size
    Сюда можно придумать функцию, вносящую хаос в объем заказа
    Только для применения в apply к pandas.DataFrame

    Args:
        row:
            [pandas.Series] Одна строка из pandas.DataFrame

    Returns:
        fact_order:
            [int] Фактическое значение order
    """

    return row.batch_size * ceil( row.recommended_order / row.batch_size )


def _dummy_apply_order_calculation_with_batch_size_v2(row):
    """Заглушка для расчета order
    Рассчитывает фактический объем заказа в зависимости от batch_size
    + учитывает projected_stock
    Сюда можно придумать функцию, вносящую хаос в объем заказа
    Только для применения в apply к pandas.DataFrame

    Args:
        row:
            [pandas.Series] Одна строка из pandas.DataFrame

    Returns:
        fact_order:
            [int] Фактическое значение order
    """

    return row.batch_size * ceil( max(0, row.recommended_order - row.projected_stock) / row.batch_size )


def _dummy_lead_time_calculation(expected_lead_time):
    """Заглушка для расчета lead_time
    Сюда можно придумать функцию, вносящую хаос в сроки доставки

    Args:
        expected_lead_time:
            [float] Плановое время доставки

    Returns:
        fact_lead_time:
            [float] Фактическое время доставки
    """

    return expected_lead_time


class RlioBasicEnv(gym.Env):
    """Среда, имитирующая поведение логистической сети.
    Разработана с использованием фреймворка OpenAI Gym

    Variables:
        * action_space: [list of pairs (ROL::int, OUL::int)] дискретный action space
        * stores_data: [pandas.DataFrame] Информация о фактических продажах товаров
        * environment_data [dict] Информация о состоянии среды в следующем формате:
            {
                store_id::int: {
                    product_id::int: {
                        'stock': [float] текущее количество товара на стоке
                        'reward_log': [list of floats] история reward для данной пары товар-магазин
                        'policy_log': [list of pairs (ROL::int, OUL::int)] история изменения политики
                        'order_queue': [list of ints] очередь заказанных товаров к доставке на 100 дней вперед
                    }
                }
            }
        * order_apply_function [function] - Функция для рассчета order
        * reward_apply_function [function] - Функция для рассчета reward
        * demand_restoration_type [string - 'dummy', 'promo' or 'window'] метод восстановления спроса в среде
        * start_date [datetime.date] Дата начала
        * finish_date [datetime.date] Последняя дата
        * current_date [datetime.date] Текущая дата

    Methods:
        * load_data(self, products_dict, demand_restoration_type='window')
        * step(self, action)
        * reset(self)
        * render(self, mode='human', show_table=True)
        * close(self)
        * compute_baseline_realworld(self)
    """

    metadata = {'render.modes': ['human']}


    def __init__(self):
        """Стандартная инициализация среды"""

        self.action_space = None

        self.stores_data = None
        self.environment_data = None
        self.order_apply_function = None
        self.reward_apply_function = None
        self.demand_restoration_type = None

        self.start_date = None
        self.finish_date = None
        self.current_date = None


    def load_data(
        self,
        products_dict,
        demand_restoration_type='window',
        reward_apply_function=_dummy_apply_reward_calculation_v2,
        order_apply_function=_dummy_apply_order_calculation_with_batch_size_v2
        ):
        """Загрузка данных в среду

        Agrs:
            products_dict:
                [dict] Словарь с описанием данных, которые необходимо загрузить в среду
                    { 'store_id': [product_ids] or 'all' }
            demand_restoration_type:
                [string - 'dummy', 'promo' or 'window'] метод восстановления спроса в среде
            reward_apply_function:
                [function] Функция для рассчета reward
            order_apply_function:
                [function] Функция для рассчета order

        Returns:
            None
        """

        # 1 - Загрузка данных
        self.stores_data = pd.DataFrame()

        for store in products_dict.keys():
            if os.path.exists( os.path.join(STORE_PROCESSED_DATA_PATH, f"STORE_{store}.csv") ):
                df_tmp = pd.read_csv( os.path.join(STORE_PROCESSED_DATA_PATH, f"STORE_{store}.csv") )
                if products_dict[store] != 'all':
                    df_tmp = df_tmp[df_tmp.product_id.isin(products_dict[store])].reset_index(drop=True)
                self.stores_data = self.stores_data.append( df_tmp )
                del df_tmp
            else:
                _store_data_preprocessing(
                    input_file=os.path.join(STORE_RAW_DATA_PATH, f"MERGE_TABLE_STORE_{store}.csv"),
                    output_file=os.path.join(STORE_PROCESSED_DATA_PATH, f"STORE_{store}.csv")
                )
                df_tmp = pd.read_csv( os.path.join(STORE_PROCESSED_DATA_PATH, f"STORE_{store}.csv") )
                if products_dict[store] != 'all':
                    df_tmp = df_tmp[df_tmp.product_id.isin(products_dict[store])].reset_index(drop=True)
                self.stores_data = self.stores_data.append( df_tmp )
                del df_tmp
        self.stores_data.curr_date = pd.to_datetime(self.stores_data.curr_date)
        self.stores_data.loc[self.stores_data.stock < 0, 'stock'] = 0

        # 2 - Восстановление спроса
        self.demand_restoration_type = demand_restoration_type

        if demand_restoration_type == 'dummy':
            self.stores_data = _dummy_demand_restoration(self.stores_data)
        elif demand_restoration_type == 'promo':
            for store in self.stores_data.store_id.unique():
                for product in self.stores_data[self.stores_data.store_id == store].product_id.unique():
                    self.stores_data = restore_demand(self.stores_data, store, product, type='promo')
        elif demand_restoration_type == 'window':
            for store in self.stores_data.store_id.unique():
                for product in self.stores_data[self.stores_data.store_id == store].product_id.unique():
                    self.stores_data = restore_demand(self.stores_data, store, product, type='window')

        self.stores_data.demand = self.stores_data.apply(
            lambda x: x.demand if (x.s_qty == x.stock) or (x.stock <= 0) else x.s_qty,
            axis=1
        )

        # 3 - Инициализация данных среды
        self.environment_data = dict()
        for store in self.stores_data.store_id.unique().tolist():
            self.environment_data[store] = dict()
            for product in self.stores_data[self.stores_data.store_id == store].product_id.unique().tolist():
                self.environment_data[store][product] = {
                    'stock': None,
                    'reward_log': list(),
                    'policy_log': list(),
                    'order_queue': [0] * 100
                }

        # 4 - Инициализация временных рамок
        self.start_date = self.stores_data.curr_date.min()
        self.finish_date = self.stores_data.curr_date.max()

        # 5 - Инициализация дискретного action_space
        self._generate_action_space()

        # 6 - Загрузка выбранной функции восстановления спроса

        self.reward_apply_function = reward_apply_function

        # 7 - Загрузка выбранной функции формирования заказа

        self.order_apply_function = order_apply_function


    def _generate_action_space(self):
        """Инициализирует дискретный action_space
        Сохраняет полученный список возможных action в self.action_space

        Делает:
            * Считает список всех возможных значений OUL (максимальный demand + 1)
            * Считает список всех возможных значений ROL (максимальный demand)
            * Сохраняет все пары вида (ROL, OUL), где ROL < OUL

        Agrs:
            None

        Returns:
            None
        """

        _maxDemand = int(
            max(
                self.stores_data.s_qty.max(),
                self.stores_data.demand.max()
            )
        )

        OUL = [i for i in range(_maxDemand + 1)]
        ROL = [i for i in range(_maxDemand)]

        _actions = []
        for prod in list( product(ROL, OUL) ):
            if prod[0] < prod[1]:
                _actions.append(prod)

        self.action_space = _actions


    def step(self, action):
        """Осуществляет один шаг среды

        Делает:
            * Получает уникальные пары shop/sku
            * Расчитывает продажи
            * Пересчитывает stock
            * Расчитывает projected_stock
            * Исходя из полученной от агента policy рассчитывает recomended_order
            * Исходя из recomended_order рассчитывает order
            * Расчитывает reward
            * Добавляет reward и выбранную агентом policy в лог
            * Исходя из expected_lead_time рассчитывает lead_time
            * Добавляет order в очередь на доставку
            * Добаляет 1 день к current_date
            * Расчитывает новый observation и reward

        Args:
            action:
                [dict] Словарь со всеми действиями, которые делает агент на этом шаге
                    {
                        store_id::int: {
                            product_id::int: (ROL::int, OUL::int)
                        }
                    }

        Returns:
            observation:
                [dict] словарь с новым наблюдением
                    {
                        store_id::int: {
                            product_id::int: {
                                'date': [datetime.date] Дата (мы смотрим на конец дня)
                                'stock': [int] Текущие остатки товара в магазине (мы смотрим на конец дня)
                                'demand': [int] Спрос за день (мы смотрим на конец дня)
                                'sales': [int] Продажи за день (мы смотрим на конец дня)
                                'flag_promo': [bool] Флаг того, было ли на этот товар промо-предложение
                                'mply_qty': [int] Кратность заказа
                                'lead_time': [int] Ожидаемое время доставки заказа
                                'batch_size': [int] Количество товара в одной упаковке
                                'service_level': [float] Значение уровня сервиса
                            }
                        }
                    }
            reward:
                [dict] словарь с наградами за текущую итерацию
                    {
                        store_id::int: {
                            product_id::int: reward::float
                        }
                    }
            is_done:
                [bool] флаг того, что цикл закончился
            metadata:
                [dict] какие-нибудь произвольные метаданные; по-умолчанию - {}
        """

        # 1 - Получить уникальные пары shop/sku
        df_currentDay = self.stores_data[self.stores_data.curr_date == self.current_date]

        for index, row in df_currentDay.iterrows():
            df_currentDay.loc[index, 'stock'] = int( self.environment_data[row.store_id][row.product_id]['stock'] )

        # 2 - Расчитать продажи
        for index, row in df_currentDay.iterrows():
            df_currentDay.loc[index, 'sales'] = min(
                row.demand,
                self.environment_data[row.store_id][row.product_id]['stock'] +\
                self.environment_data[row.store_id][row.product_id]['order_queue'][0]
            )

        # 3 - Пересчитать stock
        df_currentDay['stock'] -= df_currentDay['sales']

        for index, row in df_currentDay.iterrows():
            df_currentDay.loc[index, 'stock'] += self.environment_data[row.store_id][row.product_id]['order_queue'].pop(0)
            self.environment_data[row.store_id][row.product_id]['order_queue'].append(0)

        for index, row in df_currentDay.iterrows():
            self.environment_data[row.store_id][row.product_id]['stock'] = row.stock

        # 4 - Расчитать projected_stock
        for index, row in df_currentDay.iterrows():
            # 0 - сток, который приедет завтра
            # int(row.lead_time) - сток, который приедет в тот же день, как если мы закажем сегодня
            df_currentDay.loc[index, 'projected_stock'] = sum(self.environment_data[row.store_id][row.product_id]['order_queue'][:int(row.lead_time)])

        # 5 - Расчитать recomended_order
        for index, row in df_currentDay.iterrows():
            if row.stock <= action[row.store_id][row.product_id][0]:
                df_currentDay.loc[index, 'recommended_order'] = max(
                    # ИСПРАВИТЬ
                    # action[row.store_id][row.product_id][1] - row.stock
                    action[row.store_id][row.product_id][1] - row.stock - self.environment_data[row.store_id][row.product_id]['order_queue'][0],
                    0
                )
            else:
                df_currentDay.loc[index, 'recommended_order'] = 0

        # 6 - Расчитать order
        df_currentDay['order'] = df_currentDay.apply(self.order_apply_function, axis=1)

        # 7 - Расчитать reward
        df_currentDay['reward'] = df_currentDay.apply(self.reward_apply_function, axis=1)

        # 8 - Добавляем reward и policy в лог
        for index, row in df_currentDay.iterrows():
            self.environment_data[row.store_id][row.product_id]['reward_log'].append( row.reward )
            self.environment_data[row.store_id][row.product_id]['policy_log'].append( action[row.store_id][row.product_id] )

        # 9 - Расчитать lead time
        df_currentDay['fact_lead_time'] = df_currentDay['lead_time'].apply(_dummy_lead_time_calculation)

        # 10 - Добавить order в очередь
        for index, row in df_currentDay.iterrows():
            self.environment_data[row.store_id][row.product_id]['order_queue'][int(row.fact_lead_time) - 1] = row.order

        # 11 - Добавить 1 день
        self.current_date += pd.DateOffset(1)

        # 12 - Новый observation
        observation = dict()

        df_obs = self.stores_data[self.stores_data.curr_date == self.current_date]

        for store in df_obs.store_id.unique().tolist():
            observation[store] = dict()

        for index, row in df_obs.iterrows():
            if self.environment_data[row.store_id][row.product_id]['stock'] is None:
                self.environment_data[row.store_id][row.product_id]['stock'] = int( row.stock )

            observation[row.store_id][row.product_id] = {
                'date': self.current_date,
                'stock': self.environment_data[row.store_id][row.product_id]['stock'],
                'demand': row.demand,
                'sales': min(
                    row.demand,
                    self.environment_data[row.store_id][row.product_id]['stock'] +\
                    self.environment_data[row.store_id][row.product_id]['order_queue'][0]
                ),
                'flag_promo': row.flg_spromo,
                'mply_qty': row.mply_qty,
                'lead_time': row.lead_time,
                'batch_size': row.batch_size,
                'service_level': row.service_level
            }

        # 13 - Словарь наград
        rewards = dict()

        for index, row in df_currentDay.iterrows():
            rewards[row.store_id] = dict()

        for index, row in df_currentDay.iterrows():
            rewards[row.store_id][row.product_id] = row.reward

        is_done = self.current_date == self.finish_date

        return observation, rewards, is_done, {}


    def reset(self):
        """Сброс среды к начальному состоянию
        Возвращает первый observation

        Args:
            None

        Returns:
            observation:
                [dict] словарь с новым наблюдением
                    {
                        store_id::int: {
                            product_id::int: {
                                'date': [datetime.date] Дата (мы смотрим на конец дня)
                                'stock': [int] Текущие остатки товара в магазине (мы смотрим на конец дня)
                                'demand': [int] Спрос за день (мы смотрим на конец дня)
                                'sales': [int] Продажи за день (мы смотрим на конец дня)
                                'flag_promo': [bool] Флаг того, было ли на этот товар промо-предложение
                                'mply_qty': [int] Кратность заказа
                                'lead_time': [int] Ожидаемое время доставки заказа
                                'batch_size': [int] Количество товара в одной упаковке
                                'service_level': [float] Значение уровня сервиса
                            }
                        }
                    }
            reward:
                [dict] словарь с наградами за текущую итерацию
                    {
                        store_id::int: {
                            product_id::int: reward::float
                        }
                    }
            is_done:
                [bool] флаг того, что цикл закончился
            metadata:
                [dict] какие-нибудь произвольные метаданные; по-умолчанию - {}
        """

        # 1 - Возвращаем счетчик дней в начало
        self.current_date = self.start_date

        # 2 - Восстановление спроса
        if self.demand_restoration_type == 'dummy':
            self.stores_data = _dummy_demand_restoration(self.stores_data)
        elif self.demand_restoration_type == 'promo':
            for store in self.stores_data.store_id.unique():
                for product in self.stores_data[self.stores_data.store_id == store].product_id.unique():
                    self.stores_data = restore_demand(self.stores_data, store, product, type='promo')
        elif self.demand_restoration_type == 'window':
            for store in self.stores_data.store_id.unique():
                for product in self.stores_data[self.stores_data.store_id == store].product_id.unique():
                    self.stores_data = restore_demand(self.stores_data, store, product, type='window')

        self.stores_data.demand = self.stores_data.apply(
            lambda x: x.demand if (x.s_qty == x.stock) or (x.stock <= 0) else x.s_qty,
            axis=1
        )

        # 2 - Расчет первого observation
        observation = dict()

        df_currentDay = self.stores_data[self.stores_data.curr_date == self.current_date]
        for store in df_currentDay.store_id.unique().tolist():
            observation[store] = dict()

        for index, row in df_currentDay.iterrows():
            if self.environment_data[row.store_id][row.product_id]['stock'] is None:
                self.environment_data[row.store_id][row.product_id]['stock'] = int( row.stock )

            observation[row.store_id][row.product_id] = {
                'date': self.current_date,
                'stock': self.environment_data[row.store_id][row.product_id]['stock'],
                'demand': row.demand,
                'sales': min(
                    row.demand,
                    self.environment_data[row.store_id][row.product_id]['stock'] +\
                    self.environment_data[row.store_id][row.product_id]['order_queue'][0]
                ),
                'flag_promo': row.flg_spromo,
                'mply_qty': row.mply_qty,
                'lead_time': row.lead_time,
                'batch_size': row.batch_size,
                'service_level': row.service_level
            }

        # 3 - Дефолтный словарь наград

        rewards = dict()
        for index, row in df_currentDay.iterrows():
            rewards[row.store_id] = dict()
        for index, row in df_currentDay.iterrows():
            rewards[row.store_id][row.product_id] = 0

        return observation, rewards, False, {}


    def render(self, mode='human', show_table=True):
        """Вывод информации о текущем состоянии среды
        Если show_table=True - выводит информацию о текущем состоянии среды в виде таблицы в консоль
        Иначе - просто печатает информацию о том, что выполнение одного шага завершено

        Args:
            mode:
                [string] human (default для OpenAI Gym)
            show_table:
                [bool] Флаг о том, нужно ли выводить полный вывод

        Returns:
            None
        """

        date_range = pd.date_range(self.start_date, self.finish_date).tolist()
        print(f"{self.current_date.strftime('%d.%m.%Y')} (Day {date_range.index(self.current_date) + 1} of {len(date_range)})")
        if show_table:
            print("".join(['-'] * 103))
            print('{:10s} | {:10s} | {:13s} | {:10s} | {:11s} | {:11s} | {:20s}'.format('Store', 'SKU', 'Current Stock', 'Next Order', 'Last Policy', 'Last Reward', 'Sum Reward'))
            print("".join(['-'] * 103))
            for store in self.environment_data.keys():
                for product in self.environment_data[store].keys():
                    print(
                        '{:10d} | {:10d} | {:13s} | {:10d} | {:11s} | {:11.2f} | {:20.2f}'.format(
                            int(store),
                            int(product),
                            'N/A' if self.environment_data[store][product]['stock'] is None else str(int(self.environment_data[store][product]['stock'])),
                            int(self.environment_data[store][product]['order_queue'][0]),
                            '-' if not len(self.environment_data[store][product]['policy_log']) else str(self.environment_data[store][product]['policy_log'][-1]),
                            0 if not len(self.environment_data[store][product]['reward_log']) else self.environment_data[store][product]['reward_log'][-1],
                            0 if not len(self.environment_data[store][product]['reward_log']) else sum(self.environment_data[store][product]['reward_log'])
                        )
                    )
        print()


    def close(self):
        """Выключение среды
        Метод есть в документации OpenAI Gym, но реализован даже не во всех их средах.

        Args:
            None

        Returns:
            None
        """

        pass


    def compute_baseline_realworld(self):
        """Считает награду так, как её бы получил фактический агент, данные которого мы используем как входные

        Args:
            None

        Returns:
            df_result:
                [pandas.DataFrame] Информация о награде фактического агента за весь период; Состоит из полей:
                    'store_id' [int] ID магазина
                    'product_id' [int] ID товара
                    'first_appeared' [datetime.date] Дата первого появления товара в магазине
                    'last_appeared' [datetime.date] Дата последнего появления товара в магазине
                    'sum_reward' [float] Суммарный reward за время, когда товар был в ассортименте магазина
                    'log_reward' [float] Лог reward'а по дням
        """

        df_tmp = self.stores_data.copy()
        df_tmp['reward'] = df_tmp.apply(_dummy_apply_reward_calculation_baseline_v2, axis=1)

        df_result = df_tmp.groupby(['store_id', 'product_id']).agg(
            {
                'curr_date': ['min', 'max'],
                'reward': ['sum', list]
            }
        ).reset_index()
        df_result.columns = ['store_id', 'product_id', 'first_appeared', 'last_appeared', 'sum_reward', 'log_reward']

        return df_result
