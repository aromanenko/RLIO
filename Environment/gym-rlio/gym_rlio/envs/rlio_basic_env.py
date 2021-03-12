############## ENV IMPORTS ##############
import os
import pandas as pd
############## GYM IMPORTS ##############
import gym
from gym.utils import seeding
from gym import error, spaces, utils
################ SETTINGS ################
ECHELON_DATA_PATH = "/Users/mgcrp/Desktop/RLIO Environment/echelon_processed_data"
STORE_RAW_DATA_PATH = "/Users/mgcrp/Desktop/RLIO Environment/store_raw_data"
STORE_PROCESSED_DATA_PATH = "/Users/mgcrp/Desktop/RLIO Environment/store_processed_data"
############ DISABLE WARNINGS ############
import warnings
warnings.filterwarnings('ignore')
##########################################


def dummy_demand_restoration(df_input):
    """
    Заглушка для функции восстановления спроса

    [Input]:
        df_input
            Pandas DataFrame of columns	[product_id, store_id, curr_date, s_qty, flg_spromo, stock, mply_qty,
            lead_time, batch_size, service_level]
            Входные данные о продажах
    [Output]:
        df_output
            Pandas DataFrame of columns	[product_id, store_id, curr_date, s_qty, flg_spromo, stock, mply_qty,
            lead_time, batch_size, service_level, demand]
            Входные данные о продажах + восстановленный спрос
    """
    df_output = df_input.copy(deep=True)
    df_output["demand"] = df_output["s_qty"].astype('int')
    return df_output


def dummy_store_data_preprocessing(input, output):
    """
    Заглушка для функции нормализации данных магазина

    [Input]:
        input
            String
            Путь к сырому файлу
        output
            String
            Путь для записи обработанного файла
    [Output]: None
    """
    pass


def dummy_apply_reward_calculation(row):
    """
    Базовый вариант расчета reward
    Для применения в apply
    """
    return row.sales - row.stock * ( (1 - row.service_level) / (row.service_level) )


def dummy_order_calculation(recommended_order):
    """
    Заглушка для расчета order
    Сюда можно придумать функцию, вносящую хаос в объем заказа
    """
    return recommended_order

def dummy_lead_time_calculation(expected_lead_time):
    """
    Заглушка для расчета lead_time
    Сюда можно придумать функцию, вносящую хаос в сроки доставки
    """
    return expected_lead_time


class RlioBasicEnv(gym.Env):
    metadata = {'render.modes': ['human']}


    def __init__(self):
        """
        Стандартная инициализация среды
        """
        self.stores_data = None
        self.environment_data = None

        self.start_date = None
        self.finish_date = None
        self.current_date = None


    def load_data(self, stores, products=None):
        """
        Загрузка данных в среду

        [Input]:
            stores
                list of store_ids (int)
                Список магазинов, по которым необходимо загрузить данные
            products
                list of product_ids (int)
                Список товаров, которые надо загрузить. По умолчанию, None - загрузить все доступные
        [Output]: None
        """
        # 1 - Загрузка данных
        self.stores_data = pd.DataFrame()
        for store in stores:
            if os.path.exists( os.path.join(STORE_PROCESSED_DATA_PATH, f"STORE_{store}.csv") ):
                self.stores_data = self.stores_data.append(
                    pd.read_csv( os.path.join(STORE_PROCESSED_DATA_PATH, f"STORE_{store}.csv") )
                )
            else:
                dummy_store_data_preprocessing(
                    input=os.path.join(STORE_RAW_DATA_PATH, f"MERGE_TABLE_STORE_{store}.csv"),
                    output=os.path.join(STORE_PROCESSED_DATA_PATH, f"STORE_{store}.csv")
                )
                self.stores_data = self.stores_data.append(
                    pd.read_csv( os.path.join(STORE_PROCESSED_DATA_PATH, f"STORE_{store}.csv") )
                )
        self.stores_data.curr_date = pd.to_datetime(self.stores_data.curr_date)

        if products is not None:
            self.stores_data = self.stores_data[self.stores_data.product_id.isin(products)].reset_index(drop=True)

        # 2 - Восстановление спроса
        self.stores_data = dummy_demand_restoration(self.stores_data)

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


    def step(self, action):
        """
        Шаг среды

        [Input]: {
            'store_id': {
                'product_id': ('ROL': int, 'OUL': int)
                }
            }
        [Output]:
            observation
            reward
            is_done
            metadata
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

        # 3 - Расчитать reward
        df_currentDay['reward'] = df_currentDay.apply(dummy_apply_reward_calculation, axis=1)

        # 4 - Добавляем reward и policy в лог
        for index, row in df_currentDay.iterrows():
            self.environment_data[row.store_id][row.product_id]['reward_log'].append( row.reward )
            self.environment_data[row.store_id][row.product_id]['policy_log'].append( action[row.store_id][row.product_id] )

        # 5 - Расчитать recomended_order
        for index, row in df_currentDay.iterrows():
            if row.stock <= action[row.store_id][row.product_id][0]:
                df_currentDay.loc[index, 'recommended_order'] = max(
                    action[row.store_id][row.product_id][1] - row.stock - self.environment_data[row.store_id][row.product_id]['order_queue'][0],
                    0
                )
            else:
                df_currentDay.loc[index, 'recommended_order'] = 0

        # 6 - Расчитать order
        df_currentDay['order'] = df_currentDay['recommended_order'].apply(dummy_order_calculation)

        # 7 - Расчитать lead time
        df_currentDay['fact_lead_time'] = df_currentDay['lead_time'].apply(dummy_lead_time_calculation)

        # 8 - Пересчитать stock
        df_currentDay['stock'] -= df_currentDay['sales']

        for index, row in df_currentDay.iterrows():
            df_currentDay.loc[index, 'stock'] += self.environment_data[row.store_id][row.product_id]['order_queue'].pop(0)
            self.environment_data[row.store_id][row.product_id]['order_queue'].append(0)

        for index, row in df_currentDay.iterrows():
            self.environment_data[row.store_id][row.product_id]['stock'] = row.stock

        # 9 - Добавить order в очередь
        for index, row in df_currentDay.iterrows():
            self.environment_data[row.store_id][row.product_id]['order_queue'][int(row.fact_lead_time) - 1] = row.order

        # 10 - Добавить 1 день
        self.current_date += pd.DateOffset(1)

        # while self.current_date not in self.stores_data.curr_date.tolist():
        #     date_range = pd.date_range(self.start_date, self.finish_date).tolist()
        #     print(f"{self.current_date.strftime('%d.%m.%Y')} (Day {date_range.index(self.current_date)} of {len(date_range)}) does not exit in store data...")
        #     print("Skipping that day...")
        #     print()
        #     self.current_date += pd.DateOffset(1)

        # 11 - Новый observation
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
                'batch_size': row.batch_size
            }

        # 12 - Словарь наград
        rewards = dict()

        for index, row in df_currentDay.iterrows():
            rewards[row.store_id] = dict()

        for index, row in df_currentDay.iterrows():
            rewards[row.store_id][row.product_id] = row.reward

        is_done = self.current_date == self.finish_date

        return observation, rewards, is_done, {}


    def reset(self):
        """
        Сброс среды к начальному состоянию
        Возвращает первый observation

        [Input]: None
        [Output]:
            observation
        """
        # 1 - Возвращаем счетчик дней в начало
        self.current_date = self.start_date

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
                'batch_size': row.batch_size
            }

        # 3 - Дефолтный словарь наград

        rewards = dict()
        for index, row in df_currentDay.iterrows():
            rewards[row.store_id] = dict()
        for index, row in df_currentDay.iterrows():
            rewards[row.store_id][row.product_id] = 0

        return observation, rewards, False, {}


    def render(self, mode='human'):
        """
        Вывод информации о текущем состоянии среды
        """
        date_range = pd.date_range(self.start_date, self.finish_date).tolist()
        print(f"{self.current_date.strftime('%d.%m.%Y')} (Day {date_range.index(self.current_date) + 1} of {len(date_range)})")
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
        """
        Выключение среды
        Метод есть в документации OpenAI Gym, но реализован даже не во всех их средах.
        """
        pass
