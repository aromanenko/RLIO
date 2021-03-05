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
    df_output["demand"] = df_output["s_qty"]
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
        df_currentDay = self.stores_data[self.stores_data.curr_date == self.current_date]

        # Расчет sales
        sales = dict()
        for index, row in df_currentDay.iterrows():
            sales[row.store_id][row.product_id] = min(
                row.demand,
                self.environment_data[row.store_id][row.product_id]['stock'] +\
                self.environment_data[row.store_id][row.product_id]['order_queue'][0]
            )

        # Пересчет стоков



        # 1 - Расчет награды за прошлое действие
        reward = calculate_reward()

        # 2 - Расчет подкапотных действий
        pass

        # 3 - Расчет нового observation
        pass

        # 4 - Расчет флага is_done
        if self.curr_date == self.finish_date:
            is_done = True
        else:
            is_done = False
            self.curr_date += pd.DateOffset(1)

        return observation, reward, is_done, {}


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
                self.environment_data[row.store_id][row.product_id]['stock'] = row.stock

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

        return observation


    def render(self, mode='human'):
        """
        Вывод информации о текущем состоянии среды
        """
        date_range = pd.date_range(self.start_date, self.finish_date).tolist()
        print(f"{self.current_date.strftime('%d.%m.%Y')} (Day {date_range.index(self.current_date)} of {len(date_range)})")
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
                        '-' if not len(self.environment_data[store][product]['policy_log']) else self.environment_data[store][product]['policy_log'][-1],
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
