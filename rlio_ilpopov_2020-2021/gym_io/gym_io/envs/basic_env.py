############## GYM IMPORTS ##############
import gym
from gym.utils import seeding
from gym import error, spaces, utils
############## ENV IMPORTS ##############
import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy import stats
from itertools import product
################ SETTINGS ################
# from settings import ALPHA_DEMAND, ALPHA_STEP, MAX_STEPS, PROBA_STEP
##########################################

ALPHA_DEMAND = 1
ALPHA_STEP = 0.01
MAX_STEPS = 100
PROBA_STEP = 0.7

class IOEnv(gym.Env):
    """
    by: @mgcrp

    Базовая версия Open AI Gym Environment для проекта RLIO
    """
    metadata = {'render.modes': ['human']}

    def init(self, data_path, sku=None, loc=None):
        """
        Инициализация среды

        [Input]: { 'ROL': int, 'OUL': int }
        [Output]:
            new_observation
            reward
            is_done
            metadata
        """

        print('Loading data...')
        df = pd.read_csv(data_path, sep=';')
        df.curr_date = pd.to_datetime(df.curr_date)
        print('Data successfuly loaded!')

        if sku is None:
            _sku_to_process = df.product_id.unique()
        else:
            _sku_to_process = sku

        if loc is None:
            _loc_to_process = df.store_id.unique()
        else:
            _loc_to_process = loc

        print('Initializing lambdas...')
        self.lambdas = self._calculate_lambdas(df, sku=_sku_to_process, loc=_loc_to_process)
        print('Lambdas successfuly initialized!')

        print('Initializing initial states...')
        self.initial_states = self._initialize_states(df, sku=_sku_to_process, loc=_loc_to_process)
        print('Lambdas successfuly initialized!')

        _obs = list()

        print('Initializing observations...')
        _cnt = 0
        for _, state in tqdm( self.initial_states.iterrows(), total=len(self.initial_states) ):
            _demand = self._calculate_demand(
                df,
                sku=state.sku,
                loc=state.location,
                lambda_value=self.lambdas.loc[(self.lambdas.location == state.location) & (self.lambdas.sku == state.sku), 'lambda']
            )
            _stock = state.stock
            _order = state.order
            _actions = self._initialize_actions(_demand)

            _obs.append(
                {
                    'state_id': _cnt,
                    'demand': _demand,
                    'stock': _stock,
                    'order': _order,
                    'actions': _actions
                }
            )
            _cnt += 1
        print('Observations successfuly initialized!')

        self.df = df
        self.current_step = 1

        _tmp = pd.DataFrame(_obs).drop(columns=['actions'])
        _tmp['step'] = 1
        self.environment = _tmp

        print(self.environment)

        _reward = {i: 0 for i in self.environment.state_id}

        return _obs, _reward, False, {}

    def step(self, action):
        """
        Шаг среды

        [Input]: { 'initial': ('ROL': int, 'OUL': int) }
        [Output]:
            new_observation
            reward
            is_done
            metadata
        """

        if self.current_step < MAX_STEPS:
            pass

            self.current_step += 1


    def reset(self):
        """
        Перезапуск среды
        """
        pass

    def render(self, mode='human', close=False):
        """
        Вывод информации о состоянии среды и агента
        """
        pass

    def _initialize_actions(self, demand):
        _OUL = [i for i in range(demand + 1)]
        _ROL = [i for i in range(max(_OUL))]

        _prods = []
        for prod in list(product(_OUL, _ROL)):
            if prod[0] > prod[1]:
                _prods.append(prod)
        return _prods

    def _initialize_states(self, df, sku=None, loc=None):
        _states = pd.DataFrame(columns=['location', 'sku', 'sales', 'stock', 'sl', 'order'])

        if sku is None:
            _sku_to_process = df.product_id.unique()
        else:
            _sku_to_process = sku

        if loc is None:
            _loc_to_process = df.store_id.unique()
        else:
            _loc_to_process = loc

        for store_id in _loc_to_process:
            for product_id in _sku_to_process:
                _tmp = df[
                    (df['product_id'] == product_id) &
                    (df['store_id'] == store_id) &
                    (pd.notnull(df.s_qty)) &
                    (pd.notnull(df.sl))
                ].reset_index().groupby('curr_date').agg({'stock':np.max, 's_qty':np.max, 'sl':np.max})
                _tmp = _tmp.reindex(pd.date_range(np.min(_tmp.index), np.max(_tmp.index))).fillna(method='ffill')
                _tmp.rename(columns={'s_qty': 'sales'}, inplace=True)
                _tmp['order'] = 0
                _tmp['location'] = store_id
                _tmp['sku'] = product_id
                _states = _states.append(_tmp)

        return _states

    def _calculate_lambdas(self, df, sku=None, loc=None):
        _lambdas = pd.DataFrame(columns=['location', 'sku', 'lambda'])

        if sku is None:
            _sku_to_process = df.product_id.unique()
        else:
            _sku_to_process = sku

        if loc is None:
            _loc_to_process = df.store_id.unique()
        else:
            _loc_to_process = loc

        for store_id in _loc_to_process:
            for product_id in _sku_to_process:
                _df_stocks = df[
                    (df['product_id'] == product_id) &
                    (df['store_id'] == store_id) &
                    (pd.notnull(df.sl))
                ].reset_index().groupby('curr_date').agg({'stock':np.max})
                _df_stocks = _df_stocks.reindex(pd.date_range(np.min(_df_stocks.index), np.max(_df_stocks.index))).fillna(method='ffill')

                _df_sales = df[(df['product_id'] == product_id)].set_index('curr_date')
                _df_stocks = _df_sales[['s_qty']].merge(_df_stocks, how='right', left_index=True, right_index=True)

                _sales_start = _df_stocks[_df_stocks.max(axis=1) > 0].index[0]
                _sales_end = _df_stocks[_df_stocks.max(axis=1) > 0].index[-1]
                _df_stocks = _df_stocks[(_df_stocks.index >= _sales_start) & (_df_stocks.index <= _sales_end)]

                _zero_idx = (_df_stocks['stock'] == 0) & (_df_stocks['s_qty'] == 0)
                _greater_idx = (_df_stocks['stock'] <= _df_stocks['s_qty'])

                _df_stocks['weights'] = [1 for x in _df_stocks.index]
                _df_stocks['scalar'] = _df_stocks['s_qty'][(~_zero_idx)] * _df_stocks['weights'][(~_zero_idx)]

                _lambda = _df_stocks['scalar'].sum() / (
                    _df_stocks['weights'][(~_zero_idx) & (~_greater_idx)].sum() +
                    ALPHA_DEMAND * _df_stocks['weights'][(~_zero_idx) & _greater_idx].sum()
                )
                _lambdas = _lambdas.append({'location': store_id, 'sku': product_id, 'lambda': _lambda}, ignore_index=True)

        return _lambdas

    def _calculate_demand(self, df, sku, loc, lambda_value):
        _max_sales = df[(df['product_id'] == sku) & (df['store_id'] == loc)].s_qty.max()
        return int( min(stats.poisson.ppf(0.99, lambda_value)[0], _max_sales) )
