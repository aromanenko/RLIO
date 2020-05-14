import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import pandas as pd

class SmartEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    # ss_data - pandas.DataFrame with information on sales and stocks of columns
    #   [product_id, store_id, curr_date, s_qty, flg_spromo, stock, Timestamp]
    ss_data = None
    # sl_data - pandas.DataFrame with information on service level of columns
    #   [date_from, date_to, product_ids, location_ids, value]
    sl_data = None
    # demand_data - pandas.DataFrame with information on demand of columns
    #   [shop_id, product_id, lambda, demand]
    demand_data = None
    # pairs_data - pandas.DataFrame with location-sku pairs for SMART-algorithm to work with of columns
    #   [shop_id, product_id]
    pairs_data = None

    def initial_state(self, shop_id, product_id):
        """
        Initializes a dataframe of initial states for a given location-sku pair.

        By: @HerrMorozovDmitry & @mgcrp

        Parameters
        ----------
        shop_id : int

        product_id : int

        Returns
        -------
        state : {array-like, sparse matrix} of shape (n_samples, n_features)
            Returns the dataframe of the initial states,
            where n_samples is the number of samples and
            n_features is the number of features
            {location, sku, sales, stock, sl, order}.
        """
        # Initializing output DataFrame
        state = pd.DataFrame(columns=['location', 'sku', 'sales', 'stock', 'sl', 'order'])
        # A local copy of a part of ss_data satisfying conditions
        store_sales_t = (
            self.ss_data[(self.ss_data['store_id'] == shop_id) & (self.ss_data['product_id'] == product_id)]
            .fillna(0)
            .drop(columns=['curr_date', 'flg_spromo'])
        )
        # A local copy of a part of sl_data satisfying conditions
        sl_data_t = self.sl_data.set_index(['date_from'], drop=True)
        sl_data_t = sl_data_t[(sl_data_t['location_ids'] == shop_id) & (sl_data_t['product_ids'] == product_id)]

        if not sl_data_t.empty:
            # Adding skipped days
            all_days = pd.date_range(start=sl_data_t.index.min(), end=sl_data_t['date_to'].max())
            sl_data_t.index = pd.DatetimeIndex(sl_data_t.index)
            sl_data_t = sl_data_t.reindex(all_days, method='ffill')
            sl_data_t = sl_data_t.drop(columns=['date_to'])
            # Calculate state
            state_t = pd.merge(
                left=store_sales_t,
                right=sl_data_t['value'],
                how='right',
                left_on=store_sales_t.index,
                right_on=sl_data_t.index
            )
            state_t.rename(columns={'store_id': 'location', 'product_id': 'sku', 's_qty': 'sales', 'value': 'sl'}, inplace=True)
            state_t.drop(columns=['key_0'], inplace=True)
            state_t.drop_duplicates(inplace=True)
            state_t.dropna(inplace=True)
            state_t['order'] = 0
            # Expand state DataFrame
            state = state.append(state_t, ignore_index=True)
        return state

    def calculate_states(self):
        """
        From self.ss_data and self.sl_data computes dataframes of location-sku pairs,
        information about which is available to initialize states
        and information about which is missing.

        By: @HerrMorozovDmitry

        Returns
        -------
        not_empty_states : {array-like, sparse matrix} of shape (n_samples, n_features)
            Returns the dataframe of the not empty states pairs,
            where n_samples is the number of samples and
            n_features is the number of features
            {shop_id, product_id}.

        empty_states : {array-like, sparse matrix} of shape (n_samples, n_features)
            Returns the dataframe of the empty states pairs,
            where n_samples is the number of samples and
            n_features is the number of features
            {shop_id, product_id}.
        """
        # Initializing output DataFrames
        empty_states = pd.DataFrame(columns=['shop_id', 'product_id'])
        not_empty_states = pd.DataFrame(columns=['shop_id', 'product_id'])
        # Forming lists of unique goods and shops for each of input DataFrames
        shops_ss = np.sort(self.ss_data['store_id'].unique())
        products_ss = np.sort(self.ss_data['product_id'].unique())
        shops_sl = np.sort(self.sl_data['location_ids'].unique())
        products_sl = np.sort(self.sl_data['product_ids'].unique())

        for shop_id in [shop for shop in shops_ss if shop in shops_sl]:
            for product_id in [prod for prod in products_ss if prod in products_sl]:
                state = self.initial_state(product_id, shop_id)
                if state.empty:
                    empty_states = empty_states.append(
                        {'shop_id': shop_id, 'product_id': product_id},
                        ignore_index=True
                    )
                else:
                    not_empty_states = not_empty_states.append(
                        {'shop_id': shop_id, 'product_id': product_id},
                        ignore_index=True
                    )
        return not_empty_states, empty_states

    def load_data(self, ss_filepath, sl_filepath, demand_filepath):
        """
        Loads data into enviroment, forming a Pandas DataFrame of location-sku
        pairs for SMART-algorithm to work with

        By: @mgcrp

        Parameters
        ----------
        ss_filepath : string
            A filepath (relative or absolute) to a CSV-file with information on
            sales and stocks [MERGE_TABLE_STORE_4600.csv in example]
        sl_filepath : string
            A filepath (relative or absolute) to a CSV-file with information on
            service level [echelon_1_sl.csv in example]
        demand_filepath : string
            A filepath (relative or absolute) to a CSV-file with information on
            demand [demand_data.csv in example]

        Returns
        -------
        self
        """
        # Load sales and stocks data
        df_ss = pd.read_csv(ss_filepath, sep=';', decimal='.')
        df_ss['Timestamp'] = pd.to_datetime(df_ss['curr_date'], format='%d%b%Y')
        df_ss['curr_date'] = df_ss['Timestamp']
        df_ss = df_ss.set_index(['Timestamp'], drop=True)
        self.ss_data = df_ss
        # Load service level data
        df_sl = pd.read_csv(sl_filepath, sep=';')
        df_sl['date_from'] = pd.to_datetime(df_sl['date_from'], format='%d%b%Y')
        df_sl['date_to'] = pd.to_datetime(df_sl['date_to'], format='%d%b%Y')
        self.sl_data = df_sl
        # Load demand
        df_demand = pd.read_csv(demand_filepath, index_col=0)
        self.demand_data = df_demand
        # Calculate location-sku pairs for SMART-algorithm to work with
        self.pairs_data, _ = self.calculate_states()

        return self

    def __init__(self):
        ss_data = None
        sl_data = None
        pairs_data = None
        demand_data = None

        print('Environment initialized!')
        pass

    def step(self, action):
        print('Step successful!')
        pass

    def reset(self):
        print('Environment reset!')
        pass

    def render(self, mode='human', close=False):
        pass
