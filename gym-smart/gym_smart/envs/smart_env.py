import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import pandas as pd
import scipy.stats as stats

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
                state = self.initial_state(shop_id, product_id)
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

        return self

    def load_pairs(self, pairs_filepath):
        """
        Loads data into enviroment, forming a Pandas DataFrame of location-sku
        pairs for SMART-algorithm to work with

        By: @HerrMorozovDmitry

        Parameters
        ----------
        pairs_filepath : string
            A filepath (relative or absolute) to a CSV-file with information on
            location-sku pairs [not_empty_states.csv in example]

        Returns
        -------
        self
        """
        # Load pairs data
        self.pairs_data = pd.read_csv(pairs_filepath, index_col=0)

        return self

    def initial_demand(self, shop_id, product_id, alpha):
        """
        Computes demand for given shop_id and product_id.

        By: @HerrMorozovDmitry

        Parameters
        ----------
        shop_id : int

        product_id : int

        Returns
        -------
        demand_data : {array-like, sparse matrix} of shape (1, n_features)
            Returns the dataframe row with lambda and demand information,
            where n_features is the number of features
            {shop_id, product_id, lambda, demand}.
        """
        LAMBDA = calculate_lambda(shop_id, product_id, alpha=1)
        max_sales = self.ss_data[(self.ss_data['store_id']==shop_id) &
                                 (self.ss_data['product_id']==product_id)]['s_qty'].max()
        demand = max(stats.poisson.ppf(0.99, LAMBDA), max_sales, 1)
        demand_data = demand_data.append({'shop_id': shop_id, 'product_id': product_id, 'lambda': LAMBDA, 'demand': demand}, ignore_index=True)
        demand_data[['shop_id', 'product_id', 'demand']] = demand_data[['shop_id', 'product_id', 'demand']].astype('int')

        return demand_data
    
    def calculate_lambda(self, shop_id, product_id, alpha):
        """
        Computes lambda for demand recovery

        By: @HerrMorozovDmitry

        Parameters
        ----------
        shop_id : int

        product_id : int

        Returns
        -------
        lambda : float param
        """
        iv_ts = self.ss_data.reset_index().groupby('Timestamp').agg({'stock':np.max})
        iv_ts = iv_ts.reindex(pd.date_range(np.min(iv_ts.index), np.max(iv_ts.index))).fillna(method='ffill')

        sales_ts = self.ss_data[(self.ss_data['product_id'] == product_id) & (self.ss_data['store_id'] == shop_id)]
        iv_sales = sales_ts[['s_qty']].merge(iv_ts, how='right', left_index=True, right_index=True)

        positive_iv_sales = iv_sales[iv_sales.max(axis=1) > 0]
        life_start_date = positive_iv_sales.index[0]
        life_end_date = positive_iv_sales.index[-1]
        iv_sales = iv_sales[(iv_sales.index >= life_start_date) & (iv_sales.index <= life_end_date)]

        zero_idx = (iv_sales['stock'] == 0) & (iv_sales['s_qty'] == 0)
        sales_equal_inv_idx = (iv_sales['stock'] == iv_sales['s_qty'])
        sales_greater_i_idx = (iv_sales['stock'] <= iv_sales['s_qty'])

        demand_data = pd.DataFrame(columns=['shop_id', 'product_id', 'lambda', 'demand'])

        iv_sales['weights'] = [1 for x in iv_sales.index]
        iv_sales['scalar'] = iv_sales['s_qty'][(~zero_idx)] * iv_sales['weights'][(~zero_idx)]
        sum_k = iv_sales['scalar'].sum()
        n_k_less_m = iv_sales['weights'][(~zero_idx) & (~sales_greater_i_idx)].sum()
        n_k_equal_m = iv_sales['weights'][(~zero_idx) & sales_greater_i_idx].sum()
        
        LAMBDA = sum_k / (n_k_less_m + alpha * n_k_equal_m)
        return LAMBDA

    def calculate_demand(self):
        """
        From self.pairs_data computes dataframe with information about
        calculated lambda and demand for all pairs location-sku.

        By: @HerrMorozovDmitry

        Returns
        -------
        demand_data : {array-like, sparse matrix} of shape (n_samples, n_features)
            Returns the dataframe with demand information,
            where n_samples is the number of samples and
            n_features is the number of features
            {shop_id, product_id, lambda, demand}.
        """
        demand_data = pd.DataFrame(columns=['shop_id', 'product_id', 'lambda', 'demand'])
        for index, row in self.pairs_data.iterrows():
            shop_id = row['shop_id']
            product_id = row['product_id']
            demand_data = demand_data.append(self.initial_demand(shop_id, product_id), ignore_index=True)

        return demand_data

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
