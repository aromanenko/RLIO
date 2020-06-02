import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import pandas as pd
import scipy.stats as stats
from itertools import product

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

    state_data = None
    action_data = None
    env_data = None

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

    def product_condition(self, state_range, OUL, ROL):
        """
        Computes a list of tuples consisting of all triples
        (state number, OUL, ROL), where OUL > ROL.

        By: @HerrMorozovDmitry

        Parameters
        ----------
        state_range : array-like
            Array of state indices for which we want to initialize actions.

        OUL : array-like
            Array of Order Upto Level values.

        ROL : array-like
            Array of Re-Order Level values.

        Returns
        -------
        action : array-like
            Returns the list of tuples consisting of all triples.
        """
        prods = []
        for prod in list(product(state_range, OUL, ROL)):
            if prod[1] > prod[2]:
                prods.append(prod)

        return prods

    def initial_action(self, shop_id, product_id, state_range):
        """
        Initializes a dataframe of actions for a given location-sku pair and range of its state indices.

        By: @HerrMorozovDmitry

        Parameters
        ----------
        shop_id : int

        product_id : int

        state_range : array-like
            Array of state indices for which we want to initialize actions.

        Returns
        -------
        action : {array-like, sparse matrix} of shape (n_samples, n_features)
            Returns the dataframe of the initial actions,
            where n_samples is the number of samples and
            n_features is the number of features {location, sku, state, OUL, ROL, R}.
        """
        action = pd.DataFrame(columns=['location', 'sku', 'state', 'OUL', 'ROL'])

        max_demand = int(self.demand_data[(self.demand_data['product_id'] == product_id) & (self.demand_data['shop_id'] == shop_id)].iloc[0].demand)
        OUL = [i for i in range(max_demand + 1)]
        ROL = [i for i in range(max(OUL))]

        action_t = pd.DataFrame(self.product_condition(state_range, OUL, ROL), columns=['state', 'OUL', 'ROL'])
        action = pd.merge(action, action_t, how='outer')
        values = {'location': shop_id, 'sku': product_id}
        action = action.fillna(value=values)
        action['R'] = 0.0

        return action

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

    def calculate_lambda(self, shop_id, product_id, alpha_lambda):
        """
        Computes lambda for demand recovery

        By: @HerrMorozovDmitry

        Parameters
        ----------
        shop_id : int

        product_id : int

        alpha_lambda : float

        Returns
        -------
        lambda : float
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

        LAMBDA = sum_k / (n_k_less_m + alpha_lambda * n_k_equal_m)

        return LAMBDA

    def initial_demand(self, shop_id, product_id, alpha_lambda):
        """
        Computes demand for given shop_id and product_id.

        By: @HerrMorozovDmitry

        Parameters
        ----------
        shop_id : int

        product_id : int

        alpha_lambda : float

        Returns
        -------
        demand_data : {array-like, sparse matrix} of shape (1, n_features)
            Returns the dataframe row with lambda and demand information,
            where n_features is the number of features
            {shop_id, product_id, lambda, demand}.
        """
        LAMBDA = self.calculate_lambda(shop_id, product_id, alpha_lambda)
        max_sales = self.ss_data[(self.ss_data['store_id']==shop_id) &
                                 (self.ss_data['product_id']==product_id)]['s_qty'].max()
        demand_data = pd.DataFrame(columns=['shop_id', 'product_id', 'lambda', 'demand'])
        demand = min(stats.poisson.ppf(0.99, LAMBDA), max_sales)
        demand_data = demand_data.append({'shop_id': shop_id, 'product_id': product_id, 'lambda': LAMBDA, 'demand': demand}, ignore_index=True)
        demand_data[['shop_id', 'product_id', 'demand']] = demand_data[['shop_id', 'product_id', 'demand']].astype('int')

        return demand_data

    def calculate_demand(self, alpha_lambda):
        """
        From self.pairs_data computes dataframe with information about
        calculated lambda and demand for all pairs location-sku.

        By: @HerrMorozovDmitry

        Parameters
        ----------
        alpha_lambda : float

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
            demand_data = demand_data.append(self.initial_demand(shop_id, product_id, alpha_lambda), ignore_index=True)

        return demand_data

    def sales_update(self, data, shop_id, product_id, LT, deviation=0.01):
        """
        Computes sales using information from environment data and self.demand_data.

        By: @timothysenchenko

        Parameters
        ----------
        data : {array-like, sparse matrix} of shape (n_samples, n_features)
            Dataframe with environment information,
            where n_samples is the number of samples and
            n_features is the number of features
            {shop_id, product_id, stock, sales, order}.

        shop_id : int

        product_id : int

        LT : int

        deviation : float

        Returns
        -------
        sales : int
            Returns the value of sales.

        dem : int
            Returns the value of demand
        """
        dem = self.demand_data[(self.demand_data['shop_id'] == shop_id) &
                               (self.demand_data['product_id'] == product_id)].iloc[0].demand
        if deviation != None:
            dev = int((np.random.normal(loc=deviation * dem, size=1)[0]).round())
        else:
            dev = 0
        dem += dev
        sales = min(dem,
                    data.iloc[-1].stock + data.iloc[-LT].order)
        return sales, dem

    def stock_update(self, data, shop_id, product_id, LT, dem):
        """
        Computes stock using information from environment data and self.demand_data.

        By: @timothysenchenko

        Parameters
        ----------
        data : {array-like, sparse matrix} of shape (n_samples, n_features)
            Dataframe with environment information,
            where n_samples is the number of samples and
            n_features is the number of features
            {shop_id, product_id, stock, sales, order}.

        shop_id : int

        product_id : int

        LT : int

        dem : int

        Returns
        -------
        stock : int
            Returns the value of stock.
        """
        stock = max(0, (data.iloc[-1].stock + data.iloc[-LT].order - dem))
        return stock

    def order_update(self, stock_data, shop_id, product_id, OUL, ROL, alpha_order):
        """
        Computes order using information from environment data
        (with binomial distribution with alpha_order param)

        By: @Kirili4ik

        Parameters
        ----------
        stock_data : pd.DataFrame with 'shop_id', 'product_id', 'stock',
                     'sales' and 'order' columns

        shop_id : int

        product_id : int

        OUL : int

        ROL : int

        alpha_order : float

        Returns
        -------
        order : int
            Returns the value of order for given shop and product ids.
        """
        curr_shop_prod = stock_data.iloc[-1]

        if curr_shop_prod['stock'] <= ROL:
            rec_ord_value = OUL - curr_shop_prod['stock'] - curr_shop_prod['order']
            rec_ord_value = max(rec_ord_value, 0)
        else:
            rec_ord_value = 0

        ord_value = np.random.binomial(n=rec_ord_value, p=alpha_order)

        return ord_value

    def reward(self, environment):
        """
        Returns reward of enviroment dataframe.

        By: @sofloud

        Parameters
        ----------
        environment : {array-like, sparse matrix} of shape (n_samples, n_features)
            Dataframe with environment information,
            where n_samples is the number of samples and
            n_features is the number of features
            {shop_id, product_id, stock, sales, order}.

        Returns
        -------
        rew : float
            Reward of the system.
        """
        rew = (environment['sales'] - environment['stock'] * (1 - environment['sl']) / environment['sl']).sum()

        return rew

    def learn_single(self, shop_id, product_id, max_steps, alpha, probability, LT, alpha_order):
        """
        Running the SMART algorithm for one location-sku pair and obtaining summary
        information about all used states, actions with rewards and environment data.

        By: @HerrMorozovDmitry

        Parameters
        ----------
        shop_id : int

        product_id : int

        max_steps : int

        alpha : float

        probability : float

        LT : int

        alpha_order : float

        Returns
        -------
        state_data : {array-like, sparse matrix} of shape (n_samples, n_features)
            Returns the dataframe of the states in self,
            where n_samples is the number of samples and
            n_features is the number of features {location, sku, sales, stock, sl, order}.

        action_data : {array-like, sparse matrix} of shape (n_samples, n_features)
            Returns the dataframe of the actions in self,
            where n_samples is the number of samples and
            n_features is the number of features {location, sku, state, OUL, ROL, R}.

        env_data : {array-like, sparse matrix} of shape (n_samples, n_features)
            Returns the dataframe of the environment in self,
            where n_samples is the number of samples and
            n_features is the number of features {location, sku, order, sales, stock}.
        """
        state_data = self.initial_state(shop_id, product_id)
        action_data = self.initial_action(shop_id, product_id, np.arange(len(state_data)))
        env_data = pd.DataFrame(columns=['location', 'sku', 'order', 'sales', 'stock'])

        for starting_state in range(len(state_data)):
            m = 0 # time step
            T = 0 # total time
            C = 0 # cumulative reward / total reward
            p = 0 # reward rate / average reward

            while m < max_steps:
                if m == 0:
                    i = starting_state
                    alpha_m = alpha
                    probability_m = 1

                    env_data = env_data.append({'location': shop_id,
                                                'sku': product_id,
                                                'order': state_data.at[i, 'order'],
                                                'sales': state_data.at[i, 'sales'],
                                                'stock': state_data.at[i, 'stock']},
                                               ignore_index=True)
                else:
                    alpha_m = alpha / m
                    probability_m = probability / m

                non_exploratory = np.random.binomial(n=1, p=1-probability_m)
                if non_exploratory:
                    action_index = action_data[action_data['state']==i].R.idxmax()
                else:
                    action_index = action_data[action_data['state']==i].R.idxmax()
                    if len(action_data[action_data['state']==i]) > 1:
                        action_index = np.random.choice(action_data[(action_data['state']==i) &
                                                                    (action_data.index != action_index)].index, 1)[0]

                sales, dem = self.sales_update(env_data, shop_id, product_id, LT)
                env_data = env_data.append({'location': shop_id,
                                            'sku': product_id,
                                            'order': self.order_update(env_data, shop_id, product_id, action_data.loc[action_index].OUL, action_data.loc[action_index].ROL, alpha_order),
                                            'sales': sales,
                                            'stock': self.stock_update(env_data, shop_id, product_id, LT, dem)},
                                           ignore_index=True)

                new_state = pd.DataFrame(columns=['location', 'sku', 'sales', 'stock', 'sl', 'order'])
                new_state = new_state.append({'location': shop_id,
                                              'sku': product_id,
                                              'sales': env_data.iloc[-1].sales,
                                              'stock': env_data.iloc[-1].stock,
                                              'sl': state_data.at[i, 'sl'],
                                              'order': env_data.iloc[-1].order
                                              }, ignore_index=True)

                if state_data[(state_data.sales==new_state.sales[0]) &
                              (state_data.stock==new_state.stock[0]) &
                              (state_data.sl==new_state.sl[0]) &
                              (state_data.order==new_state.order[0])].empty:
                    state_data = state_data.append(new_state, ignore_index=True)
                    j = state_data.index.max()
                    action_data = action_data.append(self.initial_action(shop_id, product_id, [j]), ignore_index=True)
                else:
                    j = state_data[(state_data.sales==new_state.sales[0]) &
                                   (state_data.stock==new_state.stock[0]) &
                                   (state_data.sl==new_state.sl[0]) &
                                   (state_data.order==new_state.order[0])].index[0]

                r = (new_state['sales'] - new_state['stock'] * (1 - new_state['sl']) / new_state['sl']).sum()
                action_data.at[action_index, 'R'] = (1 - alpha_m) * action_data.at[action_index, 'R'] + alpha_m * (r - p * LT + action_data[action_data['state']==j].R.max())

                if non_exploratory:
                    C = C + r
                    T += LT
                    p = C / T
                i = j
                m += 1

        state_data[['sales', 'stock']] = state_data[['sales', 'stock']].astype('int')
        env_data = env_data.astype('int')
        self.state_data = state_data
        self.action_data = action_data
        self.env_data = env_data
        return self

    def learn(self, quantity, max_steps, alpha, probability, LT, alpha_order):
        """
        Running the SMART algorithm for some location-sku pairs and obtaining summary
        information about all used states, actions with rewards and environment data.

        By: @HerrMorozovDmitry

        Parameters
        ----------
        quantity : {int, 'All'}
            Sets the number of location-sku pairs starting from the zero position
            in the self.pairs_data for which you want to run the SMART algorithm.
            If 'All', then use all pairs.

        max_steps : int

        alpha : float

        probability : float

        LT : int

        alpha_order : float

        Returns
        -------
        state_data : {array-like, sparse matrix} of shape (n_samples, n_features)
            Returns the dataframe of the states in self,
            where n_samples is the number of samples and
            n_features is the number of features {location, sku, sales, stock, sl, order}.

        action_data : {array-like, sparse matrix} of shape (n_samples, n_features)
            Returns the dataframe of the actions in self,
            where n_samples is the number of samples and
            n_features is the number of features {location, sku, state, OUL, ROL, R}.

        env_data : {array-like, sparse matrix} of shape (n_samples, n_features)
            Returns the dataframe of the environment in self,
            where n_samples is the number of samples and
            n_features is the number of features {location, sku, order, sales, stock}.
        """
        state_data = pd.DataFrame(columns=['location', 'sku', 'sales', 'stock', 'sl', 'order'])
        action_data = pd.DataFrame(columns=['location', 'sku', 'state', 'OUL', 'ROL', 'R'])
        env_data = pd.DataFrame(columns=['location', 'sku', 'order', 'sales', 'stock'])

        if quantity == 'All':
            quantity = len(self.pairs_data)

        for index, row in self.pairs_data.iloc[:quantity].iterrows():
            shop_id = row['shop_id']
            product_id = row['product_id']

            state = self.initial_state(shop_id, product_id)
            action = self.initial_action(shop_id, product_id, np.arange(len(state)))
            env = pd.DataFrame(columns=['location', 'sku', 'order', 'sales', 'stock'])

            for starting_state in range(len(state)):
                m = 0 # time step
                T = 0 # total time
                C = 0 # cumulative reward / total reward
                p = 0 # reward rate / average reward

                while m < max_steps:
                    if m == 0:
                        i = starting_state
                        alpha_m = alpha
                        probability_m = 1

                        env = env.append({'location': shop_id,
                                          'sku': product_id,
                                          'order': state.at[i, 'order'],
                                          'sales': state.at[i, 'sales'],
                                          'stock': state.at[i, 'stock']},
                                         ignore_index=True)
                    else:
                        alpha_m = alpha / m
                        probability_m = probability / m

                    non_exploratory = np.random.binomial(n=1, p=1-probability_m)
                    if non_exploratory:
                        action_index = action[action['state']==i].R.idxmax()
                    else:
                        action_index = action[action['state']==i].R.idxmax()
                        if len(action[action['state']==i]) > 1:
                            action_index = np.random.choice(action[(action['state']==i) &
                                                                   (action.index != action_index)].index, 1)[0]

                    sales, dem = self.sales_update(env, shop_id, product_id, LT)
                    env = env.append({'location': shop_id,
                                      'sku': product_id,
                                      'order': self.order_update(env, shop_id, product_id, action.loc[action_index].OUL, action.loc[action_index].ROL, alpha_order),
                                      'sales': sales,
                                      'stock': self.stock_update(env, shop_id, product_id, LT, dem)},
                                     ignore_index=True)

                    new_state = pd.DataFrame(columns=['location', 'sku', 'sales', 'stock', 'sl', 'order'])
                    new_state = new_state.append({'location': shop_id,
                                                  'sku': product_id,
                                                  'sales': env.iloc[-1].sales,
                                                  'stock': env.iloc[-1].stock,
                                                  'sl': state.at[i, 'sl'],
                                                  'order': env.iloc[-1].order
                                                  }, ignore_index=True)

                    if state[(state.sales==new_state.sales[0]) &
                             (state.stock==new_state.stock[0]) &
                             (state.sl==new_state.sl[0]) &
                             (state.order==new_state.order[0])].empty:
                        state = state.append(new_state, ignore_index=True)
                        j = state.index.max()
                        action = action.append(self.initial_action(shop_id, product_id, [j]), ignore_index=True)
                    else:
                        j = state[(state.sales==new_state.sales[0]) &
                                  (state.stock==new_state.stock[0]) &
                                  (state.sl==new_state.sl[0]) &
                                  (state.order==new_state.order[0])].index[0]

                    r = (new_state['sales'] - new_state['stock'] * (1 - new_state['sl']) / new_state['sl']).sum()
                    action.at[action_index, 'R'] = (1 - alpha_m) * action.at[action_index, 'R'] + alpha_m * max(r - p * LT + action[action['state']==j].R.max(), 0)

                    if non_exploratory:
                        C = C + r
                        T += LT
                        p = C / T
                    i = j
                    m += 1

            state_data = state_data.append(state)
            action_data = action_data.append(action)
            env_data = env_data.append(env)

        state_data[['sales', 'stock']] = state_data[['sales', 'stock']].astype('int')
        env_data = env_data.astype('int')
        self.state_data = state_data
        self.action_data = action_data
        self.env_data = env_data
        return self

    def __init__(self):
        ss_data = None
        sl_data = None
        pairs_data = None
        demand_data = None
        state_data = None
        action_data = None
        env_data = None
        print('Environment initialized!')

    def reset(self):
        """
        Return summory environment information from SMART.

        By: @HerrMorozovDmitry

        Returns
        -------
        state_data : {array-like, sparse matrix} of shape (n_samples, n_features)
            Returns the dataframe of the states,
            where n_samples is the number of samples and
            n_features is the number of features {location, sku, sales, stock, sl, order}.

        action_data : {array-like, sparse matrix} of shape (n_samples, n_features)
            Returns the dataframe of the actions,
            where n_samples is the number of samples and
            n_features is the number of features {location, sku, state, OUL, ROL, R}.

        env_data : {array-like, sparse matrix} of shape (n_samples, n_features)
            Returns the dataframe of the environment,
            where n_samples is the number of samples and
            n_features is the number of features {location, sku, order, sales, stock}.
        """
        print('Environment reset!')
        return self.state_data, self.action_data, self.env_data

    def predict(self, obs):
        """
        Predict optimal action for current state.

        By: @HerrMorozovDmitry

        Parameters
        ----------
        obs : {array-like, sparse matrix} of shape (n_samples, n_features)
            The dataframe of the environment,
            where n_samples is the number of samples and
            n_features is the number of features {location, sku, order, sales, stock, sl}.

        Returns
        -------
        Pair (OUL : int, ROL : int)
        """
        shop_id = obs.iloc[-1].location
        product_id = obs.iloc[-1].sku
        sales = obs.iloc[-1].sales
        stock = obs.iloc[-1].stock
        order = obs.iloc[-1].order
        sl = obs.iloc[-1].sl

        if not self.state_data[(self.state_data['location']==shop_id) & (self.state_data['sku']==product_id) &
                          (self.state_data['sales']==sales) & (self.state_data['stock']==stock) &
                          (self.state_data['sl']==sl) & (self.state_data['order']==order)].empty:

            state_index = self.state_data[(self.state_data['location']==shop_id) & (self.state_data['sku']==product_id) &
                                     (self.state_data['sales']==sales) & (self.state_data['stock']==stock) &
                                     (self.state_data['sl']==sl) & (self.state_data['order']==order)].index[0]

            action_index = self.action_data[(self.action_data['location']==shop_id) &
                                       (self.action_data['sku']==product_id) &
                                       (self.action_data['state']==state_index)].R.idxmax()

            OUL = self.action_data[(self.action_data['location']==shop_id) &
                              (self.action_data['sku']==product_id)].loc[action_index].OUL
            ROL = self.action_data[(self.action_data['location']==shop_id) &
                              (self.action_data['sku']==product_id)].loc[action_index].ROL
        else:
            OUL = 0
            ROL = 0

        action = [OUL, ROL]
        return action

    def step(self, obs, action, LT, alpha_order):
        """
        Predict optimal action for current state.

        By: @HerrMorozovDmitry

        Parameters
        ----------
        obs : {array-like, sparse matrix} of shape (n_samples, n_features)
            The dataframe of the environment,
            where n_samples is the number of samples and
            n_features is the number of features {location, sku, order, sales, stock, sl}.

        action : Pair (OUL : int, ROL : int)

        LT : int

        alpha_order : float

        Returns
        -------
        obs : {array-like, sparse matrix} of shape (n_samples, n_features)
            The updated dataframe of the environment,
            where n_samples is the number of samples and
            n_features is the number of features {location, sku, order, sales, stock, sl}.

        rew : float
            Reward for new state.
        """
        shop_id = obs.iloc[-1].location
        product_id = obs.iloc[-1].sku
        sales = obs.iloc[-1].sales
        stock = obs.iloc[-1].stock
        order = obs.iloc[-1].order
        sl = obs.iloc[-1].sl

        sales, dem = self.sales_update(obs, shop_id, product_id, LT)
        obs = obs.append({'location': shop_id,
                          'sku': product_id,
                          'sl': sl,
                          'order': self.order_update(obs, shop_id, product_id, action[0], action[1], alpha_order),
                          'sales': sales,
                          'stock': self.stock_update(obs, shop_id, product_id, LT, dem)},
                          ignore_index=True)

        rew = self.reward(obs.iloc[-1])
        print('\nStep successful!')
        return obs, rew

    def render(self, action, enviroment, reward, mode='human', close=False):
        """
        Print summory inforamation about current step or enviroment.

        By: @HerrMorozovDmitry

        Parameters
        ----------
        action : Pair (OUL : int, ROL : int)

        enviroment : {array-like, sparse matrix} of shape (n_samples, n_features)
            The dataframe of the environment,
            where n_samples is the number of samples and
            n_features is the number of features {location, sku, order, sales, stock, sl}.

        rew : float
            Reward of the system.
        """
        if action != None:
            print('Action (OUL, ROL):')
            print(action)
            print('State:')
            print(enviroment.tail(1))
            print('Reward:')
            print(reward)
        else:
            print('\nEnvironment:\n', enviroment)
            print('Final Reward:')
            print(reward, '\n')
