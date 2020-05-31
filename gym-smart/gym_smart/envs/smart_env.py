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

    def calculate_lambda(self, shop_id, product_id, alpha_lambda):
        """
        Computes lambda for demand recovery

        By: @HerrMorozovDmitry

        Parameters
        ----------
        shop_id : int

        product_id : int

        alpha_lambda

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

        alpha_lambda

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

    def order_update(stock_data, shop_id, product_id, OUL, ROL, alpha_order):
        """
        Computes order using information from environment data (with binomial distribution with alpha_order param)

        By: @Kirili4ik

        Parameters
        ----------
        stock_data : pd.DataFrame with 'shop_id', 'product_id', 'stock' and 'value' columns

        shop_id : int

        product_id : int

        OUL

        ROL

        alpha_order

        Returns
        -------
        order : int
            Returns the value of order for given shop and product ids.
        """
        curr_shop_prod = stock_data[(stock_data['shop_id'] == shop_id)
                                    & (stock_data['product_id'] == product_id)].iloc[-1]

        if curr_shop_prod['stock'] <= ROL:
            rec_ord_value = OUL - curr_shop_prod['stock'] - curr_shop_prod['value']
            rec_ord_value = max(rec_ord_value, 0)
        else:
            rec_ord_value = 0

        ord_value = np.random.binomial(n=rec_ord_value, p=alpha_order)

        return ord_value

    def learn_single(self, shop_id, product_idm, max_steps, alpha, probability):
        """
        Running the SMART algorithm for one location-sku pair and obtaining summary
        information about all used states, actions with rewards and environment data.

        By: @HerrMorozovDmitry

        Parameters
        ----------
        shop_id : int

        product_id : int

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

                env_data = env_data.append({'location': shop_id,
                                            'sku': product_id,
                                            'order': self.order_update(env_data, shop_id, product_id, action_data.loc[action_index].OUL, action_data.loc[action_index].ROL),
                                            'sales': self.sales_update(env_data, shop_id, product_id),
                                            'stock': self.stock_update(env_data, shop_id, product_id)},
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
                action_data.at[action_index, 'R'] = (1 - alpha_m) * action_data.at[action_index, 'R'] + alpha_m * (r - p * self.lt + action_data[action_data['state']==j].R.max())

                if non_exploratory:
                    C = C + r
                    T += self.lt
                    p = C / T
                i = j
                m += 1

        state_data[['sales', 'stock']] = state_data[['sales', 'stock']].astype('int')
        env_data = env_data.astype('int')
        return state_data, action_data, env_data

    def learn(self, quantity='All', max_steps, alpha, probability):
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

                while m < self.max_steps:
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

                    env = env.append({'location': shop_id,
                                      'sku': product_id,
                                      'order': self.order_update(env, shop_id, product_id, action.loc[action_index].OUL, action.loc[action_index].ROL),
                                      'sales': self.sales_update(env, shop_id, product_id),
                                      'stock': self.stock_update(env, shop_id, product_id)},
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
                    action.at[action_index, 'R'] = (1 - alpha_m) * action.at[action_index, 'R'] + alpha_m * max(r - p * self.lt + action[action['state']==j].R.max(), 0)

                    if non_exploratory:
                        C = C + r
                        T += self.lt
                        p = C / T
                    i = j
                    m += 1

            state_data = state_data.append(state)
            action_data = action_data.append(action)
            env_data = env_data.append(env)

        state_data[['sales', 'stock']] = state_data[['sales', 'stock']].astype('int')
        env_data = env_data.astype('int')
        return state_data, action_data, env_data
    
    def simulation(self, state_data, action_data, env_data, quantity, date_start, date_end):
        """
        Simulates the operation of the algorithm and returns summary
        information about itand the operation of the original system.
        
        By: @sofloud

        Parameters
        ----------
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

        quantity : {int}
            Sets the number of location-sku pairs starting from the zero position in the
            self.pairs_data for which you want to simulate the SMART algorithm operation.

        date_start : {Timestamp}
            Initial date of the environment simulation.

        date_end : {Timestamp}
            Final date of the environment simulation.

        Returns
        -------
        starting_state : {array-like, sparse matrix} of shape (n_samples, n_features)
            Returns the dataframe of the initial states,
            where n_samples is the number of samples and
            n_features is the number of features {location, sku, sales, stock, sl, order, curr_date}.

        old_states : {array-like, sparse matrix} of shape (n_samples, n_features)
            Returns the dataframe of the original states,
            where n_samples is the number of samples and
            n_features is the number of features {location, sku, sales, stock, sl, order, curr_date}.

        new_states : {array-like, sparse matrix} of shape (n_samples, n_features)
            Returns the dataframe of the states obtained
            during the application of the SMART algorithm,
            where n_samples is the number of samples and
            n_features is the number of features {location, sku, sales, stock, sl, order, curr_date}.

        """
        old_states = pd.DataFrame(columns=['location', 'sku', 'sales', 'stock', 'sl', 'order', 'curr_date'])
        new_states = pd.DataFrame(columns=['location', 'sku', 'sales', 'stock', 'sl', 'order', 'curr_date'])
        starting_state = pd.DataFrame(columns=['location', 'sku', 'sales', 'stock', 'sl', 'order', 'curr_date'])

        for index, row in model.pairs_data.iloc[:quantity].iterrows():
            shop_id = row['shop_id']
            product_id = row['product_id']

            state = pd.DataFrame(columns=['location', 'sku', 'sales', 'stock', 'sl', 'order'])

            store_sales_t = model.ss_data[model.ss_data['store_id']==shop_id]
            store_sales_t = store_sales_t[store_sales_t['product_id']==product_id].fillna(0)
            store_sales_t = store_sales_t.drop(columns=['flg_spromo'])

            sl_data_t = model.sl_data.set_index(['date_from'], drop=True)
            sl_data_t = sl_data_t[sl_data_t['location_ids']==shop_id]
            sl_data_t = sl_data_t[sl_data_t['product_ids']==product_id]

            all_days = pd.date_range(start=sl_data_t.index.min(), end=sl_data_t['date_to'].max())
            sl_data_t.index = pd.DatetimeIndex(sl_data_t.index)
            sl_data_t = sl_data_t.reindex(all_days, method='ffill')
            sl_data_t = sl_data_t.drop(columns=['date_to'])

            state_t = pd.merge(store_sales_t, sl_data_t['value'], how='right', left_on=store_sales_t.index, right_on=sl_data_t.index)
            state_t = state_t.rename(columns={'store_id': 'location', 'product_id': 'sku', 's_qty': 'sales', 'value': 'sl'})
            state_t = state_t.drop(columns=['key_0'])
            state_t = state_t.drop_duplicates()
            state_t = state_t.dropna()
            state_t['order'] = 0
            state = state.append(state_t, ignore_index=True)

            for date in pd.date_range(start=date_start, end=date_end):
                old_states = old_states.append(state[state['curr_date']==date], ignore_index=True)

            starting_state = starting_state.append(state[state['curr_date']==pd.to_datetime(date_start - timedelta(days=1))], ignore_index=True)

            new_states_t = state[state['curr_date']==pd.to_datetime(date_start - timedelta(days=1))]

            for date in pd.date_range(start=date_start, end=date_end):
                if not new_states_t.empty:
                    sales = new_states_t.iloc[-1].sales
                    stock = new_states_t.iloc[-1].stock
                    order = new_states_t.iloc[-1].order
                    sl = new_states_t.iloc[-1].sl 

                    if not state_data[(state_data['location']==shop_id) & (state_data['sku']==product_id) &
                                      (state_data['sales']==sales) & (state_data['stock']==stock) &
                                      (state_data['sl']==sl) & (state_data['order']==order)].empty:

                        state_index = state_data[(state_data['location']==shop_id) & (state_data['sku']==product_id) &
                                                 (state_data['sales']==sales) & (state_data['stock']==stock) &
                                                 (state_data['sl']==sl) & (state_data['order']==order)].index[0]

                        action_index = action_data[(action_data['location']==shop_id) &
                                                   (action_data['sku']==product_id) &
                                                   (action_data['state']==state_index)].R.idxmax()

                        OUL = action_data[(action_data['location']==shop_id) &
                                          (action_data['sku']==product_id)].loc[action_index].OUL
                        ROL = action_data[(action_data['location']==shop_id) &
                                          (action_data['sku']==product_id)].loc[action_index].ROL

                        new_states_t = new_states_t.append({'location': shop_id,
                                                            'sku': product_id,
                                                            'sales': model.sales_update(new_states_t, shop_id, product_id),
                                                            'stock': model.stock_update(new_states_t, shop_id, product_id),
                                                            'sl': sl,
                                                            'order': model.order_update(new_states_t, shop_id, product_id, OUL, ROL),
                                                            'curr_date': date
                                                            }, ignore_index=True)
                    else:
                        new_states_t = new_states_t.append({'location': shop_id,
                                                            'sku': product_id,
                                                            'sales': 0,
                                                            'stock': 0,
                                                            'sl': 0,
                                                            'order': 0,
                                                            'curr_date': date
                                                            }, ignore_index=True)
                else:
                    new_states_t = new_states_t.append({'location': shop_id,
                                                        'sku': product_id,
                                                        'sales': 0,
                                                        'stock': 0,
                                                        'sl': 0,
                                                        'order': 0,
                                                        'curr_date': date
                                                        }, ignore_index=True)

            new_states = new_states.append(new_states_t.iloc[1:], ignore_index=True)

        starting_state[['location', 'sku', 'sales', 'stock']] = starting_state[['location', 'sku', 'sales', 'stock']].astype('int')
        old_states[['location', 'sku', 'sales', 'stock']] = old_states[['location', 'sku', 'sales', 'stock']].astype('int')
        new_states[['location', 'sku', 'sales', 'stock']] = new_states[['location', 'sku', 'sales', 'stock']].astype('int')

        
        return starting_state, old_states, new_states      

    def reward(self, environment):
        """
        Computes and returns reward.
        
        By: @sofloud

        Parameters
        ----------
        environment : {array-like, sparse matrix} of shape (n_samples, n_features)   
            n_features is the number of features {'location', 'sku', 'sales', 'stock', 'sl', 'order'}.

        Returns
        -------
        reward : {float}
            Returns reward.

        """
        reward = (environment['sales'] - environment['stock'] * (1 - environment['sl']) / environment['sl']).sum()
        return reward


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
