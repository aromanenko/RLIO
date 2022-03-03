import pandas as pd
from tqdm import tqdm

MPLY_QTY_LOAD_PATH = 'echelon/echelon_1_mply_qty.csv'
LEAD_TIME_LOAD_PATH = 'echelon/echelon_1_lt.csv'
BATCH_SIZE_LOAD_PATH = 'echelon/echelon_1_batch_size.csv'
SERVICE_LEVEL_LOAD_PATH = 'echelon/echelon_1_sl.csv'

MPLY_QTY_SAVE_PATH = 'normalized_data/mply_qty_normalized.csv'
LEAD_TIME_SAVE_PATH = 'normalized_data/lead_time_normalized.csv'
BATCH_SIZE_SAVE_PATH = 'normalized_data/batch_size_normalized.csv'
SERVICE_LEVEL_SAVE_PATH = 'normalized_data/service_level_normalized.csv'


def normalize_data(load_path, save_path, columns):
    df = pd.read_csv(load_path, sep=';')
    with open(save_path, 'w') as file:
        file.write(f'{";".join(columns)}\n')
        for _, row in tqdm( df.iterrows(), total=df.shape[0] ):
            for date in pd.date_range(row.date_from, row.date_to):
                file.write(f'{row.product_ids};{row.location_ids};{date};{row["value"]}\n')


print('1 - Normalizing mply_qty data...')

normalize_data(MPLY_QTY_LOAD_PATH, MPLY_QTY_SAVE_PATH, ['product_id', 'store_id', 'curr_date', 'mply_qty'])

print('Done!')

print('2 - Normalizing lead_time data...')

normalize_data(LEAD_TIME_LOAD_PATH, LEAD_TIME_SAVE_PATH, ['product_id', 'store_id', 'curr_date', 'lead_time'])

print('Done!')

print('3 - Normalizing batch_size data...')

normalize_data(BATCH_SIZE_LOAD_PATH, BATCH_SIZE_SAVE_PATH, ['product_id', 'store_id', 'curr_date', 'batch_size'])

print('Done!')

print('4 - Normalizing service_level data...')

normalize_data(SERVICE_LEVEL_LOAD_PATH, SERVICE_LEVEL_SAVE_PATH, ['product_id', 'store_id', 'curr_date', 'service_level'])

print('Done!')
