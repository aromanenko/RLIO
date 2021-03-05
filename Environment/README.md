# OpenAI Gym Environment для проекта RLIO

## Назначение папок

* __echelon_raw_data__

Содержит сырые файлы `echelon_1_batch_size.csv`, `echelon_1_lt.csv`, `echelon_1_mply_qty.csv`, `echelon_1_sl.csv`

* __echelon_processed_data__

Содержит файлы `echelon_1_batch_size.csv`, `echelon_1_lt.csv`, `echelon_1_mply_qty.csv`, `echelon_1_sl.csv`, обработанные с помощью скрипта `preprocessing.py` - `batch_size_normalized.csv`, `lead_time_normalized.csv`, `mply_qty_normalized.csv` и `service_level_normalized.csv` соответственно.

* __store_raw_data__

Содержит сырые файлы вида `MERGE_TABLE_STORE_N.csv`, где N - id расположения

* __store_processed_data__

Содержит обработанные файлы вида `STORE_N.csv`, где N - id расположения. По сравнению с `MERGE_TABLE_STORE_N.csv`, в этих файлах заполены пропуски, а также добавлена информация о batch_size, lead_time, mply_qty и service_level.

* __experiments__

Содержит ноутбуки с экспериментами
