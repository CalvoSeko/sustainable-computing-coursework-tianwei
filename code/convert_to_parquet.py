import sys
import pandas as pd
import os
import time
sys.path.append(os.path.abspath('LibreHardwareMonitor-net472'))
from power_monitor import PowerMonitor

with PowerMonitor() as pmon:
    start_time = time.time()
    file_path = "data/df_fuel_ckan.csv"
    store_path = "data/data_CI.parquet"
    df = pd.read_csv(file_path)
    df_subset = df[['DATETIME', 'CARBON_INTENSITY']]
    df_subset = df_subset[df_subset['DATETIME'] >= '2015-01-01']

    df_subset.to_parquet(store_path, engine="pyarrow", index = False)
    print("Execution Time: %s seconds" % (time.time() - start_time))
print(pmon.stats())