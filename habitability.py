import pandas as pd
df = pd.read_csv('hwc.csv')
df.dropna(subset=['P_TEMP_SURF', 'P_TEMP_EQUIL', 'S_HZ_OPT_MIN', 'S_HZ_OPT_MAX', 'P_ESI'], inplace=True)
print(df.head(), df.dtypes)
features = ['P_TEMP_SURF', 'P_TEMP_EQUIL', 'S_HZ_OPT_MIN', 'S_HZ_OPT_MAX', 'P_ES']
