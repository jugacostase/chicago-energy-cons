import pandas as pd


df_comtime = pd.read_csv('data/census/commute_times/ACSDT5Y2019.B08012-Data.csv', skiprows=[1])
df_cnames = pd.read_csv('data/census/commute_times/column_names.csv')
df_zip2tract = pd.read_excel('data/census/ZIP_TRACT_032019.xlsx')

dict_comtime_names = df_cnames.set_index('column').to_dict()['label']
columns_comtime = list(dict_comtime_names.keys())
df_comtime = df_comtime[columns_comtime].rename(columns=dict_comtime_names)
columns_comtime = list(df_comtime.columns)
df_comtime['GEOID'] = df_comtime.GEOID.str[9:]

df_zip2tract['zip'] = df_zip2tract.zip.astype(str)
df_zip2tract['tract'] = df_zip2tract.tract.astype(str)

df = pd.merge(df_comtime, df_zip2tract, left_on='GEOID', right_on='tract')
df.loc[:, columns_comtime[1:]] = df[columns_comtime[1:]].multiply(df['res_ratio'], axis='index')
df = df.groupby('zip')[columns_comtime[1:]].sum().reset_index()

#df.to_csv('data/census/commute_times/commute_times_zc_il.csv', index=False)