import numpy as np
import pandas as pd
import geopandas as gpd

import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler

from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

import plotly.graph_objects as go

import shap

columns_sex = ["DP05_0003E"]
columns_age = ["DP05_0005E","DP05_0006E","DP05_0007E","DP05_0008E","DP05_0009E","DP05_0010E","DP05_0011E","DP05_0012E","DP05_0013E","DP05_0014E","DP05_0015E","DP05_0016E","DP05_0017E"]
columns_sex_age = ["DP05_0026E","DP05_0027E", "DP05_0030E","DP05_0031E"]
columns_race = ["DP05_0037E", 'DP05_0038E', 'DP05_0039E', 'DP05_0044E', 'DP05_0052E', 'DP05_0057E']
columns_income = ['S1901_C01_012E', 'S1901_C01_013E']
columns_education = ['S1501_C01_002E', 'S1501_C01_003E', 'S1501_C01_004E', 'S1501_C01_005E', 'S1501_C01_007E',
                     'S1501_C01_008E', 'S1501_C01_009E', 'S1501_C01_010E', 'S1501_C01_011E', 'S1501_C01_012E',
                     'S1501_C01_013E', 'S1501_C01_014E', 'S1501_C01_015E', 'S1501_C01_017E', 'S1501_C01_018E',
                     'S1501_C01_020E', 'S1501_C01_021E', 'S1501_C01_023E', 'S1501_C01_024E', 'S1501_C01_026E',
                     'S1501_C01_027E']

best_columns_gb = ['DP05_0007E', 'S1501_C01_020E',
       'DP05_0015E', 'DP05_0009E', 'DP05_0014E', 'DP05_0013E',
       'DP05_0005E', 'DP05_0012E', 'S1501_C01_014E', 'DP05_0038E',
       'S1501_C01_018E', 'DP05_0044E', 'DP05_0003E', 'DP05_0010E',
       'S1501_C01_017E', 'Urban', 'DP05_0001E']

best_columns_rf = ['S1501_C01_007E', 'DP05_0009E', 'DP05_0038E',
       'S1501_C01_020E', 'DP05_0003E', 'DP05_0010E', 'DP05_0005E',
       'DP05_0015E', 'DP05_0044E', 'S1501_C01_009E', 'DP05_0012E',
       'S1501_C01_018E', 'S1501_C01_017E', 'Urban', 'DP05_0001E']


data_urls = [
    'census/Census_Clean_Zip5_IL_Household&Family_Married&Nonmarried_Income_2018.csv',
    'census/Census_Clean_Zip5_IL_Sex_Age_Ethnicity_2018.csv',
    'census/Census_Clean_Zip5_IL_EducationLevel_byAge_byIncome_Ethnicity_bySex.csv',
    'census/commute_times_mode/ACSDT5Y2019.B08134-Data.csv',
    'census/buildings/ACSST5Y2019.S2504-Data.csv'
]

mode_cols = ['Car, truck, or van',
'Other',
'Public transportation',
'Walked',
]

time_mode_cols = [
'B08134_036E',
'B08134_035E',
'B08134_034E',
'B08134_033E',
'B08134_032E',
'B08134_031E',
'B08134_030E',
'B08134_029E',
'B08134_028E',
'B08134_027E',
'B08134_026E',
'B08134_025E',
'B08134_024E',
'B08134_023E',
'B08134_022E',
'B08134_021E',
'B08134_020E',
'B08134_019E',
'B08134_018E',
'B08134_017E',
'B08134_016E',
'B08134_015E',
'B08134_014E',
'B08134_013E',
'B08134_012E',
'B08134_011E',
'B08134_010E',
'B08134_009E',
'B08134_008E',
'B08134_007E',
'B08134_006E',
'B08134_005E',
'B08134_004E',
'B08134_003E',
'B08134_002E',

]

building_cols = ['S2504_C01_002E',
'S2504_C01_003E',
'S2504_C01_004E',
'S2504_C01_005E',
'S2504_C01_006E',
'S2504_C01_007E',
'S2504_C01_009E',
'S2504_C01_010E',
'S2504_C01_011E',
'S2504_C01_012E',
'S2504_C01_013E',
'S2504_C01_014E',
'S2504_C01_015E',
'S2504_C01_016E',
'S2504_C01_017E',
'S2504_C01_018E',
'S2504_C01_019E',
'S2504_C01_020E',
'S2504_C01_027E',
]

best_cols = ['S1501_C01_017E', 'DP05_0044E', 'S1501_C01_018E', 'DP05_0030E',
             'B08134_017E', 'DP05_0031E', 'B08134_019E', 'Urban', 'DP05_0057E',
             'S2504_C01_017E', 'S2504_C01_019E', 'S2504_C01_018E', 'S2504_C01_014E']

dict_groups = {
    'sex': columns_sex,
    'age': columns_age,
    'sex_age': columns_sex_age,
    'race': columns_race,
    'income':columns_income,
    'education':columns_education,
    'mode': mode_cols,
    'time_mode': time_mode_cols,
    'building': building_cols,
    'best': best_cols,
    'weekday': ['weekday'],
    'urban': ['Urban']
}

dict_col_names = pd.read_excel('data/census/variable_names.xlsx', sheet_name='raw').set_index('Code').to_dict()['Name']

def generate_filename(year, month=None, day=None, day_type=None):
    if (day == None) & (month !=None):
        if month < 10:
            month = f'0{month}'
        filename = f'{year}{month}'
    elif day != None:
        if day < 10:
            day = f'0{day}'
        filename = f'{year}{month}{day}'
    else:
        filename = f'{year}'

    if day_type != None:
        filename = f'{filename}_{day_type}'

    return filename


def plot_accuracy_scatter(reg, X_train, y_train, X_test, y_test, log_scale=True, size=3):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=y_train, y=reg.predict(X_train),
                             mode='markers',
                             name='Train data',
                             marker={'color': '#A34184',
                                     'size': size}))

    fig.add_trace(go.Scatter(x=y_test, y=reg.predict(X_test),
                             mode='markers',
                             name='Test data',
                             marker={'color': '#49A1A3',
                                     'size': size}))

    if log_scale:
        ax_range = np.arange(0, y_train.max(), 1000)
    else:
        ax_range = np.arange(0, y_train.max(), 1000)
    fig.add_trace(go.Scatter(x=ax_range, y=ax_range,
                             mode='lines',
                             name='Theoretical solution',
                             line=dict(color='black',
                                       width=1,
                                       dash='dash')))

    fig.update_layout(
        width=900,
        height=700,
        title="Gradient Boosting Model",
        xaxis_title="Actual electricity consumption (kWh)",
        yaxis_title="Predicted electricity consumption (kWh)"
    )

    if log_scale:
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")

    fig.show()

def print_metrics(reg, X_train, y_train, X_test, y_test):
    mse_train = mean_squared_error(y_train, reg.predict(X_train))
    mse_test = mean_squared_error(y_test, reg.predict(X_test))
    mape_train = mean_absolute_percentage_error(y_train, reg.predict(X_train))
    mape_test = mean_absolute_percentage_error(y_test, reg.predict(X_test))
    r2_train = r2_score(y_train, reg.predict(X_train))
    r2_test = r2_score(y_test, reg.predict(X_test))
    print("The mean squared error (MSE) on train set: {:.4f}".format(mse_train))
    print("The mean squared error (MSE) on test set: {:.4f}".format(mse_test))
    print("The mean squared error (MAPE) on train set: {:.4f}".format(mape_train))
    print("The mean squared error (MAPE) on test set: {:.4f}".format(mape_test))
    print("The R2 score (R2) on train set: {:.4f}".format(r2_train))
    print("The R2 score (R2) on test set: {:.4f}".format(r2_test))

def plot_importances(reg, X_train, y_train, X_test, y_test):
    feature_importance = reg.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + 0.5
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.barh(pos, feature_importance[sorted_idx], align="center")
    plt.yticks(pos, np.array(X_train.columns)[sorted_idx])
    plt.title("Feature Importance (MDI)")

    result = permutation_importance(
        reg, X_train, y_train, n_repeats=10, random_state=42, n_jobs=2
    )
    sorted_idx = result.importances_mean.argsort()
    plt.subplot(1, 2, 2)
    plt.boxplot(
        result.importances[sorted_idx].T,
        vert=False,
        labels=np.array(X_train.columns)[sorted_idx],
    )
    plt.title("Permutation Importance (train set)")
    fig.tight_layout()
    plt.show()

def shap_analysis(reg, X, X_train, X_test, max_display=10, cmap='BrBG'):
    explainer = shap.TreeExplainer(reg)
    shap_values = explainer(X)  # Entire dataset
    shap_train = explainer(X_train)  # Train set
    shap_test = explainer(X_test)  # Test set - You can count the 14 dots per feature (i.e., size of test set)

    ## Global Bar Plot
    #fig = plt.figure()  # Define an empty figure
    #ax0 = fig.add_subplot(131)
    #shap.plots.bar(shap_values, show=False, max_display=max_display)
    #plt.title("Entire Dataset")
    #ax0 = fig.add_subplot(132)
    #shap.plots.bar(shap_train, show=False, max_display=max_display)
    #plt.title("Train Set")
    #ax0 = fig.add_subplot(133)
    #shap.plots.bar(shap_test, show=False, max_display=max_display)
    #plt.title("Test Set")
    #fig.set_size_inches(18, 6)
    #plt.tight_layout()
    ## plt.savefig(file_name + '_SHAP_bar.png') #Saving the plot
    #plt.show()

    shap.summary_plot(shap_values, X.values, show=False, max_display=max_display, cmap=cmap)
    plt.title('SHAP', fontsize=20)
    plt.show()




def prepare_data(date='201901'):
    df_data = pd.read_csv(f'data/comed_month/comed_{date}.csv')
    df_data['date_time'] = pd.to_datetime(df_data.date_time)

    if 'service_name' in df_data.columns:
        df_data = df_data[df_data['service_name'].str.contains('RESIDENTIAL')]

    df_data = df_data.groupby(['zip5', 'date_time']).energy.sum().reset_index()
    df_data['weekday'] = df_data.date_time.dt.weekday
    df_data['day'] = df_data.date_time.dt.day
    gdf_zc = gpd.read_file('data/geo/Chicago_ZC.geojson')
    gdf_zc['GEOID20'] = gdf_zc['GEOID20'].astype(int)

    gdf_data = pd.merge(gdf_zc, df_data, left_on='GEOID20', right_on='zip5')
    gdf_data = gdf_data.groupby(['zip5', 'weekday', 'day'])['energy'].sum().reset_index()
    gdf_data = gdf_data.groupby(['zip5'])['energy'].mean().reset_index()

    gdf_data['zip5'] = gdf_data['zip5'].astype(str)

    for url in data_urls:
        df_acs = pd.read_csv(f'data/{url}', low_memory=False)
        # dict_names = dict(zip(df_acs.columns, df_acs.loc[0].values))
        df_acs = df_acs.drop(0)
        df_acs['zip5'] = df_acs.NAME.str[6:]
        df_acs = df_acs.drop(columns=['GEO_ID', 'NAME'])
        gdf_data = pd.merge(gdf_data, df_acs, on='zip5')

    gdf_data = gdf_data.drop_duplicates()

    gdf_counties = gpd.read_file('data/geo/US_counties.json')
    gdf_zc_c = gdf_zc.to_crs(epsg='4326').sjoin(gdf_counties, how='left')
    city_zc = gdf_zc_c.query('NAME=="Cook"').GEOID20.astype(str).values

    gdf_data['Urban'] = np.where(gdf_data.zip5.isin(city_zc), 1, 0)

    gdf_data['DP05_0001E'] = gdf_data['DP05_0001E'].astype(float)
    gdf_data['energy_scaled'] = gdf_data.energy / gdf_data['DP05_0001E']

    for col in gdf_data.columns:
        mean_val = np.mean(pd.to_numeric(gdf_data[col], errors='coerce'))
        gdf_data[col] = (pd.to_numeric(gdf_data[col], errors='coerce').fillna(mean_val))

    return gdf_data

def get_train_test_data(gdf_data, use_cols, train_size=.7):
    selected_cols = []

    for cols_group in use_cols:
        selected_cols = selected_cols + dict_groups[cols_group]

    X, y = gdf_data[selected_cols], gdf_data.energy

    X = X.rename(columns=dict_col_names)


    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=42)
    groups = gdf_data.zip5
    split_ind = gss.split(X, y, groups)

    train_ind, test_ind = next(split_ind)

    X_train = X.iloc[train_ind, :]
    y_train = y.iloc[train_ind]
    X_test = X.iloc[test_ind, :]
    y_test = y.iloc[test_ind]

    return X_train, y_train, X_test, y_test, X

def generate_landuse_geojson():
    df_land = gpd.read_file('data/geo/2018_Land_Use_Inventory_for_Northeastern_Illinois.geojson')
    land_uses = ['11', '12', '13', '14']
    df_land = df_land[df_land.LANDUSE.str[:2].isin(land_uses)]

    gdf_zc = gpd.read_file('data/geo/Chicago_ZC.geojson')
    gdf_zc['GEOID20'] = gdf_zc['GEOID20'].astype(int)
    df_land_z = df_land.sjoin(gdf_zc.to_crs(4326))
    df_land_z['residential'] = np.where(df_land_z.LANDUSE.str[:2] == '11', 1, 0)
    df_land_z = df_land_z[['OBJECTID', 'GEOID20', 'residential', 'geometry']].rename(
        columns={'OBJECTID': 'id', 'GEOID20': 'zip5'})
    df_land_z.to_file('data/geo/landuse_zc.geojson', driver='GeoJSON')
