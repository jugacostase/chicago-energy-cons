import math
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler


from minisom import MiniSom
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from sklearn.cluster import KMeans

from sklearn.decomposition import PCA
from IPython.display import clear_output

import matplotlib.dates as mdates


def prepare_whole_year(year=2019):
    df_total = pd.DataFrame()
    for month in range(1, 13):
        if month < 10:
            file = f'{year}0{month}'
        else:
            file = f'{year}{month}'
        df_data = pd.read_csv(f'data/comed_month/comed_{file}.csv')
            # Filtering to get only residential accounts
        df_data = df_data[df_data.service_name.str.contains('RESIDENTIAL')]
        df_data['date_time'] = pd.to_datetime(df_data.date_time)
        # Getting only the desired day of the month
        df_data['date_time'] = df_data['date_time'].dt.floor('7D')
        # Grouping observations at the zipcode level
        df_data = df_data.groupby(['zip5', 'date_time']).energy.sum().reset_index()
        df_total = pd.concat([df_total, df_data])
    
    df_data = df_total
    df_data = df_data.groupby(['zip5', 'date_time']).energy.sum().reset_index()
    ## Excluding zipcodes with not enough data
    #ind_counts = df_data.zip5.value_counts().index
    #mode_counts = df_data.zip5.value_counts().mode().values[0]
    #bad_obs = ind_counts[df_data.zip5.value_counts() != mode_counts]
    #df_data = df_data[~(df_data.zip5.isin(bad_obs))]
    #
    #if len(bad_obs>0):
    #    print(f'Excluded the following zip codes {bad_obs.values}')
    ## Getting each zipcode series into a list of series

    df_data = fill_bad_dates(df_data)

    namesofMySeries = list(df_data.zip5.unique())
    mySeries = []

    for zip_code in namesofMySeries:
        df = df_data.query('zip5==@zip_code')[['date_time', 'energy']].set_index('date_time').sort_index()
        mySeries.append(df)
        # Prints something if not all series have the same length
        #if df.shape[0] != 48:
        #    print(df.shape)

    # Scaling the data with respect to each series itself

    for i in range(len(mySeries)):
        scaler = MinMaxScaler()
        mySeries[i] = scaler.fit_transform(mySeries[i])
        mySeries[i]= mySeries[i].reshape(len(mySeries[i]))


    return df_data, mySeries, namesofMySeries
        

def prepare_TS(year=2019, month=1, day=None):
    if month < 10:
        month = f'0{month}'
    file = f'{year}{month}'

    df_data = pd.read_csv(f'data/comed_month/comed_{file}.csv')
    # Filtering to get only residential accounts
    df_data = df_data[df_data.service_name.str.contains('RESIDENTIAL')]
    df_data['date_time'] = pd.to_datetime(df_data.date_time)
    # Getting only the desired day of the month
    if day != None:
        df_data = df_data[df_data.date_time.dt.day == day]
    else:
        df_data['date_time'] = df_data['date_time'].dt.floor('d')
    # Grouping observations at the zipcode level
    df_data = df_data.groupby(['zip5', 'date_time']).energy.sum().reset_index()

    df_data = fill_bad_dates(df_data)

    namesofMySeries = list(df_data.zip5.unique())
    mySeries = []

    for zip_code in namesofMySeries:
        df = df_data.query('zip5==@zip_code')[['date_time', 'energy']].set_index('date_time').sort_index()
        mySeries.append(df)

    for i in range(len(mySeries)):
        scaler = MinMaxScaler()
        mySeries[i] = scaler.fit_transform(mySeries[i])
        mySeries[i]= mySeries[i].reshape(len(mySeries[i]))

    return df_data, mySeries, namesofMySeries




def estimate_knn_clusters(year=2019, month=None,  day=None, n_clusters=7, metric='dtw', plot=False, df=False):

    gdf_zc = gpd.read_file('data/geo/Chicago_ZC.geojson')
    gdf_zc['GEOID20'] = gdf_zc['GEOID20'].astype(str)

    if month == None:
        df_data, mySeries, namesofMySeries = prepare_whole_year(year)
    else:
        df_data, mySeries, namesofMySeries = prepare_TS(year=year, month=month,  day=day)

    best_n_cluster = n_clusters
    km = TimeSeriesKMeans(n_clusters=best_n_cluster, metric=metric, n_init=5, random_state=1234)
    labels = km.fit_predict(mySeries)

    if df:
        fancy_names_for_labels = [f"Cluster {label+1}" for label in labels]
        df_labels = pd.DataFrame(zip(namesofMySeries,fancy_names_for_labels),columns=["zip5","Cluster"]).sort_values(by="Cluster").set_index("zip5").reset_index()
        df_labels['zip5'] = df_labels.zip5.astype(str)
        gdf_data = pd.merge(gdf_zc, df_labels, left_on='GEOID20', right_on='zip5')
        gdf_data = gdf_data[['GEOID20', 'Cluster', 'geometry']]
        return gdf_data

    if plot:
        plot_count = math.ceil(math.sqrt(n_clusters))
        fig, axs = plt.subplots(plot_count,plot_count,figsize=(25,25))
        fig.suptitle('Clusters')
        row_i=0
        column_j=0
        for label in set(labels):
            cluster = []
            for i in range(len(labels)):
                    if(labels[i]==label):
                        axs[row_i, column_j].plot(mySeries[i],c="gray",alpha=0.4)
                        cluster.append(mySeries[i])
            if len(cluster) > 0:
                axs[row_i, column_j].plot(np.average(np.vstack(cluster),axis=0),c="red")
            axs[row_i, column_j].set_title("Cluster "+str(row_i*plot_count+column_j))
            column_j+=1
            if column_j%plot_count == 0:
                row_i+=1
                column_j=0

        plt.show()
    



def fill_bad_dates(df_data):
    ind_counts = df_data.zip5.value_counts().index
    mode_counts = df_data.zip5.value_counts().mode().values[0]
    bad_zips = ind_counts[df_data.zip5.value_counts() != mode_counts]
    #df_data = df_data[~(df_data.zip5.isin(bad_obs))]
    for bad_zip in bad_zips:
        good_zipcode = df_data[~(df_data.zip5.isin(bad_zips))].reset_index().zip5[0]
        all_dates = df_data[df_data.zip5==good_zipcode].date_time
        available_dates = df_data[df_data.zip5==bad_zip].date_time
        fill_dates = list(set(all_dates) - set(available_dates))
        for fill_date in fill_dates:
            df_new_row = pd.DataFrame([[bad_zip, fill_date, np.nan]], columns=['zip5', 'date_time', 'energy'])
            df_data = pd.concat([df_data, df_new_row])
            df_data = df_data.sort_values(['zip5', 'date_time']).reset_index(drop=True)
            df_data.loc[df_data.zip5==bad_zip, 'energy'] = df_data[df_data.zip5==bad_zip].energy.interpolate(limit_direction="both")

    return df_data



def estimate_som_clusters(year=2019, month=None,  day=None, som_x=None, som_y=None, sigma=0.5, learning_rate=0.5, plot=False, df=False, 
                          neighborhood_function='gaussian'):

    gdf_zc = gpd.read_file('data/geo/Chicago_ZC.geojson')
    gdf_zc['GEOID20'] = gdf_zc['GEOID20'].astype(str)

    if month == None:
        df_data, mySeries, namesofMySeries = prepare_whole_year(year)
    else:
        df_data, mySeries, namesofMySeries = prepare_TS(year=year, month=month,  day=day)
    # Plot for best number of clusters

    if som_x == None:
        som_x =  math.ceil(math.sqrt(math.sqrt(len(mySeries))))

    if som_y == None:
        som_y =  math.ceil(math.sqrt(math.sqrt(len(mySeries))))

    som = MiniSom(som_x, som_y, len(mySeries[0]), sigma, learning_rate, neighborhood_function=neighborhood_function, random_seed=1234)
    som.pca_weights_init(mySeries)
    som.train(mySeries, 50000)

    win_map = som.win_map(mySeries)

    if plot:
        #fig, axs = plt.subplots(som_x,som_y,figsize=(25,25))
        #fig.suptitle('Clusters')
        #for x in range(som_x):
        #    for y in range(som_y):
        #        cluster = (x,y)
        #        if cluster in win_map.keys():
        #            for series in win_map[cluster]:
        #                axs[cluster].plot(series,c="gray",alpha=0.5) 
        #            axs[cluster].plot(np.average(np.vstack(win_map[cluster]),axis=0),c="red")
        #        cluster_number = x*som_y+y+1
        #        axs[cluster].set_title(f"Cluster {cluster_number}")
#
        #plt.show()

        dates = df_data.date_time.unique()
        for cluster in win_map.keys():
            print(f'Cluster: {cluster}')
            fig, ax = plt.subplots(figsize=(12, 6))
            for series in win_map[cluster]:
                ax.plot(dates, series,c="gray",alpha=0.5)
            ax.plot(dates, dtw_barycenter_averaging(np.vstack(win_map[cluster])),c="red")
            fig.show()



        #for label in set(labels):
        #    dates = df_data.date_time.unique()
        #    fig, ax = plt.subplots(figsize=(12, 6))
        #    for i in range(len(labels)):
        #        if(labels[i]==label):
        #            cluster = []
        #            ax.plot(dates[1:], mySeries[i][1:],c="gray",alpha=0.2)
        #            cluster.append(mySeries[i])
        #    if len(cluster) > 0:
        #            ax.plot(dates[1:], dtw_barycenter_averaging(np.vstack(cluster))[1:],c="red")
#
        #    ax.xaxis.set_major_formatter(
        #        mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
#
        #    ax.set_ylabel(r'Normalized energy consumption')
        #    ax.grid(linewidth=0.25)
        #    plt.show()



    if df:
        cluster_map = []
        for idx in range(len(mySeries)):
            winner_node = som.winner(mySeries[idx])
            cluster_map.append((namesofMySeries[idx],f"Cluster {winner_node[0]*som_y+winner_node[1]+1}"))

        df_labels = pd.DataFrame(cluster_map,columns=["zip5","Cluster"]).sort_values(by="Cluster").set_index("zip5").reset_index()
        df_labels['zip5'] = df_labels.zip5.astype(str)
        gdf_data = pd.merge(gdf_zc, df_labels, left_on='GEOID20', right_on='zip5')
        gdf_data = gdf_data[['GEOID20', 'Cluster', 'geometry']]
        return gdf_data
