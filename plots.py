import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models import estimate_knn_clusters, estimate_som_clusters, prepare_whole_year, prepare_TS
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from IPython.display import clear_output


def inertia_plot(year=2019, month=None,  day=None, metric='dtw'):
    if month == None:
        df_data, mySeries, namesofMySeries = prepare_whole_year(year)
    else:
        df_data, mySeries, namesofMySeries = prepare_TS(year=year, month=month,  day=day)

    intertias = []
    n_clusters = []
    for i in range(1, 16):
        clear_output(wait=True)
        print(f'{round(100*i/15, 1)}%')
        km = TimeSeriesKMeans(n_clusters=i, metric=metric, n_init=5, random_state=1234)
        labels = km.fit_predict(mySeries)
        
        n_clusters.append(i)
        intertias.append(km.inertia_)

    df = pd.DataFrame({'clusters': n_clusters, 'inertia': intertias})
    for i in np.arange(0.0, 0.1, 0.02):
        scaled_intertias = (np.array(intertias)/intertias[0])+i*np.array(n_clusters)
        df[f'scld_intertia_{i}'] = scaled_intertias
        plt.plot(n_clusters, scaled_intertias, linestyle="-",marker=".", label=i)
    plt.legend()
    plt.show()

    if (day == None) & (month !=None) :
        filename = f'{year}{month}'
    elif day != None:
        filename = f'{year}{month}{day}'
    else:
        filename = f'{year}'

    df.to_csv(f'outputs/{filename}_inertias_plot.csv', index=False)



def get_silhouette_plot(year=2019, month=None,  day=None, metric='dtw'):

    if month == None:
        df_data, mySeries, namesofMySeries = prepare_whole_year(year)
    else:
        df_data, mySeries, namesofMySeries = prepare_TS(year=year, month=month,  day=day)

    ## K-Means Clustering
    print('Generating silhouette plot...')
    sil_sc = []
    cluster_counts = [2,5,7,10,13,15,17,20,25,30,35]

    for cluster_count in cluster_counts:
        print(cluster_count)
        km = TimeSeriesKMeans(n_clusters=cluster_count, metric=metric, random_state=1234)
        labels = km.fit_predict(np.array(mySeries))
        sil_sc.append(silhouette_score(np.array(mySeries), labels, metric=metric))
    clear_output(wait=True)
    plt.plot(cluster_counts, sil_sc)

    df_plot = pd.DataFrame({'clusters': cluster_count, 's_score': sil_sc})
    if (day == None) & (month !=None) :
        filename = f'{year}{month}'
    elif day != None:
        filename = f'{year}{month}{day}'
    else:
        filename = f'{year}'
    df_plot.to_csv(f'outpus/{filename}.csv', index=False)