import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np
import folium.features


def city_detection_enhanced(coordsXY: list) -> pd.Series:
    """ Computes the probability of base stations' city-ness using H-DBScan.
        
        Parameters
        ----------
        coordsXY : list
            [x, y] coordinates of all points (lambert-93 projection).
        Returns
        -------
        probaCity : pd.Dataframe
            A dataframe containing labels of the clusters and associated probabilities.
    """

    clusterer = HDBSCAN(min_cluster_size=4, min_samples=7, alpha=10)
    clusterer.fit(coordsXY)

    data = {'labels' : clusterer.labels_, 'probas' : clusterer.probabilities_}

    return pd.DataFrame(data = data, index = coordsXY.index)

def city_detection(coordsXY: list) -> pd.Series:
    """ Computes the probability of base stations' city-ness using H-DBScan.
        
        Parameters
        ----------
        coordsXY : list
            [x, y] coordinates of all points (lambert-93 projection).

        Returns
        -------
        probaCity : pd.Dataframe
            A dataframe containing labels of the clusters.
    """

    clusterer = DBSCAN(eps = 2600, min_samples = 10)
    clusterer.fit(coordsXY)

    data = {'labels' : clusterer.labels_}

    return pd.DataFrame(data = data, index = coordsXY.index)



def city_comparison(labels1, labels2):
    l1 = np.array(labels1)
    l2 = np.array(labels2)
    nb = len(l1)
    a = np.sum((l1 == -1) & (l2 == -1))/nb
    b = np.sum((l1 == -1) & (l2 != -1))/nb
    c = np.sum((l1 != -1) & (l2 == -1))/nb
    d = np.sum((l1 != -1) & (l2 != -1))/nb
    return (a,b,c,d)

def plotMapWithColors(df, countryside, colors, title):
    map = folium.Map(location=np.mean(df[['latitude','longitude']], axis=0), zoom_start=7, tiles="Cartodb Positron")
    points = folium.FeatureGroup(f"Points").add_to(map)

    for bs_id, latitude, longitude in df[['latitude', 'longitude']].itertuples():
        if(bs_id in countryside):
            color = colors[bs_id]
            points.add_child(folium.CircleMarker(location=[latitude, longitude], color=color, radius=1))
        # else:
        #     color = mean_distance_choice(bs_id, mean_distances, mean_distance_params, 'colour')
        #     points.add_child(folium.CircleMarker(location=[latitude, longitude], color=color, radius=1))

    folium.LayerControl().add_to(map)

    map.save(f"../../out/maps/{title}.html")
    return map

def mean_distance_to_NN(coordsXY: pd.DataFrame, n_neighbours: int = 4) -> pd.Series:
    """ Computes the mean distance to the n_neighbours.
        
        Parameters
        ----------
        coordsXY : DataFrame
            [x, y] coordinates of all points (lambert-93 projection).
        n_neighbours : int (default=4)
            Number of nearest neighbours.

        Returns
        -------
        mean_distances : pd.Series
            A Series containing the mean_distances to base stations' nearest neighbours.
    """
    nbrs = NearestNeighbors(n_neighbors=n_neighbours+1, metric='euclidean').fit(coordsXY)  # n_neighbors+1 because considering himself
    distances, _ = nbrs.kneighbors(coordsXY)
    
    mean_distances = np.round(np.mean(distances[:, 1:]/1000, axis=1), decimals=3) # we exclude the first element (distance to ourself is 0) and round to 3 decimals to be 1-meter precise

    return pd.Series(data=mean_distances, index=coordsXY.index)

def mean_distance_choice(node: int, mean_distances: pd.Series, mean_distance_params: dict, param: str):
    values = [elem[param] for elem in mean_distance_params.values()]
    if(mean_distances[node] <= 1.0):
        return values[0]
    elif((mean_distances[node] > 1) and (mean_distances[node] <= 2)):
        return values[1]
    elif((mean_distances[node] > 2) and (mean_distances[node] <= 4)):
        return values[2]
    else:
        return values[3]