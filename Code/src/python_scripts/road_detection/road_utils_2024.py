import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.cluster import DBSCAN
import numpy as np
import folium.features
from sklearn.cluster import DBSCAN
from sklearn.cluster import HDBSCAN
from sklearn.cluster import OPTICS
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures
from pyproj import Transformer
import seaborn as sns
import branca

from python_scripts.city.city_utils import mean_distance_to_NN

def plotMapWithColorsAndLayers(df, countryside, colors, linearModels, xBounds, clusters, layersContent, layersLabel, mapName = "Cartodb Positron"):
    map = folium.Map(location=np.mean(df[['latitude','longitude']], axis=0), zoom_start=7, tiles=mapName)
    layers = [folium.FeatureGroup(layerLabel) for layerLabel in layersLabel]
    for layer in layers : layer.add_to(map)
    layers = {clustId : layer for layer, layerContent in zip(layers, layersContent) for clustId in layerContent}

    for bs_id, latitude, longitude in df[['latitude', 'longitude']].itertuples():
        if(bs_id in countryside):
            color = colors[bs_id]
            layers.get(clusters[bs_id]).add_child(folium.CircleMarker(location=[latitude, longitude], color=color, radius=1, popup=f"cluster : {clusters[bs_id]}"))
        # else:
        #     color = mean_distance_choice(bs_id, mean_distances, mean_distance_params, 'colour')
        #     points.add_child(folium.CircleMarker(location=[latitude, longitude], color=color, radius=1))
    for clustId in clusters.unique():
        if (clustId != -1):
            xBound = xBounds[clustId]
            x = np.linspace(xBound[0], xBound[1], 100)
            pr = PolynomialFeatures(degree = len(linearModels[clustId].coef_[0])-1)
            X_poly = pr.fit_transform(x.reshape(-1, 1))
            y = linearModels[clustId].predict(X_poly)

            # print(f"x : {x}")
            # print(f"y : {y}")

            transformer = Transformer.from_crs("epsg:2154", "epsg:4326")
            latitudes, longitudes = transformer.transform(x, y)

            layers.get(clustId).add_child(folium.PolyLine(list(zip(latitudes, longitudes)), color='black', popup=f"id du cluster : {clustId}"))


    folium.LayerControl().add_to(map)

    return map

def road_get_clust_hdbscan(df_extracted, countryside):
    hdbscan = HDBSCAN(cluster_selection_epsilon=4500, min_cluster_size=20, min_samples=2, max_cluster_size=100).fit(df_extracted[['x','y']].loc[countryside])
    clust_hdbscan = np.array(hdbscan.labels_)
    np.where(hdbscan.probabilities_ < 0.9, -1, clust_hdbscan)
    clust_hdbscan = pd.Series(clust_hdbscan, index = countryside)
    return clust_hdbscan

def road_get_clust_dbscan(df_extracted, countryside):
    return  pd.Series(DBSCAN(eps=6000, min_samples=6).fit(df_extracted[['x','y']].loc[countryside]).labels_, index = countryside)

def road_get_clust_optics(df_extracted, countryside):
    return pd.Series(OPTICS(max_eps=20000, min_samples=6).fit(df_extracted[['x','y']].loc[countryside]).labels_, index = countryside)

def detect_roads_based_on_clusters(clusters, df_extracted):
    excluded_clusters = []
    linearModels = {}
    xBounds = {}
    for clustId in clusters.unique():
        if (clustId != -1):
            clust = df_extracted.loc[clusters[clusters == clustId].index]
            if (len(clust)>0):
                pr = PolynomialFeatures(degree = 10)
                X_poly = pr.fit_transform(clust[['x']])
                lr_2 = LinearRegression()
                lr_2.fit(X_poly, clust[['y']])
                linearModels[clustId] = lr_2

                x = clust['x'].values

                xBounds[clustId] = (np.min(x), np.max(x))
                if (lr_2.score(X_poly, clust[['y']]) < 0.15 ):
                    excluded_clusters.append(clustId)
    not_exluded_clusters = [clustId for clustId in clusters if clustId not in excluded_clusters and clustId != -1 ]
    return (excluded_clusters, not_exluded_clusters, linearModels, xBounds)


def road_candidates(df_extracted):
    mean_distances = mean_distance_to_NN(df_extracted[['x', 'y']], n_neighbours=3) # 3 to have more neighbours
    countryside = []
    for station in mean_distances.index:
        if(mean_distances[station]>2):
            countryside.append(station)

    clust_dbscan = pd.Series(DBSCAN(eps=4500, min_samples=4).fit(df_extracted[['x','y']].loc[countryside]).labels_, index = countryside)
    clust_hdbscan = pd.Series(HDBSCAN(cluster_selection_epsilon=4500, min_cluster_size=2, min_samples=2, alpha=75).fit(df_extracted[['x','y']].loc[countryside]).labels_, index = countryside)
    clust_optics = pd.Series(OPTICS(max_eps=4500, min_samples=4).fit(df_extracted[['x','y']].loc[countryside]).labels_, index = countryside)

    for clustering in [clust_dbscan, clust_hdbscan, clust_optics]:
        for cluster in clustering.loc[clustering!=-1].unique():
            nb_elem = list(clustering).count(cluster)
            if(nb_elem <= 10):
                clustering.loc[clustering==cluster] = -1

    possible_points = []
    for bsId in df_extracted.index:
        if (bsId not in countryside):
            possible_points.append(bsId)
        elif((clust_dbscan[bsId]!=-1) or (clust_dbscan[bsId]!=-1) or (clust_dbscan[bsId]!=-1)):
            possible_points.append(bsId)
    
    return possible_points

def distance_to_segment(A, B, C):
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    
    P = A + (np.dot(B-A, C-A))/(np.linalg.norm(B-A)**2) * (B-A)
    
    k = np.dot(B-A, P-A) / np.linalg.norm(B-A)**2

    dist = np.linalg.norm(P-C) if ((k >= 0) and (k <= 1)) else np.inf
    
    return dist, P