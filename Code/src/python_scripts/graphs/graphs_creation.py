#=======================#
# Libraries importation #
#=======================#

import networkx as nx # type: ignore
from pandas import DataFrame
from scipy.spatial import Delaunay # type: ignore
from itertools import combinations
from sklearn.neighbors import NearestNeighbors
import numpy as np

#=======================================================#
# Creation of a graph based on a delaunay triangulation #
#=======================================================#

def delaunay_graph(df: DataFrame) -> tuple[nx.Graph, dict]:
    """ Returns a Networkx Graph based on the Delaunay triangulation and the position of each node.
        
        Parameters
        ----------
        df : DataFrame
            The pandas DataFrame of your data.

        Returns
        -------
        del_G : Graph
            A Delaunay triangulation graph.
        pos : dict
            The position of del_G's nodes.
    """
    points_pos = df[['latitude', 'longitude']].values

    delaunay_triangulation = Delaunay(points_pos)

    del_G = nx.Graph()
    nodes = range(len(delaunay_triangulation.points))
    del_G.add_nodes_from(nodes)

    for simplex in delaunay_triangulation.simplices:
        del_G.add_edges_from(combinations(simplex, 2))

    del_G = nx.relabel_nodes(del_G, dict(zip(nodes, df.index))) # renaming nodes according to the indexes of the dataframe

    pos = dict(zip(df.index, points_pos)) # gives each node his own position

    return del_G, pos

#=================================================================#
# Creation of the gabriel graph based on a delaunay triangulation #
#=================================================================#

def gabriel_graph(df: DataFrame) -> tuple[nx.Graph, dict]:
    """ Returns a Gabriel Graph based on the Delaunay triangulation and the position of each node.
        
        Parameters
        ----------
        df : DataFrame
            The pandas DataFrame of your data.

        Returns
        -------
        gab_G : Graph
            A Gabriel Graph.
        pos : dict
            The position of gab_G's nodes.
    """
    gab_G, pos = delaunay_graph(df)

    coordsXY = df[['x','y']]

    for edge in gab_G.edges:
        pt1 = edge[0]
        pt2 = edge[1]

        middle_point = (coordsXY.loc[pt1] + coordsXY.loc[pt2])/2

        neigh = NearestNeighbors(radius=np.sqrt(np.sum((coordsXY.loc[pt1] - coordsXY.loc[pt2])**2, axis=0))/2)
        neigh.fit(coordsXY.values)

        if(len(coordsXY.iloc[neigh.radius_neighbors([middle_point], sort_results=True)[1][0][:-2]].index)>0):
            gab_G.remove_edges_from([edge])

    return gab_G, pos

#==============================================#
# Creation of a graph based on the k-NN method #
#==============================================#

def kNN_graph(df: DataFrame, k: int = 15) -> tuple[nx.Graph, dict]:
    """ Returns a graph based on the k-NN method and the position of each node.
        
        Parameters
        ----------
        df : DataFrame
            The pandas DataFrame of your data.
        k : int (default=15)
            The number of neighbours kept for creating the graph.

        Returns
        -------
        kNN_G : Graph
            A k-NN-based Graph.
        pos : dict
            The position of kNN_G's nodes.
    """
    kNN_G = nx.Graph()
    nodes = df.index
    kNN_G.add_nodes_from(nodes)

    coordsXY = df[['x', 'y']]

    nbrs = NearestNeighbors(n_neighbors=k+1).fit(coordsXY) # +1 for taking in account the bs
    kNN_neighs = nbrs.kneighbors(coordsXY, return_distance=False)

    neighs = np.array([[None] * (k+1) for i in range(len(kNN_neighs))])
    for row in range(len(kNN_neighs)):
        for col in range(k+1):
            neighs[row, col] = coordsXY.index[kNN_neighs[row, col]]

    for row in neighs:
        bs_id = row[0]
        for bs_id_neigh in row[1:]:
            edge = [bs_id, bs_id_neigh]
            kNN_G.add_edges_from([edge])

    pos = dict(zip(nodes, df[['latitude', 'longitude']].values)) # gives each node his own position

    return kNN_G, pos