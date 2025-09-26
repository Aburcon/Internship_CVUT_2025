#=======================#
# Libraries importation #
#=======================#


import numpy as np # type: ignore
import pandas as pd # type: ignore
from tqdm import tqdm # progression bar # type: ignore
from copy import deepcopy

from .miscellaneous_for_neighbouring import *
from python_scripts.city.city_utils import mean_distance_to_NN, mean_distance_choice

#==========#
# Criteria #
#==========#

def distance_criterion_enhanced(G: nx.Graph, pos: dict, params: dict, mean_distances: pd.Series, **kwargs) -> nx.Graph:
    """ Removes all the edges of G wich are longer than the distance_range.
        
        Parameters
        ----------
        G : Graph
            A Networkx Graph graph.
        pos : dict
            The position of G's nodes.
        params : dict<dict>
            ??? .

        Returns
        -------
        modif_G : Graph
            The modified graph.
    """
    modif_G = deepcopy(G)
    
    for bs_id in tqdm(pos.keys(), desc="nodes - distance"):
        max_distance = mean_distance_choice(bs_id, mean_distances, params, 'max_distance')
        
        for [_, neigh_id] in G.edges(bs_id):
            dist = km_distance(pos[bs_id],pos[neigh_id])
            neigh_max_distance = mean_distance_choice(neigh_id, mean_distances, params, 'max_distance')
            if((dist > max_distance) and (dist > neigh_max_distance)):
                modif_G.remove_edges_from([[bs_id, neigh_id]])

    return modif_G

def quadrant_criterion_enhanced(G: nx.Graph, pos: dict, params: dict, mean_distances: pd.Series, k_nn: int = 1, **kwargs) -> nx.Graph:
    """ Removes all the edges of G wich doesn't respect the quadrant criterion.
        
        Parameters
        ----------
        G : Graph
            A Networkx Graph graph.
        pos : dict
            The position of G's nodes.
        k_nn: int
            How much neighbours you want to keep per quadrant.

        Returns
        -------
        modif_G : Graph
            The modified graph.
    """
    modif_G = deepcopy(G)

    for bs_id in tqdm(pos.keys(), desc="nodes - quadrant"):
        neighbours = [edge[1] for edge in modif_G.edges(bs_id)]
        edges_to_remove = list(modif_G.edges(bs_id))
        quadrants = create_6_quadrants_enhanced(bs_id, neighbours, pos)

        nb_neighbours = 0
        while(nb_neighbours < k_nn):
            for quad in quadrants.values():
                nearest_neighbour = nearestNeighbour(bs_id, quad, pos)
                if(nearest_neighbour != bs_id):
                    edges_to_remove.remove((bs_id, nearest_neighbour)) # removing the nearest neighbour in the quadrant centered on bs_id from the edges to remove
                    quad.remove(nearest_neighbour)
            nb_neighbours += 1
        
        modif_G.remove_edges_from(edges_to_remove)

    return modif_G

def quadrant_criterion_enhanced_v2(G: nx.Graph, pos: dict, params: dict, mean_distances: pd.Series, k_nn: int = 1, **kwargs) -> nx.Graph:
    """ Removes all the edges of G wich doesn't respect the quadrant criterion v2.
        
        Parameters
        ----------
        G : Graph
            A Networkx Graph graph.
        pos : dict
            The position of G's nodes.
        k_nn: int
            How much neighbours you want to keep per quadrant.

        Returns
        -------
        modif_G : Graph
            The modified graph.
    """
    df_az = kwargs.get('df_azimuth', None)

    modif_G = deepcopy(G)

    for bs_id in tqdm(pos.keys(), desc="nodes - quadrant"):
        neighbours = [edge[1] for edge in modif_G.edges(bs_id)]
        edges_to_remove = list(modif_G.edges(bs_id))
        quadrants = create_azimuth_quadrants(bs_id, neighbours, pos, np.unique(df_az.loc[df_az['id_station_anfr']==bs_id,'angle_azimuth']))

        nb_neighbours = 0
        while(nb_neighbours < k_nn):
            for quad in quadrants.values():
                nearest_neighbour = nearestNeighbour(bs_id, quad, pos)
                if((nearest_neighbour != bs_id) and ((bs_id, nearest_neighbour) in edges_to_remove)):
                    edges_to_remove.remove((bs_id, nearest_neighbour)) # removing the nearest neighbour in the quadrant centered on bs_id from the edges to remove
                    quad.remove(nearest_neighbour)
            nb_neighbours += 1
        
        modif_G.remove_edges_from(edges_to_remove)

    return modif_G

def angle_criterion_enhanced(G: nx.Graph, pos: dict, params: dict, mean_distances: pd.Series, **kwargs) -> nx.Graph:
    """ Removes all the edges of G wich doesn't respect the angle criterion.
        
        Parameters
        ----------
        G : Graph
            A Networkx Graph graph.
        pos : dict
            The position of G's nodes.
        params : dict<dict>
            ??? .

        Returns
        -------
        modif_G : Graph
            The modified graph.
    """
    modif_G = deepcopy(G)

    for bs_id in tqdm(pos.keys(), desc="nodes - angles"):
        min_angle = mean_distance_choice(bs_id, mean_distances, params, 'min_angle')

        neighbours = [edge[1] for edge in modif_G.edges(bs_id)]
        angles = compute_angles(bs_id, neighbours, pos)
        idx_angles = np.argsort(angles) # angles of bs_id's neighbours

        for i, id in enumerate(idx_angles):
            if(i < len(idx_angles) - 1):
                next_id = idx_angles[i + 1]
                if((angles[next_id] - angles[id]) <= min_angle):
                    angle_elim(modif_G, pos, bs_id, neighbours, id, next_id)
                
        if(idx_angles.size!=0):
            next_id = idx_angles[-1]
            id = idx_angles[0]
            if((360 - angles[next_id] + angles[id]) <= min_angle):
                angle_elim(modif_G, pos, bs_id, neighbours, id, next_id)
        
    return modif_G