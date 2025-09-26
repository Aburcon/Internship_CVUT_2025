#=======================#
# Libraries importation #
#=======================#

import numpy as np # type: ignore
from geopy import distance # type: ignore
import pandas as pd # type: ignore
import networkx as nx # type: ignore
import math

#==================#
# Useful functions #
#==================#

def km_distance(pt1: list, pt2: list) -> float:
    """ Computes the distance in km between pt1 and pt2.
        
        Parameters
        ----------
        pt1, pt2 : list
            The ['latitude', 'longitude'] coordinates of the points.

        Returns
        -------
        km_distance : float
            The distance in km between pt1 and pt2.
    """
    return distance.distance(pt1, pt2).km

def compute_angles(ref_point: int, adj: list, pos: dict) -> np.array:
    """ Computes the angle position of ref_point's neighbours.
        
        Parameters
        ----------
        ref_point : int
            The reference of the central point.
        adj : list
            An array containing references of ref_point's adjacent nodes.
        pos : dict
            A dictionnary referencing all points' positions.

        Returns
        -------
        angles : np.array
            The angle position of ref_point's neighbours (from 0 to 360°).
    """
    angles = []
    for neighbour in adj:
        x_neighbour = pos[neighbour][0]
        y_neighbour = pos[neighbour][1]

        angles.append(np.degrees(np.arctan2(y_neighbour - pos[ref_point][1], x_neighbour - pos[ref_point][0])))
    angles = [np.round((k + 360) % 360, decimals=2) for k in angles]

    return np.array(angles)

def create_6_quadrants(ref_point: int, adj: list, pos: dict) -> dict:
    """ Creates 6 quadrants around ref_point.
        
        Parameters
        ----------
        ref_point : int
            The reference of the central point.
        adj : list
            An array containing references of ref_point's adjacent nodes.
        pos : dict
            A dictionnary referencing all points' positions.

        Returns
        -------
        quadrants : dict
            A dictionnary referencing in which quadrant adj's point are.
    """
    angles = compute_angles(ref_point, adj, pos)

    quadrants = dict()
    for ind in range(0, 301, 60):
        quadrants[f"{ind}_{ind+60}"] = [adj[k] for k in np.where((angles >= ind) & (angles < ind + 60))[0]] # to have the real name of the node

    return quadrants

def create_azimuth_quadrants(ref_point: int, adj: list, pos: dict, azimuths: pd.Series) -> dict:
    """ Creates 6 quadrants around ref_point.
        
        Parameters
        ----------
        ref_point : int
            The reference of the central point.
        adj : list
            An array containing references of ref_point's adjacent nodes.
        pos : dict
            A dictionnary referencing all points' positions.

        Returns
        -------
        quadrants : dict
            A dictionnary referencing in which quadrant adj's point are.
    """
    angles = compute_angles(ref_point, adj, pos)

    quadrants = dict()
    nb_az = len(azimuths)

    if(nb_az>0):
        for id, az in enumerate(azimuths):
            next_id = (id+1) % nb_az
            next_az = azimuths[next_id]
            biss = int(((((next_az - az) / 2) % 180) + az) % 360)
            quadrants[f"{biss}_{next_az}"] = [adj[k] for k in np.where((angles >= biss) & (angles < next_az))[0]]
            quadrants[f"{az}_{biss}"] = [adj[k] for k in np.where((angles >= az) & (angles < biss))[0]]

    return quadrants

# for az in azimuths:
#     sup_angle = (az + semi_cover_angle) % 360
#     inf_angle = (az - semi_cover_angle) % 360
#     quadrants[f"{inf_angle}_{az}"] = [adj[k] for k in np.where((angles >= (inf_angle)) & (angles < az))[0]] # to have the real name of the node
#     quadrants[f"{az}_{sup_angle}"] = [adj[k] for k in np.where((angles >= az) & (angles < (sup_angle)))[0]] # to have the real name of the node

def compute_distance_to_quadrant(ref_point: int, adj: list, pos: dict, angles: list, gap: int) -> float:
    """ Computes the sum of the distances of each point to its closest quadrant edge.
        
        Parameters
        ----------
        ref_point : int
            The reference of the central point.
        adj : list
            An array containing references of ref_point's adjacent nodes.
        pos : dict
            A dictionnary referencing all points' positions.
        angles : list
            An array containing angles from ref_point to ref_point's adjacent nodes.
        gap : int
            The angle gap of the quadrant.

        Returns
        -------
        distance : float
            The sum of the distances of each point to its closest quadrant edge.    
    """
    quadrant_angles = [angle + gap for angle in range(0, 301, 60)]
    closest_quadrant_delimations = [quadrant_angles[np.argmin((np.abs((quadrant_angles-angle) % 360)))] for angle in angles]
    distances = [math.dist(pos[ref_point], pos[point]) * math.sin(abs(angle-quadrant_angle)) for angle, quadrant_angle, point in zip(angles, closest_quadrant_delimations, adj)]
    
    return np.sum(distances)

def create_6_quadrants_enhanced(ref_point: int, adj: list, pos: dict) -> dict:
    """ Creates 6 quadrants around ref_point, with an angle optimized angle gap (quadrant doesn't always start at 0 degrees).
        
        Parameters
        ----------
        ref_point : int
            The reference of the central point.
        adj : list
            An array containing references of ref_point's adjacent nodes.
        pos : dict
            A dictionnary referencing all points' positions.

        Returns
        -------
        quadrants : dict
            A dictionnary referencing in which quadrant adj's point are.
    """
    angles = compute_angles(ref_point, adj, pos)
    gaps = range(0, 60, 5)
    distances = [compute_distance_to_quadrant(ref_point, adj, pos, angles, gap) for gap in gaps]
    selected_gap = gaps[np.argmax(distances)]

    angles = angles + selected_gap
    angles = angles % 360 
    quadrants = dict()
    for ind in range(0, 301, 60):
        quadrants[f"{ind}_{ind+60}"] = [adj[k] for k in np.where((angles >= ind) & (angles < ind + 60))[0]] # to have the real name of the node

    return quadrants

def nearestNeighbour(ref_point: int, adj: list, pos: dict) -> int:
    """ Gives the nearest neighbour around ref_point.
        
        Parameters
        ----------
        ref_point : int
            The reference of the central point.
        adj : list
            An array containing references of ref_point's adjacent nodes.
        pos : dict
            A dictionnary referencing all points' positions.

        Returns
        -------
        nearestNeighbour : int
            The reference of the nearest adjacent node, if adj is empty it returns ref_point.
    """
    min = np.inf
    nearestNeighbour = ref_point
    
    for pt2 in adj:
        dist = km_distance(pos[ref_point], pos[pt2])
        if((dist > 0) and (dist < min)):
            min = dist
            nearestNeighbour = pt2

    return nearestNeighbour

#=======================#
# Criteria eliminations #
#=======================#

def distance_elim(G, pos, edge, max_distance):
    if(km_distance(pos[edge[0]],pos[edge[1]]) > max_distance):
        G.remove_edges_from([edge])

def angle_elim(G: nx.Graph, pos: dict, ref_point: int, adj: list, id: int, next_id: int):
    """ Computes the angle elimination for angle criterion (keeps the nearest point).
        
        Parameters
        ----------
        G : Graph
            A Networkx Graph graph.
        pos : dict
            The position of G's nodes.
        ref_point : int
            The reference of the central point.
        adj : list
            An array containing references of ref_point's adjacent nodes.
        id, next_id : int
            The indices of the adjacent nodes ind adj.
    """
    if(km_distance(pos[ref_point], pos[adj[id]]) > km_distance(pos[ref_point], pos[adj[next_id]])):
        G.remove_edges_from([(ref_point, adj[id])])
    else:
        G.remove_edges_from([(ref_point, adj[next_id])])   
