#=======================#
# Libraries importation #
#=======================#

import folium
import numpy as np
from networkx import Graph
from pandas import DataFrame, Series

from python_scripts.city.city_utils import mean_distance_choice

#==================#
# Useful functions #
#==================#

def addLegend(map, labelsToColors):
    legend_html = f'''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 200px; height: 120px; 
                border:2px solid grey; z-index:9999; font-size:14px;
                background-color: white;
                "><strong>Legend</strong> <br>
    '''
    for label in labelsToColors.keys():
        legend_html += f"\n&nbsp; <span style='color:{labelsToColors.get(label).get('colour')}'>&#9632;</span> {label} <br>\n"
    legend_html += '</div>'
    map.get_root().html.add_child(folium.Element(legend_html))

def add_graph_edges(G_base: Graph, G: Graph, df: DataFrame, fg: folium.FeatureGroup, colour: str):
    created_edges = []
    for edge in G_base.edges:
        stations = []

        if((not(edge in created_edges)) and (not(edge in G.edges))):
            created_edges.append(edge)
            stations.append(df.loc[edge[0], ['latitude', 'longitude']])
            stations.append(df.loc[edge[1], ['latitude', 'longitude']])

            folium.PolyLine(np.array(stations), color=colour, weight=2.5, opacity=1).add_to(fg)

#==============#
# Map creation #
#==============#

def create_method_illustation_map(df: DataFrame, base_graph: Graph, base_graph_name: str, neigh_graph: Graph, mean_distances: Series, mean_distance_params: dict, save_as: str, **kwargs):
    G_dis = kwargs.get('dis_filt', None)
    G_ang = kwargs.get('ang_filt', None)
    G_qua = kwargs.get('qua_filt', None)
    
    map = folium.Map(location=list(np.mean(df[['latitude','longitude']], axis=0)), zoom_start=8.5, tiles="Cartodb Positron")

    edges_base = folium.FeatureGroup(f"Edges - {base_graph_name} ({len(base_graph.edges)})", show=True).add_to(map)
    edges_neigh = folium.FeatureGroup(f"Edges - neighbouring graph ({len(neigh_graph.edges)})", show=False).add_to(map)
    add_graph_edges(base_graph, Graph(), df, edges_base, colour="lightblue")
    add_graph_edges(neigh_graph, Graph(), df, edges_neigh, colour="#AAA662")

    if(G_dis):
        edges_dis = folium.FeatureGroup(f"Edges - distance filtering ({len(G_dis.edges)})", show=False).add_to(map)
        add_graph_edges(base_graph, G_dis, df, edges_dis, colour="red")
    if(G_ang):
        edges_ang = folium.FeatureGroup(f"Edges - angle filtering ({len(G_ang.edges)})", show=False).add_to(map)
        add_graph_edges(base_graph, G_ang, df, edges_ang, colour="orange")
    if(G_qua):
        edges_qua = folium.FeatureGroup(f"Edges - quadrant filtering ({len(G_qua.edges)})", show=False).add_to(map)
        add_graph_edges(base_graph, G_qua, df, edges_qua, colour="green")

    points = folium.FeatureGroup(f"Base stations ({len(df)})").add_to(map)

    for ind, latitude, longitude in df[['latitude', 'longitude']].itertuples():
        popup_text = folium.Popup(
            f"Id_anfr: {df.loc[ind, 'id_station_anfr']}<br>"
            f"Commune: {df.loc[ind, 'nom_com']}<br>"
            f"3NN distance: {mean_distances.get(ind)}<br>"
            , max_width=150)

        color = mean_distance_choice(ind, mean_distances, mean_distance_params, 'colour')
        points.add_child(folium.CircleMarker(location=[latitude, longitude], color=color, radius=2.5, popup=popup_text, fillOpacity=1, fill=True))
    folium.LayerControl().add_to(map)

    addLegend(map, mean_distance_params)

    map.save(f"../../out/maps/neighbours_finding/{save_as}.html")