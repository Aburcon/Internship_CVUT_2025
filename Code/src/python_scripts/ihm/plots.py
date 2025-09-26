#=======================#
# Libraries importation #
#=======================#

import matplotlib.pyplot as plt # type: ignore
import networkx as nx # type: ignore
import pandas as pd # type: ignore
# import geopandas as gpd # type: ignore
from copy import deepcopy

#==================#
# Global variables #
#==================#

EDGE_COLOR = "lightblue"
NODE_COLOR = "green"

#=====================#
# Plotting parameters #
#=====================#

def plot_params(title, ax, long_points, lat_points):
    """ Sets the right parameters for subplots axes.
        
        Parameters
        ----------
        title : string
            Title of the subplot.
        ax : Matplotlib Axes object (default=None)
            Draw the graph in the specified Matplotlib axes.
        long_points : array
            Longitude coordinates of the points.
        lat_points : array
            Latitude coordinates of the points.
    """
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    
    ax.set_title(title)

    ax.tick_params(
        reset=True,
        top=False,
        right=False
    )
    
    # ax.set_xlim(min(long_points)-0.05, max(long_points)+0.05)
    # ax.set_ylim(min(lat_points)-0.05, max(lat_points)+0.05)

# #====================#
# # Department borders #
# #====================#

# # Importing database
# df = pd.read_csv("../../database/data.csv", sep=";", decimal=",")
# df.head()

# # Importing the borders database
# chemin = '../../local/travail_anneePrec/departements-20180101.shx'
# geo_df_shx = gpd.read_file(chemin)

# # Removing useless columns and renaming
# geo_df = geo_df_shx.drop([ 'wikipedia', 'nuts3', 'surf_km2'], axis = 1)
# geo_df = geo_df.rename(columns={'code_insee': 'insee_dep', 'nom': 'nom_dep'})
# geo_df = geo_df.drop(geo_df[(geo_df['insee_dep'] == '973') | (geo_df['insee_dep'] == '972') | (geo_df['insee_dep'] == '974') | (geo_df['insee_dep'] == '976') | (geo_df['insee_dep'] == '971')].index)
# geo_df.index = geo_df.index.astype(str)

# # On joint la Métropole de Lyon et le Rhône
# polygone_rhone = geo_df[geo_df['insee_dep'] == "69D"]['geometry'].values[0]
# polygone_lyon = geo_df[geo_df['insee_dep'] == "69M"]['geometry'].values[0]


# # Créer un MultiPolygon avec les deux polygones
# multipolygone_69 = polygone_rhone.union(polygone_lyon)

# # Créer une nouvelle ligne avec insee_dep = '69' et le MultiPolygon
# dep_rhone = {
#     'insee_dep': '69',
#     'nom_dep': 'Rhône',
#     'geometry': multipolygone_69
# }

# # Convertir la nouvelle ligne en GeoDataFrame
# nouveau_gdf = gpd.GeoDataFrame([dep_rhone], geometry='geometry')

# # Supprimer les anciennes lignes avec insee_dep = '69'
# geo_df = geo_df[geo_df['insee_dep'] != '69D']
# geo_df = geo_df[geo_df['insee_dep'] != '69M']

# # Ajouter la nouvelle ligne au GeoDataFrame
# geo_df = gpd.GeoDataFrame(pd.concat([geo_df, nouveau_gdf], ignore_index=True), crs=geo_df.crs)

# geo_df = geo_df.drop_duplicates()

# # Replacing department names by those from the database
# for code in geo_df['insee_dep']:
#     geo_df.loc[geo_df['insee_dep']==code, 'nom_dep'] = df.loc[df['insee_dep']==code, 'nom_dep'].iloc[0]

# def plot_borders(department, ax):
#     if(department != None):
#         geo_df.loc[geo_df['nom_dep']==department,:].plot(edgecolor = "black", linewidth = 1, color="white", ax=ax)


#=================================#
# Plotting Delaunay triangulation #
#=================================#

def plot_delaunay(delaunay_triangulation, show=True, **kwargs):
    """ Plots a Delaunay triangulation.
        
        Parameters
        ----------
        delaunay_triangulation : Delaunay
            Result of the Delaunay triangulation.
        show : bool (default=True)
            If True, shows the plot.
        ax : Matplotlib Axes object (default=None)
            Draw the graph in the specified Matplotlib axes.
    """
    ax = kwargs.get('ax', plt)
    # department = kwargs.get('department', None)

    # plot_borders(department, ax)

    ax.triplot(delaunay_triangulation.points[:,0], delaunay_triangulation.points[:,1], delaunay_triangulation.simplices, linewidth=1, c=EDGE_COLOR)
    ax.plot(delaunay_triangulation.points[:,0], delaunay_triangulation.points[:,1], 'o', markersize=3, c=NODE_COLOR)
    
    if(ax!=plt):
        plot_params("Delaunay Triangulation", ax, delaunay_triangulation.points[:,0], delaunay_triangulation.points[:,1])
    else:
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.tick_params(
            reset=True,
            top=False,
            right=False)
    
    if(show):
        plt.show()


#==================#
# Plotting a graph #
#==================#

def plot_graph(G, pos, show=True, **kwargs):
    """ Plots a Networkx Graph.
        
        Parameters
        ----------
        G : Graph
            The Networkx Graph graph you want to plot.
        pos : dict
            The position of G's nodes.
        show : bool (default=True)
            If True, shows the plot.
        ax : Matplotlib Axes object (default=None)
            Draw the graph in the specified Matplotlib axes.
        title : string
            The title of the subplot (works with ax).
    """
    ax = kwargs.get('ax', None)
    title = kwargs.get('title', "Delaunay Graph")
    # department = kwargs.get('department', None)

    pos_copy = deepcopy(pos)

    for k in pos_copy.keys():
        pos_copy[k] = pos_copy[k][::-1] # to have values in pos shaped like ['long', 'lat'] instead of ['lat', 'long']

    # plot_borders(department, ax)

    nx.draw_networkx(G, pos_copy, node_size=10, with_labels=False, node_color=NODE_COLOR, edge_color=EDGE_COLOR, ax=ax)

    if(ax):
        plot_params(title, ax, [pos_copy[k][0] for k in pos], [pos_copy[k][1] for k in pos])
    else:
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.tick_params(
            reset=True,
            top=False,
            right=False)
    
    if(show):
        plt.show()