import customtkinter as ctk
import folium
import pandas as pd
import webbrowser
from sklearn.cluster import DBSCAN
import tkinter as tk
from scipy.spatial import Voronoi, Delaunay
import numpy as np
from functools import partial

import mapUtils as mu
from copy import deepcopy

ctk.set_default_color_theme("dark-blue")

df = pd.read_csv("../../../../database/data.csv", sep=";", decimal=',')


df = df.sample(frac=1).reset_index(drop=True)

# Sample data for regions and providers
providers = df['nom_op'].unique()
regions = df['nom_reg'].unique()    
technologies = pd.Series(['site_2g', 'site_3g', 'site_4g', 'site_5g'])

def select_all(variables, select_all_state):
    for var in variables:
        var.set(select_all_state)

def plotMap():
    selected_option = action_var.get()
    if(selected_option in action_options):
        selected_providers = [providers[i] for i in range(len(providers)) if provider_vars[i].get() == 1]
        selected_regions = [regions[i] for i in range(len(regions)) if region_vars[i].get() == 1]
        selected_technologies = [technologies[i] for i in range(len(technologies)) if technology_vars[i].get() == 1]

        map = mu.getMap(data = deepcopy(df), selected_providers = selected_providers, selected_regions = selected_regions, selected_technologies = selected_technologies, option = action_var.get(), epsilon=float(epsilon_entry.get()), nmin=int(n_min_entry.get()))
        # Save map to HTML file
        map_file = "temp_map"
        map.save(f"../../../../out/maps/{map_file}.html")

        # Open the map file in a web browser
        webbrowser.open(f"../../../../out/maps/{map_file}.html")

if(__name__=='__main__'):
    # Création de la fenêtre principale
    window = ctk.CTk()
    window.minsize(1400, 1000)  # Taille minimale de la fenêtre
    window.maxsize(2500, 1500)  # Taille maximale de la fenêtre
    window.title("Visualisator 3000")
    window.grid_rowconfigure(0, weight=1)  # Centrer verticalement
    window.grid_columnconfigure(0, weight=1)  # Centrer horizontalement

    # Titre
    title_label = ctk.CTkLabel(window, text="Outil de visualisation des stations téléphoniques françaises", font=("Helvetica", 31, 'bold'), anchor="center", pady=15, padx=20)
    title_label.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

    # Menu déroulant pour les actions
    action_options = ["Opérateurs", "Technologies", "Villes", "Delaunay", "Voronoi"]
    action_var = tk.StringVar(window)
    action_var.set(action_options[0])  # Valeur par défaut
    action_combo_box = ctk.CTkComboBox(window, values=action_options, font=("Helvetica", 25), variable=action_var, width=200, height=50, dropdown_font=("Helvetica", 25))
    action_combo_box.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

    # Layout pour les régions, opérateurs et technologies
    selection_frame = ctk.CTkFrame(window)
    selection_frame.grid(row=2, column=0, padx=10, pady=10)

    # En tête de colonne pour les régions avec case à cocher pour sélectionner toute la colonne
    region_label = ctk.CTkLabel(selection_frame, text="Régions:", font=("Helvetica", 30, 'bold'), anchor="center")
    region_label.grid(row=0, column=0, sticky="w", pady=20, padx=35)
    region_select_all_var = tk.BooleanVar()
    region_select_all_checkbox = ctk.CTkCheckBox(selection_frame, text="Tout sélectionner", font=("Helvetica", 20), variable=region_select_all_var, command=lambda: select_all(region_vars, region_select_all_var.get()))
    region_select_all_checkbox.grid(row=0, column=1, sticky="w", pady=20, padx=10)

    # ComboBox pour les régions avec case à cocher
    region_vars = [tk.IntVar() for _ in range(len(regions))]
    region_checkboxes = [ctk.CTkCheckBox(selection_frame, text=region, font=("Helvetica", 20), variable=region_vars[i], checkbox_height=30, checkbox_width=30) for i, region in enumerate(regions)]
    for i, checkbox in enumerate(region_checkboxes):
        checkbox.grid(row=i+1, column=0, sticky="w", padx=40, pady=10)

    # En tête de colonne pour les opérateurs avec case à cocher pour sélectionner toute la colonne
    provider_label = ctk.CTkLabel(selection_frame, text="Opérateurs:", font=("Helvetica", 30, 'bold'), anchor="center")
    provider_label.grid(row=0, column=2, sticky="w", pady=20, padx=35)
    provider_select_all_var = tk.BooleanVar()
    provider_select_all_checkbox = ctk.CTkCheckBox(selection_frame, text="Tout sélectionner", font=("Helvetica", 20), variable=provider_select_all_var, command=lambda: select_all(provider_vars, provider_select_all_var.get()))
    provider_select_all_checkbox.grid(row=0, column=3, sticky="w", pady=20, padx=10)

    # ComboBox pour les opérateurs avec case à cocher
    provider_vars = [tk.IntVar() for _ in range(len(providers))]
    provider_checkboxes = [ctk.CTkCheckBox(selection_frame, text=provider, font=("Helvetica", 20), variable=provider_vars[i], checkbox_height=30, checkbox_width=30) for i, provider in enumerate(providers)]
    for i, checkbox in enumerate(provider_checkboxes):
        checkbox.grid(row=i+1, column=2, sticky="w", padx=40, pady=10)

    # Liste des technologies disponibles
    display_technologies = technologies.apply(lambda tech : tech.split('_')[1].upper())

    # En tête de colonne pour les technologies avec case à cocher pour sélectionner toute la colonne
    technology_label = ctk.CTkLabel(selection_frame, text="Technologies:", font=("Helvetica", 30, 'bold'), anchor="center")
    technology_label.grid(row=0, column=4, sticky="w", pady=20, padx=35)
    technology_select_all_var = tk.BooleanVar()
    technology_select_all_checkbox = ctk.CTkCheckBox(selection_frame, text="Tout sélectionner", font=("Helvetica", 20), variable=technology_select_all_var, command=lambda: select_all(technology_vars, technology_select_all_var.get()))
    technology_select_all_checkbox.grid(row=0, column=5, sticky="w", pady=20, padx=10)

    # Création de variables pour stocker les valeurs sélectionnées de la technologie
    technology_vars = [tk.IntVar() for _ in range(len(technologies))]

    # CheckBox pour les technologies
    technology_checkboxes = [ctk.CTkCheckBox(selection_frame, text=tech, font=("Helvetica", 20), variable=technology_vars[i], checkbox_height=30, checkbox_width=30) for i, tech in enumerate(display_technologies)]
    for i, checkbox in enumerate(technology_checkboxes):
        checkbox.grid(row=i+1, column=4, sticky="w", padx=40, pady=10)

    # Layout pour les paramètres numériques (epsilon et n_min)
    param_frame = ctk.CTkFrame(selection_frame)
    param_frame.grid(row=0, column=6, rowspan=len(providers)+1, padx=10, pady=10)

    # Labels et champs de saisie pour epsilon et n_min avec des valeurs par défaut
    epsilon_label = ctk.CTkLabel(param_frame, text="Epsilon:", font=("Helvetica", 25), anchor="center")
    epsilon_label.grid(row=0, column=0, sticky="e", padx=10, pady=10)
    epsilon_entry = ctk.CTkEntry(param_frame, font=("Helvetica", 25), width=100)
    epsilon_entry.grid(row=0, column=1, sticky="w", padx=10, pady=10)
    epsilon_entry.insert(0, "0.03")  # Valeur par défaut pour epsilon

    n_min_label = ctk.CTkLabel(param_frame, text="N Min:", font=("Helvetica", 25), anchor="center")
    n_min_label.grid(row=1, column=0, sticky="e", padx=10, pady=10)
    n_min_entry = ctk.CTkEntry(param_frame, font=("Helvetica", 25), width=100)
    n_min_entry.grid(row=1, column=1, sticky="w", padx=10, pady=10)
    n_min_entry.insert(0, "15")  # Valeur par défaut pour n_min

    param_frame.grid_forget()

    # Fonction pour afficher ou cacher les paramètres en fonction de l'option sélectionnée
    def toggle_params(option):
        if option == "Villes":
            param_frame.grid(row=0, column=6, rowspan=len(regions)+1, padx=10, pady=10)
        else:
            param_frame.grid_forget()

    # Lier la fonction de bascule au changement de sélection dans la liste déroulante
    action_var.trace_add("write", lambda *args: toggle_params(action_var.get()))

    # Bouton Afficher
    afficher_button = ctk.CTkButton(window, text="Afficher", font=("Helvetica", 30), command=plotMap, height=75, width=500)
    afficher_button.grid(row=3, column=0, columnspan=2, padx=10, pady=20)

    window.update_idletasks()
    window.mainloop()
