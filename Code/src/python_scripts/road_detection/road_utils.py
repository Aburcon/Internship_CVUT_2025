import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import ConvexHull
from collections import defaultdict, Counter

##### Fonctions de base

# Masque des stations hors ville
def points_hors_ville(xy, k = 5, seuil_dist_ville = 2000):  # Si la moyenne des distances aux k plus proches voisins d'une station est inférieure, alors c'est en ville
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(xy)
    distances, _ = nbrs.kneighbors(xy)
    distances_to_k = distances[:, 1:]     # On exclut la première colonne (distance à soi-même = 0)

    distances_moyenne = distances_to_k.mean(axis=1)
    masque = distances_moyenne >= seuil_dist_ville
    return xy[masque], masque

# Angle entre 2 vecteurs
def angle_between(v1, v2):
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0
    cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))

# Angle entre 3 points (p1, p2, p3)
def angle_triplet(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    return angle_between(v1, v2)

# Affichage
def affichage_chains(xy, chains, title):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.scatter(xy[:, 0], xy[:, 1], s=2, c='lightgray', label='Stations')

    for chain in chains:
        coords = xy[chain]
        ax.plot(coords[:, 0], coords[:, 1], '-', lw=2)

    ax.set_title(title)
    ax.set_aspect('equal')
    plt.show()
    
# Direction globale d'une chaîne
def chain_direction(chain, xy):
    vec = xy[chain[-1]] - xy[chain[0]]
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else vec
    
# Direction d'un vecteur
def direction(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else vec

# Transformer des chaînes formées par département en chaînes avec indices globaux
def to_global_chains(chains_by_dep, dep_to_indices):   
    global_chains = []
    for dep, chains in chains_by_dep.items():
        global_indices = dep_to_indices[dep]
        for chain in chains:
            try:
                global_chain = [global_indices[i] for i in chain]
                global_chains.append(global_chain)
            except IndexError:
                print(f"Indice hors limite dans le département {dep}")
    return global_chains
    
##### Fonctions d'erreurs et métriques

df_routes = pd.read_csv("database/data_len_routes.csv", sep=",", decimal='.')
df_chemin_fer = pd.read_csv("database/data_len_train.csv", sep=";", decimal=',')

# ref_total_length = (
#     df_routes['Autoroutes'].sum() +
#     df_routes['Routes nationales'].sum() +
#     # df_routes['Routes départementales et voies communales'].sum() +
#     # 382491 +    # Routes départementales, d'après statista
#     # 714,883 +  # Routes communales, d'après statista
#     df_chemin_fer['long_ferre'].sum()
# )

def compute_total_chain_length(xy, chains):
    total_length = 0.0
    for chain in chains:
        coords = xy[chain]
        segment_lengths = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        total_length += segment_lengths.sum()
    return total_length / 1000  # conversion en km

# Erreur globale
def evaluate_global(xy, chains, ref_total_length):
    detected_length = compute_total_chain_length(xy, chains)
    error = abs(detected_length - ref_total_length)
    relative_error = error / ref_total_length if ref_total_length > 0 else float('inf')
    print(f"Longueur détectée : {detected_length:.1f} km")
    print(f"Longueur réelle (référence) : {ref_total_length:.1f} km")
    print(f"Erreur absolue : {error:.1f} km")
    print(f"Erreur relative : {relative_error:.2%}")
    return error, relative_error

# Erreur par département
def evaluate_by_department(xy, chains, departements): # departements de la même longueur que xy, pas unique
    """
    Évalue la qualité de la détection par département.
    Retourne un DataFrame avec les erreurs par département dans l'ordre décroissant.
    """

    # Longueur détectée par département
    detected_by_dep = defaultdict(float)

    for chain in chains:
        coords = xy[chain]
        seg_lengths = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        seg_indices = [(chain[i], chain[i+1]) for i in range(len(chain)-1)]

        for (i1, i2), length in zip(seg_indices, seg_lengths):
            dep1 = departements.iloc[i1]
            dep2 = departements.iloc[i2]
            if dep1 == dep2:
                detected_by_dep[dep1] += length / 1000  # en km
            else:
                # si segment entre 2 départements, on le partage à moitié
                detected_by_dep[dep1] += length / 2000
                detected_by_dep[dep2] += length / 2000

    # Longueur réelle par département et comparaison
    results = []
    deps = set(departements.unique())

    for dep in deps:
        road_row = df_routes[df_routes['nom_dep'] == dep]
        rail_row = df_chemin_fer[df_chemin_fer['libgeo'] == dep]

        ref_length = (
            road_row['Autoroutes'].sum() +
            road_row['Routes nationales'].sum() +
           # road_row['Routes départementales et voies communales'].sum() +
            rail_row['long_ferre'].sum()
        )

        detected = detected_by_dep.get(dep, 0)
        abs_error = abs(detected - ref_length)
        rel_error = abs_error / ref_length if ref_length > 0 else float('inf')

        results.append({
            "Département": dep,
            "Réel (km)": round(ref_length, 1),
            "Détecté (km)": round(detected, 1),
            "Erreur (km)": round(abs_error, 1),
            "Erreur relative": rel_error,           
            "Erreur relative (%)": f"{rel_error:.1%}" if ref_length > 0 else "-"
        })

    df_result = pd.DataFrame(results).sort_values(by="Erreur relative", ascending=False)
    return df_result
    
# Evaluation
def evaluate_chain_detection(xy, chains, departements, ref_total_length, alpha=0.8, beta=0.2):
    """
    Évalue la qualité de détection des chaînes en combinant :
    - longueur globale
    - homogénéité géographique (écart type des erreurs par département)

    alpha et beta sont les poids du score combiné.
    """

    # Longueur détectée globale
    detected_length = compute_total_chain_length(xy, chains)
    error = abs(detected_length - ref_total_length)
    relative_error = error / ref_total_length if ref_total_length > 0 else float('inf')

    # Erreur par département
    df_deps = evaluate_by_department(xy, chains, departements)
    rel_errors_deps = df_deps["Erreur relative"].replace([np.inf, -np.inf], np.nan).dropna().tolist()

    std_rel_error_deps = np.std(rel_errors_deps) if rel_errors_deps else float('inf')

    # Score combiné
    score = alpha * relative_error + beta * std_rel_error_deps

    print(f"\nÉvaluation globale :")
    print(f"  - Longueur détectée   : {detected_length:.1f} km")
    print(f"  - Longueur réelle     : {ref_total_length:.1f} km")
    print(f"  - Erreur absolue             : {error:.1f} km")
    print(f"  - Erreur relative globale    : {relative_error:.2%}")
    print(f"  - Écart-type err. rel. dép.  : {std_rel_error_deps:.2%}")
    print(f"  - Score final  : {score:.4f} ")

    return score

# Calcul de multiples indicateurs
def compute_chain_metrics(xy, chains, departements, ref_total_length):
    
    print("🔹 Nombre de chaînes trouvées : ", len(chains))
    
    # Longueur des chaînes (km ET nb stations)
    chain_lengths_km = [np.sum(np.linalg.norm(np.diff(xy[chain], axis=0), axis=1)) for chain in chains]
    mean_chain_length_km = np.mean(chain_lengths_km)/1000 if chain_lengths_km else 0
    std_chain_length_km = np.std(chain_lengths_km)/1000 if chain_lengths_km else 0
    print(f"🔹 Longueur moyenne des chaînes : {mean_chain_length_km:.3f} km")
    print(f"🔹 Écart-type longueur des chaînes : {std_chain_length_km:.3f} km")    
    
    chain_lengths = [len(chain) for chain in chains]
    mean_chain_length = np.mean(chain_lengths) if chain_lengths else 0
    std_chain_length = np.std(chain_lengths) if chain_lengths else 0
    print(f"🔹 Longueur moyenne des chaînes : {mean_chain_length:.1f} stations")
    print(f"🔹 Écart-type longueur des chaînes : {std_chain_length:.2f}") 
    
    
    print("\n🔹 Évaluation globale :")
    abs_error, rel_error = evaluate_global(xy, chains, ref_total_length)


    print("\n🔹 Évaluation par département :")
    df_deps = evaluate_by_department(xy, chains, departements)
    print(df_deps.drop(columns=["Erreur relative"]).head())

    rel_errors_deps = df_deps["Erreur relative"].dropna().values
    std_rel_error_deps = np.std(rel_errors_deps) if len(rel_errors_deps) > 0 else float('inf')
    max_rel_error_dep = np.max(rel_errors_deps) if len(rel_errors_deps) > 0 else float('inf')
    high_error_deps = sum(rel_errors_deps > 0.5)
    print(f"\nÉcart-type des erreurs départementales : {std_rel_error_deps:.3f}")
    print(f"Max erreur relative d’un département : {max_rel_error_dep:.2%}")
    print(f"Nombre de départements à forte erreur (>50%) : {high_error_deps}")

    detected_deps = set(df_deps.loc[df_deps["Détecté (km)"] > 0, "Département"])        # Départements détectés (au moins un peu de longueur détectée)
    all_deps = set(df_deps["Département"])
    undetected_dep = all_deps - detected_deps
    print("Départements non détectés :", ", ".join(sorted(undetected_dep)))


    print("\n🔹 Redondance :")
    def jaccard(c1, c2):        # IOU (inter over union)
        s1, s2 = set(c1), set(c2)
        inter = len(s1 & s2)
        union = len(s1 | s2)
        return inter / union if union else 0

    redundant_pairs_9 = 0
    #redundant_pairs_75 = 0
    #redundant_pairs_5 = 0
    for i in range(len(chains)):
        for j in range(i + 1, len(chains)):
            if jaccard(chains[i], chains[j]) > 0.9:
                redundant_pairs_9 += 1
    #        if jaccard(chains[i], chains[j]) > 0.75:
    #            redundant_pairs_75 += 1
    #        if jaccard(chains[i], chains[j]) > 0.5:
    #            redundant_pairs_5 += 1

    redundancy_rate_9 = redundant_pairs_9 / max(len(chains), 1)
    #redundancy_rate_75 = redundant_pairs_75 / max(len(chains), 1)
    #redundancy_rate_5 = redundant_pairs_5 / max(len(chains), 1)
    print(f"\nProportion de chaînes quasi-identiques, >90% : {redundancy_rate_9:.2%}")
    #print(f"🔹 Taux de redondance (chaînes quasi-identiques, >75%) : {redundancy_rate_75:.2%}")
    #print(f"🔹 Taux de redondance (chaînes quasi-identiques, >50%) : {redundancy_rate_5:.2%}")
    
    segment_counter = Counter()
    for chain in chains:
        for i in range(len(chain) - 1):
            segment = tuple(sorted((chain[i], chain[i+1])))
            segment_counter[segment] += 1
    
    total_segments = len(segment_counter)
    redundant_segments = sum(1 for c in segment_counter.values() if c > 1)          # nb de segments dans plusieurs chaînes
    redundancy_ratio = redundant_segments / total_segments
    print(f"Proportion de segments dans plusieurs chaînes : {redundancy_ratio:.2%}")
    
    point_counts = Counter([pt for chain in chains for pt in chain])
    n_repeated_points = sum(1 for count in point_counts.values() if count > 1)      # nb de points dans plusieurs chaînes
    used_points = set(point_counts.keys())
    n_used_points = len(used_points)
    n_unused_points = len(xy) - n_used_points                                       # nb de points dans aucune chaîne
    
    overlap_ratio = n_repeated_points / n_used_points if n_used_points > 0 else 0   # proportion des points qui sont dans plusieurs chaînes
    unused_points_ratio = n_unused_points / len(xy) if len(xy) > 0 else 0
    print(f"Proportion de points dans plusieurs chaînes : {overlap_ratio:.2%}")
    
    print(f"\n🔹 Proportion de points non utilisés : {unused_points_ratio:.2%}")

    # Étendue spatiale
#    all_coords = np.concatenate([xy[chain] for chain in chains]) if chains else np.empty((0, 2))
#    if len(all_coords) > 0:
#        min_x, min_y = np.min(all_coords, axis=0)
#        max_x, max_y = np.max(all_coords, axis=0)
#        area_km2 = ((max_x - min_x) / 1000) * ((max_y - min_y) / 1000)
#    else:
#        area_km2 = 0
#    print(f"\n🔹 Surface couverte approx : {area_km2:.1f} km²")

    return {
        "Erreur absolue (km)": round(abs_error, 1),
        "Erreur relative globale": round(rel_error, 3),
        "Écart-type erreurs relatives dép.": round(std_rel_error_deps, 3),
        "Max erreurs relatives dép.": round(max_rel_error_dep, 3),
        "Nb départements à forte erreur": int(high_error_deps),
        "Taux de redondance (≈doublons)": round(redundancy_rate_9, 3),
#        "Surface couverte approx (km²)": round(area_km2, 1)
    }


##### Fonctions spécifiques Méthode Triplets

def construct_triplets_alignés(xy, k, max_dist, angle_threshold):
    '''retourne un dictionnaire avec clé = index central, valeur = liste de (prev, next) indices des stations alignées'''
    triplet_dict = {}  

    nbrs = NearestNeighbors(n_neighbors=k+1).fit(xy)
    distances, indices = nbrs.kneighbors(xy)

    for i, neighbors in enumerate(indices):
        neighbors = neighbors[1:]  # on enlève soi-même
        for j in range(len(neighbors)):
            for k_ in range(j + 1, len(neighbors)):
                a = xy[neighbors[j]]
                b = xy[i]
                c = xy[neighbors[k_]]
                if np.linalg.norm(a - b) > max_dist or np.linalg.norm(c - b) > max_dist:
                    continue
                angle = angle_between(a - b, c - b)
                if abs(angle - 180) < angle_threshold:
                    triplet_dict.setdefault(i, []).append((neighbors[j], neighbors[k_]))
    return triplet_dict


def extend_chain_right(chain, triplet_dict):
    """Essaie d'étendre une chaîne existante vers la droite"""
    last = chain[-1]
    for (prev, nxt) in triplet_dict.get(last, []):
        if prev == chain[-2] and nxt not in chain:
            return chain + [nxt]
    return chain

def extend_chain_left(chain,triplet_dict):
    """Essaie d'étendre une chaîne existante vers la gauche"""
    first = chain[0]
    for (prev, nxt) in triplet_dict.get(first, []):
        if nxt == chain[1] and prev not in chain:
            return [prev] + chain
    return chain

def extend_chain(chain, triplet_dict):
    """Essaie d'étendre une chaîne existante"""
    extended = True
    while extended:
        extended = False
        new_chain = extend_chain_right(chain,triplet_dict)
        if new_chain != chain:
            chain = new_chain
            extended = True

        new_chain = extend_chain_left(chain,triplet_dict)
        if new_chain != chain:
            chain = new_chain
            extended = True
    return chain
    
def is_valid_triplet(xy, i, j, k, dist_max, angle_max, seen_triplets):
    res = True
    if len(set([i, j, k])) < 3:
       res = False
    else: 
        if np.linalg.norm(xy[i] - xy[j]) > dist_max or np.linalg.norm(xy[j] - xy[k]) > dist_max:
           res = False
        else: 
            angle = angle_triplet(xy[i], xy[j], xy[k])
            if abs(angle - 180) > angle_max:
               res = False
            else: 
                triplet_key = tuple(sorted([i, j, k]))
                if triplet_key in seen_triplets:
                   res = False
    return res
    
#### Fusion

# Test de compatibilité entre deux chaînes
def can_fuse(c1, c2, xy, fusion_angle_thresh, fusion_dist_thresh):
    p1a, p1b = xy[c1[0]], xy[c1[-1]]
    p2a, p2b = xy[c2[0]], xy[c2[-1]]

    candidates = [
        (p1a, p2a, 'rev1+rev2'),
        (p1a, p2b, 'rev1'),
        (p1b, p2a, 'rev2'),
        (p1b, p2b, 'none')
    ]

    for pt1, pt2, mode in candidates:
        dist = np.linalg.norm(pt1 - pt2)
        if dist < fusion_dist_thresh:
            dir1 = chain_direction(c1[::-1] if 'rev1' in mode else c1, xy)
            dir2 = chain_direction(c2[::-1] if 'rev2' in mode else c2, xy)
            angle = angle_between(dir1, dir2)
            if abs(angle - 180) < fusion_angle_thresh:
                return True, mode
    return False, None

# Fusion de deux chaînes dans le bon ordre
def merge_chains(c1, c2, mode, xy):
    if 'rev1' in mode:
        c1 = c1[::-1]
    if 'rev2' in mode:
        c2 = c2[::-1]
    if np.linalg.norm(xy[c1[-1]] - xy[c2[0]]) < np.linalg.norm(xy[c1[0]] - xy[c2[0]]):
        return c1 + c2
    else:
        return c2 + c1

# Fusion de toutes les chaînes
def fuse_all_chains(chains, xy, fusion_angle_thresh, fusion_dist_thresh):
    merged = []
    used = set()
    for i in range(len(chains)):
        if i in used:
            continue
        base = chains[i]
        changed = True
        while changed:
            changed = False
            for j in range(len(chains)):
                if j == i or j in used:
                    continue
                can_merge, mode = can_fuse(base, chains[j], xy, fusion_angle_thresh, fusion_dist_thresh)
                if can_merge:
                    base = merge_chains(base, chains[j], mode, xy)
                    used.add(j)
                    changed = True
        merged.append(base)
    return merged

##### Fonctions spécifiques Méthode Suivi de chemin

# Décide des directions de départ, celles des plus proches voisins en filtrant légérement pour ne pas aller 2 fois dans la même direction
def find_best_directions(xy, idx, neighbors_idx, dist_max, angle_tol_deg, n_directions):
    """
    Params:
    - xy : array Nx2 des coordonnées des stations
    - idx : index de la station source
    - neighbors_idx : liste ou array des indices voisins à tester
    - dist_max : distance max entre stations pour considérer un lien
    - angle_tol_deg : tolérance angulaire minimale entre directions (en degrés)
    - n_directions : nombre max de directions à retourner
    """
    origin = xy[idx]
    directions = []
    used_angles = []

    for j in neighbors_idx:
        if j == idx:
            continue
        vec = xy[j] - origin
        dist = np.linalg.norm(vec)
        if dist == 0 or dist > dist_max:
            continue
        vec_unit = vec / dist
        angle = np.degrees(np.arctan2(vec_unit[1], vec_unit[0])) % 180  # angle dans [0,180)

        # Vérifier que cette direction est assez différente des directions déjà sélectionnées
        if all(abs((angle - a + 90) % 180 - 90) > angle_tol_deg for a in used_angles):
            directions.append(vec_unit)
            used_angles.append(angle)

        if len(directions) >= n_directions:
            break

    return directions
    
# Construit une chaîne à partir d'un point de départ
def follow_line(xy, idx_start, dir_vec, nn_model, dist_max, angle_max):
    """
    Suit une ligne en utilisant soit des scalaires pour les paramètres, soit des tableaux de paramètres locaux.
    """
    path = [idx_start]
    current = idx_start

    # Détection si dist_max et angle_max sont des scalaires ou arrays
    use_local = hasattr(dist_max, "__len__") and hasattr(angle_max, "__len__")

    while True:
        if use_local:
            d_max = dist_max[current]
            a_max = angle_max[current]
        else:
            d_max = dist_max
            a_max = angle_max

        distances, indices = nn_model.kneighbors([xy[current]], return_distance=True)
        best = None
        best_angle = a_max
        for dist, j in zip(distances[0][1:], indices[0][1:]):
            if j in path:
                continue
            vec = xy[j] - xy[current]
            if np.linalg.norm(vec) > d_max:
                continue
            angle = angle_between(vec, dir_vec)
            if angle < best_angle:
                best_angle = angle
                best = j
        if best is None:
            break
        path.append(best)
        current = best
    return path
    

def add_chain_if_unique(chain, chains):
    """
    Ajoute la chaîne si elle est non incluse dans une existante,
    et remplace si des existantes sont incluses dedans.
    """
    chain_set = set(chain)
    to_remove = []
    for i, existing in enumerate(chains):
        existing_set = set(existing)

        # Cas 1 : la chaîne est déjà incluse
        if chain_set <= existing_set:
            return  # Ne pas ajouter
        # Cas 2 : une chaîne existante est incluse → on la retire
        if existing_set <= chain_set:
            to_remove.append(i)

    # Supprimer les chaînes incluses
    for i in reversed(to_remove):  # reverse pour ne pas casser les indices
        del chains[i]
    chains.append(chain)
    
    
def compute_density(xy, r_density):
    """
    Densité corrigée au bord : densité = nb de voisins / surface réellement occupée autour du point
    """
    nn = NearestNeighbors(radius=r_density)
    nn.fit(xy)
    
    densities = []
    for pt in xy:
        indices = nn.radius_neighbors([pt], return_distance=False)[0]
        if len(indices) < 3:
            densities.append(0)  # Pas assez pour un polygone
            continue
        
        neighbors = xy[indices]
        
        try:
            hull = ConvexHull(neighbors)
            area = hull.area  # ou hull.volume pour 2D
            density = len(indices) / area if area > 0 else 0
        except:
            density = 0  # Si la convex hull échoue (rare cas)
        
        densities.append(density)
    
    return np.array(densities)

# Version linéaire
# def normalize(arr, min_val, max_val):
#    arr = np.clip(arr, np.min(arr), np.max(arr))
#    scaled = (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-5)
#    return min_val + (1 - scaled) * (max_val - min_val)

# Normalisation selon différents modes  
def normalize(arr, min_val, max_val, mode="linear", **kwargs):
    
    if mode == "log":
        factor = kwargs.get("factor", 5)
        arr = np.log1p(arr * factor)
    elif mode == "power":
        gamma = kwargs.get("gamma", 2)
        arr = arr ** gamma
    elif mode == "sigmoid":
        k = kwargs.get("k", 10)
        arr = 1 / (1 + np.exp(-k * (arr - np.mean(arr))))  # centrage
    
    # Normalisation linéaire finale
    scaled = (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-5)
    return min_val + (1 - scaled) * (max_val - min_val)


##### Méthodes complètes

def detect_chains_by_triplets(xy, k, angle_threshold, min_chain_length, max_dist):
  
    triplet_dict = construct_triplets_alignés(xy, k, max_dist, angle_threshold)  # retourne dictionnaire : clé = index central, valeur = liste de (prev, next) indices des stations alignées

    chains = []
#    visited = []
    
    for center, pairs in triplet_dict.items():
        for (a, c) in pairs:
            chain = [a, center, c]
#            if any(i in visited for i in chain):
#                continue

            chain = extend_chain(chain, triplet_dict)

            if len(chain) >= min_chain_length:
                add_chain_if_unique(chain, chains)
    return chains


def detect_chains_by_triplets_enhanced(xy, k, angle_max, angle_max_ext, dist_max, min_chain_len):
    # Voisinage
    nn = NearestNeighbors(n_neighbors=k+1).fit(xy)
    _, indices = nn.kneighbors(xy)

    chains = []
    seen_triplets = set()

    # Explorer tous les triplets initiaux (i, j, k) voisins
    for i in range(len(xy)):
        ni = indices[i][1:]
        for j in ni:
            nj = indices[j][1:]
            for k in nj:
                if not(is_valid_triplet(xy, i, j, k, dist_max, angle_max, seen_triplets)):
                    continue
                triplet_key = tuple(sorted([i, j, k]))
                seen_triplets.add(triplet_key)

                # Initier une chaîne avec ce triplet
                chain = [i, j, k]
                prev = k
                before_prev = j

                # Étendre la chaîne tant qu'on reste aligné
                while True:
                    candidates = indices[prev][1:]
                    best = None
#                    best_angle = angle_max
                    for nxt in candidates:
                        if nxt in chain:
                            continue
                        if np.linalg.norm(xy[nxt] - xy[prev]) > dist_max:
                            continue
                        angle = angle_triplet(xy[before_prev], xy[prev], xy[nxt])
                        if abs(angle - 180) < angle_max_ext:
                            best = nxt
#                            best_angle = abs(angle - 180)
                    if best is None:
                        break
                    chain.append(best)
                    before_prev = prev
                    prev = best

                if len(chain) >= min_chain_len:
                    add_chain_if_unique(chain, chains)
                    
    return chains
    
def detect_chains_by_follow_line(xy, n_neighbors=10, n_directions=5, angle_tol=5, dist_max=6000, angle_max=90, min_len=10):
    """
    Détection de chaînes par suivi de chemin

    Paramètres :
        xy : np.ndarray, coordonnées des stations (N x 2)
        n_neighbors : int, nombre de voisins à considérer dans le suivi de ligne
        n_directions : nombre de directions à tester par point
        angle_tol : float, tolérance angulaire de départ (degrés)
        dist_max : seuil pour continuer une route (en m)
        angle_max : seuil pour continuer une route (en degrés)
        min_len : longueur minimale pour valider une chaîne

    Retourne :
        chains : liste de listes d'indices formant des chaînes
    """

    nn_model = NearestNeighbors(n_neighbors=n_neighbors)
    nn_model.fit(xy)

    chains = []

    for idx in range(len(xy)):
        distances, indices = nn_model.kneighbors([xy[idx]])

        local_dirs = find_best_directions(xy, idx, indices[0], dist_max, angle_tol, n_directions)

        for dir_vec in local_dirs:
            chain = follow_line(xy, idx, dir_vec, nn_model, dist_max, angle_max)

            if len(chain) >= min_len:
                add_chain_if_unique(chain, chains)

    print(f"Nombre total de chaînes : {len(chains)}")
    return chains
    
    
def detect_chains_by_density( xy, r_density=20000, n_neighbors=10, dist_min=5, dist_max=6000, angle_min=20, angle_max=90, angle_tol=5, n_directions=4, min_len=10, factor=2, verbose=False):
    """
    Détection de chaînes par densité de stations

    Paramètres :
        xy : np.ndarray, coordonnées des stations (N x 2)
        r_density : float, rayon pour estimer la densité locale
        n_neighbors : int, nombre de voisins à considérer dans le suivi de ligne
        dist_min, dist_max : bornes pour la distance max adaptative (en m)
        angle_min, angle_max : bornes pour la tolérance angulaire adaptative (en degrés)
        angle_tol : float, tolérance angulaire de départ (degrés)
        n_directions : nombre de directions à tester par point
        min_len : longueur minimale pour valider une chaîne
        verbose : bool, affiche des infos de debug si True

    Retourne :
        chains : liste de listes d'indices formant des chaînes
    """

    if verbose:
        print('Calculs de la densité et paramètres locaux...')
    density = compute_density(xy, r_density)
    local_dist = normalize(density, dist_min, dist_max, mode="linear", kwargs = factor)
    local_angle = normalize(density, angle_min, angle_max, mode="linear", kwargs = factor)

    nn_model = NearestNeighbors(n_neighbors=n_neighbors)
    nn_model.fit(xy)

    chains = []

    if verbose:
        print('Construction des chaînes...')
    for idx in range(len(xy)):
        distances, indices = nn_model.kneighbors([xy[idx]])

        local_dirs = find_best_directions( xy, idx, indices[0], dist_max=local_dist[idx], angle_tol_deg=angle_tol, n_directions=n_directions)

        for dir_vec in local_dirs:
            chain = follow_line( xy, idx, dir_vec, nn_model, dist_max=local_dist, angle_max=local_angle)
            if len(chain) >= min_len:
                add_chain_if_unique(chain, chains)

    if verbose:
        print(f"Nombre total de chaînes : {len(chains)}")

    return chains
    