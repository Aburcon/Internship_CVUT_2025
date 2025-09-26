# Parameters for criteria using 3NN method (angle in degrees and distance in kilometer)
MEAN_DISTANCE_PARAMS = {
    ']0, 1] km': {'colour': '#030464', 'min_angle': 5, 'max_distance': 2},
    ']1, 2] km': {'colour': '#069AF3', 'min_angle': 10, 'max_distance': 5},
    ']2, 4] km': {'colour': '#02D4BB', 'min_angle': 15, 'max_distance': 10},
    ']4, inf] km': {'colour': '#0DBF75', 'min_angle': 20, 'max_distance': 15},
}

# Parameters for simple criteria
MAX_DISTANCE = 15
MIN_ANGLE = 20

# Number of neighbours in the kNN city detection method
N_NEIGH = 3

# Informations for data extraction
PROVIDER = 'Orange'
TECHNO='4g'
REGION='Normandie'