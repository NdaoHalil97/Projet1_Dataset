import numpy as np
from scipy.spatial import distance

def retrieve_similar_images(signatures, query_features, distance_measure, num_images):
    distances = []
    for idx, sig in enumerate(signatures):
        if distance_measure == "euclidean":
            dist = distance.euclidean(sig, query_features)
        elif distance_measure == "manhattan":
            dist = distance.cityblock(sig, query_features)
        elif distance_measure == "chebyshev":
            dist = distance.chebyshev(sig, query_features)
        elif distance_measure == "canberra":
            dist = distance.canberra(sig, query_features)
        distances.append((idx, dist))
    
    distances.sort(key=lambda x: x[1])
    return distances[:num_images]

def manhattan(v1, v2):
    """This function computes the Manhattan / cityblock distance
    V1 and V2 must be of the same size

    Args:
        v1 (list or np.ndarray): list or array of the first object
        v2 (list or np.ndarray): list or array of the second object
    """
    v1 = np.array(v1).astype('float')
    v2 = np.array(v2).astype('float')
    return np.sum(np.abs(v1 - v2))

def euclidean(v1, v2):
    """This function computes the Euclidean distance
    V1 and V2 must be of the same size

    Args:
        v1 (list or np.ndarray): list or array of the first object
        v2 (list or np.ndarray): list or array of the second object
    """
    v1 = np.array(v1).astype('float')
    v2 = np.array(v2).astype('float')
    return np.sqrt(np.sum((v1 - v2)**2))

def chebyshev(v1, v2):
    """This function computes the Chebyshev distance
    V1 and V2 must be of the same size

    Args:
        v1 (list or np.ndarray): list or array of the first object
        v2 (list or np.ndarray): list or array of the second object
    """
    v1 = np.array(v1).astype('float')
    v2 = np.array(v2).astype('float')
    return np.max(np.abs(v1 - v2))

def canberra(v1, v2):
    """This function computes the Canberra distance
    V1 and V2 must be of the same size

    Args:
        v1 (list or np.ndarray): list or array of the first object
        v2 (list or np.ndarray): list or array of the second object
    """
    return distance.canberra(v1, v2)

def retrieve_similar_image(features_db, query_features, distance, num_results):
    distances = []
    for instance in features_db:
        features, label, img_path = instance[:-2], instance[-2], instance[-1]
        if distance == 'manhattan':
            dist = manhattan(query_features, features)
        elif distance == 'euclidean':
            dist = euclidean(query_features, features)
        elif distance == 'chebyshev':
            dist = chebyshev(query_features, features)
        elif distance == 'canberra':
            dist = canberra(query_features, features)
        distances.append((img_path, dist, label))
    distances.sort(key=lambda x: x[1])
    return distances[:num_results]
