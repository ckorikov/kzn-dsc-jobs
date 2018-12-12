import numpy as np


def feature_select(total_features, max_features):
    selected = []
    while len(selected) < max_features:
        element = np.random.randint(0, total_features, 1)
        if element not in selected:
            selected.append(element[0])

    selected.sort()
    return selected

print(feature_select(10, 5))

print(feature_select(10, 5))

print(feature_select(15, 15))