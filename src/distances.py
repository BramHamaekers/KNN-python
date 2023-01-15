from math import sqrt


# Euclidean distance = sqrt(sum i to N (x_i – y_i)^2)
def euclidean(x, y):
    distance = 0.0
    for a, b in zip(x, y):
        distance += (sum([(pow((a-b), 2))]))
    return sqrt(distance)


# Manhattan distance = sum i to N abs(x_i – y_i)
def manhattan(x, y):
    distance = 0.0
    for a, b in zip(x, y):
        distance += sum([abs(a-b)])
    return distance


# Chebyshev distance = max(|y_i – y_i|)
def chebyshev(x, y):
    distance = []
    for a, b in zip(x, y):
        distance.append(abs(a-b))
    return max(distance)