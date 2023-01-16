from math import log2


def entropy(dataset):
    p_list = dataset.value_counts() / dataset.shape[0]
    return -sum(p * log2(p) for p in p_list)


class D_TREE_model:
    def __init__(self):
        self.x_train = None
        self.y_train = None

    def fit(self, X_train, Y_train):
        #TODO: learning algorithm
        return 0