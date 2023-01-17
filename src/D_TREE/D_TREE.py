from math import log2


def variance(Y_train):
    return Y_train.var()


def entropy(Y_train):
    p_list = Y_train.value_counts() / Y_train.shape[0]
    return -sum(p * log2(p) for p in p_list)


def information_gain(X_train, Y_train, test):
    # test must be binary test
    # si = |S_i| / |S|
    s1 = sum(test)/X_train.shape[0]
    s2 = 1 - s1

    # IG_Regression(S,t) = Var(S) - sum((|S_i| / |S|)*Var(S_i)
    # IG_Classification(S,t) = CE(S) - sum((|S_i| / |S|)*CE(S_i)
    if s1 == 0 or s2 == 0: return 0
    if Y_train.dtypes != 'object': variance(Y_train) - (s1 * variance(Y_train[test])) - (s2 * variance(Y_train[-test]))
    return entropy(Y_train) - (s1 * entropy(Y_train[test])) - (s2 * entropy(Y_train[-test]))


class D_TREE_model:
    def __init__(self):
        self.x_train = None
        self.y_train = None

    def fit(self, X_train, Y_train):
        #TODO: learning algorithm
        return 0