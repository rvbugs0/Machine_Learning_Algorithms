from enum import Enum
import numpy as np
import pandas as pd
import requests
from collections import Counter


class Criterion(Enum):
    MISCLASSIFICATION_RATE = 1
    GINI_IMPURITY = 2
    ENTROPY = 3


class Node(object):
    def __init__(self, X, Y, depth=0, is_leaf=False, node_type="root", rule=""):
        self.X = X
        self.Y = Y
        self.n = len(self.Y)
        self.depth = depth
        self.split_feature = None
        self.split_value = None
        self.is_leaf = is_leaf
        self.left = None
        self.right = None
        self.node_type = node_type
        self.rule = rule
        self.counts = Counter(Y)
        counts_sorted = list(
            sorted(self.counts.items(), key=lambda item: item[1]))

        yhat = None
        if len(counts_sorted) > 0:
            yhat = counts_sorted[-1][0]
        self.yhat = yhat

    def gini(self):
        
        counts = Counter(self.Y)
        c1, c2 = counts.get(
            0, 0), counts.get(1, 0)
        
        return self.gini_impurity(c1, c2)

    def gini_impurity(self, count_of_class_1, count_of_class_2):
        if count_of_class_1 is None:
            count_of_class_1 = 0

        if count_of_class_2 is None:
            count_of_class_2 = 0

        if count_of_class_1 + count_of_class_2 == 0:
            return 0.0

        probability_of_class_1 = count_of_class_1 / (count_of_class_1 + count_of_class_2)
        probability_of_class_2 = count_of_class_2 / (count_of_class_1 + count_of_class_2)

        gini_impurity = 1 - (probability_of_class_1 ** 2 + probability_of_class_2 ** 2)

        return gini_impurity

    def average(self, x: np.array, window: int) -> np.array:
        cumsum = np.cumsum(x)
        cumsum[window:] = cumsum[window:] - cumsum[:-window]
        return cumsum[window - 1:] / window


class decision_tree_classifier(object):

    def __init__(self, criterion: Criterion =Criterion.GINI_IMPURITY, min_samples_split=10,  max_depth=5, min_samples_leaf=1):
        self.root = None
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.criterion = criterion
        self.X = None
        self.y = None
        self.features = None

    def fit(self, X, y):

        self.X = X
        self.Y = y
        self.features = list(self.X.columns)
        self.root = self.initialize_tree()

        return self.root

    def initialize_tree(self):
        root_node = Node(self.X, self.Y, depth=0,
                         is_leaf=False)
        self.grow_tree(root_node)
        return root_node

    def entropy(self, counts):
        total = sum(counts.values())
        entropy = 0
        for count in counts.values():
            probability = count / total
            if probability > 0:
                entropy += -probability * np.log2(probability)
        return entropy

    def best_split(self, node):
        
        d = node.X.copy()
        d['Y'] = node.Y

        
        b_gini = node.gini()

        
        m_gain = 0

        
        b_feature = None
        b_value = None

        for f in self.features:
            
            df = d.dropna().sort_values(f)

            
            m = node.average(df[f].unique(), 2)

            for v in m:
                
                left_d = df[df[f] < v]
                right_d = df[df[f] >= v]

                
                lc = left_d['Y'].value_counts().to_dict()
                rc = right_d['Y'].value_counts().to_dict()

                y0l, y1l = lc.get(0, 0), lc.get(1, 0)
                y0r, y1r = rc.get(0, 0), rc.get(1, 0)

                
                gini_l = node.gini_impurity(y0l, y1l)
                gini_r = node.gini_impurity(y0r, y1r)

                
                n_l = len(left_d)
                n_r = len(right_d)

                
                w_l = n_l / (n_l + n_r)
                w_r = n_r / (n_l + n_r)

                
                w_gini = w_l * gini_l + w_r * gini_r

            
                g_gain = b_gini - w_gini

                
                if g_gain > m_gain:
                    b_feature = f
                    b_value = v

                    
                    m_gain = g_gain

        return b_feature, b_value


    def grow_tree(self, node):

        df = node.X.copy()
        df['Y'] = node.Y


        if (node.depth < self.max_depth) and (node.n >= self.min_samples_split) :

            
            best_feature, best_value = None, None

            if self.criterion == Criterion.MISCLASSIFICATION_RATE:
                best_feature, best_value = self.misclassification_split(node)
            elif self.criterion == Criterion.GINI_IMPURITY:
                best_feature, best_value = self.best_split(node)
            elif self.criterion == Criterion.ENTROPY:
                best_feature, best_value = self.best_split_entropy(node)

            if best_feature is not None:
                

                
                left_df, right_df = df[df[best_feature] <= best_value].copy(
                ), df[df[best_feature] > best_value].copy()



                if(left_df.shape[0]>=self.min_samples_leaf and right_df.shape[0]>=self.min_samples_leaf):
                    
                    node.split_feature = best_feature
                    node.split_value = best_value
                    left = Node(

                        left_df[self.features],
                        left_df['Y'].values.tolist(),
                        depth=node.depth + 1,
                        node_type='left_node',
                        rule=f"{best_feature} <= {round(best_value, 3)}"
                    )

                    node.left = left
                    self.grow_tree(node.left)

                
                    

                    right = Node(
                        right_df[self.features],
                        right_df['Y'].values.tolist(),
                        depth=node.depth + 1,
                        node_type='right_node',
                        rule=f"{best_feature} > {round(best_value, 3)}"
                    )

                    node.right = right
                    self.grow_tree(node.right)

    def print_node(self, node, width=4):
        
        # Defining the number of spaces
        const = int(node.depth * width ** 1.5)
        spaces = "-" * const

        if node.node_type == 'root':
            print("Root")
        else:
            print(f"{'-' * const} Split rule: {node.rule}")


    def print_tree(self, node):
        self.print_node(node)

        if node.left is not None:
            self.print_tree(node.left)

        if node.right is not None:
            self.print_tree(node.right)

    def predict(self, X: pd.DataFrame):
        predictions = []

        for _, x in X.iterrows():
            values = {}
            for feature in self.features:
                values.update({feature: x[feature]})

            # print(values)
            predictions.append(self.predict_obs(values))

        return np.array(predictions)

    def predict_obs(self, values: dict) -> int:
        cur_node = self.root
        while cur_node.depth < self.max_depth:

            if cur_node.n < self.min_samples_split:
                break

            if (cur_node.split_feature == None):
                break

            best_feature = cur_node.split_feature

            best_value = cur_node.split_value

            if (values.get(best_feature) < best_value):
                if cur_node.left is not None:
                    cur_node = cur_node.left
            else:
                if cur_node.right is not None:
                    cur_node = cur_node.right

        return cur_node.yhat

    def best_split_entropy(self, node):
        df = node.X.copy()
        df['Y'] = node.Y

        max_gain = 0
        best_feature = None
        best_value = None

        for feature in self.features:
            Xdf = df.dropna().sort_values(feature)

            xmeans = node.average(Xdf[feature].unique(), 2)

            for value in xmeans:
                
                left_counts = Counter(Xdf[Xdf[feature] < value]['Y'])
                right_counts = Counter(Xdf[Xdf[feature] >= value]['Y'])

                
                entropy_gain = self.entropy_gain(
                    left_counts, right_counts, node)

                
                if entropy_gain > max_gain:
                    max_gain = entropy_gain
                    best_feature = feature
                    best_value = value

        return best_feature, best_value

    def entropy_gain(self, left_counts, right_counts, node):
        
        n_left = sum(left_counts.values())
        n_right = sum(right_counts.values())
        n_total = n_left + n_right

        
        weighted_entropy = (n_left / n_total) * self.entropy(left_counts) + \
                           (n_right / n_total) * self.entropy(right_counts)

        
        entropy_gain = self.entropy(node.counts) - weighted_entropy
        return entropy_gain

    def misclassification_split(self, node):
        
        df = node.X.copy()
        df['Y'] = node.Y

        
        misclassification_base = 1 - max(node.counts.values()) / len(node.Y)

        
        max_reduction = 0

        
        best_feature = None
        best_value = None

        for feature in self.features:
            
            Xdf = df.dropna().sort_values(feature)

            
            xmeans = node.average(Xdf[feature].unique(), 2)

            for value in xmeans:
                
                left_counts = Counter(Xdf[Xdf[feature] < value]['Y'])
                right_counts = Counter(Xdf[Xdf[feature] >= value]['Y'])

                
                y0_left, y1_left, y0_right, y1_right = left_counts.get(0, 0), left_counts.get(
                    1, 0), right_counts.get(0, 0), right_counts.get(1, 0)

                
                misclassification_left = 1 - \
                    max(y0_left, y1_left) / (y0_left + y1_left + 0.000000001)
                misclassification_right = 1 - \
                    max(y0_right, y1_right) / (y0_right + y1_right + 0.000000001)

                
                n_left = y0_left + y1_left
                n_right = y0_right + y1_right

                
                w_left = n_left / (n_left + n_right)
                w_right = n_right / (n_left + n_right)

                
                wMisclassification = w_left * (misclassification_base - misclassification_left) + w_right * (
                    misclassification_base - misclassification_right)

                
                if wMisclassification > max_reduction:
                    max_reduction = wMisclassification
                    best_feature = feature
                    best_value = value

        
        if best_feature is None or best_value is None:
            return None, None

        return best_feature, best_value


if __name__ == '__main__':
    # Reading data
    d = pd.read_csv(
        "data/train.csv")[['Age', 'Sex', 'Fare', 'Pclass', 'Survived']].dropna()
    d = d.assign(Sex=d.Sex.eq('male').astype(int))

    # Constructing the X and Y matrices
    X = d[['Age', 'Sex', 'Fare', 'Pclass']]
    Y = d['Survived'].values.tolist()

    # Initiating the Node

    dt = decision_tree_classifier(
        Criterion.ENTROPY, min_samples_split=10, max_depth=5, min_samples_leaf=1)
    root_node = dt.fit(X, Y)
    dt.print_tree(root_node)

    # Predicting
    Xsubset = X.copy()
    Xsubset['yhat'] = dt.predict(Xsubset)
    yhat = Xsubset['yhat'].values

    # print(Xsubset)
    same = 0
    for i in range(len(yhat)):
        if yhat[i] == Y[i]:
            same += 1

    print("ACCURACY: ", same/len(yhat))
