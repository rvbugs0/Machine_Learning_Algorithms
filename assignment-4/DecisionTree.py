from enum import Enum
import numpy as np
import pandas as pd
import requests


class Criterion(Enum):
    MISCLASSIFICATION_RATE = 1
    GINI_IMPURITY = 2
    ENTROPY = 3


class Node(object):
    def __init__(self, value, is_leaf=False):
        self.value = value
        self.left = None
        self.right = None
        self.gini_value = None
        self.col_index = None
        self.is_leaf = is_leaf


class decision_tree_classifier(object):

    def __init__(self,criterion : Criterion, min_samples_split=10, min_gini=0.2, max_depth=5 ):
        self.root = None
        self.min_samples_split = min_samples_split
        self.min_gini = min_gini
        self.max_depth = max_depth
        self.classes = None
        self.criterion = criterion

    def fit(self, data):
        self.classes = list(set(int(row[-1]) for row in data))
        self.root = self.create_node(data)
        self.build_tree(self.root, current_depth=0)
        return self.root

    def create_node(self, data):
        node = Node(None)

        # checking if node can be further split using minimum gini as criterion
        gini_score = self.gini_index(data)
        if gini_score <= self.min_gini:
            node.is_leaf = True
            node.value = np.bincount([row[-1] for row in data]).argmax()
            node.gini_value = gini_score
            return node

        # checking if node has enough samples to be split again
        if len(data) <= self.min_samples_split:
            node.is_leaf = True
            node.value = np.bincount([row[-1] for row in data]).argmax()
            node.gini_value = gini_score
            return node

        # finding minimum gini impurity split
        gini_index = 1.0
        for col_index in range(len(data[0])-1):
            for row_index in range(len(data)):
                value = data[row_index][col_index]
                child = self.get_split(data, col_index, value)
                node_gini_index = self.calculate_gini_index(value, child)
                if node_gini_index < gini_index:
                    gini_index = node_gini_index
                    node.value = value
                    node.gini_value = node_gini_index
                    node.col_index = col_index
                    node.left = child['l']
                    node.right = child['r']
        return node

    def gini_index(self, data):
        size = len(data)
        instances = [0] * len(self.classes)
        for row in data:
            instances[int(row[-1])] += 1
        return 1 - np.sum([(val/size)**2 for val in instances]) if size > 0 else 1

    def calculate_gini_index(self, value, child):
        group_size = [len(child['l']), len(child['r'])]
        left_gini = self.gini_index(child['l'])
        right_gini = self.gini_index(child['r'])
        return (group_size[0]/np.sum(group_size) * left_gini) + (group_size[1]/np.sum(group_size) * right_gini)

    def get_split(self, data, col_index, value):
        left_child, right_child = [], []
        for index in range(len(data)):
            if data[index][col_index] < value:
                left_child.append(data[index])
            if data[index][col_index] > value:
                right_child.append(data[index])
        return {'l': left_child, 'r': right_child}

    def build_tree(self, node, current_depth):
        # create left subtree for the node if possible under constraints
        if current_depth < self.max_depth:
            # creating left node
            if node.left is not None and isinstance(node.left, list):
                left_node = self.create_node(node.left)
                node.left = left_node
                if node.left.is_leaf is not True:
                    self.build_tree(node.left, current_depth+1)
            if node.right is not None and isinstance(node.right, list):
                # creating right node
                right_node = self.create_node(node.right)
                node.right = right_node
                if node.right.is_leaf is not True:
                    self.build_tree(node.right, current_depth+1)

    def traverse(self):
        tree = {'root': str(self.root.value)}
        stack = [self.root]
        node = self.root
        while len(stack) > 0:
            while node is not None:
                stack.append(node)
                node = node.left
            node = stack.pop(-1)
            if node.is_leaf is not True:
                tree[str(node.value)] = {'left': node.left.value, 'right': node.right.value,
                                         'feature': node.col_index, 'gini value': node.gini_value}
            else:
                tree[str(node.value)] = {'left': 'None', 'right': 'None',
                                         'class label': node.value, 'gini value': node.gini_value, 'leaf': True}
            node = node.right
        return tree

    def predict(self, sample):
        predictions = []
        for row in sample:
            node = self.root
            while node.is_leaf is not True:
                if row[node.col_index] < node.value:
                    node = node.left
                    continue
                if row[node.col_index] >= node.value:
                    node = node.right
            predictions.append(node.value)
        return predictions


def accuracy(data):
    dt = decision_tree_classifier(Criterion.GINI_IMPURITY)
    tree = dt.fit(data)
    print('<========= decision tree ===========>')
    print(dt.traverse())
    predictions = dt.predict(data[0:-1][0:-1])
    true_values = [row[-1] for row in data]
    return '{:1f}'.format(sum([t == p for t, p in zip(true_values, predictions)])/len(true_values) * 100) + '% accuracy'


def flower_to_id(value):
    if value == 'Iris-virginica':
        return 2
    if value == 'Iris-versicolor':
        return 0
    if value == 'Iris-setosa':
        return 1


def get_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    values = requests.get(url).content.decode('utf-8').split('\n')
    data_set = [val.split(',') for val in values]
    data_df = pd.DataFrame(data=data_set, columns=[
                           'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'flower'])
    data_df = data_df.dropna()
    data_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']] = data_df[[
        'sepal_length', 'sepal_width', 'petal_length', 'petal_width']].astype('float')
    data_df['flower'] = data_df['flower'].map(flower_to_id)
    return data_df


# data = get_data()
# print(accuracy(data.to_numpy()))



        
if __name__ == '__main__':
    # Reading data
    d = pd.read_csv("data/train.csv")[['Age', 'Fare', 'Survived']].dropna()

    # Constructing the X and Y matrices
    X = d[['Age', 'Sex']]
    Y = d['Survived'].values.tolist()

    # Initiating the Node
    root = Node(Y, X, max_depth=3, min_samples_split=100)

    # Getting teh best split
    root.grow_tree()

    # Printing the tree information 
    root.print_tree()

    # Predicting 
    Xsubset = X.copy()
    Xsubset['yhat'] = root.predict(Xsubset)
    print(Xsubset)