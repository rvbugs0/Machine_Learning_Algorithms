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
    def __init__(self, X, Y, depth=0, is_leaf=False, node_type = "root",rule = ""):
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
        self.rule =rule
        self.counts = Counter(Y)
        counts_sorted = list(sorted(self.counts.items(), key=lambda item: item[1]))

        # Getting the last item
        # i.e. which has more number of occurences  
        yhat = None
        if len(counts_sorted) > 0:
            yhat = counts_sorted[-1][0]
        self.yhat = yhat

    def gini(self):
        # calculates gini for base (without splits)
        counts = Counter(self.Y)
        class_1_count, class_2_count = counts.get(
            0, 0), counts.get(1, 0)
        # Getting the GINI impurity
        return self.gini_impurity(class_1_count, class_2_count)

    def gini_impurity(self, class_1_count, class_2_count):
        if class_1_count is None:
            class_1_count = 0

        if class_2_count is None:
            class_2_count = 0

        if class_1_count+class_2_count == 0:
            return 0.0

        # Getting the probability to see each of the classes
        p1 = class_1_count / (class_1_count+class_2_count)
        p2 = class_2_count / (class_1_count+class_2_count)

        # Calculating GINI
        gini = 1 - (p1 ** 2 + p2 ** 2)

        # Returning the gini impurity
        return gini

    def ma(self, x: np.array, window: int) -> np.array:
        """
        Calculates the moving average of the given list. 
        """
        return np.convolve(x, np.ones(window), 'valid') / window



class decision_tree_classifier(object):

    def __init__(self, criterion: Criterion, min_samples_split=10,  max_depth=5, min_samples_leaf=1):
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
        """
        Calculates the entropy given a dictionary of class counts.
        """
        total = sum(counts.values())
        entropy = 0
        for count in counts.values():
            probability = count / total
            if probability > 0:
                entropy += -probability * np.log2(probability)
        return entropy



    def best_split(self, node):
        # Creating a dataset for spliting
        df = node.X.copy()
        df['Y'] = node.Y

        # Getting the GINI impurity for the base input
        GINI_base = node.gini()

        # Finding which split yields the best GINI gain
        max_gain = 0

        # Default best feature and split
        best_feature = None
        best_value = None

        for feature in self.features:
            # Droping missing values
            Xdf = df.dropna().sort_values(feature)

            # Sorting the values and getting the rolling average
            xmeans = node.ma(Xdf[feature].unique(), 2)

            for value in xmeans:
                # Spliting the dataset
                left_counts = Counter(Xdf[Xdf[feature] < value]['Y'])
                right_counts = Counter(Xdf[Xdf[feature] >= value]['Y'])

                # Getting the Y distribution from the dicts
                y0_left, y1_left, y0_right, y1_right = left_counts.get(0, 0), left_counts.get(
                    1, 0), right_counts.get(0, 0), right_counts.get(1, 0)

                # Getting the left and right gini impurities
                gini_left = node.gini_impurity(y0_left, y1_left)
                gini_right = node.gini_impurity(y0_right, y1_right)

                # Getting the obs count from the left and the right data splits
                n_left = y0_left + y1_left
                n_right = y0_right + y1_right

                # Calculating the weights for each of the nodes
                w_left = n_left / (n_left + n_right)
                w_right = n_right / (n_left + n_right)

                # Calculating the weighted GINI impurity
                wGINI = w_left * gini_left + w_right * gini_right

                # Calculating the GINI gain
                GINIgain = GINI_base - wGINI

                # Checking if this is the best split so far
                if GINIgain > max_gain:
                    best_feature = feature
                    best_value = value

                    # Setting the best gain to the current one
                    max_gain = GINIgain

        return (best_feature, best_value)



    def grow_tree(self,node):
        """
        Recursive method to create the decision tree
        """
        # Making a df from the data
        df = node.X.copy()
        df['Y'] = node.Y

        # If there is GINI to be gained, we split further
        if (node.depth < self.max_depth) and (node.n >= self.min_samples_split) and (node.n>self.min_samples_leaf):

            # Getting the best split
            best_feature, best_value =None,None

            if self.criterion == Criterion.MISCLASSIFICATION_RATE:
                pass
            elif self.criterion == Criterion.GINI_IMPURITY:
                best_feature, best_value = self.best_split(node)
            elif self.criterion == Criterion.ENTROPY:
                best_feature, best_value = self.best_split_entropy(node)


            

            if best_feature is not None:
                # Saving the best split to the current node
                node.split_feature = best_feature
                node.split_value = best_value

                # Getting the left and right nodes
                left_df, right_df = df[df[best_feature] <= best_value].copy(
                ), df[df[best_feature] > best_value].copy()

                # Creating the left and right nodes
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


    def print_info(self, node ,width=4):
        """
        Method to print the infromation about the tree
        """
        # Defining the number of spaces 
        const = int(node.depth * width ** 1.5)
        spaces = "-" * const
        
        if node.node_type == 'root':
            print("Root")
        else:
            print(f"|{spaces} Split rule: {node.rule}")
        # print(f"{' ' * const}   | GINI impurity of the node: {round(node.gini_impurity, 2)}")
        # print(f"{' ' * const}   | Class distribution in the node: {dict(node.counts)}")
        # print(f"{' ' * const}   | Predicted class: {node.yhat}")   

    def print_tree(self,node):
        """
        Prints the whole tree from the current node to the bottom
        """
        self.print_info(node) 
        
        if node.left is not None: 
            self.print_tree(node.left)
        
        if node.right is not None:
            self.print_tree(node.right)




    def predict(self, X:pd.DataFrame):
        """
        Batch prediction method
        """
        predictions = []

        for _, x in X.iterrows():
            values = {}
            for feature in self.features:
                values.update({feature: x[feature]})

            # print(values)
            predictions.append(self.predict_obs(values))
        

        return predictions

    def predict_obs(self, values: dict) -> int:
        """
        Method to predict the class given a set of features
        """
        cur_node = self.root
        while cur_node.depth < self.max_depth:
            # Traversing the nodes all the way to the bottom
            
            if cur_node.n < self.min_samples_split :
                break 

            if(cur_node.split_feature == None):
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
        """
        Finds the best split based on entropy gain.
        """
        # Creating a dataset for splitting
        df = node.X.copy()
        df['Y'] = node.Y

        # Finding which split yields the best entropy gain
        max_gain = 0
        best_feature = None
        best_value = None

        for feature in self.features:
            # Dropping missing values
            Xdf = df.dropna().sort_values(feature)

            # Sorting the values and getting the rolling average
            xmeans = node.ma(Xdf[feature].unique(), 2)

            for value in xmeans:
                # Splitting the dataset
                left_counts = Counter(Xdf[Xdf[feature] < value]['Y'])
                right_counts = Counter(Xdf[Xdf[feature] >= value]['Y'])

                # Getting the entropy gain for the split
                entropy_gain = self.entropy_gain(left_counts, right_counts, node)

                # Updating the best split if necessary
                if entropy_gain > max_gain:
                    max_gain = entropy_gain
                    best_feature = feature
                    best_value = value

        return best_feature, best_value
    

    def entropy_gain(self, left_counts, right_counts, node):
        """
        Calculates the entropy gain for a given split using the left and right class counts.
        """
        n_left = sum(left_counts.values())
        n_right = sum(right_counts.values())
        n_total = n_left + n_right

        # Calculating the weighted entropy
        weighted_entropy = (n_left / n_total) * self.entropy(left_counts) + \
                           (n_right / n_total) * self.entropy(right_counts)

        # Calculating the entropy gain
        entropy_gain = self.entropy(node.counts) - weighted_entropy
        return entropy_gain


if __name__ == '__main__':
    # Reading data
    d = pd.read_csv("decision-tree/data/train.csv")[['Age', 'Sex', 'Fare', 'Pclass', 'Survived']].dropna()
    d = d.assign(Sex=d.Sex.eq('male').astype(int))

    # Constructing the X and Y matrices
    X = d[['Age', 'Sex', 'Fare', 'Pclass']]
    Y = d['Survived'].values.tolist()

    # Initiating the Node
    dt = decision_tree_classifier(
        Criterion.ENTROPY, min_samples_split=10, max_depth=5, min_samples_leaf=1)

    root_node  = dt.fit(X, Y)

    
    dt.print_tree(root_node)

    # # Getting the best split
    # root.grow_tree()

    # # Printing the tree information
    # root.print_tree()

    # Predicting
    Xsubset = X.copy()
    Xsubset['yhat'] = dt.predict(Xsubset)
    yhat = Xsubset['yhat'].values

    # print(Xsubset)
    same  = 0
    for i in range(len(yhat)):
        if yhat[i] == Y[i]:
            same += 1

    print("ACCURACY: ",same/len(yhat))
