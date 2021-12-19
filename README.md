# DecTree

DecTree is the python implementation of a decision tree which can be used for 
predictions of both classification and regression cases in machine learning context. 
DecTree classifier is capable of handling both numerical and categorical attributes.

## Structure

Decision tree is implemented using the object-oriented structure where each sub-tree
is generated recursively as a property of its parent root node. This linked recursive
structure is then used for traversals accordingly in order to predict for new cases.

Several sample data-sets are provided in data folder which are used for validating the
decision tree implementation. Accuracy, root mean square error and R-square error are 
reported for each case along with optional graphs.

## Configurations

Minimum threshold for the entropy validation (used for termination of recursions thus
creating a leaf node) and maximum allowed depth of the tree can be configured using the
configuration values. Please note that lower thresholds and higher depths can lead to
an over-fitted model.