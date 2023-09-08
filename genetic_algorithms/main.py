from deap import base, creator, tools
import random
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def fitness(individual, data_inputs, data_outputs):
    """
    Here fitness function is accuracy. Could try with other fitness functions
    """


    # get the features subset
    features = [idx for idx in range(len(individual)) if individual[idx]==1]
    features_data = data_inputs[:, features]

    # splitting the dataset
    data_inputs_train, data_inputs_val, data_outputs_train, data_outputs_val = train_test_split(features_data, data_outputs, test_size=0.2)

    # classifier
    classifier = DecisionTreeClassifier()
    classifier.fit(data_inputs_train, data_outputs_train)

    # Make predictions and get the accuracy for the validation set
    predictions = classifier.predict(data_inputs_val)
    accuracy = accuracy_score(data_outputs_val, predictions)
    
    return accuracy