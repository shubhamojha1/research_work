from deap import base, creator, tools
import random

def fitness(individual, data_inputs, data_outputs):
    features = [idx for idx in range(len(individual)) if individual[idx]==1]
    features_data = data_inputs[:, features]

    