# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.metrics import mean_squared_error

# class Ant:
#     def __init__(self, num_features):
#         self.num_features = num_features
#         self.visited_features = np.zeros(num_features, dtype=bool)

#     def select_feature(self, feature_pheromones):
#         probabilities = feature_pheromones * ~self.visited_features
#         probabilities /= probabilities.sum()
#         feature = np.random.choice(self.num_features, p=probabilities)
#         self.visited_features[feature] = True
#         return feature
    
# class ACO:
#     def __init__(self, num_ants, num_features, rho=0.1, fitness_func=None):
#         self.num_ants = num_ants
#         self.num_features = num_features
#         self.rho = rho
#         self.fitness_func = fitness_func
#         self.ants = [Ant(num_features) for _ in range(num_ants)]
#         self.feature_pheromones = np.ones(num_features)

#     def update_pheromones(self, solutions):
#         for features, fitness in solutions:
#             self.feature_pheromones[features] += fitness
#         self.feature_pheromones *= (1-self.rho)
    
#     def find_best_features(self, num_iterations):
#         for iteration in range(num_iterations):
#             solutions = []
#             for ant in self.ants:
#                 features = [ant.select_feature(self.feature_pheromones) for _ in range(self.num_features)]
#                 fitness = self.fitness_func(features)
#                 solutions.append((features, fitness))

#             self.update_pheromones(solutions)
#         return self.feature_pheromones.argmax()
    
# df = pd.read_csv("../data/borg_traces_data_preprocessed_small.csv")
# df = df.fillna(0)
    
# def fitness(features):
# # Select the columns corresponding to the features
#     X = df.drop(['failed', 'timeCorr', 'event'], axis=1)#[features]
#     y = df['failed']

#     # Split the data into training and validation sets
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Train the model on the training set
#     model = GradientBoostingRegressor(random_state=42)
#     model.fit(X_train, y_train)

#     # Evaluate the model on the validation set
#     y_pred = model.predict(X_val)
#     score = -mean_squared_error(y_val, y_pred)  # Negate because we want to maximize the fitness

#     return score
    

# aco = ACO(num_ants=10, num_features=df.shape[1], fitness_func=fitness)
# best_features = aco.find_best_features(num_iterations=100)
# print(best_features)


