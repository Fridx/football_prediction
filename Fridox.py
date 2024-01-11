import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier

# Scores précédents (données historiques)
previous_scores = ["1-1", "4-1", "0-5", "0-2", "2-1", "0-5", "0-1", "0-0", "1-1", "1-3", "0-2", "1-5"]
previous_scores = [score.split("-") for score in previous_scores]

# Données d'entraînement (80% des données historiques)
X_train = np.array([[1, 2, int(score[0]), int(score[1])] for score in previous_scores[:8]])
y_train = np.array([int(score[0]) for score in previous_scores[:8]])

# Données de test (20% des données historiques)
X_test = np.array([[5, 1, int(previous_scores[4][0]), int(previous_scores[4][1])]])
y_test = np.array([int(previous_scores[4][0])])

# Modèles de régression
regression_models = [
    LinearRegression(),
    KNeighborsRegressor(n_neighbors=3),
    RandomForestRegressor(n_estimators=100),
    MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1000),
    SVR(kernel='rbf'),
    GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3),
    DecisionTreeRegressor()
]

# Modèle de classification pour prédire l'équipe gagnante
classification_model = RandomForestClassifier(n_estimators=100)

# Prédictions des modèles de régression
regression_predictions = []

for model in regression_models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test).astype(int)
    regression_predictions.append(y_pred)

# Combinaison des prédictions des modèles de régression avec le vote majoritaire
combined_regression_predictions = np.vstack(regression_predictions)
final_regression_prediction = np.round(np.mean(combined_regression_predictions, axis=0)).astype(int)

# Prédiction de l'équipe gagnante
classification_model.fit(X_train, np.where(y_train > y_test, 1, 0))
winner_prediction = classification_model.predict(X_test)

# Affichage des prédictions finales
print("Prédictions finales :")
print(f"Score exact : {final_regression_prediction[0]}-{y_test[0]}")
print(f"Nombre total de buts : {final_regression_prediction[0] + y_test[0]}")
print(f"Équipe gagnante : {'Équipe A' if winner_prediction[0] == 1 else 'Équipe B'}")
