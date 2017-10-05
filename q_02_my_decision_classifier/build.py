from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("./data/loan_prediction.csv")
np.random.seed(9)
X = data.iloc[:,0:4]
y = data.iloc[:,4]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)


def my_decision_classifier(X_train, X_test, y_train, y_test, param_grid, n_iter_search):
    decision_class = DecisionTreeClassifier(random_state=9)

    rs_cv = RandomizedSearchCV(decision_class, param_distributions=param_grid, n_iter=n_iter_search)
    rs_cv.fit(X_train, y_train)

    best_params = rs_cv.best_params_
    y_pred = rs_cv.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    return accuracy, best_params