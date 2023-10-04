# Encoding:
from category_encoders import TargetEncoder
from feature_engine.encoding import CountFrequencyEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn import preprocessing
import pandas as pd

# Balance
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks

# Optimization
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from skopt.searchcv import BayesSearchCV
import optuna


def encoding(train, test, cols, encoding):

    train_encoded = train.copy()
    test_encoded = test.copy()
    
    if encoding == 'OHE':  # One-Hot Encoding (Dummies)
        train_encoded = pd.get_dummies(train, columns=cols, drop_first=True)
        test_encoded = pd.get_dummies(test, columns=cols, drop_first=True)

        # Asegurar que las columnas en train y test sean las mismas
        missing_cols = set(train_encoded.columns) - set(test_encoded.columns)
        for c in missing_cols:
            test_encoded[c] = 0
        test_encoded = test_encoded[train_encoded.columns]

    elif encoding == 'LE':  # Label Encoding
        le = LabelEncoder()
        for col in cols:
            train_encoded[col] = le.fit_transform(train[col])
            test_encoded[col] = le.transform(test[col].map(lambda s: '<unknown>' if s not in le.classes_ else s))
            le_classes = le.classes_.tolist()
            le_classes_index = dict(zip(le.classes_, range(len(le.classes_))))
            le_classes_index['<unknown>'] = -1
            test_encoded[col] = test_encoded[col].map(le_classes_index).fillna(-1).astype(int)

    elif encoding == 'OE':  # Ordinal Encoding
        oe = OrdinalEncoder()
        train_encoded[cols] = oe.fit_transform(train[cols])
        test_encoded[cols] = oe.transform(test[cols].applymap(lambda s: '<unknown>' if s not in oe.categories_ else s))

    elif encoding == 'FE':  # Frequency Encoding
        for col in cols:
            fe = train[col].value_counts(normalize=True)
            train_encoded[col] = train[col].map(fe)
            test_encoded[col] = test[col].map(fe).fillna(fe.max())

    elif encoding == 'TE':  # Target Encoding
        te = TargetEncoder(cols=cols)
        train_encoded[cols] = te.fit_transform(train[cols], train['conversion'])
        test_encoded[cols] = te.transform(test[cols])
        
    else:
        raise ValueError("Tipo de codificación no válido. Usa 'OHE', 'LE', 'OE', 'FE' o 'TE'.")

    return train_encoded, test_encoded



def balance(X_train, y_train, technique):
    if technique == 'OS':  # Oversampling
        resampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
        
    elif technique == 'US':  # Undersampling
        resampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
        
    elif technique == 'LT':  # Links de Tomek
        resampler = TomekLinks(sampling_strategy='majority')
        
    elif technique == 'NM':  # Near Miss
        resampler = NearMiss(version=1)
        
    elif technique == 'SM':  # SMOTE
        resampler = SMOTE(random_state=42)
        
    else:
        raise ValueError(f"Técnica de balanceo no reconocida: {technique}")
        
    X_balanced, y_balanced = resampler.fit_resample(X_train, y_train)
    return X_balanced, y_balanced



class Optimization:
    def __init__(self, model, param_grid=None, n_iter=10, cv=5, random_state=None, bayes_search=False,
                 optuna_search=False):
        """
        Initialize the Optimization object.

        Parameters:
        - model: The machine learning model for which you want to perform hyperparameter optimization.
        - param_grid: The hyperparameter grid to search. Only used for Grid Search and Random Search.
        - n_iter: Number of random parameter combinations to try. Only used for Random Search.
        - cv: Number of cross-validation folds.
        - random_state: Seed for random number generation in Random Search.
        - bayes_search: Flag to indicate whether to use Bayesian optimization.
        """
        self.model = model
        self.param_grid = param_grid
        self.n_iter = n_iter
        self.cv = cv
        self.random_state = random_state
        self.bayes_search = bayes_search
        self.optuna_search = optuna_search

    def grid_search(self, X, y):
        """
        Perform Grid Search for hyperparameter optimization.

        Parameters:
        - X: The feature matrix.
        - y: The target vector.

        Returns:
        - best_params: The best hyperparameters found.
        """
        if self.param_grid is None:
            raise ValueError("param_grid is required for Grid Search")

        grid_search = GridSearchCV(self.model, param_grid=self.param_grid, cv=self.cv)
        grid_search.fit(X, y)
        best_params = grid_search.best_params_
        return best_params

    def random_search(self, X, y):
        """
        Perform Random Search for hyperparameter optimization.

        Parameters:
        - X: The feature matrix.
        - y: The target vector.

        Returns:
        - best_params: The best hyperparameters found.
        """
        random_search = RandomizedSearchCV(
            self.model,
            param_distributions=self.param_grid,
            n_iter=self.n_iter,
            cv=self.cv,
            random_state=self.random_state,
        )
        random_search.fit(X, y)
        # best_params = random_search.best_params_
        return random_search #best_params

    def bayesian_optimization(self, x, y, search_spaces):
        """
        Perform Bayesian Optimization for hyperparameter optimization.

        Parameters:
        - X: The feature matrix.
        - y: The target vector.
        - search_spaces: Dictionary defining the search space for Bayesian Optimization.

        Returns:
        - best_params: The best hyperparameters found.
        """
        if not self.bayes_search:
            raise ValueError("Bayesian optimization is not enabled. Set bayes_search=True to use it.")

        bayes_search = BayesSearchCV(
            self.model,
            search_spaces=search_spaces,
            n_iter=self.n_iter,
            cv=self.cv,
            random_state=self.random_state,
        )
        bayes_search.fit(x, y)
        # best_params = bayes_search.best_params_
        return bayes_search #best_params

    def optuna_optimization(self, X, y, n_trials=100):
        """
        Perform Optuna Optimization for hyperparameter optimization.

        Parameters:
        - X: The feature matrix.
        - y: The target vector.
        - n_trials: Number of optimization trials.

        Returns:
        - best_params: The best hyperparameters found.
        """
        if not self.optuna_search:
            raise ValueError("Optuna optimization is not enabled. Set optuna_search=True to use it.")

        def objective(trial):
            params = {}
            for key, value in self.param_grid.items():
                if isinstance(value, list):
                    params[key] = trial.suggest_categorical(key, value)
                elif isinstance(value, int):
                    params[key] = trial.suggest_int(key, value[0], value[1])
                elif isinstance(value, float):
                    params[key] = trial.suggest_float(key, value[0], value[1])
                else:
                    raise ValueError(f"Unsupported parameter type: {type(value)}")

            self.model.set_params(**params)
            scores = cross_val_score(self.model, X, y, cv=self.cv)
            return -scores.mean()  # Minimize negative mean cross-validation score

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        # best_params = study.best_params
        return study #best_params
    
