# Encoding:
from category_encoders import TargetEncoder
from feature_engine.encoding import CountFrequencyEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn import preprocessing
import pandas as pd

# Balance
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks

# Optimization
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from skopt.searchcv import BayesSearchCV
import optuna



def encoding(df, cols, encoding):
    """
    Aplica la codificación especificada a las columnas seleccionadas del DataFrame.

    Args:
    df (DataFrame): El DataFrame que contiene los datos.
    cols (list): Lista de nombres de columnas a codificar.
    encoding (str): El tipo de codificación a aplicar ('OHE', 'LE', 'OE', 'FE', 'TE').

    Returns:
    DataFrame: Un nuevo DataFrame con las columnas codificadas.
    """

    train_df = df[df["ROW_ID"].isna()]

    if encoding == 'OHE':  # One-Hot Encoding (Dummies)
        df_encoded = pd.get_dummies(df, columns=cols, prefix=cols)
        train_encoded = pd.get_dummies(train_df, columns=cols, prefix=cols)

        # Asegurar que las columnas coincidan
        for col in train_encoded.columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[train_encoded.columns]
    
    elif encoding == 'LE':  # Label Encoding
        df_encoded = df.copy()
        for col in cols:
            encoder = preprocessing.LabelEncoder().fit(train_df[col])
            df_encoded[col] = df[col].map(lambda s: encoder.transform([s])[0] if s in encoder.classes_ else -1)
    
    elif encoding == 'OE':  # Ordinal Encoding
        encoder = OrdinalEncoder()
        encoder.fit(train_df[cols])
        df_encoded = df.copy()
        for col in cols:
            known_vals = encoder.categories_[cols.index(col)]
            df_encoded[col] = df[col].apply(lambda x: known_vals.tolist().index(x) if x in known_vals else -1)

    elif encoding == 'FE':  # Frequency Encoding
        df_encoded = df.copy()
        for col in cols:
            freq = train_df[col].value_counts(normalize=True)
            df_encoded[col] = df[col].map(freq).fillna(0)

    elif encoding == 'TE':  # Target Encoding
        encoder = TargetEncoder(cols=cols)
        df_encoded = df.copy()
        if 'conversion' in df.columns:
            encoder.fit(train_df[cols], train_df['conversion'])
            df_encoded[cols] = encoder.transform(df[cols])
        else:
            df_encoded[cols] = encoder.transform(df[cols]) 
    
    else:
        raise ValueError("Tipo de codificación no válido. Usa 'OHE', 'LE', 'OE', 'FE' o 'TE'.")

    return df_encoded



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
    
