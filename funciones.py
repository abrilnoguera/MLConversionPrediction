# Encoding:
from category_encoders import TargetEncoder
from feature_engine.encoding import CountFrequencyEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn import preprocessing
import pandas as pd

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

    if encoding == 'OHE':  # One-Hot Encoding (Dummies)
        df_encoded = pd.get_dummies(df, columns=cols, prefix=cols)
    
    elif encoding == 'LE':  # Label Encoding
        encoder = preprocessing.LabelEncoder()
        df_encoded = df.copy()
        df_encoded[cols] = encoder.fit_transform(df[cols])
    
    elif encoding == 'OE':  # Ordinal Encoding
        encoder = OrdinalEncoder()
        df_encoded = df.copy()
        df_encoded[cols] = encoder.fit_transform(df[cols])

    elif encoding == 'FE':  # Frequency Encoding
        encoder = CountFrequencyEncoder(encoding_method='count', variables=cols)
        df_encoded = encoder.fit_transform(df)
    
    elif encoding == 'TE':  # Target Encoding
        encoder = TargetEncoder(cols=cols)
        df_encoded = encoder.fit_transform(df, df[cols])  
    
    else:
        raise ValueError("Tipo de codificación no válido. Usa 'OHE', 'LE', 'OE', 'FE' o 'TE'.")

    return df_encoded