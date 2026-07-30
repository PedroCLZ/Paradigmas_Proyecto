"""Requerimientos del análisis:
- Detección de tipos de variables
- Análisis de datos
- Visualización de resultados

Análisis:
- Estadísticas descriptivas básicas
- Análisis de correlación
- Detección de valores atípicos
- Clustering automático
- Relaciones relevantes entre variables
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def detect_variable_types(summary: pd.DataFrame) -> dict:
    """Detecta los tipos de variables en un DataFrame. 
    Args: summary (pd.DataFrame): DataFrame a analizar. 
    Returns: dict: Diccionario con los tipos de variables. """
    variable_types = {}

    for column in summary.columns:
        if pd.api.types.is_numeric_dtype(summary[column]):
            variable_types[column] = "numeric"
        elif pd.api.types.is_string_dtype(summary[column]):
            variable_types[column] = "categorical"
        else:
            variable_types[column] = "other"

    return variable_types


def descriptive_statistics(summary: pd.DataFrame) -> pd.DataFrame:
    """Calcula estadísticas descriptivas básicas para un DataFrame. 
    Args: summary (pd.DataFrame): DataFrame a analizar. 
    Returns: pd.DataFrame: DataFrame con estadísticas descriptivas. """
    return summary.describe(include="all")


def correlation_analysis(summary: pd.DataFrame) -> pd.DataFrame:
    """Calcula la matriz de correlación de Pearson.
    Args: summary (pd.DataFrame): DataFrame a analizar.
    Returns: pd.DataFrame: DataFrame con la matriz de correlación."""
    numeric_data = summary.select_dtypes(include="number")
    return numeric_data.corr(method="pearson")


def atipical_value_detection(summary: pd.DataFrame) -> pd.DataFrame:
   #Detecta valores atípicos en un DataFrame utilizando el método del rango intercuartílico (IQR). 
    numeric_data = summary.select_dtypes(include="number")

    q1 = numeric_data.quantile(0.25)
    q3 = numeric_data.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    return numeric_data[(numeric_data < lower_bound) | (numeric_data > upper_bound)]


def clustering_analysis(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza clustering automático utilizando K-Means y
    devuelve la proyección PCA junto con el cluster asignado.
    """

    numeric_data = summary.select_dtypes(include="number")

    if numeric_data.empty:
        return pd.DataFrame({
            "message": ["No existen variables numéricas para realizar clustering."]
        })

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)

    # Número de clusters
    n_clusters = min(3, len(numeric_data))

    if n_clusters < 2:
        return pd.DataFrame({
            "message": ["No hay suficientes registros para realizar clustering."]
        })

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init="auto"
    )

    clusters = kmeans.fit_predict(scaled_data)

    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)

    result = pd.DataFrame(
        pca_data,
        columns=["PCA1", "PCA2"],
        index=summary.index
    )

    result["Cluster"] = clusters

    return result


def relevant_relationships(summary: pd.DataFrame) -> pd.DataFrame:
    """Detecta relaciones relevantes entre variables utilizando la correlación de Pearson. 
    Args: summary (pd.DataFrame): DataFrame a analizar. 
    Returns: pd.DataFrame: DataFrame con las relaciones relevantes. """

    correlation_matrix = correlation_analysis(summary)

    return correlation_matrix.where(
        (correlation_matrix >= 0.5) |
        (correlation_matrix <= -0.5)
    )