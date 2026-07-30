"""Analisis de los datos de las universidades:
-Contar cuantos masters tiene cada una
-Contar cuantos doctorados tiene cada una
-Contar cuantos programas (departamentos) de grado tiene cada una
-Visualizar en que idioma se imparten las carreras
-Visualizar el field of study de las carreras
-Visualizar el nivel de topik de admisión del programa
-Buscar datos por busqueda de palabras clave en el nombre del programa"""

import pandas as pd


def _is_offered(value) -> bool:
    """Las columnas 'masters' y 'doctoral' marcan disponibilidad con 'O'
    (a veces como el caracter unicode 'O') o 'X', con espacios sueltos.
    Esta funcion normaliza esos valores a un booleano."""
    if pd.isna(value):
        return False
    return str(value).strip().upper() in ("O", "O")  # "O" ascii y "○" unicode


def count_total_universities(df: pd.DataFrame) -> int:
    """Funcion para contar el total de universidades"""
    return df["university"].nunique()


def count_masters_per_university(df: pd.DataFrame) -> pd.Series:
    """Funcion para contar cuantos programas de maestria tiene cada universidad"""
    ofrece_masters = df["masters"].apply(_is_offered).astype(bool)
    return df.loc[ofrece_masters].groupby("university").size()


def count_phd_per_university(df: pd.DataFrame) -> pd.Series:
    """Funcion para contar cuantos programas de doctorado tiene cada universidad"""
    ofrece_phd = df["doctoral"].apply(_is_offered).astype(bool)
    return df.loc[ofrece_phd].groupby("university").size()


def count_departments_per_university(df: pd.DataFrame) -> pd.Series:
    """Funcion para contar cuantos programas (departamentos) tiene cada universidad"""
    return df.groupby("university")["department"].nunique()

def count_departments_per_field_and_university(df: pd.DataFrame) -> pd.Series:
    """Funcion para contar cuantos programas (departamentos) tiene cada universidad"""
    return df.groupby(["field", "university"])["department"].nunique()



def topik_distribution(df: pd.DataFrame):
    """Funcion para visualizar el nivel de topik de admisión del programa por universidad"""
    if "topik" not in df.columns:
        return pd.DataFrame({"message": ["No hay datos de topik para este departamento"]})
    else:
        return df.groupby("department")["topik"].value_counts()


def language_distribution(df: pd.DataFrame):
    """Funcion para visualizar en que idioma se imparten las carreras por universidad"""
    if "language" not in df.columns:
        return pd.DataFrame({"message": ["No hay datos de idioma para este departamento"]})
    else:
        return df.groupby("university")["language"].value_counts()
    
def language_distribution_by_department(df: pd.DataFrame):
    """Funcion para visualizar en que idioma se imparten las carreras por universidad"""
    if "language" not in df.columns:
        return pd.DataFrame({"message": ["No hay datos de idioma para este departamento"]})
    else:
        return df.groupby("department")["language"].value_counts()


def field_distribution(df: pd.DataFrame):
    """Funcion para visualizar el field of study de las carreras por universidad"""
    if "field" not in df.columns:
        return pd.DataFrame({"message": ["No hay datos de field of study para este departamento"]})
    else:
        return df.groupby("university")["field"].value_counts()

def location_distribution(df: pd.DataFrame):
    """Funcion para visualizar la distribucion de ubicacion de las universidades"""
    if "campus_location" not in df.columns:
        return pd.DataFrame({"message": ["No hay datos de ubicacion para este departamento"]})
    else:
        return df.groupby(["university", "campus_location"]).size()

def search_by_keyword(df: pd.DataFrame, keyword: str) :
    """Funcion para buscar programas cuyo nombre de departamento contenga
    una palabra clave (busqueda insensible a mayusculas/minusculas)."""
    return df[df["department"].str.contains(keyword, case=False, na=False)]


"""Funcion para crear el dataframe con los datos númericos de cada universidad, incluyendo el total de programas de maestría, doctorado y departamentos"""
def create_university_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Crea un resumen de cada universidad con el total de programas de maestría, doctorado y departamentos."""
    summary = pd.DataFrame({
        "Maestrías": count_masters_per_university(df),
        "Doctorado": count_phd_per_university(df),
        "Departamentos": count_departments_per_university(df),
        "Campos": df.groupby("university")["field"].nunique(),
        "Idiomas": df.groupby("university")["language"].nunique(),
        "Campus": df.groupby("university")["campus_location"].nunique(),
        
    }).fillna(0).astype(int)

    return summary

