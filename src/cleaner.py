import pandas as pd

class Cleaner:
    def __init__(self):
        # Si una columna tiene más del 95% de valores nulos, se considera irrelevante.
        self.null_threshold = 0.95

    def clean(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if dataframe.empty:
            raise ValueError("El DataFrame recibido está vacío.")

        # 1. Eliminar filas completamente vacías y duplicados
        dataframe = dataframe.dropna(how="all").drop_duplicates()

        # 2. Limpiar nombres de columnas y quitar 'Unnamed'
        dataframe.columns = dataframe.columns.astype(str).str.strip().str.replace("\n", " ", regex=False)
        
        unnamed_cols = [col for col in dataframe.columns if col.lower().startswith("unnamed")]
        if unnamed_cols:
            dataframe = dataframe.drop(columns=unnamed_cols)

        # 3. Eliminar filas donde se haya colado un encabezado repetido
        if "No." in dataframe.columns:
            dataframe = dataframe[dataframe["No."] != "No."]

        # 4. Limpiar espacios de texto y manejar nulos
        object_columns = dataframe.select_dtypes(include="object").columns
        for col in object_columns:
            dataframe[col] = (
                dataframe[col]
                .astype(str)
                .str.strip()
                # Se agrega "<NA>" al diccionario por si acaso astype(str) lo convierte a texto
                .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "NaN": pd.NA, "<NA>": pd.NA})
            )

        # 5. Eliminar columnas casi vacías según el umbral
        min_vals = int(len(dataframe) * (1 - self.null_threshold))
        dataframe = dataframe.dropna(axis=1, thresh=min_vals)

        # 6. Eliminar filas donde TODAS las variables de interés estén vacías
        ignore_cols = ["Universidad", "Archivo_Origen"]
        cols_to_validate = [c for c in dataframe.columns if c not in ignore_cols]
        dataframe = dataframe.dropna(subset=cols_to_validate, how="all")


        for col in dataframe.columns:
            if col not in ignore_cols:
                try:
                    # Convertir columnas numericas en tipo numerico
                    dataframe[col] = pd.to_numeric(dataframe[col], errors='ignore')
                except Exception:
                    pass

        # Reiniciar índices
        dataframe = dataframe.reset_index(drop=True)
        return dataframe