import pandas as pd

class Cleaner:
    COLUMNS_TO_DROP = [
        "대학명",
        "학과명",
        "학과계열",
        "Remarks",
        "Website URL for detailed information",
        "단과대학",   # Facultad, solo aparece en Jeonbuk National University
        "Major",     # Solo aparece en Hankuk Univ. (Graduate School)
    ]

    COLUMN_MAPPING = {

        "university (대학알리미 공시 명칭)": "university_name",
        "campus location": "campus_location",
        "department (대학알리미 공시 명칭)": "department",
        "field of study (division)": "field",
        "masters'": "masters",
        "medium of instruction": "language",
        "required topik for admission after language program": "topik",
        "program starts": "program_starts",
        "doctoral": "doctoral",
        "medium of instruction.1": "language.1",
        "required topik for admission after language program.1": "topik.1",
        "program starts.1": "program_starts.1",
        }


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
                .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "NaN": pd.NA, "<NA>": pd.NA}) # type: ignore
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
                    dataframe[col] = pd.to_numeric(dataframe[col], errors='ignore') # type: ignore
                except Exception:
                    pass

       
          # 9. Eliminar columnas irrelevantes según la lista definida
        dataframe = dataframe.drop(columns=self.COLUMNS_TO_DROP, errors="ignore")
        unnamed_cols = [c for c in dataframe.columns if str(c).startswith("Unnamed")]
        dataframe = dataframe.drop(columns=unnamed_cols, errors="ignore")           
      
         #8.Normalizar nombres de columnas según el mapeo definido
        dataframe.columns = dataframe.columns.str.strip().str.lower().str.replace("\n", "_")

         # 7. Renombrar columnas según el mapeo definido
        dataframe = dataframe.rename(columns=self.COLUMN_MAPPING)
        



        # Reiniciar índices
        dataframe = dataframe.reset_index(drop=True)
        return dataframe