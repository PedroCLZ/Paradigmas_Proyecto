from pathlib import Path
import pandas as pd


class Reader:

    def __init__(self):

        self.supported_extensions = [".xlsx", ".xls", ".csv"]

    # -------------------------------------------------
    # VALIDACIONES
    # -------------------------------------------------

    def validate_path(self, path: str) -> bool:

        return Path(path).exists()

    def _validate_extension(self, path: Path):

        if path.suffix.lower() not in self.supported_extensions:

            raise ValueError(
                f"Formato '{path.suffix}' no soportado."
            )

    def _validate_dataframe(
        self,
        dataframe: pd.DataFrame,
        filename: str
    ):

        if dataframe.empty:

            raise ValueError(
                f"El archivo '{filename}' está vacío."
            )

        if dataframe.columns.empty:

            raise ValueError(
                f"El archivo '{filename}' no contiene columnas."
            )

    # -------------------------------------------------
    # BUSCAR ARCHIVOS
    # -------------------------------------------------

    def get_files(self, folder_path: str) -> list[Path]:

        folder = Path(folder_path)

        if not folder.exists():

            raise FileNotFoundError(
                f"La carpeta '{folder_path}' no existe."
            )

        if not folder.is_dir():

            raise NotADirectoryError(
                f"'{folder_path}' no es una carpeta."
            )

        files = sorted([

            file

            for file in folder.rglob("*")

            if file.is_file()

            and file.suffix.lower() in self.supported_extensions

        ])

        return files

    # -------------------------------------------------
    # LEER UN ARCHIVO
    # -------------------------------------------------

    def read_file(self, file_path: str) -> pd.DataFrame:

        path = Path(file_path)

        if not path.exists():

            raise FileNotFoundError(
                f"El archivo '{path.name}' no existe."
            )

        self._validate_extension(path)

        try:

            if path.suffix.lower() == ".csv":

                dataframe = pd.read_csv(path)

            else:

                # Encabezados comienza en fila 3
                preview = pd.read_excel(
                    path,
                    header=None,
                    nrows=10
                )

                header = 0

                for index, row in preview.iterrows():

                    values = row.astype(str).tolist()

                    if "No." in values:

                        header = index

                        break

                dataframe = pd.read_excel(
                    path,
                    header=header
                )

        except Exception as error:

            raise Exception(

                f"No fue posible leer '{path.name}'.\n{error}"

            )

        self._validate_dataframe(
            dataframe,
            path.name
        )

        # Eliminar columnas completamente vacías
        dataframe = dataframe.dropna(
            axis=1,
            how="all"
        )

        print(
            f"Archivo leído correctamente: {path.name}"
        )

        return dataframe

    # -------------------------------------------------
    # LEER TODA LA CARPETA
    # -------------------------------------------------

    def read_folder(self, folder_path: str) -> pd.DataFrame:

        files = self.get_files(folder_path)

        if len(files) == 0:

            raise FileNotFoundError(

                "No se encontraron archivos compatibles."

            )

        dataframes = []

        for file in files:

            dataframe = self.read_file(file)

            dataframe["University"] = file.parent.name

            dataframe["Archivo_Origen"] = file.name

            dataframes.append(dataframe)

        final_dataframe = pd.concat(

            dataframes,

            ignore_index=True,

            sort=False

        )

        print("\nResumen de lectura")
        print("----------------------------")
        print(f"Archivos procesados : {len(files)}")
        print(f"Registros totales   : {len(final_dataframe)}")
        print(f"Columnas            : {len(final_dataframe.columns)}")
        print("----------------------------")

        return final_dataframe

    # -------------------------------------------------
    # MOSTRAR INFORMACIÓN (REPORTE EXPLORATORIO)
    # -------------------------------------------------
    def show_info(self, dataframe: pd.DataFrame):
        print("\n" + "="*50)
        print("REPORTE BÁSICO DEL READER")
        print("="*50)

        print(f"\n   Dimensiones del Dataset:")
        print(f"   - Registros Totales: {dataframe.shape[0]}")
        print(f"   - Columnas Totales: {dataframe.shape[1]}")

        # Separar variables por tipo detectado automáticamente
        numeric_cols = dataframe.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = dataframe.select_dtypes(include=['object', 'category']).columns.tolist()

        print(f"\n  Tipos de Variables Detectadas:")
        print(f"   - Numéricas ({len(numeric_cols)}): {', '.join(numeric_cols[:4])}..." if numeric_cols else "   - Numéricas: 0 (Revise si los números vienen en formato texto)")
        print(f"   - Categóricas/Texto ({len(categorical_cols)}): {', '.join(categorical_cols[:4])}..." if categorical_cols else "   - Categóricas: 0")

        print("\n  Calidad de Datos (Top Columnas con Valores Nulos):")
        null_counts = dataframe.isnull().sum()
        null_cols = null_counts[null_counts > 0].sort_values(ascending=False)
        if not null_cols.empty:
            for col, count in null_cols.head(5).items():
                pct = (count / len(dataframe)) * 100
                print(f"   - {col}: {count} nulos ({pct:.1f}%)")
        else:
            print("   -  Dataset completo, sin valores nulos.")

        # Generar Estadísticas Descriptivas Automáticas
        if numeric_cols:
            print("\n Resumen Estadístico (Variables Numéricas):")
            stats = dataframe[numeric_cols].describe().T[['mean', 'min', 'max', 'std']].round(2)
            print(stats.to_string())
            
        if categorical_cols:
            print("\n Muestra de Variables Categóricas Principales:")
            for col in categorical_cols[:3]: # Muestra valores más frecuentes de las primeras 3
                top = dataframe[col].value_counts().head(2)
                top_str = " | ".join([f"{k} ({v})" for k, v in top.items()])
                print(f"   - {col}: {top_str}")

        print("\n" + "="*50)