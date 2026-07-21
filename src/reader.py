from pathlib import Path
import pandas as pd


class Reader:

    def __init__(self):
        self.supported_extensions = [".xlsx", ".xls", ".csv"]

    def validate_path(self, path: str) -> bool:
        return Path(path).exists()

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

        files = [
            file
            for file in folder.iterdir()
            if file.is_file() and file.suffix.lower() in self.supported_extensions
        ]

        return files

    def read_file(self, file_path: str) -> pd.DataFrame:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"El archivo '{path.name}' no existe."
            )

        self._validate_extension(path)

        try:

            if path.suffix.lower() == ".csv":
                df = pd.read_csv(path)

            else:
                df = pd.read_excel(path)

        except Exception as e:
            raise Exception(
                f"No fue posible leer '{path.name}'.\n{e}"
            )

        self._validate_dataframe(df, path.name)

        print(f"Archivo leido correctamente: {path.name}")

        return df

    def read_folder(self, folder_path: str) -> pd.DataFrame:
        files = self.get_files(folder_path)

        if len(files) == 0:
            raise FileNotFoundError(
                "No se encontraron archivos Excel o CSV."
            )

        dataframes = []

        for file in files:

            df = self.read_file(file)

            df["Archivo_Origen"] = file.name

            dataframes.append(df)

        final_df = pd.concat(
            dataframes,
            ignore_index=True
        )

        print("\nResumen de lectura")
        print("----------------------------")
        print(f"Archivos procesados : {len(files)}")
        print(f"Total de registros  : {len(final_df)}")
        print(f"Total de columnas   : {len(final_df.columns)}")
        print("----------------------------")

        return final_df

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
                f"El archivo '{filename}' esta vacio."
            )

        if dataframe.columns.empty:

            raise ValueError(
                f"El archivo '{filename}' no posee columnas."
            )

    def show_info(self, dataframe: pd.DataFrame):
        print("\nInformacion del DataFrame")
        print("----------------------------")
        print(dataframe.info())

        print("\nPrimeras filas")
        print("----------------------------")
        print(dataframe.head())

        print("\nValores nulos")
        print("----------------------------")
        print(dataframe.isnull().sum())