from zipfile import Path

from reader import Reader

def main():

    reader = Reader()

    folder = Path(__file__).resolve().parent / "data"

    if not reader.validate_path(folder):
        print("La ruta indicada no existe.")
        return

    try:

        print("\nBuscando archivos...")

        files = reader.get_files(folder)

        print(f"Se encontraron {len(files)} archivos.\n")

        for file in files:
            print(f"- {file.name}")

        print("\nLeyendo archivos...\n")

        dataframe = reader.read_folder(folder)

        reader.show_info(dataframe)

    except Exception as error:

        print("\nOcurrio un error:")
        print(error)


if __name__ == "__main__":
    main()