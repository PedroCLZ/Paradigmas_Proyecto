from reader import Reader
from cleaner import Cleaner

def main():
    reader = Reader()
    cleaner = Cleaner()
    folder = "data"

    print("==================================================")
    print("INICIANDO SISTEMA DE ANÁLISIS AUTOMATIZADO ")
    print("==================================================")

    if not reader.validate_path(folder):
        print(f"Error: La ruta '{folder}' no existe.")
        return

    try:
        print("\nBuscando archivos...")
        files = reader.get_files(folder)
        print(f"Se encontraron {len(files)} archivos compatibles.")

        print("\nLeyendo e integrando archivos...")
        raw_dataframe = reader.read_folder(folder)

        print("Iniciando proceso inteligente de limpieza y tipado de datos...")
        clean_dataframe = cleaner.clean(raw_dataframe)

        # Mostrar insights de valor
        reader.show_info(clean_dataframe)

        print("Proceso completado exitosamente.")
        print("Dataset consolidado y estructurado. Listo para la fase de Análisis")

    except Exception as error:
        print(f"\nOcurrió un error inesperado:\n{error}")

if __name__ == "__main__":
    main()