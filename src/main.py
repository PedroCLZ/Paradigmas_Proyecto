from reader import Reader
from cleaner import Cleaner
from visualization import Visualizer
import university_analyzer as ua
import general_analyzer as ga

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

        print("\nCOLUMNAS DESPUÉS DEL MAPEO:")
        print(clean_dataframe.columns.tolist())

        print("Proceso completado exitosamente.")
        print("Dataset consolidado y estructurado. Listo para la fase de Análisis")

        print("==================================================")
        print("INICIANDO Análisis de Datos")
        print("==================================================")

        print ("Analisis de Universidades")

        print("\n" + "=" * 60)
        print("TOTAL DE UNIVERSIDADES")
        print("=" * 60)
        print(ua.count_total_universities(clean_dataframe))

        print("\n" + "=" * 60)
        print("MASTERS POR UNIVERSIDAD")
        print("=" * 60)
        print(ua.count_masters_per_university(clean_dataframe))

        print("\n" + "=" * 60)
        print("DOCTORADOS (PhD) POR UNIVERSIDAD")
        print("=" * 60)
        print(ua.count_phd_per_university(clean_dataframe))

        print("\n" + "=" * 60)
        print("DEPARTAMENTOS POR UNIVERSIDAD")
        print("=" * 60)
        print(ua.count_departments_per_university(clean_dataframe))

        print("\n" + "=" * 60)
        print("DISTRIBUCION DE TOPIK (primeros 10)")
        print("=" * 60)
        print(ua.topik_distribution(clean_dataframe).head(10))

        print("\n" + "=" * 60)
        print("DISTRIBUCION DE IDIOMA (primeros 10)")
        print("=" * 60)
        print(ua.language_distribution(clean_dataframe).head(10))

        print("\n" + "=" * 60)
        print("DISTRIBUCION DE FIELD OF STUDY (primeros 10)")
        print("=" * 60)
        print(ua.field_distribution(clean_dataframe))
        print("\n" + "=" * 60)
        print("DISTRIBUCION DE DEPARTAMENTOS POR FIELD OF STUDY")
        print("=" * 100)
        print(ua.count_departments_per_field_and_university(clean_dataframe))

        print("\n" + "=" * 60)
        print("DISTRIBUCION DE UBICACION DE LAS UNIVERSIDADES")
        print("=" * 60)
        print(ua.location_distribution(clean_dataframe))

        print("\n" + "=" * 60)
        print("DISTRIBUCION DE IDIOMA POR DEPARTAMENTO ")
        print("=" * 60)
        print(ua.language_distribution_by_department(clean_dataframe))

        print("\n" + "=" * 60)
        print("BUSQUEDA POR PALABRA CLAVE: 'Artificial Intelligence'")
        print("=" * 60)

        resultados = ua.search_by_keyword(clean_dataframe, "Artificial Intelligence")
        print(resultados[["university", "field"]].head(10))
        print(f"Total de coincidencias: {len(resultados)}")


        print("===============================================")
        print("Iniciando Análisis General de Datos")
        print("===============================================")


        print("\n" + "=" * 60)
        print("RESUMEN NUMÉRICO DE UNIVERSIDADES")
        print("=" * 60)

        summary = ua.create_university_summary(clean_dataframe)
        print(summary)

        print("\n" + "=" * 60)
        print("TIPOS DE VARIABLES")
        print("=" * 60)

        print(ga.detect_variable_types(summary))

        print("\n" + "=" * 60)
        print("ESTADÍSTICAS DESCRIPTIVAS")
        print("=" * 60)

        print(ga.descriptive_statistics(summary))

        print("\n" + "=" * 60)
        print("MATRIZ DE CORRELACIÓN")
        print("=" * 60)

        print(ga.correlation_analysis(summary))

        print("\n" + "=" * 60)
        print("VALORES ATÍPICOS")
        print("=" * 60)

        print(ga.atipical_value_detection(summary))

        clusters = ga.clustering_analysis(summary)

        print("\n" + "=" * 60)
        print("CLUSTERING")
        print("=" * 60)
        print(clusters)


        print(ga.clustering_analysis(summary))

        print("\n" + "=" * 60)
        print("RELACIONES RELEVANTES")
        print("=" * 60)

        print(ga.relevant_relationships(summary))

        visualizer = Visualizer()
        visualizer.export_all(clean_dataframe, summary)

    except Exception as error:
        print(f"\nOcurrió un error inesperado:\n{error}")

if __name__ == "__main__":
    main()