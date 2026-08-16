from reader import Reader
from cleaner import Cleaner
from visualization import Visualizer

import university_analyzer as ua
import general_analyzer as ga
import ai_analysis as ai


def main():
    reader = Reader()
    cleaner = Cleaner()
    visualizer = Visualizer()

    # La ruta es relativa a la carpeta desde donde
    # se ejecuta el programa.
    folder = "data"

    print("=" * 60)
    print("INICIANDO SISTEMA DE ANÁLISIS AUTOMATIZADO")
    print("=" * 60)

    # ========================================================
    # 1. VALIDAR CARPETA DE DATOS
    # ========================================================

    if not reader.validate_path(folder):
        print(f"Error: La ruta '{folder}' no existe.")
        return

    try:
        # ====================================================
        # 2. BUSCAR ARCHIVOS
        # ====================================================

        print("\nBuscando archivos...")

        files = reader.get_files(folder)

        if not files:
            print(
                f"No se encontraron archivos Excel o CSV "
                f"dentro de la carpeta '{folder}'."
            )
            return

        print(
            f"Se encontraron {len(files)} "
            f"archivos compatibles."
        )

        # ====================================================
        # 3. LEER E INTEGRAR ARCHIVOS
        # ====================================================

        print("\nLeyendo e integrando archivos...")

        raw_dataframe = reader.read_folder(folder)

        print(
            f"DataFrame inicial: "
            f"{raw_dataframe.shape[0]} filas y "
            f"{raw_dataframe.shape[1]} columnas."
        )

        # ====================================================
        # 4. LIMPIAR LOS DATOS
        # ====================================================

        print(
            "\nIniciando proceso de limpieza "
            "y tipado de datos..."
        )

        clean_dataframe = cleaner.clean(
            raw_dataframe
        )

        print(
            f"DataFrame limpio: "
            f"{clean_dataframe.shape[0]} filas y "
            f"{clean_dataframe.shape[1]} columnas."
        )

        # Mostrar reporte exploratorio básico
        reader.show_info(clean_dataframe)

        print("\n" + "=" * 60)
        print("COLUMNAS DESPUÉS DE LA LIMPIEZA Y EL MAPEO")
        print("=" * 60)

        for column in clean_dataframe.columns:
            print(f"- {column}")

        print(
            "\nProceso de limpieza completado exitosamente."
        )

        print(
            "El dataset está listo para la fase de análisis."
        )

        # ====================================================
        # 5. ANÁLISIS ESPECÍFICO DE UNIVERSIDADES
        # ====================================================

        print("\n" + "=" * 60)
        print("INICIANDO ANÁLISIS DE UNIVERSIDADES")
        print("=" * 60)

        # ----------------------------------------------------
        # Total de universidades
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print("TOTAL DE UNIVERSIDADES")
        print("=" * 60)

        total_universities = (
            ua.count_total_universities(
                clean_dataframe
            )
        )

        print(total_universities)

        # ----------------------------------------------------
        # Maestrías
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print("MAESTRÍAS POR UNIVERSIDAD")
        print("=" * 60)

        print(
            ua.count_masters_per_university(
                clean_dataframe
            )
        )

        # ----------------------------------------------------
        # Doctorados
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print("DOCTORADOS POR UNIVERSIDAD")
        print("=" * 60)

        print(
            ua.count_phd_per_university(
                clean_dataframe
            )
        )

        # ----------------------------------------------------
        # Departamentos
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print("DEPARTAMENTOS POR UNIVERSIDAD")
        print("=" * 60)

        print(
            ua.count_departments_per_university(
                clean_dataframe
            )
        )

        # ----------------------------------------------------
        # TOPIK
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print(
            "DISTRIBUCIÓN DE TOPIK "
            "- PRIMEROS 10 RESULTADOS"
        )
        print("=" * 60)

        topik_distribution = (
            ua.topik_distribution(
                clean_dataframe
            )
        )

        print(topik_distribution.head(10))

        # ----------------------------------------------------
        # Idiomas
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print(
            "DISTRIBUCIÓN DE IDIOMAS "
            "- PRIMEROS 10 RESULTADOS"
        )
        print("=" * 60)

        language_distribution = (
            ua.language_distribution(
                clean_dataframe
            )
        )

        print(language_distribution.head(10))

        # ----------------------------------------------------
        # Campos de estudio
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print("DISTRIBUCIÓN DE CAMPOS DE ESTUDIO")
        print("=" * 60)

        print(
            ua.field_distribution(
                clean_dataframe
            )
        )

        # ----------------------------------------------------
        # Departamentos por campo
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print(
            "DEPARTAMENTOS POR CAMPO "
            "DE ESTUDIO Y UNIVERSIDAD"
        )
        print("=" * 60)

        print(
            ua.count_departments_per_field_and_university(
                clean_dataframe
            )
        )

        # ----------------------------------------------------
        # Ubicaciones
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print("DISTRIBUCIÓN DE UBICACIONES")
        print("=" * 60)

        print(
            ua.location_distribution(
                clean_dataframe
            )
        )

        # ----------------------------------------------------
        # Idiomas por departamento
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print("DISTRIBUCIÓN DE IDIOMAS POR DEPARTAMENTO")
        print("=" * 60)

        print(
            ua.language_distribution_by_department(
                clean_dataframe
            )
        )

        # ====================================================
        # 6. BÚSQUEDA POR PALABRA CLAVE
        # ====================================================

        keyword = "Artificial Intelligence"

        print("\n" + "=" * 60)
        print(
            f"BÚSQUEDA POR PALABRA CLAVE: "
            f"'{keyword}'"
        )
        print("=" * 60)

        search_results = ua.search_by_keyword(
            clean_dataframe,
            keyword,
        )

        if search_results.empty:
            print(
                f"No se encontraron programas "
                f"relacionados con '{keyword}'."
            )

        else:
            columns_to_show = [
                column
                for column in [
                    "university",
                    "department",
                    "field",
                ]
                if column in search_results.columns
            ]

            print(
                search_results[
                    columns_to_show
                ].head(10)
            )

        print(
            f"Total de coincidencias: "
            f"{len(search_results)}"
        )

        # ====================================================
        # 7. CREAR RESUMEN NUMÉRICO
        # ====================================================

        print("\n" + "=" * 60)
        print("RESUMEN NUMÉRICO DE UNIVERSIDADES")
        print("=" * 60)

        summary = ua.create_university_summary(
            clean_dataframe
        )

        if summary.empty:
            print(
                "No fue posible crear el resumen "
                "numérico de universidades."
            )
            return

        print(summary)

        # ====================================================
        # 8. ANÁLISIS GENERAL DE DATOS
        # ====================================================

        print("\n" + "=" * 60)
        print("INICIANDO ANÁLISIS GENERAL DE DATOS")
        print("=" * 60)

        # ----------------------------------------------------
        # Tipos de variables
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print("TIPOS DE VARIABLES")
        print("=" * 60)

        variable_types = ga.detect_variable_types(
            summary
        )

        print(variable_types)

        # ----------------------------------------------------
        # Estadísticas descriptivas
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print("ESTADÍSTICAS DESCRIPTIVAS")
        print("=" * 60)

        descriptive_statistics = (
            ga.descriptive_statistics(
                summary
            )
        )

        print(descriptive_statistics)

        # ----------------------------------------------------
        # Correlación
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print("MATRIZ DE CORRELACIÓN")
        print("=" * 60)

        correlation_matrix = (
            ga.correlation_analysis(
                summary
            )
        )

        print(correlation_matrix)

        # ----------------------------------------------------
        # Valores atípicos
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print("VALORES ATÍPICOS")
        print("=" * 60)

        outliers = ga.atipical_value_detection(
            summary
        )

        outlier_rows = outliers.dropna(
            how="all"
        )

        if outlier_rows.empty:
            print(
                "No se detectaron valores atípicos."
            )
        else:
            print(outlier_rows)

        # ----------------------------------------------------
        # Relaciones relevantes
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print("RELACIONES RELEVANTES")
        print("=" * 60)

        relevant_relationships = (
            ga.relevant_relationships(
                summary
            )
        )

        print(relevant_relationships)

        # ====================================================
        # 9. GENERAR VISUALIZACIONES
        # ====================================================

        # Los gráficos deben generarse antes del informe PDF,
        # porque ai_analysis.py los inserta en el documento.

        print("\n" + "=" * 60)
        print("GENERANDO VISUALIZACIONES")
        print("=" * 60)

        visualizer.export_all(
            clean_dataframe,
            summary,
        )

        # ====================================================
        # 10. INTELIGENCIA ARTIFICIAL E INFORMES
        # ====================================================

        print("\n" + "=" * 60)
        print("ANÁLISIS DE INTELIGENCIA ARTIFICIAL")
        print("=" * 60)

        university_clusters, interpretations = (
            ai.run_ai_analysis(
                clean_dataframe,
                summary,
            )
        )

        # ----------------------------------------------------
        # Mostrar clusters
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print("UNIVERSIDADES Y CLUSTERS ASIGNADOS")
        print("=" * 60)

        if university_clusters.empty:
            print(
                "No fue posible realizar el clustering "
                "de las universidades."
            )
        else:
            print(university_clusters)

        # ----------------------------------------------------
        # Mostrar interpretaciones
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print("INTERPRETACIONES AUTOMÁTICAS")
        print("=" * 60)

        if not interpretations:
            print(
                "No fue posible generar interpretaciones."
            )
        else:
            for interpretation in interpretations:
                print(f"\n{interpretation}")

        # ====================================================
        # 11. RESULTADO FINAL
        # ====================================================

        print("\n" + "=" * 60)
        print("PROCESO COMPLETADO EXITOSAMENTE")
        print("=" * 60)

        print("\nResultados generados:")

        print("\n- Informe en texto:")
        print("  output/informe_final.txt")

        print("\n- Informe en PDF:")
        print("  output/informe_final.pdf")

        print("\n- Visualizaciones:")
        print("  output/images/")

        print(
            f"\n- Archivos procesados: "
            f"{len(files)}"
        )

        print(
            f"- Registros analizados: "
            f"{len(clean_dataframe)}"
        )

        print(
            f"- Universidades analizadas: "
            f"{len(summary)}"
        )

    # ========================================================
    # MANEJO DE ERRORES
    # ========================================================

    except FileNotFoundError as error:
        print(
            f"\nError de archivo o carpeta:\n{error}"
        )

    except ModuleNotFoundError as error:
        print(
            f"\nFalta instalar una dependencia:\n{error}"
        )

    except ValueError as error:
        print(
            f"\nError en los datos recibidos:\n{error}"
        )

    except KeyError as error:
        print(
            "\nNo se encontró una columna necesaria "
            f"para realizar el análisis:\n{error}"
        )

    except Exception as error:
        print(
            f"\nOcurrió un error inesperado:\n{error}"
        )


if __name__ == "__main__":
    main()