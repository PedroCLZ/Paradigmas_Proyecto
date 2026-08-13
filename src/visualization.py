from pathlib import Path
from typing import Any, Optional

import pandas as pd

import general_analyzer as ga
import university_analyzer as ua


class Visualizer:
    """Genera visualizaciones a partir de los resultados del pipeline.

    Esta clase es independiente del resto de la lógica de análisis y solo se
    encarga de crear gráficos y exportarlos a archivos PNG.
    """

    def __init__(self, output_folder: str = "output/images") -> None:
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self._plt = self._initialize_matplotlib()

    @staticmethod
    def _initialize_matplotlib() -> Optional[Any]:
        try:
            import matplotlib.pyplot as plt
            return plt
        except ModuleNotFoundError:
            print("Advertencia: matplotlib no está disponible. No se generarán gráficos.")
            return None

    def export_all(self, clean_dataframe: pd.DataFrame, summary: pd.DataFrame) -> None:
        """Genera y exporta todas las visualizaciones definidas."""
        if self._plt is None:
            print("Visualización omitida porque matplotlib no está instalado.")
            return

        self.create_departments_chart(clean_dataframe)
        self.create_languages_chart(clean_dataframe)
        self.create_fields_chart(clean_dataframe)
        self.create_correlation_heatmap(summary)
        self.create_clusters_chart(summary)
        print("Proceso de visualización finalizado.")

    def create_departments_chart(self, dataframe: pd.DataFrame) -> None:
        """Genera el gráfico de cantidad de departamentos por universidad."""
        print("✓ Generando gráfico de departamentos...")
        if not self._has_columns(dataframe, {"university", "department"}):
            print("Omitido departments_per_university: faltan columnas necesarias.")
            return

        try:
            data = ua.count_departments_per_university(dataframe)
            if data.empty:
                print("Omitido departments_per_university: no hay datos para graficar.")
                return

            fig, ax = self._plt.subplots(figsize=(12, 6))
            data.sort_values(ascending=False).plot(
                kind="bar",
                ax=ax,
                color="#4c72b0",
                edgecolor="black"
            )
            ax.set_title("Cantidad de departamentos por universidad")
            ax.set_xlabel("Universidad")
            ax.set_ylabel("Departamentos")
            ax.tick_params(axis="x", rotation=45) 
            for label in ax.get_xticklabels():
                label.set_horizontalalignment("right")

            self._save_plot(fig, "departments_per_university.png")
        except Exception as error:
            print(f"Omitido departments_per_university: {error}")

    def create_languages_chart(self, dataframe: pd.DataFrame) -> None:
        """Genera el gráfico de distribución de idiomas."""
        print("✓ Generando gráfico de idiomas...")
        if not self._has_columns(dataframe, {"university", "language"}):
            print("Omitido languages: faltan columnas necesarias.")
            return

        try:
            data = ua.language_distribution(dataframe)
            counts = self._prepare_distribution_series(data, "language")
            if counts.empty:
                print("Omitido languages: no hay datos para graficar.")
                return

            fig, ax = self._plt.subplots(figsize=(10, 6))
            counts.sort_values(ascending=False).plot(
                kind="bar",
                ax=ax,
                color="#55a868",
                edgecolor="black"
            )
            ax.set_title("Distribución de idiomas")
            ax.set_xlabel("Idioma")
            ax.set_ylabel("Cantidad de programas")
            ax.tick_params(axis="x", rotation=45) 
            for label in ax.get_xticklabels():
                label.set_horizontalalignment("right")

            self._save_plot(fig, "languages.png")
        except Exception as error:
            print(f"Omitido languages: {error}")

    def create_fields_chart(self, dataframe: pd.DataFrame) -> None:
        """Genera el gráfico de distribución de campos de estudio."""
        print("✓ Generando gráfico de campos...")
        if not self._has_columns(dataframe, {"university", "field"}):
            print("Omitido fields: faltan columnas necesarias.")
            return

        try:
            data = ua.field_distribution(dataframe)
            counts = self._prepare_distribution_series(data, "field")
            if counts.empty:
                print("Omitido fields: no hay datos para graficar.")
                return

            fig, ax = self._plt.subplots(figsize=(10, 6))
            counts.sort_values(ascending=False).plot(
                kind="bar",
                ax=ax,
                color="#c44e52",
                edgecolor="black"
            )
            ax.set_title("Distribución de campos de estudio")
            ax.set_xlabel("Campo de estudio")
            ax.set_ylabel("Cantidad de programas")
            ax.tick_params(axis="x", rotation=45) 
            for label in ax.get_xticklabels():
                label.set_horizontalalignment("right")

            self._save_plot(fig, "fields.png")
        except Exception as error:
            print(f"Omitido fields: {error}")

    def create_correlation_heatmap(self, summary: pd.DataFrame) -> None:
        """Genera un heatmap de la matriz de correlación."""
        print("✓ Generando Heatmap...")
        if summary is None or summary.empty:
            print("Omitido correlation_heatmap: summary vacío.")
            return

        try:
            correlation = ga.correlation_analysis(summary)
            if correlation.empty:
                print("Omitido correlation_heatmap: no hay datos numéricos para la correlación.")
                return

            fig, ax = self._plt.subplots(figsize=(10, 8))
            heatmap = ax.imshow(correlation, cmap="viridis", aspect="auto")
            ax.set_xticks(range(len(correlation.columns)))
            ax.set_xticklabels(correlation.columns, rotation=45, ha="right")
            ax.set_yticks(range(len(correlation.index)))
            ax.set_yticklabels(correlation.index)

            for row in range(len(correlation.index)):
                for col in range(len(correlation.columns)):
                    ax.text(
                        col,
                        row,
                        f"{correlation.iat[row, col]:.2f}",
                        ha="center",
                        va="center",
                        color="white" if abs(correlation.iat[row, col]) > 0.5 else "black",
                        fontsize=8,
                    )

            fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title("Mapa de calor de correlación")

            self._save_plot(fig, "correlation_heatmap.png")
        except Exception as error:
            print(f"Omitido correlation_heatmap: {error}")

    def create_clusters_chart(self, summary: pd.DataFrame) -> None:
        """Genera el gráfico de clusters basado en PCA."""
        print("✓ Generando gráfico de Clusters...")
        if summary is None or summary.empty:
            print("Omitido clusters: summary vacío.")
            return

        try:
            cluster_df = ga.clustering_analysis(summary)
            if cluster_df.empty:
                print("Omitido clusters: no hay datos de clustering.")
                return

            required_columns = {"PCA1", "PCA2", "Cluster"}
            if not required_columns.issubset(cluster_df.columns):
                print("Omitido clusters: salida de clustering no válida.")
                return

            fig, ax = self._plt.subplots(figsize=(10, 7))
            scatter = ax.scatter(
                cluster_df["PCA1"],
                cluster_df["PCA2"],
                c=cluster_df["Cluster"],
                cmap="tab10",
                s=80,
                edgecolor="black",
                alpha=0.8,
            )

            ax.set_title("Clusters PCA")
            ax.set_xlabel("PCA1")
            ax.set_ylabel("PCA2")
            ax.grid(True, linestyle="--", alpha=0.3)

            for index, label in enumerate(cluster_df.index):
                ax.annotate(
                    str(label),
                    (cluster_df["PCA1"].iat[index], cluster_df["PCA2"].iat[index]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=7,
                )

            legend = ax.legend(*scatter.legend_elements(), title="Cluster", loc="best")
            ax.add_artist(legend)
            self._save_plot(fig, "clusters.png")
        except Exception as error:
            print(f"Omitido clusters: {error}")

    def _prepare_distribution_series(self, result: Any, level_name: str) -> pd.Series:
        """Convierte la salida de distribution en una serie agregada por el nivel deseado."""
        if isinstance(result, pd.DataFrame):
            if "message" in result.columns:
                return pd.Series(dtype=int)
            return result.squeeze()

        if isinstance(result, pd.Series):
            if result.empty:
                return result
            if isinstance(result.index, pd.MultiIndex):
                if level_name in result.index.names:
                    return result.groupby(level=level_name).sum()
            return result

        return pd.Series(dtype=int)

    def _has_columns(self, dataframe: pd.DataFrame, columns: set[str]) -> bool:
        """Verifica que el DataFrame contenga todas las columnas necesarias."""
        return columns.issubset(set(dataframe.columns))

    def _save_plot(self, figure: Any, filename: str) -> None:
        """Guarda la figura en disco con los parámetros solicitados."""
        path = self.output_folder / filename
        figure.savefig(path, dpi=300, bbox_inches="tight")
        self._plt.close(figure)
        print(f"✓ Imagen guardada: {path}")
