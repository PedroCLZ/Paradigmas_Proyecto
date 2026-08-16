from pathlib import Path
from typing import Any

import pandas as pd

import general_analyzer as ga


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

REPORT_TITLE = "Informe final del análisis de universidades GKS"

TEXT_REPORT_PATH = "output/informe_final.txt"
PDF_REPORT_PATH = "output/informe_final.pdf"
IMAGES_FOLDER = "output/images"


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def _normalized_value_counts(
    series: pd.Series,
    title_case: bool = False,
) -> pd.Series:
    """
    Limpia y cuenta valores categóricos.

    Normaliza:
    - Espacios innecesarios.
    - Diferencias alrededor de "/".
    - Diferencias de mayúsculas en ubicaciones.
    """

    clean_series = (
        series
        .dropna()
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace(r"\s*/\s*", " / ", regex=True)
    )

    # Los guiones representan ausencia de información
    clean_series = clean_series[
        ~clean_series.isin(["", "-", "nan", "None"])
    ]

    if title_case:
        clean_series = clean_series.str.title()

    return clean_series.value_counts()


def _format_distribution(
    distribution: pd.Series,
    limit: int = 10,
) -> str:
    """
    Convierte una distribución en una tabla de texto.
    """

    if distribution.empty:
        return "No hay datos disponibles."

    result = (
        distribution
        .head(limit)
        .rename_axis("Categoría")
        .reset_index(name="Cantidad")
    )

    return result.to_string(index=False)


def _get_top_category(
    distribution: pd.Series,
) -> tuple[str, int]:
    """
    Retorna la categoría más frecuente y su cantidad.
    """

    if distribution.empty:
        return "Sin datos", 0

    category = str(distribution.index[0])
    quantity = int(distribution.iloc[0])

    return category, quantity


# ============================================================
# CLUSTERING
# ============================================================

def create_cluster_analysis(
    summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ejecuta K-Means y agrega el cluster asignado
    al resumen numérico de cada universidad.

    Retorna:
    - Universidades con su cluster.
    - Perfil promedio de cada cluster.
    """

    clusters = ga.clustering_analysis(summary)

    if "Cluster" not in clusters.columns:
        return pd.DataFrame(), pd.DataFrame()

    university_clusters = summary.copy()

    university_clusters["Cluster"] = clusters[
        "Cluster"
    ]

    cluster_profiles = (
        university_clusters
        .groupby("Cluster")
        .mean(numeric_only=True)
        .round(2)
    )

    return university_clusters, cluster_profiles


def generate_cluster_interpretations(
    summary: pd.DataFrame,
    university_clusters: pd.DataFrame,
    cluster_profiles: pd.DataFrame,
) -> list[str]:
    """
    Genera una interpretación automática para cada cluster.
    """

    if university_clusters.empty or cluster_profiles.empty:
        return [
            "No fue posible generar interpretaciones "
            "de los clusters."
        ]

    interpretations = []

    general_means = summary.mean(numeric_only=True)

    for cluster_id, profile in cluster_profiles.iterrows():
        universities = university_clusters[
            university_clusters["Cluster"] == cluster_id
        ].index.tolist()

        masters = profile.get("Maestrías", 0)
        doctorates = profile.get("Doctorado", 0)
        departments = profile.get("Departamentos", 0)
        fields = profile.get("Campos", 0)
        languages = profile.get("Idiomas", 0)
        campuses = profile.get("Campus", 0)

        if (
            masters > general_means.get("Maestrías", 0)
            and departments
            > general_means.get("Departamentos", 0)
        ):
            description = (
                "universidades con una oferta académica amplia, "
                "con una cantidad elevada de maestrías "
                "y departamentos"
            )

        elif (
            departments
            < general_means.get("Departamentos", 0)
            and masters
            < general_means.get("Maestrías", 0)
        ):
            description = (
                "universidades con una oferta académica "
                "comparativamente reducida"
            )

        elif fields < general_means.get("Campos", 0):
            description = (
                "universidades especializadas, concentradas "
                "en una menor variedad de campos de estudio"
            )

        elif doctorates > general_means.get("Doctorado", 0):
            description = (
                "universidades con una presencia destacada "
                "de programas doctorales"
            )

        else:
            description = (
                "universidades con una oferta académica intermedia"
            )

        university_text = ", ".join(
            map(str, universities)
        )

        interpretation = (
            f"Grupo {cluster_id}: {description}.\n"
            f"Universidades: {university_text}.\n"
            f"Promedios del grupo: "
            f"{masters:.2f} maestrías, "
            f"{doctorates:.2f} doctorados, "
            f"{departments:.2f} departamentos, "
            f"{fields:.2f} campos, "
            f"{languages:.2f} idiomas y "
            f"{campuses:.2f} campus."
        )

        interpretations.append(interpretation)

    return interpretations


# ============================================================
# ANÁLISIS DESCRIPTIVO
# ============================================================

def create_descriptive_analysis(
    clean_dataframe: pd.DataFrame,
) -> dict[str, pd.Series]:
    """
    Calcula distribuciones de las variables categóricas
    relevantes.
    """

    analysis: dict[str, pd.Series] = {}

    if "campus_location" in clean_dataframe.columns:
        analysis["locations"] = _normalized_value_counts(
            clean_dataframe["campus_location"],
            title_case=True,
        )

    if "field" in clean_dataframe.columns:
        fields = clean_dataframe["field"].replace({
            "Engneering": "Engineering",
        })

        analysis["fields"] = _normalized_value_counts(
            fields
        )

    if "language" in clean_dataframe.columns:
        analysis["masters_languages"] = (
            _normalized_value_counts(
                clean_dataframe["language"]
            )
        )

    if "language.1" in clean_dataframe.columns:
        analysis["doctoral_languages"] = (
            _normalized_value_counts(
                clean_dataframe["language.1"]
            )
        )

    if "topik" in clean_dataframe.columns:
        analysis["masters_topik"] = (
            _normalized_value_counts(
                clean_dataframe["topik"]
            )
        )

    if "topik.1" in clean_dataframe.columns:
        analysis["doctoral_topik"] = (
            _normalized_value_counts(
                clean_dataframe["topik.1"]
            )
        )

    if "program_starts" in clean_dataframe.columns:
        analysis["masters_starts"] = (
            _normalized_value_counts(
                clean_dataframe["program_starts"]
            )
        )

    if "program_starts.1" in clean_dataframe.columns:
        analysis["doctoral_starts"] = (
            _normalized_value_counts(
                clean_dataframe["program_starts.1"]
            )
        )

    if "embassy track type" in clean_dataframe.columns:
        analysis["embassy_tracks"] = (
            _normalized_value_counts(
                clean_dataframe["embassy track type"]
            )
        )

    if (
        "univ. track applicable programs"
        in clean_dataframe.columns
    ):
        analysis["university_tracks"] = (
            _normalized_value_counts(
                clean_dataframe[
                    "univ. track applicable programs"
                ]
            )
        )

    return analysis


def generate_general_interpretations(
    clean_dataframe: pd.DataFrame,
    descriptive_analysis: dict[str, pd.Series],
) -> list[str]:
    """
    Genera interpretaciones de las distribuciones
    categóricas del dataset.
    """

    interpretations = []

    # --------------------------------------------------------
    # Ubicaciones
    # --------------------------------------------------------

    locations = descriptive_analysis.get(
        "locations",
        pd.Series(dtype=int),
    )

    location, location_count = _get_top_category(
        locations
    )

    if location_count > 0:
        interpretations.append(
            f"Ubicación: {location} es la ubicación con "
            f"mayor cantidad de registros de programas "
            f"({location_count}). Esto indica una concentración "
            f"importante de la oferta académica en esa zona."
        )

    # --------------------------------------------------------
    # Campos de estudio
    # --------------------------------------------------------

    fields = descriptive_analysis.get(
        "fields",
        pd.Series(dtype=int),
    )

    field, field_count = _get_top_category(fields)

    if field_count > 0:
        interpretations.append(
            f"Campo de estudio: {field} es el campo con "
            f"mayor presencia, con {field_count} registros. "
            f"Esto permite identificar una de las áreas "
            f"académicas predominantes."
        )

    # --------------------------------------------------------
    # Idiomas de maestría
    # --------------------------------------------------------

    masters_languages = descriptive_analysis.get(
        "masters_languages",
        pd.Series(dtype=int),
    )

    language, language_count = _get_top_category(
        masters_languages
    )

    if language_count > 0:
        interpretations.append(
            f"Idioma de maestría: la modalidad más frecuente "
            f"es '{language}', presente en "
            f"{language_count} registros. El idioma de enseñanza "
            f"es relevante para valorar la accesibilidad de "
            f"los programas para estudiantes internacionales."
        )

    # --------------------------------------------------------
    # Idiomas de doctorado
    # --------------------------------------------------------

    doctoral_languages = descriptive_analysis.get(
        "doctoral_languages",
        pd.Series(dtype=int),
    )

    language, language_count = _get_top_category(
        doctoral_languages
    )

    if language_count > 0:
        interpretations.append(
            f"Idioma de doctorado: la modalidad más frecuente "
            f"es '{language}', presente en "
            f"{language_count} registros."
        )

    # --------------------------------------------------------
    # TOPIK de maestría
    # --------------------------------------------------------

    masters_topik = descriptive_analysis.get(
        "masters_topik",
        pd.Series(dtype=int),
    )

    topik, topik_count = _get_top_category(
        masters_topik
    )

    if topik_count > 0:
        interpretations.append(
            f"TOPIK para maestría: el requisito más común es "
            f"'{topik}', registrado en "
            f"{topik_count} programas."
        )

    # --------------------------------------------------------
    # TOPIK de doctorado
    # --------------------------------------------------------

    doctoral_topik = descriptive_analysis.get(
        "doctoral_topik",
        pd.Series(dtype=int),
    )

    topik, topik_count = _get_top_category(
        doctoral_topik
    )

    if topik_count > 0:
        interpretations.append(
            f"TOPIK para doctorado: el requisito más común es "
            f"'{topik}', presente en "
            f"{topik_count} registros."
        )

    # --------------------------------------------------------
    # Inicio de maestrías
    # --------------------------------------------------------

    masters_starts = descriptive_analysis.get(
        "masters_starts",
        pd.Series(dtype=int),
    )

    start, start_count = _get_top_category(
        masters_starts
    )

    if start_count > 0:
        interpretations.append(
            f"Inicio de maestrías: la opción más frecuente es "
            f"'{start}', presente en "
            f"{start_count} registros."
        )

    # --------------------------------------------------------
    # Inicio de doctorados
    # --------------------------------------------------------

    doctoral_starts = descriptive_analysis.get(
        "doctoral_starts",
        pd.Series(dtype=int),
    )

    start, start_count = _get_top_category(
        doctoral_starts
    )

    if start_count > 0:
        interpretations.append(
            f"Inicio de doctorados: la opción más frecuente es "
            f"'{start}', presente en "
            f"{start_count} registros."
        )

    # --------------------------------------------------------
    # Programas relacionados con IA
    # --------------------------------------------------------

    if "department" in clean_dataframe.columns:
        ai_programs = clean_dataframe[
            clean_dataframe["department"].str.contains(
                r"Artificial Intelligence|\bAI\b",
                case=False,
                na=False,
                regex=True,
            )
        ]

        interpretations.append(
            f"Programas relacionados con Inteligencia Artificial: "
            f"se encontraron {len(ai_programs)} registros cuyo "
            f"departamento contiene 'Artificial Intelligence' "
            f"o el término independiente 'AI'."
        )

    return interpretations


# ============================================================
# INFORME TXT
# ============================================================

def export_text_report(
    clean_dataframe: pd.DataFrame,
    summary: pd.DataFrame,
    university_clusters: pd.DataFrame,
    cluster_profiles: pd.DataFrame,
    cluster_interpretations: list[str],
    descriptive_analysis: dict[str, pd.Series],
    general_interpretations: list[str],
    output_path: str = TEXT_REPORT_PATH,
) -> None:
    """
    Genera el informe completo en formato TXT.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "INFORME FINAL DEL ANÁLISIS DE UNIVERSIDADES GKS",
        "=" * 70,
        "",
        "1. INFORMACIÓN GENERAL",
        "-" * 70,
        f"Registros analizados: {len(clean_dataframe)}",
        f"Universidades analizadas: {len(summary)}",
        f"Variables analizadas: {len(clean_dataframe.columns)}",
        "",
        "2. RESUMEN NUMÉRICO POR UNIVERSIDAD",
        "-" * 70,
        summary.to_string(),
        "",
        "3. DISTRIBUCIÓN DE UBICACIONES",
        "-" * 70,
        _format_distribution(
            descriptive_analysis.get(
                "locations",
                pd.Series(dtype=int),
            )
        ),
        "",
        "4. DISTRIBUCIÓN DE CAMPOS DE ESTUDIO",
        "-" * 70,
        _format_distribution(
            descriptive_analysis.get(
                "fields",
                pd.Series(dtype=int),
            )
        ),
        "",
        "5. IDIOMAS DE MAESTRÍA",
        "-" * 70,
        _format_distribution(
            descriptive_analysis.get(
                "masters_languages",
                pd.Series(dtype=int),
            )
        ),
        "",
        "6. IDIOMAS DE DOCTORADO",
        "-" * 70,
        _format_distribution(
            descriptive_analysis.get(
                "doctoral_languages",
                pd.Series(dtype=int),
            )
        ),
        "",
        "7. REQUISITOS TOPIK PARA MAESTRÍA",
        "-" * 70,
        _format_distribution(
            descriptive_analysis.get(
                "masters_topik",
                pd.Series(dtype=int),
            )
        ),
        "",
        "8. REQUISITOS TOPIK PARA DOCTORADO",
        "-" * 70,
        _format_distribution(
            descriptive_analysis.get(
                "doctoral_topik",
                pd.Series(dtype=int),
            )
        ),
        "",
        "9. INICIO DE MAESTRÍAS",
        "-" * 70,
        _format_distribution(
            descriptive_analysis.get(
                "masters_starts",
                pd.Series(dtype=int),
            )
        ),
        "",
        "10. INICIO DE DOCTORADOS",
        "-" * 70,
        _format_distribution(
            descriptive_analysis.get(
                "doctoral_starts",
                pd.Series(dtype=int),
            )
        ),
        "",
        "11. INTERPRETACIÓN GENERAL",
        "-" * 70,
    ]

    for interpretation in general_interpretations:
        lines.append(f"- {interpretation}")

    lines.extend([
        "",
        "12. UNIVERSIDADES Y CLUSTERS",
        "-" * 70,
        (
            university_clusters.to_string()
            if not university_clusters.empty
            else "No fue posible realizar el clustering."
        ),
        "",
        "13. PERFIL PROMEDIO DE LOS CLUSTERS",
        "-" * 70,
        (
            cluster_profiles.to_string()
            if not cluster_profiles.empty
            else "No hay perfiles disponibles."
        ),
        "",
        "14. INTERPRETACIÓN DE LOS CLUSTERS",
        "-" * 70,
    ])

    for interpretation in cluster_interpretations:
        lines.append(interpretation)
        lines.append("")

    lines.extend([
        "15. CONCLUSIÓN",
        "-" * 70,
        (
            "El análisis permitió estudiar la oferta académica "
            "de las universidades GKS considerando maestrías, "
            "doctorados, departamentos, campos de estudio, "
            "idiomas, ubicaciones, requisitos TOPIK y fechas "
            "de inicio."
        ),
        (
            "El algoritmo K-Means agrupó las universidades "
            "según similitudes en su oferta académica. Las "
            "interpretaciones permiten comprender las "
            "características principales de cada grupo."
        ),
        "",
        "Los gráficos se encuentran en:",
        IMAGES_FOLDER,
    ])

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(f"Informe TXT guardado en: {path}")


# ============================================================
# FUNCIONES AUXILIARES PARA EL PDF
# ============================================================

def _create_pdf_styles() -> dict[str, Any]:
    """
    Crea los estilos visuales utilizados en el PDF.
    """

    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.styles import getSampleStyleSheet

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=29,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#17365D"),
            spaceAfter=18,
        )
    )

    styles.add(
        ParagraphStyle(
            name="ReportSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=11,
            leading=16,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#52606D"),
            spaceAfter=12,
        )
    )

    styles.add(
        ParagraphStyle(
            name="SectionTitle",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=19,
            textColor=colors.HexColor("#17365D"),
            spaceBefore=8,
            spaceAfter=10,
        )
    )

    styles.add(
        ParagraphStyle(
            name="SubsectionTitle",
            parent=styles["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#2F5597"),
            spaceBefore=6,
            spaceAfter=7,
        )
    )

    styles.add(
        ParagraphStyle(
            name="ReportBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#263238"),
            spaceAfter=7,
        )
    )

    styles.add(
        ParagraphStyle(
            name="InsightText",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            textColor=colors.HexColor("#17365D"),
        )
    )

    styles.add(
        ParagraphStyle(
            name="TableHeader",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            alignment=TA_CENTER,
            textColor=colors.white,
        )
    )

    styles.add(
        ParagraphStyle(
            name="TableCell",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=9.5,
            textColor=colors.HexColor("#263238"),
        )
    )

    return styles


def _create_dataframe_table(
    dataframe: pd.DataFrame,
    styles: dict[str, Any],
    max_width: float,
    first_column_width: float | None = None,
):
    """
    Convierte un DataFrame en una tabla de ReportLab.
    """

    from reportlab.lib import colors
    from reportlab.platypus import Paragraph, Table, TableStyle

    if dataframe.empty:
        return Paragraph(
            "No hay datos disponibles.",
            styles["ReportBody"],
        )

    table_dataframe = dataframe.reset_index()

    headers = [
        Paragraph(
            str(column),
            styles["TableHeader"],
        )
        for column in table_dataframe.columns
    ]

    table_data = [headers]

    for row in table_dataframe.itertuples(
        index=False,
        name=None,
    ):
        formatted_row = []

        for value in row:
            if pd.isna(value):
                text = "-"
            elif isinstance(value, float):
                text = f"{value:.2f}"
            else:
                text = str(value)

            formatted_row.append(
                Paragraph(
                    text,
                    styles["TableCell"],
                )
            )

        table_data.append(formatted_row)

    column_count = len(table_dataframe.columns)

    if first_column_width and column_count > 1:
        remaining_width = max_width - first_column_width

        column_widths = [
            first_column_width
        ] + [
            remaining_width / (column_count - 1)
        ] * (column_count - 1)
    else:
        column_widths = [
            max_width / column_count
        ] * column_count

    table = Table(
        table_data,
        colWidths=column_widths,
        repeatRows=1,
        hAlign="CENTER",
    )

    table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#2F5597"),
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white,
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.35,
                colors.HexColor("#B0BEC5"),
            ),
            (
                "BACKGROUND",
                (0, 1),
                (-1, -1),
                colors.HexColor("#FFFFFF"),
            ),
            (
                "ROWBACKGROUNDS",
                (0, 1),
                (-1, -1),
                [
                    colors.HexColor("#FFFFFF"),
                    colors.HexColor("#EEF3F8"),
                ],
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE",
            ),
            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                5,
            ),
            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                5,
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                5,
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                5,
            ),
        ])
    )

    return table


def _create_distribution_table(
    distribution: pd.Series,
    styles: dict[str, Any],
    max_width: float,
    limit: int = 10,
):
    """
    Crea una tabla PDF a partir de una distribución.
    """

    if distribution.empty:
        from reportlab.platypus import Paragraph

        return Paragraph(
            "No hay datos disponibles.",
            styles["ReportBody"],
        )

    dataframe = (
        distribution
        .head(limit)
        .rename_axis("Categoría")
        .reset_index(name="Cantidad")
    )

    # Evita agregar otro índice al construir la tabla
    dataframe.index.name = None

    return _create_dataframe_table(
        dataframe.set_index("Categoría"),
        styles,
        max_width=max_width,
        first_column_width=max_width * 0.80,
    )


def _create_metric_cards(
    total_records: int,
    total_universities: int,
    total_variables: int,
    styles: dict[str, Any],
    max_width: float,
):
    """
    Crea tarjetas con los indicadores generales.
    """

    from reportlab.lib import colors
    from reportlab.platypus import Paragraph, Table, TableStyle

    card_width = max_width / 3

    data = [[
        Paragraph(
            f"<b>{total_records:,}</b><br/>Registros",
            styles["InsightText"],
        ),
        Paragraph(
            f"<b>{total_universities}</b><br/>Universidades",
            styles["InsightText"],
        ),
        Paragraph(
            f"<b>{total_variables}</b><br/>Variables",
            styles["InsightText"],
        ),
    ]]

    table = Table(
        data,
        colWidths=[card_width] * 3,
        rowHeights=[55],
    )

    table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                colors.HexColor("#EAF1F8"),
            ),
            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.8,
                colors.HexColor("#8EA9C1"),
            ),
            (
                "INNERGRID",
                (0, 0),
                (-1, -1),
                0.8,
                colors.HexColor("#8EA9C1"),
            ),
            (
                "ALIGN",
                (0, 0),
                (-1, -1),
                "CENTER",
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE",
            ),
        ])
    )

    return table


def _create_insight_box(
    text: str,
    styles: dict[str, Any],
    max_width: float,
):
    """
    Crea un recuadro para una interpretación.
    """

    from reportlab.lib import colors
    from reportlab.platypus import Paragraph, Table, TableStyle

    paragraph = Paragraph(
        text.replace("\n", "<br/>"),
        styles["InsightText"],
    )

    table = Table(
        [[paragraph]],
        colWidths=[max_width],
    )

    table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                colors.HexColor("#EAF1F8"),
            ),
            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.8,
                colors.HexColor("#8EA9C1"),
            ),
            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                10,
            ),
            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                10,
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                9,
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                9,
            ),
        ])
    )

    return table


def _create_report_image(
    image_path: str,
    max_width: float,
    max_height: float = 330,
):
    """
    Crea una imagen manteniendo su proporción.
    """

    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import Image as ReportImage

    path = Path(image_path)

    if not path.exists():
        return None

    reader = ImageReader(str(path))
    width, height = reader.getSize()

    scale = min(
        max_width / width,
        max_height / height,
    )

    return ReportImage(
        str(path),
        width=width * scale,
        height=height * scale,
    )


def _draw_page_number(canvas, document) -> None:
    """
    Dibuja encabezado y número de página.
    """

    from reportlab.lib import colors

    canvas.saveState()

    page_width, page_height = document.pagesize

    canvas.setStrokeColor(
        colors.HexColor("#D0D7DE")
    )

    canvas.line(
        document.leftMargin,
        25,
        page_width - document.rightMargin,
        25,
    )

    canvas.setFont(
        "Helvetica",
        8,
    )

    canvas.setFillColor(
        colors.HexColor("#607D8B")
    )

    canvas.drawString(
        document.leftMargin,
        14,
        "Análisis de universidades GKS",
    )

    canvas.drawRightString(
        page_width - document.rightMargin,
        14,
        f"Página {canvas.getPageNumber()}",
    )

    canvas.restoreState()


# ============================================================
# INFORME PDF
# ============================================================

def export_pdf_report(
    clean_dataframe: pd.DataFrame,
    summary: pd.DataFrame,
    university_clusters: pd.DataFrame,
    cluster_profiles: pd.DataFrame,
    cluster_interpretations: list[str],
    descriptive_analysis: dict[str, pd.Series],
    general_interpretations: list[str],
    output_path: str = PDF_REPORT_PATH,
    images_folder: str = IMAGES_FOLDER,
) -> None:
    """
    Genera el informe final en formato PDF.

    Incluye:
    - Indicadores generales.
    - Interpretaciones.
    - Tablas.
    - Gráficos.
    - Resultados del clustering.
    """

    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            KeepTogether,
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
        )

    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "ReportLab no está instalado. Ejecuta: "
            "py -m pip install reportlab"
        ) from error

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    styles = _create_pdf_styles()

    document = SimpleDocTemplate(
        str(path),
        pagesize=landscape(A4),
        rightMargin=14 * mm,
        leftMargin=14 * mm,
        topMargin=14 * mm,
        bottomMargin=16 * mm,
        title=REPORT_TITLE,
        author="Sistema de análisis GKS",
        subject=(
            "Análisis descriptivo y clustering "
            "de universidades GKS"
        ),
    )

    max_width = document.width

    story = []

    # ========================================================
    # PORTADA
    # ========================================================

    story.append(Spacer(1, 35 * mm))

    story.append(
        Paragraph(
            "INFORME FINAL",
            styles["ReportTitle"],
        )
    )

    story.append(
        Paragraph(
            "Análisis automatizado de universidades GKS",
            styles["ReportTitle"],
        )
    )

    story.append(Spacer(1, 8 * mm))

    story.append(
        Paragraph(
            (
                "Análisis descriptivo, detección de patrones "
                "y agrupamiento automático mediante K-Means"
            ),
            styles["ReportSubtitle"],
        )
    )

    story.append(Spacer(1, 14 * mm))

    story.append(
        _create_metric_cards(
            total_records=len(clean_dataframe),
            total_universities=len(summary),
            total_variables=len(clean_dataframe.columns),
            styles=styles,
            max_width=max_width,
        )
    )

    story.append(PageBreak())

    # ========================================================
    # RESUMEN EJECUTIVO
    # ========================================================

    story.append(
        Paragraph(
            "1. Resumen ejecutivo",
            styles["SectionTitle"],
        )
    )

    story.append(
        Paragraph(
            (
                "Este informe presenta los principales resultados "
                "obtenidos a partir de los datos de universidades "
                "participantes en el programa GKS. El análisis "
                "considera la oferta de maestrías y doctorados, "
                "departamentos, campos de estudio, idiomas, "
                "ubicaciones, requisitos TOPIK y fechas de inicio."
            ),
            styles["ReportBody"],
        )
    )

    story.append(
        Paragraph(
            (
                "También se utilizó el algoritmo K-Means para "
                "identificar grupos de universidades con perfiles "
                "académicos similares. PCA se utilizó únicamente "
                "para representar visualmente esos grupos en dos "
                "dimensiones."
            ),
            styles["ReportBody"],
        )
    )

    story.append(Spacer(1, 4 * mm))

    for interpretation in general_interpretations:
        story.append(
            _create_insight_box(
                interpretation,
                styles,
                max_width,
            )
        )

        story.append(Spacer(1, 3 * mm))

    # ========================================================
    # RESUMEN POR UNIVERSIDAD
    # ========================================================

    story.append(PageBreak())

    story.append(
        Paragraph(
            "2. Resumen numérico por universidad",
            styles["SectionTitle"],
        )
    )

    story.append(
        Paragraph(
            (
                "La siguiente tabla resume la cantidad de "
                "maestrías, doctorados, departamentos, campos, "
                "idiomas y campus identificados para cada "
                "universidad."
            ),
            styles["ReportBody"],
        )
    )

    story.append(
        _create_dataframe_table(
            summary,
            styles,
            max_width=max_width,
            first_column_width=190,
        )
    )

    # ========================================================
    # UBICACIONES
    # ========================================================

    story.append(PageBreak())

    story.append(
        Paragraph(
            "3. Distribución geográfica",
            styles["SectionTitle"],
        )
    )

    story.append(
        Paragraph(
            (
                "La distribución representa la cantidad de "
                "registros de programas asociados con cada "
                "ubicación. No corresponde al número de "
                "universidades por ciudad."
            ),
            styles["ReportBody"],
        )
    )

    locations = descriptive_analysis.get(
        "locations",
        pd.Series(dtype=int),
    )

    story.append(
        _create_distribution_table(
            locations,
            styles,
            max_width=max_width,
        )
    )

    story.append(Spacer(1, 5 * mm))

    # ========================================================
    # CAMPOS DE ESTUDIO
    # ========================================================

    story.append(
        Paragraph(
            "4. Campos de estudio",
            styles["SectionTitle"],
        )
    )

    fields = descriptive_analysis.get(
        "fields",
        pd.Series(dtype=int),
    )

    story.append(
        _create_distribution_table(
            fields,
            styles,
            max_width=max_width,
        )
    )

    fields_image = _create_report_image(
        f"{images_folder}/fields.png",
        max_width=max_width,
    )

    if fields_image is not None:
        story.append(PageBreak())
        story.append(
            Paragraph(
                "Distribución visual de campos de estudio",
                styles["SubsectionTitle"],
            )
        )
        story.append(fields_image)

    # ========================================================
    # IDIOMAS
    # ========================================================

    story.append(PageBreak())

    story.append(
        Paragraph(
            "5. Idiomas de enseñanza",
            styles["SectionTitle"],
        )
    )

    story.append(
        Paragraph(
            "Idiomas de los programas de maestría",
            styles["SubsectionTitle"],
        )
    )

    story.append(
        _create_distribution_table(
            descriptive_analysis.get(
                "masters_languages",
                pd.Series(dtype=int),
            ),
            styles,
            max_width=max_width,
        )
    )

    story.append(Spacer(1, 5 * mm))

    story.append(
        Paragraph(
            "Idiomas de los programas de doctorado",
            styles["SubsectionTitle"],
        )
    )

    story.append(
        _create_distribution_table(
            descriptive_analysis.get(
                "doctoral_languages",
                pd.Series(dtype=int),
            ),
            styles,
            max_width=max_width,
        )
    )

    languages_image = _create_report_image(
        f"{images_folder}/languages.png",
        max_width=max_width,
    )

    if languages_image is not None:
        story.append(PageBreak())
        story.append(
            Paragraph(
                "Distribución visual de idiomas",
                styles["SubsectionTitle"],
            )
        )
        story.append(languages_image)

    # ========================================================
    # TOPIK
    # ========================================================

    story.append(PageBreak())

    story.append(
        Paragraph(
            "6. Requisitos TOPIK",
            styles["SectionTitle"],
        )
    )

    story.append(
        Paragraph(
            "Requisitos TOPIK para maestría",
            styles["SubsectionTitle"],
        )
    )

    story.append(
        _create_distribution_table(
            descriptive_analysis.get(
                "masters_topik",
                pd.Series(dtype=int),
            ),
            styles,
            max_width=max_width,
        )
    )

    story.append(Spacer(1, 5 * mm))

    story.append(
        Paragraph(
            "Requisitos TOPIK para doctorado",
            styles["SubsectionTitle"],
        )
    )

    story.append(
        _create_distribution_table(
            descriptive_analysis.get(
                "doctoral_topik",
                pd.Series(dtype=int),
            ),
            styles,
            max_width=max_width,
        )
    )

    # ========================================================
    # FECHAS DE INICIO
    # ========================================================

    story.append(PageBreak())

    story.append(
        Paragraph(
            "7. Fechas de inicio",
            styles["SectionTitle"],
        )
    )

    story.append(
        Paragraph(
            "Inicio de programas de maestría",
            styles["SubsectionTitle"],
        )
    )

    story.append(
        _create_distribution_table(
            descriptive_analysis.get(
                "masters_starts",
                pd.Series(dtype=int),
            ),
            styles,
            max_width=max_width,
        )
    )

    story.append(Spacer(1, 5 * mm))

    story.append(
        Paragraph(
            "Inicio de programas de doctorado",
            styles["SubsectionTitle"],
        )
    )

    story.append(
        _create_distribution_table(
            descriptive_analysis.get(
                "doctoral_starts",
                pd.Series(dtype=int),
            ),
            styles,
            max_width=max_width,
        )
    )

    # ========================================================
    # CLUSTERS
    # ========================================================

    story.append(PageBreak())

    story.append(
        Paragraph(
            "8. Análisis de clusters",
            styles["SectionTitle"],
        )
    )

    story.append(
        Paragraph(
            (
                "K-Means agrupó las universidades utilizando "
                "la cantidad de maestrías, doctorados, "
                "departamentos, campos, idiomas y campus. "
                "Los números de cluster son identificadores y "
                "no representan una clasificación de calidad."
            ),
            styles["ReportBody"],
        )
    )

    for interpretation in cluster_interpretations:
        story.append(
            _create_insight_box(
                interpretation,
                styles,
                max_width,
            )
        )

        story.append(Spacer(1, 3 * mm))

    story.append(Spacer(1, 4 * mm))

    story.append(
        Paragraph(
            "Perfil promedio de los clusters",
            styles["SubsectionTitle"],
        )
    )

    if cluster_profiles.empty:
        story.append(
            Paragraph(
                "No hay perfiles disponibles.",
                styles["ReportBody"],
            )
        )
    else:
        story.append(
            _create_dataframe_table(
                cluster_profiles,
                styles,
                max_width=max_width,
                first_column_width=90,
            )
        )

    # ========================================================
    # GRÁFICO DE CLUSTERS
    # ========================================================

    clusters_image = _create_report_image(
        f"{images_folder}/clusters.png",
        max_width=max_width,
    )

    if clusters_image is not None:
        story.append(PageBreak())

        story.append(
            Paragraph(
                "9. Visualización PCA de los clusters",
                styles["SectionTitle"],
            )
        )

        story.append(
            Paragraph(
                (
                    "PCA reduce las características numéricas "
                    "a dos componentes para permitir su "
                    "representación gráfica. Las universidades "
                    "cercanas tienen perfiles similares y el "
                    "color representa el cluster asignado."
                ),
                styles["ReportBody"],
            )
        )

        story.append(clusters_image)

    # ========================================================
    # CORRELACIÓN
    # ========================================================

    correlation_image = _create_report_image(
        f"{images_folder}/correlation_heatmap.png",
        max_width=max_width,
    )

    if correlation_image is not None:
        story.append(PageBreak())

        story.append(
            Paragraph(
                "10. Correlación entre variables",
                styles["SectionTitle"],
            )
        )

        story.append(
            Paragraph(
                (
                    "El mapa de calor muestra la intensidad y "
                    "dirección de las relaciones lineales entre "
                    "las variables numéricas. Una correlación "
                    "alta no implica necesariamente causalidad."
                ),
                styles["ReportBody"],
            )
        )

        story.append(correlation_image)

    # ========================================================
    # UNIVERSIDADES Y CLUSTERS
    # ========================================================

    story.append(PageBreak())

    story.append(
        Paragraph(
            "11. Universidades y clusters asignados",
            styles["SectionTitle"],
        )
    )

    if university_clusters.empty:
        story.append(
            Paragraph(
                "No fue posible realizar el clustering.",
                styles["ReportBody"],
            )
        )
    else:
        story.append(
            _create_dataframe_table(
                university_clusters,
                styles,
                max_width=max_width,
                first_column_width=190,
            )
        )

    # ========================================================
    # CONCLUSIÓN
    # ========================================================

    story.append(PageBreak())

    story.append(
        Paragraph(
            "12. Conclusión",
            styles["SectionTitle"],
        )
    )

    story.append(
        Paragraph(
            (
                "El análisis permitió estudiar la oferta "
                "académica de las universidades GKS mediante "
                "indicadores relacionados con maestrías, "
                "doctorados, departamentos, campos de estudio, "
                "idiomas, ubicaciones, requisitos TOPIK y "
                "fechas de inicio."
            ),
            styles["ReportBody"],
        )
    )

    story.append(
        Paragraph(
            (
                "El algoritmo K-Means identificó universidades "
                "con perfiles académicos similares. El análisis "
                "descriptivo y las reglas de interpretación "
                "transformaron los resultados numéricos en "
                "información comprensible para usuarios sin "
                "experiencia técnica."
            ),
            styles["ReportBody"],
        )
    )

    story.append(
        Paragraph(
            (
                "Los resultados deben interpretarse como un "
                "análisis exploratorio basado en los archivos "
                "procesados. Los números de cluster no indican "
                "que un grupo sea mejor que otro."
            ),
            styles["ReportBody"],
        )
    )

    document.build(
        story,
        onFirstPage=_draw_page_number,
        onLaterPages=_draw_page_number,
    )

    print(f"Informe PDF guardado en: {path}")


# ============================================================
# EJECUCIÓN COMPLETA
# ============================================================

def run_ai_analysis(
    clean_dataframe: pd.DataFrame,
    summary: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Ejecuta:
    - Clustering.
    - Análisis descriptivo.
    - Interpretaciones.
    - Informe TXT.
    - Informe PDF.
    """

    university_clusters, cluster_profiles = (
        create_cluster_analysis(summary)
    )

    cluster_interpretations = (
        generate_cluster_interpretations(
            summary,
            university_clusters,
            cluster_profiles,
        )
    )

    descriptive_analysis = create_descriptive_analysis(
        clean_dataframe
    )

    general_interpretations = (
        generate_general_interpretations(
            clean_dataframe,
            descriptive_analysis,
        )
    )

    export_text_report(
        clean_dataframe=clean_dataframe,
        summary=summary,
        university_clusters=university_clusters,
        cluster_profiles=cluster_profiles,
        cluster_interpretations=cluster_interpretations,
        descriptive_analysis=descriptive_analysis,
        general_interpretations=general_interpretations,
    )

    export_pdf_report(
        clean_dataframe=clean_dataframe,
        summary=summary,
        university_clusters=university_clusters,
        cluster_profiles=cluster_profiles,
        cluster_interpretations=cluster_interpretations,
        descriptive_analysis=descriptive_analysis,
        general_interpretations=general_interpretations,
    )

    all_interpretations = (
        general_interpretations
        + cluster_interpretations
    )

    return university_clusters, all_interpretations