from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image as RLImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .real_pipeline import load_real_config, verify_real_pipeline


NAVY = colors.HexColor("#12304A")
BLUE = colors.HexColor("#2563EB")
TEAL = colors.HexColor("#0F766E")
LIGHT = colors.HexColor("#EAF1F7")
MUTED = colors.HexColor("#475569")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _scaled_image(path: Path, max_width: float, max_height: float) -> RLImage:
    image = RLImage(str(path))
    ratio = min(max_width / image.imageWidth, max_height / image.imageHeight)
    image.drawWidth = image.imageWidth * ratio
    image.drawHeight = image.imageHeight * ratio
    return image


def _footer(canvas: Any, document: Any) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
    canvas.line(1.6 * cm, 1.25 * cm, A4[0] - 1.6 * cm, 1.25 * cm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(1.6 * cm, 0.8 * cm, "GeoAI Rabat - rapport court de vérification")
    canvas.drawRightString(A4[0] - 1.6 * cm, 0.8 * cm, f"Page {document.page}")
    canvas.restoreState()


def generate_real_report(
    config_path: str | Path = "configs/rabat_real_pilot.json",
) -> dict[str, Any]:
    config = load_real_config(config_path)
    verification = verify_real_pipeline(config_path)
    paths = {name: Path(value) for name, value in config["paths"].items() if name != "report_pdf"}
    manifest = _read_json(paths["processed"] / "real_dataset_manifest.json")
    audit = _read_json(paths["processed"] / "quality_audit_real.json")
    metadata = _read_json(paths["artifacts"] / "model_real_metadata.json")
    metrics = pd.read_csv(paths["artifacts"] / "metrics_real_validation.csv")
    rows_text = f"{manifest['rows']:,}".replace(",", " ")
    pixels_text = f"{manifest['unique_pixels']:,}".replace(",", " ")
    output = Path(config["paths"]["report_pdf"])
    output.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "TitleCustom",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=23,
            leading=28,
            textColor=NAVY,
            alignment=TA_CENTER,
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            "Subtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=11,
            leading=16,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceAfter=18,
        )
    )
    styles.add(
        ParagraphStyle(
            "Section",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=19,
            textColor=NAVY,
            spaceBefore=8,
            spaceAfter=9,
        )
    )
    styles.add(
        ParagraphStyle(
            "BodySmall",
            parent=styles["BodyText"],
            fontSize=9.5,
            leading=14,
            textColor=colors.HexColor("#1F2937"),
        )
    )
    styles.add(
        ParagraphStyle(
            "Caption",
            parent=styles["Normal"],
            fontSize=8.3,
            leading=11,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceBefore=4,
        )
    )
    styles.add(
        ParagraphStyle(
            "Callout",
            parent=styles["BodyText"],
            fontSize=10,
            leading=15,
            textColor=NAVY,
            borderColor=TEAL,
            borderWidth=1,
            borderPadding=9,
            backColor=colors.HexColor("#ECFDF5"),
            spaceBefore=8,
            spaceAfter=10,
        )
    )

    document = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        rightMargin=1.6 * cm,
        leftMargin=1.6 * cm,
        topMargin=1.45 * cm,
        bottomMargin=1.6 * cm,
        title="Rapport court GeoAI Rabat - semaines 1 à 5",
        author="Projet GeoAI Rabat",
    )
    story: list[Any] = []
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("GeoAI Rabat", styles["TitleCustom"]))
    story.append(
        Paragraph(
            "Vérification des semaines 1 à 5 sur données réelles<br/>"
            "Zone pilote - grille 30 m - Landsat, Sentinel-2, Copernicus DEM et météo",
            styles["Subtitle"],
        )
    )
    story.append(
        Paragraph(
            f"<b>Résultat :</b> pipeline réel exécuté avec succès le "
            f"{datetime.now().strftime('%d/%m/%Y')}. "
            f"La table contient <b>{rows_text} observations</b>, "
            f"<b>{manifest['dates']} dates Landsat</b> et "
            f"<b>{pixels_text} pixels uniques</b>.",
            styles["Callout"],
        )
    )
    story.append(Paragraph("État d’avancement vérifié", styles["Section"]))
    status_rows = [
        ["Semaine", "État", "Preuve obtenue"],
        ["1", "Réalisée", "Zone, limite OSM, EPSG:32629 et grille 30 m fixés"],
        ["2", "Réalisée - pilote", "14 Landsat, Sentinel-2, 2 tuiles DEM et météo réelles"],
        ["3", "Réalisée - pilote", "QA_PIXEL, masque, reprojection et alignement exécutés"],
        ["4", "Réalisée - pilote", f"Table réelle : {rows_text} lignes, OSM, ERA5-Land"],
        ["5", "Réalisée - pilote", "2 baselines, linéaire, Random Forest, boosting et XGBoost"],
    ]
    status_table = Table(status_rows, colWidths=[1.6 * cm, 3.1 * cm, 11.9 * cm], repeatRows=1)
    status_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("LEADING", (0, 0), (-1, -1), 11),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.append(status_table)
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("Sources réelles utilisées", styles["Section"]))
    source_text = (
        "<b>Landsat 8/9 Collection 2 Level-2 :</b> ST_B10 et QA_PIXEL via Microsoft "
        "Planetary Computer / USGS.<br/>"
        "<b>Sentinel-2 Level-2A :</b> B02, B03, B04, B08, B11 et SCL.<br/>"
        "<b>OpenStreetMap :</b> densités de bâtiments et de routes rasterisées.<br/>"
        "<b>Relief :</b> Copernicus DEM GLO-30, deux tuiles mosaïquées.<br/>"
        "<b>Météo :</b> ERA5-Land (température, humidité) complété par ERA5 "
        "(vent, rayonnement), distribué via Open-Meteo.<br/><br/>"
        "<font color='#475569'>Références : planetarycomputer.microsoft.com/docs ; "
        "usgs.gov/landsat-missions ; documentation.dataspace.copernicus.eu ; "
        "open-meteo.com/en/docs</font>"
    )
    story.append(Paragraph(source_text, styles["BodySmall"]))

    story.append(PageBreak())
    story.append(Paragraph("Captures des données et traitements", styles["Section"]))
    sentinel_image = _scaled_image(
        paths["figures"] / "sentinel2_rgb_zone_pilote.png", 5.2 * cm, 8.5 * cm
    )
    landsat_image = _scaled_image(
        paths["figures"] / "landsat_lst_reelle.png", 5.2 * cm, 8.5 * cm
    )
    osm_image = _scaled_image(
        paths["figures"] / "osm_densites_urbaines.png", 5.2 * cm, 8.5 * cm
    )
    image_table = Table(
        [
            [sentinel_image, landsat_image, osm_image],
            [
                Paragraph(
                    "Sentinel-2 L2A réel. Composition RGB sur la zone pilote après "
                    "masque SCL et alignement.",
                    styles["Caption"],
                ),
                Paragraph(
                    "Landsat LST réelle en °C après conversion ST_B10 et exclusion "
                    "des pixels QA non valides.",
                    styles["Caption"],
                ),
                Paragraph(
                    "Densités réelles de bâtiments et routes OpenStreetMap, "
                    "rasterisées à 10 m puis agrégées à 30 m.",
                    styles["Caption"],
                ),
            ],
        ],
        colWidths=[5.7 * cm, 5.7 * cm, 5.7 * cm],
    )
    image_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("BOX", (0, 0), (0, 0), 0.5, colors.HexColor("#CBD5E1")),
                ("BOX", (1, 0), (1, 0), 0.5, colors.HexColor("#CBD5E1")),
                ("BOX", (2, 0), (2, 0), 0.5, colors.HexColor("#CBD5E1")),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(image_table)
    story.append(Spacer(1, 0.35 * cm))
    story.append(
        Paragraph(
            f"Tous les rasters de contrôle partagent la même projection "
            f"<b>{config['crs']}</b>, la même résolution de <b>{config['resolution_m']} m</b> "
            f"et une grille de <b>{manifest['grid']['width']} × {manifest['grid']['height']}</b> pixels. "
            f"Le contrôle automatique confirme {verification['aligned_rasters']} rasters alignés.",
            styles["Callout"],
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("Dataset réel et entraînement de semaine 5", styles["Section"]))
    metric_rows = [["Modèle", "MAE (°C)", "RMSE (°C)", "R²"]]
    for row in metrics.itertuples(index=False):
        metric_rows.append(
            [row.model, f"{row.mae_c:.3f}", f"{row.rmse_c:.3f}", f"{row.r2:.3f}"]
        )
    metric_table = Table(metric_rows, colWidths=[7.0 * cm, 3.0 * cm, 3.0 * cm, 2.5 * cm])
    metric_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.append(metric_table)
    story.append(Spacer(1, 0.35 * cm))
    chart = _scaled_image(paths["figures"] / "metrics_modeles_reels.png", 16.5 * cm, 7.2 * cm)
    story.append(chart)
    story.append(
        Paragraph(
            f"Le meilleur modèle candidat est <b>{metadata['selected_model']}</b> "
            f"(MAE {metadata['selected_metrics']['mae_c']:.2f} °C), mais la "
            f"<b>{metadata['best_baseline_model']}</b> reste meilleure. "
            "Aucun candidat ne doit donc être promu avant l’analyse de semaine 6. "
            "Ce résultat est conservé honnêtement sans ouvrir les deux dates de test.",
            styles["Callout"],
        )
    )
    story.append(Paragraph("Limites et conclusion", styles["Section"]))
    limitations = (
        "Cette réalisation utilise de vraies observations mais reste une <b>zone pilote</b>, "
        "pas encore toute la commune. Les indices Sentinel-2 sont issus d’une scène statique "
        "2024. Les densités de bâtiments et routes proviennent d’OpenStreetMap ; la météo "
        "historique combine ERA5-Land et ERA5 avec provenance explicite. Les dates de test 2025 restent "
        "isolées, conformément au planning.<br/><br/>"
        f"<b>Contrôle qualité :</b> {audit['duplicate_pixel_date']} doublon pixel-date, "
        f"cible LST de {audit['target_summary_c']['min']:.1f} à "
        f"{audit['target_summary_c']['max']:.1f} °C, aucune cible manquante.<br/><br/>"
        "<b>Conclusion :</b> les tâches demandées pour les semaines 1 à 5 sont désormais "
        "exécutées sur données réelles à l’échelle pilote, avec provenance, rasters alignés, "
        "table Parquet, modèles entraînés et artefacts vérifiables."
    )
    story.append(Paragraph(limitations, styles["BodySmall"]))

    document.build(story, onFirstPage=_footer, onLaterPages=_footer)
    reader = PdfReader(output)
    if len(reader.pages) < 3:
        raise AssertionError("Le rapport court devrait contenir au moins trois pages.")
    return {
        "status": "ok",
        "path": output.as_posix(),
        "pages": len(reader.pages),
        "bytes": output.stat().st_size,
        "verification": verification,
    }
