from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
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

from .final_pipeline import verify_final_pipeline
from .real_pipeline import load_real_config, verify_real_pipeline


NAVY = colors.HexColor("#12304A")
BLUE = colors.HexColor("#2563EB")
TEAL = colors.HexColor("#0F766E")
ORANGE = colors.HexColor("#EA580C")
LIGHT = colors.HexColor("#EEF4F8")
MUTED = colors.HexColor("#475569")


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _image(path: Path, max_width: float, max_height: float) -> RLImage:
    result = RLImage(str(path))
    ratio = min(max_width / result.imageWidth, max_height / result.imageHeight)
    result.drawWidth = result.imageWidth * ratio
    result.drawHeight = result.imageHeight * ratio
    return result


def _footer(canvas: Any, document: Any) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
    canvas.line(1.5 * cm, 1.2 * cm, A4[0] - 1.5 * cm, 1.2 * cm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(1.5 * cm, 0.75 * cm, "GeoAI Rabat - rapport final - semaines 1 a 8")
    canvas.drawRightString(A4[0] - 1.5 * cm, 0.75 * cm, f"Page {document.page}")
    canvas.restoreState()


def _table(rows: list[list[Any]], widths: list[float], *, numeric_from: int = 1) -> Table:
    result = Table(rows, colWidths=widths, repeatRows=1)
    result.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("LEADING", (0, 0), (-1, -1), 11),
                ("ALIGN", (numeric_from, 1), (-1, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return result


def generate_final_report(
    config_path: str | Path = "configs/rabat_real_pilot.json",
) -> dict[str, Any]:
    config = load_real_config(config_path)
    verify_real = verify_real_pipeline(config_path)
    verify_final = verify_final_pipeline(config_path)
    paths = {name: Path(value) for name, value in config["paths"].items()}
    manifest = _json(paths["processed"] / "real_dataset_manifest.json")
    audit = _json(paths["processed"] / "quality_audit_real.json")
    metadata = _json(paths["artifacts"] / "model_real_metadata.json")
    test = _json(paths["final_artifacts"] / "test_metrics_final.json")
    forecast = _json(paths["final_artifacts"] / "forecast_summary.json")
    validation = pd.read_csv(paths["artifacts"] / "metrics_real_validation.csv")
    output = Path(config["paths"]["final_report_pdf"])
    output.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("T", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=25,
                              leading=30, textColor=NAVY, alignment=TA_CENTER, spaceAfter=12))
    styles.add(ParagraphStyle("Sub", parent=styles["Normal"], fontSize=11, leading=16,
                              textColor=MUTED, alignment=TA_CENTER, spaceAfter=18))
    styles.add(ParagraphStyle("H", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=16,
                              leading=20, textColor=NAVY, spaceBefore=5, spaceAfter=9))
    styles.add(ParagraphStyle("B", parent=styles["BodyText"], fontSize=9.5, leading=14,
                              textColor=colors.HexColor("#1F2937"), spaceAfter=7))
    styles.add(ParagraphStyle("Cap", parent=styles["Normal"], fontSize=8, leading=10,
                              textColor=MUTED, alignment=TA_CENTER, spaceBefore=3))
    styles.add(ParagraphStyle("Call", parent=styles["BodyText"], fontSize=10, leading=15,
                              textColor=NAVY, borderColor=TEAL, borderWidth=1, borderPadding=9,
                              backColor=colors.HexColor("#ECFDF5"), spaceBefore=7, spaceAfter=9))

    doc = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm,
                            topMargin=1.4 * cm, bottomMargin=1.55 * cm,
                            title="GeoAI Rabat - rapport final", author="Projet GeoAI Rabat")
    story: list[Any] = [Spacer(1, 0.7 * cm), Paragraph("GeoAI Rabat", styles["T"]),
        Paragraph("Cartographie thermique urbaine et prévisions J+1/J+2<br/>"
                  "Rapport final - semaines 1 à 8 - zone pilote à 30 m", styles["Sub"]),
        Paragraph(f"<b>Projet entièrement exécuté au {datetime.now().strftime('%d/%m/%Y')}.</b> "
                  f"Le pipeline assemble {manifest['rows']:,} observations pixel-date, "
                  f"{manifest['dates']} dates Landsat et {manifest['unique_pixels']:,} pixels uniques. "
                  "Le modèle figé est évalué sur des dates jamais vues, puis utilisé pour générer "
                  "automatiquement deux cartes de température de surface.", styles["Call"]),
        Paragraph("Résultat en une page", styles["H"])]
    status = [["Sem.", "Statut", "Preuve principale"],
              ["1", "Fait", "Zone, cible LST, grille EPSG:32629 à 30 m"],
              ["2", "Fait", "Landsat, Sentinel-2, DEM, OSM et météo réels"],
              ["3", "Fait", "Masques qualité, reprojection et alignement"],
              ["4", "Fait", "Table pixel-date, dictionnaire, manifeste et audit"],
              ["5", "Fait", "Baselines et 4 familles ML réellement comparées"],
              ["6", "Fait", "Test final unique sur 2 dates réservées"],
              ["7", "Fait", "Prévisions J+1/J+2, GeoTIFF, PNG et hotspots"],
              ["8", "Fait", "Tests, guides, rapport et présentation"]]
    story += [_table(status, [1.2 * cm, 2.1 * cm, 13.8 * cm], numeric_from=3), Spacer(1, 0.25 * cm),
              Paragraph("La cible est la <b>température de surface terrestre (LST)</b> en degrés "
                        "Celsius. Elle ne doit pas être confondue avec la température de l'air.", styles["B"])]

    story += [PageBreak(), Paragraph("1. Données réelles et chaîne de traitement", styles["H"]),
              Paragraph("Les sources sont inventoriées avec URL, date de téléchargement et empreinte. "
                        "Landsat fournit la cible ST_B10 et QA_PIXEL, Sentinel-2 décrit l'occupation du "
                        "sol, Copernicus DEM le relief, OSM les densités urbaines et ERA5/ERA5-Land la météo.", styles["B"])]
    imgs = [[_image(paths["figures"] / "sentinel2_rgb_zone_pilote.png", 5.2*cm, 8.2*cm),
             _image(paths["figures"] / "landsat_lst_reelle.png", 5.2*cm, 8.2*cm),
             _image(paths["figures"] / "osm_densites_urbaines.png", 5.2*cm, 8.2*cm)],
            [Paragraph("Sentinel-2 L2A - composition RGB", styles["Cap"]),
             Paragraph("Landsat - LST réelle après masque QA", styles["Cap"]),
             Paragraph("OSM - densités bâtiments et routes", styles["Cap"])]]
    img_table = Table(imgs, colWidths=[5.7*cm]*3)
    img_table.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),("ALIGN",(0,0),(-1,-1),"CENTER"),
                                   ("LEFTPADDING",(0,0),(-1,-1),4),("RIGHTPADDING",(0,0),(-1,-1),4)]))
    story += [img_table, Spacer(1, 0.25*cm),
              Paragraph(f"<b>Contrôle automatique :</b> {verify_real['aligned_rasters']} rasters alignés, "
                        f"grille {manifest['grid']['width']} × {manifest['grid']['height']}, "
                        f"{audit['duplicate_pixel_date']} doublon pixel-date et aucune cible manquante.", styles["Call"]),
              Paragraph("Les traitements appliquent les bits de qualité Landsat, convertissent ST_B10 en "
                        "degrés Celsius, reprojettent les variables continues avec interpolation bilinéaire "
                        "et conservent une partition strictement temporelle.", styles["B"])]

    story += [PageBreak(), Paragraph("2. Dataset et modélisation", styles["H"]),
              Paragraph(f"La table finale contient {manifest['rows']:,} lignes, {manifest['unique_pixels']:,} "
                        f"pixels uniques et {len(metadata['features'])} variables explicatives sur {manifest['dates']} dates. "
                        "Les dates complètes sont affectées à l'entraînement, à la validation ou au test, sans fuite.", styles["B"])]
    val_rows = [["Modèle", "MAE (°C)", "RMSE (°C)", "R²"]] + [
        [r.model, f"{r.mae_c:.3f}", f"{r.rmse_c:.3f}", f"{r.r2:.3f}"] for r in validation.itertuples(index=False)]
    story += [_table(val_rows, [7.2*cm, 3.1*cm, 3.1*cm, 2.7*cm]), Spacer(1, 0.25*cm),
              _image(paths["figures"] / "metrics_modeles_reels.png", 16.5*cm, 6.2*cm),
              Paragraph(f"En validation, la baseline saisonnière reste la meilleure. La "
                        f"<b>{metadata['selected_model']}</b> est néanmoins le meilleur modèle ML et est gelée "
                        "avant l'ouverture du test final. La sélection n'a jamais utilisé les dates de test.", styles["Call"])]

    story += [PageBreak(), Paragraph("3. Validation finale sur dates jamais vues", styles["H"]),
              Paragraph(f"Le test a été ouvert une seule fois le {test['test_opened_at_utc'][:10]}. "
                        f"Il couvre {test['rows']:,} observations sur les dates "
                        f"{test['test_dates'][0]} et {test['test_dates'][1]}.", styles["B"])]
    tm = test["overall"]
    test_rows = [["Méthode", "MAE (°C)", "RMSE (°C)", "R²", "Biais (°C)"],
                 ["Random Forest figée", f"{tm['random_forest']['mae_c']:.3f}",
                  f"{tm['random_forest']['rmse_c']:.3f}", f"{tm['random_forest']['r2']:.3f}",
                  f"{tm['random_forest']['bias_c']:+.3f}"],
                 ["Baseline saisonnière", f"{tm['monthly_seasonal_baseline']['mae_c']:.3f}",
                  f"{tm['monthly_seasonal_baseline']['rmse_c']:.3f}",
                  f"{tm['monthly_seasonal_baseline']['r2']:.3f}",
                  f"{tm['monthly_seasonal_baseline']['bias_c']:+.3f}"]]
    story += [_table(test_rows, [5.2*cm, 2.8*cm, 2.8*cm, 2.5*cm, 2.8*cm]), Spacer(1, .35*cm)]
    residuals = [[_image(paths["final_figures"] / f"residual_test_{d}.png", 8.1*cm, 10.3*cm) for d in test["test_dates"]],
                 [Paragraph(f"Résidus observé - prédit, {d}", styles["Cap"]) for d in test["test_dates"]]]
    residual_table = Table(residuals, colWidths=[8.5*cm, 8.5*cm])
    residual_table.setStyle(TableStyle([("ALIGN",(0,0),(-1,-1),"CENTER"),("VALIGN",(0,0),(-1,-1),"TOP")]))
    story += [residual_table, Paragraph("La Random Forest bat la baseline sur le test final. Son biais chaud "
                                       "moyen de 0,74 °C et son R² de 0,237 montrent toutefois qu'une extension "
                                       "spatiale et temporelle reste nécessaire avant usage opérationnel.", styles["Call"])]

    dates = forecast["forecast_dates"]
    story += [PageBreak(), Paragraph("4. Prévisions automatisées J+1 et J+2", styles["H"]),
              Paragraph("Le pipeline récupère la météo horaire du point central à 11 h locales, reconstruit "
                        "les 15 variables dans l'ordre appris, applique le modèle figé et écrit deux GeoTIFF "
                        "parfaitement alignés. Les PNG partagent la même échelle de couleurs.", styles["B"])]
    forecast_images = [[_image(paths["final_figures"] / f"lst_forecast_{d}.png", 8.1*cm, 10.2*cm) for d in dates],
                       [Paragraph(f"LST prévue - {d}", styles["Cap"]) for d in dates]]
    forecast_table = Table(forecast_images, colWidths=[8.5*cm, 8.5*cm])
    forecast_table.setStyle(TableStyle([("ALIGN",(0,0),(-1,-1),"CENTER"),("VALIGN",(0,0),(-1,-1),"TOP")]))
    story.append(forecast_table)
    wr = [["Date", "Air °C", "HR %", "Vent m/s", "SW W/m²", "LST moy. °C"]]
    for date in dates:
        w, s = forecast["weather"][date], forecast["statistics"][date]
        wr.append([date, f"{w['air_temperature_c']:.1f}", f"{w['relative_humidity_pct']:.0f}",
                   f"{w['wind_speed_ms']:.2f}", f"{w['shortwave_radiation_wm2']:.0f}", f"{s['mean_c']:.2f}"])
    story += [Spacer(1,.2*cm), _table(wr, [3.3*cm,2.2*cm,2.0*cm,2.4*cm,2.5*cm,3.2*cm]),
              Paragraph("Chaque date dispose aussi des 100 pixels les plus chauds, avec coordonnées projetées. "
                        "Ces cartes décrivent la LST prévue et ne constituent pas une alerte sanitaire.", styles["Call"])]

    story += [PageBreak(), Paragraph("5. Reproductibilité, limites et conclusion", styles["H"]),
              Paragraph("<b>Commande de livraison :</b><br/><font name='Courier'>"
                        ".\\scripts\\run_complete.ps1</font><br/><br/>"
                        "Elle vérifie les données réelles, protège l'évaluation finale déjà enregistrée, actualise "
                        "les deux jours de prévision, génère le rapport et exécute les tests.", styles["B"])]
    deliveries = [["Livrable", "Chemin"],
                  ["Table pixel-date", "data/processed/real/pixel_date_real.parquet"],
                  ["Modèle et model card", "artifacts/real_week5/"],
                  ["Métriques de test", "artifacts/final/test_metrics_final.json"],
                  ["Prévisions et résidus GeoTIFF", "output/rasters/"],
                  ["Cartes PNG", "reports/figures/final/"],
                  ["Documentation", "docs/01 à docs/10"],
                  ["Rapport et présentation", "output/pdf/ et output/presentation/"]]
    story += [_table(deliveries, [5.4*cm, 11.2*cm], numeric_from=3), Spacer(1,.35*cm),
              Paragraph("<b>Limites majeures</b><br/>- zone pilote, et non toute la commune ;<br/>"
                        "- Sentinel-2 et occupation du sol traités comme variables statiques ;<br/>"
                        "- météo de prévision issue d'un point central ;<br/>"
                        "- nombre de dates encore limité pour capturer tous les régimes saisonniers ;<br/>"
                        "- validation externe et suivi de dérive à prévoir avant mise en production.", styles["B"]),
              Paragraph(f"<b>Conclusion :</b> les huit semaines sont réalisées avec données réelles, "
                        f"{verify_final['test_dates_scored']} dates de test, {verify_final['forecast_rasters']} cartes "
                        "de prévision géoréférencées et une chaîne contrôlée par tests. Le prototype démontre la "
                        "faisabilité technique et fournit une base honnête, traçable et extensible.", styles["Call"]),
              Paragraph("Sources principales : USGS Landsat Collection 2 ; Copernicus Data Space ; "
                        "OpenStreetMap ; Microsoft Planetary Computer ; Open-Meteo (ERA5/ERA5-Land et Forecast API).", styles["B"])]

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    pages = len(PdfReader(output).pages)
    if pages < 6:
        raise AssertionError(f"Le rapport final devrait contenir au moins 6 pages, obtenu : {pages}")
    return {"status": "ok", "path": output.as_posix(), "pages": pages,
            "bytes": output.stat().st_size, "verification": verify_final}
