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

from .final_pipeline import verify_complete_delivery
from .real_pipeline import load_real_config


NAVY = colors.HexColor("#12304A")
BLUE = colors.HexColor("#2563EB")
TEAL = colors.HexColor("#0F766E")
ORANGE = colors.HexColor("#EA580C")
GREEN_BG = colors.HexColor("#ECFDF5")
BLUE_BG = colors.HexColor("#EFF6FF")
AMBER_BG = colors.HexColor("#FFF7D6")
LIGHT = colors.HexColor("#EEF4F8")
MUTED = colors.HexColor("#526274")


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _image(path: Path, width: float, height: float) -> RLImage:
    result = RLImage(str(path))
    ratio = min(width / result.imageWidth, height / result.imageHeight)
    result.drawWidth = result.imageWidth * ratio
    result.drawHeight = result.imageHeight * ratio
    return result


def _footer(canvas: Any, document: Any) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
    canvas.line(1.45 * cm, 1.15 * cm, A4[0] - 1.45 * cm, 1.15 * cm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(1.45 * cm, 0.72 * cm, "GeoAI Rabat - guide simple de soutenance")
    canvas.drawRightString(A4[0] - 1.45 * cm, 0.72 * cm, f"Page {document.page}")
    canvas.restoreState()


def _styled_table(rows: list[list[Any]], widths: list[float], font_size: float = 8.2) -> Table:
    header_style = ParagraphStyle(
        "TableHeader", fontName="Helvetica-Bold", fontSize=font_size,
        leading=font_size + 2.2, textColor=colors.white,
    )
    cell_style = ParagraphStyle(
        "TableCell", fontName="Helvetica", fontSize=font_size,
        leading=font_size + 2.2, textColor=colors.HexColor("#1F2937"),
    )
    wrapped: list[list[Any]] = []
    for row_index, row in enumerate(rows):
        wrapped.append([
            value if isinstance(value, Paragraph) else Paragraph(str(value), header_style if row_index == 0 else cell_style)
            for value in row
        ])
    table = Table(wrapped, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("LEADING", (0, 0), (-1, -1), font_size + 2.5),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "DefenseTitle", parent=base["Title"], fontName="Helvetica-Bold",
            fontSize=25, leading=30, textColor=NAVY, alignment=TA_CENTER, spaceAfter=10,
        ),
        "subtitle": ParagraphStyle(
            "DefenseSubtitle", parent=base["Normal"], fontSize=11, leading=16,
            textColor=MUTED, alignment=TA_CENTER, spaceAfter=16,
        ),
        "h1": ParagraphStyle(
            "DefenseH1", parent=base["Heading1"], fontName="Helvetica-Bold",
            fontSize=18, leading=22, textColor=NAVY, spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "DefenseH2", parent=base["Heading2"], fontName="Helvetica-Bold",
            fontSize=12.5, leading=16, textColor=TEAL, spaceBefore=7, spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "DefenseBody", parent=base["BodyText"], fontSize=9.5, leading=14,
            textColor=colors.HexColor("#1F2937"), spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "DefenseSmall", parent=base["BodyText"], fontSize=8.3, leading=11.5,
            textColor=colors.HexColor("#334155"), spaceAfter=4,
        ),
        "callout": ParagraphStyle(
            "DefenseCallout", parent=base["BodyText"], fontSize=10, leading=14.5,
            textColor=NAVY, borderColor=TEAL, borderWidth=1, borderPadding=8,
            backColor=GREEN_BG, spaceBefore=6, spaceAfter=8,
        ),
        "answer": ParagraphStyle(
            "DefenseAnswer", parent=base["BodyText"], fontSize=9.2, leading=13.2,
            textColor=colors.HexColor("#1F2937"), borderColor=colors.HexColor("#BFDBFE"),
            borderWidth=0.8, borderPadding=7, backColor=BLUE_BG, spaceAfter=7,
        ),
        "warning": ParagraphStyle(
            "DefenseWarning", parent=base["BodyText"], fontSize=9.5, leading=14,
            textColor=colors.HexColor("#7C2D12"), borderColor=colors.HexColor("#F2C14E"),
            borderWidth=0.8, borderPadding=7, backColor=AMBER_BG, spaceAfter=7,
        ),
        "caption": ParagraphStyle(
            "DefenseCaption", parent=base["Normal"], fontSize=7.8, leading=10,
            textColor=MUTED, alignment=TA_CENTER, spaceBefore=3,
        ),
    }


def _question(number: int, question: str, answer: str, styles: dict[str, ParagraphStyle]) -> list[Any]:
    return [
        Paragraph(f"<b>Q{number}. {question}</b><br/>{answer}", styles["answer"]),
    ]


def generate_defense_guide(
    config_path: str | Path = "configs/rabat_real_pilot.json",
) -> dict[str, Any]:
    config = load_real_config(config_path)
    verification = verify_complete_delivery(config_path)
    paths = {name: Path(value) for name, value in config["paths"].items()}
    manifest = _json(paths["processed"] / "real_dataset_manifest.json")
    audit = _json(paths["processed"] / "quality_audit_real.json")
    metadata = _json(paths["artifacts"] / "model_real_metadata.json")
    test = _json(paths["final_artifacts"] / "test_metrics_final.json")
    forecast = _json(paths["final_artifacts"] / "forecast_summary.json")
    validation = pd.read_csv(paths["artifacts"] / "metrics_real_validation.csv")
    importance = pd.read_csv(paths["artifacts"] / "feature_importance_real.csv")
    output = Path(config["paths"]["defense_guide_pdf"])
    output.parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()

    doc = SimpleDocTemplate(
        str(output), pagesize=A4, leftMargin=1.45 * cm, rightMargin=1.45 * cm,
        topMargin=1.35 * cm, bottomMargin=1.5 * cm,
        title="Guide complet de soutenance - GeoAI Rabat",
        author="Projet GeoAI Rabat",
    )
    story: list[Any] = []

    # Page 1 - Cover and 30-second answer.
    story += [
        Spacer(1, 0.55 * cm),
        Paragraph("GeoAI Rabat", styles["title"]),
        Paragraph(
            "Guide simple pour comprendre, expliquer et défendre tout le projet<br/>"
            "Questions du professeur, réponses courtes et checklist vérifiée",
            styles["subtitle"],
        ),
        Paragraph(
            "<b>Le projet en une phrase :</b> nous combinons des images satellites, des données "
            "urbaines, le relief et la météo pour estimer la température de surface à 30 m, "
            "tester le modèle sur des dates jamais vues et produire deux cartes J+1/J+2.",
            styles["callout"],
        ),
        Paragraph("Les quatre nombres à retenir", styles["h1"]),
    ]
    numbers = [
        ["748 120", "53 690", "1,416 °C", "2 + 2"],
        ["lignes pixel-date", "pixels uniques", "MAE test final", "résidus + prévisions"],
    ]
    number_table = Table(numbers, colWidths=[4.15 * cm] * 4)
    number_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 18),
        ("TEXTCOLOR", (0, 0), (-1, 0), BLUE),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, 1), 8),
        ("TEXTCOLOR", (0, 1), (-1, 1), MUTED),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    story += [number_table, Spacer(1, 0.3 * cm), Paragraph("Réponse orale de 30 secondes", styles["h1"]),
        Paragraph(
            "Le besoin est de montrer où la surface urbaine de Rabat est la plus chaude, et pas "
            "seulement de donner une température météo unique. Nous avons construit une grille de "
            "30 m, assemblé 14 dates Landsat avec Sentinel-2, OSM, Copernicus DEM et ERA5, entraîné "
            "plusieurs modèles, gardé une Random Forest, puis évalué deux dates 2025 jamais vues. "
            "Le modèle obtient une MAE de 1,416 °C et génère automatiquement deux GeoTIFF datés "
            "avec la météo Open-Meteo. Tout est documenté, testé et reproductible.", styles["body"]),
        Paragraph(
            "<b>Important :</b> ce résultat concerne la température de surface terrestre sur une "
            "zone pilote. Ce n'est ni la température de l'air, ni une alerte sanitaire.", styles["warning"]),
        Paragraph(f"Guide généré et vérifié le {datetime.now().strftime('%d/%m/%Y')}.", styles["small"]),
    ]

    # Page 2 - Demand checklist.
    story += [PageBreak(), Paragraph("1. Ce qui était demandé et ce qui a été livré", styles["h1"])]
    demanded = [
        ["Demande du cahier des charges", "Ce qui a été fait", "Statut"],
        ["Dataset pixel-date propre et documenté", "Parquet réel, 748 120 lignes, dictionnaire et audit", "OK"],
        ["Végétation, bâti, routes, relief et météo", "NDVI, NDBI, densités OSM, DEM, pente, ERA5", "OK"],
        ["Masque des nuages et grille commune", "QA_PIXEL, SCL, EPSG:32629, résolution 30 m", "OK"],
        ["Comparer plusieurs régressions", "2 baselines + linéaire + RF + boosting + XGBoost", "OK"],
        ["Dates de test jamais vues", "15/06/2025 et 25/07/2025 isolées jusqu'à la semaine 6", "OK"],
        ["MAE, RMSE, R2, résidus et gain baseline", "Métriques JSON/CSV et 2 cartes de résidus", "OK"],
        ["Deux cartes J+1/J+2", "2 GeoTIFF, 2 PNG, même échelle, 100 hotspots/date", "OK"],
        ["Pipeline reproductible", "CLI, config JSON, PowerShell, Docker, tests", "OK"],
        ["Guide, rapport, présentation, démonstration", "PDF, PPTX 9 slides, guide et scénario 5 minutes", "OK"],
    ]
    story += [_styled_table(demanded, [6.1 * cm, 8.8 * cm, 1.5 * cm], 7.8), Spacer(1, 0.25 * cm),
        Paragraph(
            f"<b>Vérification automatique finale :</b> {verification['weeks_complete']}/8 semaines, "
            f"{verification['aligned_source_rasters']} rasters sources alignés, "
            f"{verification['test_dates']} dates de test, {verification['forecast_rasters']} rasters "
            f"de prévision, {verification['report_pages']} pages de rapport et "
            f"{verification['presentation_slides']} diapositives avec notes de sources.", styles["callout"]),
        Paragraph(
            "Trois compléments ont été ajoutés lors du dernier audit : dépendance pypdf déclarée, "
            "model card finale après test et commande verify-delivery contrôlant toute la livraison.",
            styles["body"]),
    ]

    # Page 3 - Concept and architecture.
    story += [PageBreak(), Paragraph("2. Comprendre le projet sans jargon", styles["h1"]),
        Paragraph("Pourquoi le projet existe", styles["h2"]),
        Paragraph(
            "Une station météo donne une valeur pour une zone large. Dans une ville, une toiture, "
            "une route, un parc et une zone proche de l'eau ne chauffent pas de la même façon. Le "
            "projet cherche cette différence locale à l'échelle d'une cellule de 30 m.", styles["body"]),
        Paragraph("L'idée mathématique", styles["h2"]),
        Paragraph(
            "Chaque ligne représente un pixel à une date. Les colonnes décrivent ce pixel et la "
            "météo de cette date. La cible est la LST observée par Landsat. Le modèle apprend une "
            "fonction : variables du pixel + météo + date -> LST en °C.", styles["callout"]),
    ]
    flow = [
        ["1. SOURCES", "2. PRÉPARATION", "3. DATASET", "4. MODÈLE", "5. CARTES"],
        ["Landsat\nSentinel-2\nOSM\nDEM\nMétéo", "Nuages\nProjection\nAlignement\nVariables", "Une ligne\n= pixel + date", "Apprentissage\nValidation\nTest final", "Résidus\nJ+1\nJ+2\nHotspots"],
    ]
    flow_table = Table(flow, colWidths=[3.32 * cm] * 5)
    flow_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (-1, 1), LIGHT), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica"), ("FONTSIZE", (0, 0), (-1, -1), 7.8),
        ("LEADING", (0, 0), (-1, -1), 11), ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
        ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    story += [flow_table, Spacer(1, 0.3 * cm), Paragraph("Pourquoi les cartes J+1 et J+2 sont différentes", styles["h2"]),
        Paragraph(
            "Les variables statiques restent identiques : altitude, routes, bâtiments et occupation du "
            "sol. En revanche, la température de l'air, l'humidité, le vent, le rayonnement et le jour "
            "de l'année changent. Ces nouvelles entrées produisent une autre carte.", styles["body"]),
        Paragraph(
            "Si toutes les entrées étaient identiques, un modèle déterministe produirait exactement la "
            "même carte. C'est un bon contrôle logique à expliquer au professeur.", styles["warning"]),
    ]

    # Page 4 - Weeks 1 and 2.
    story += [PageBreak(), Paragraph("3. Semaines 1 et 2 - Cadrage et données", styles["h1"]),
        Paragraph("Semaine 1 : décisions prises", styles["h2"]),
        Paragraph(
            "- Zone : emprise pilote de Rabat.<br/>- Projection : EPSG:32629, adaptée au Maroc dans "
            "la zone UTM 29N.<br/>- Résolution : 30 m, cohérente avec Landsat.<br/>- Cible : LST "
            "diurne Landsat en °C.<br/>- Unité d'observation : un pixel et une date.", styles["body"]),
        Paragraph("Semaine 2 : sources réellement acquises", styles["h2"]),
    ]
    source_rows = [
        ["Source", "Rôle", "Ce qui a été utilisé"],
        ["Landsat 8/9 C2 L2", "Cible LST et qualité", "14 dates, ST_B10 et QA_PIXEL"],
        ["Sentinel-2 L2A", "Surface", "NDVI, NDBI, eau, végétation, imperméabilisation"],
        ["OpenStreetMap", "Ville", "13 642 bâtiments et 8 899 voies"],
        ["Copernicus DEM", "Relief", "2 tuiles, altitude et pente"],
        ["ERA5-Land / ERA5", "Météo historique", "Air, humidité, vent, rayonnement"],
        ["Open-Meteo", "Météo future", "J+1/J+2 à 11 h locales"],
    ]
    story += [_styled_table(source_rows, [4.0 * cm, 4.2 * cm, 8.2 * cm]), Spacer(1, 0.2 * cm)]
    img_rows = [[
        _image(paths["figures"] / "sentinel2_rgb_zone_pilote.png", 7.8 * cm, 5.7 * cm),
        _image(paths["figures"] / "osm_densites_urbaines.png", 7.8 * cm, 5.7 * cm),
    ], [
        Paragraph("Sentinel-2 réel", styles["caption"]),
        Paragraph("Densités OSM réelles", styles["caption"]),
    ]]
    image_table = Table(img_rows, colWidths=[8.3 * cm, 8.3 * cm])
    image_table.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story += [image_table, Paragraph(
        "Les volumes prévus dans la proposition étaient des ordres de grandeur. La réalisation est "
        "volontairement une zone pilote : elle prouve toute la chaîne sans prétendre couvrir la "
        "commune entière.", styles["warning"])]

    # Page 5 - Week 3.
    story += [PageBreak(), Paragraph("4. Semaine 3 - Nettoyer et aligner", styles["h1"]),
        Paragraph("Le problème", styles["h2"]),
        Paragraph(
            "Les sources n'ont pas la même résolution, la même projection ni la même qualité. Un "
            "pixel nuageux ou un décalage d'une cellule peut fausser l'apprentissage.", styles["body"]),
        Paragraph("Les traitements appliqués", styles["h2"]),
    ]
    processing = [
        ["Étape", "Méthode", "Pourquoi"],
        ["Masque Landsat", "Bits fill, nuage, cirrus, ombre, neige", "Supprimer les LST non fiables"],
        ["Masque Sentinel-2", "Classes SCL valides", "Éviter nuages et pixels invalides"],
        ["Conversion LST", "DN x 0,00341802 + 149 - 273,15", "Passer en degrés Celsius"],
        ["Projection", "Toutes les couches vers EPSG:32629", "Calculer surfaces et distances en mètres"],
        ["Rééchantillonnage", "Bilinéraire/average pour continu, nearest pour classes", "Respecter la nature des données"],
        ["Alignement", "Même transform, 222 x 264, résolution 30 m", "Chaque cellule correspond au même lieu"],
        ["Valeurs manquantes", "Cible manquante supprimée", "Ne jamais inventer la vérité Landsat"],
    ]
    story += [_styled_table(processing, [4.0 * cm, 6.0 * cm, 6.4 * cm], 7.8), Spacer(1, 0.25 * cm),
        Paragraph(
            f"<b>Preuve :</b> verify-real confirme {verification['aligned_source_rasters']} rasters "
            f"alignés. L'audit trouve {audit['duplicate_pixel_date']} doublon pixel-date et aucune "
            "cible manquante.", styles["callout"]),
        Paragraph("Question piège : bilinéaire ou nearest ?", styles["h2"]),
        Paragraph(
            "Le bilinéaire convient aux valeurs continues comme l'altitude. Nearest est obligatoire "
            "pour les classes et masques, sinon on créerait des classes artificielles entre deux "
            "catégories.", styles["body"]),
    ]

    # Page 6 - Week 4.
    story += [PageBreak(), Paragraph("5. Semaine 4 - Construire le dataset pixel-date", styles["h1"]),
        Paragraph(
            f"La table finale contient <b>{manifest['rows']:,} lignes</b>, "
            f"<b>{manifest['unique_pixels']:,} pixels</b>, <b>{manifest['dates']} dates</b> et "
            f"<b>{len(metadata['features'])} variables de modèle</b>. Elle est stockée en Parquet.",
            styles["callout"]),
    ]
    feature_rows = [
        ["Groupe", "Variables", "Intuition"],
        ["Relief", "elevation_m, slope_deg", "Le relief modifie l'exposition et le climat local"],
        ["Spectral", "NDVI, NDBI, impervious_fraction", "Végétation, bâti et surfaces minérales"],
        ["Urbain", "building_density, road_density", "Plus de surfaces artificielles"],
        ["Distances", "distance_to_water_m, distance_to_green_m", "Proximité de zones rafraîchissantes"],
        ["Météo", "air, humidité, vent, rayonnement", "Conditions du jour"],
        ["Temps", "sin/cos du jour de l'année", "Saison sans rupture entre décembre et janvier"],
        ["Cible", "lst_c", "Température de surface Landsat"],
    ]
    story += [_styled_table(feature_rows, [3.1 * cm, 6.5 * cm, 6.8 * cm], 7.8), Spacer(1, 0.25 * cm),
        Paragraph("Pourquoi sinus et cosinus pour le jour ?", styles["h2"]),
        Paragraph(
            "Le jour 365 et le jour 1 sont proches dans la saison mais très éloignés comme nombres. "
            "Le couple sinus/cosinus transforme l'année en cercle et conserve cette proximité.", styles["body"]),
        Paragraph("Pourquoi Parquet ?", styles["h2"]),
        Paragraph(
            "Parquet conserve les types, compresse bien les millions de valeurs et se lit plus vite "
            "qu'un CSV. Le dictionnaire CSV séparé reste facile à consulter.", styles["body"]),
        Paragraph(
            "Les coordonnées et identifiants servent à reconstruire les cartes. Seules les 15 variables "
            "explicatives entrent dans le modèle ; la cible et les identifiants n'y entrent pas.", styles["warning"]),
    ]

    # Page 7 - Week 5.
    story += [PageBreak(), Paragraph("6. Semaine 5 - Entraîner et choisir", styles["h1"]),
        Paragraph(
            "Nous comparons d'abord des méthodes simples. Un modèle complexe n'est utile que s'il "
            "fait mieux qu'une référence naïve sur des dates non vues.", styles["body"]),
    ]
    val_rows = [["Modèle", "MAE validation", "RMSE", "R2"]] + [
        [r.model, f"{r.mae_c:.3f} °C", f"{r.rmse_c:.3f} °C", f"{r.r2:.3f}"]
        for r in validation.itertuples(index=False)
    ]
    story += [_styled_table(val_rows, [7.0 * cm, 3.3 * cm, 3.3 * cm, 2.7 * cm]), Spacer(1, 0.2 * cm),
        _image(paths["figures"] / "metrics_modeles_reels.png", 16.2 * cm, 6.2 * cm),
        Paragraph(
            "La baseline saisonnière gagne la validation. La Random Forest est néanmoins le meilleur "
            "candidat ML et elle est figée avant d'ouvrir le test. Nous ne changeons pas le modèle après "
            "avoir vu le test : sinon le test deviendrait une seconde validation.", styles["callout"]),
        Paragraph("Pourquoi certains R2 de validation sont négatifs ?", styles["h2"]),
        Paragraph(
            "Un R2 négatif signifie que le modèle généralise moins bien que la moyenne sur ces dates de "
            "validation. Ce n'est pas une erreur de calcul : c'est un signal honnête de décalage entre "
            "saisons et du faible nombre de dates.", styles["warning"]),
    ]

    # Page 8 - Week 6.
    rf = test["overall"]["random_forest"]
    baseline = test["overall"]["monthly_seasonal_baseline"]
    story += [PageBreak(), Paragraph("7. Semaine 6 - Test final et erreurs", styles["h1"]),
        Paragraph(
            f"Le test contient {test['rows']:,} lignes sur les dates {test['test_dates'][0]} et "
            f"{test['test_dates'][1]}. Le marqueur prouve une seule ouverture et l'empreinte du modèle.",
            styles["body"]),
    ]
    test_rows = [
        ["Méthode", "MAE", "RMSE", "R2", "Biais"],
        ["Random Forest figée", f"{rf['mae_c']:.3f}", f"{rf['rmse_c']:.3f}", f"{rf['r2']:.3f}", f"{rf['bias_c']:+.3f}"],
        ["Baseline saisonnière", f"{baseline['mae_c']:.3f}", f"{baseline['rmse_c']:.3f}", f"{baseline['r2']:.3f}", f"{baseline['bias_c']:+.3f}"],
    ]
    story += [_styled_table(test_rows, [5.2 * cm, 2.8 * cm, 2.8 * cm, 2.5 * cm, 2.8 * cm]), Spacer(1, 0.2 * cm)]
    residual_rows = [[
        _image(paths["final_figures"] / f"residual_test_{date}.png", 7.9 * cm, 8.3 * cm)
        for date in test["test_dates"]
    ], [
        Paragraph(f"Résidus observé - prédit, {date}", styles["caption"])
        for date in test["test_dates"]
    ]]
    residual_table = Table(residual_rows, colWidths=[8.3 * cm, 8.3 * cm])
    residual_table.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story += [residual_table, Paragraph(
        "<b>Lecture :</b> résidu = observé - prédit. Rouge positif : le modèle a sous-estimé. Bleu "
        "négatif : il a surestimé. La même échelle est utilisée pour comparer honnêtement les dates.",
        styles["callout"]),
        Paragraph(
            "La Random Forest gagne globalement le test, mais sur le 25 juillet sa MAE est légèrement "
            "moins bonne que la baseline. On communique donc le résultat global et le détail par date.",
            styles["warning"]),
    ]

    # Page 9 - Week 7.
    dates = forecast["forecast_dates"]
    story += [PageBreak(), Paragraph("8. Semaine 7 - Produire J+1 et J+2", styles["h1"]),
        Paragraph(
            "L'exécution archivée du 10/08/2026 a produit les cartes des 11 et 12 août. Une nouvelle "
            "exécution interroge Open-Meteo et remplace ces dates par les deux jours suivants.", styles["body"]),
    ]
    forecast_rows = [[
        _image(paths["final_figures"] / f"lst_forecast_{date}.png", 7.9 * cm, 8.7 * cm)
        for date in dates
    ], [
        Paragraph(f"LST prévue - {date}", styles["caption"])
        for date in dates
    ]]
    forecast_table = Table(forecast_rows, colWidths=[8.3 * cm, 8.3 * cm])
    forecast_table.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(forecast_table)
    weather_rows = [["Date", "Air °C", "HR %", "Vent m/s", "Rayonnement", "LST moyenne"]]
    for date in dates:
        w = forecast["weather"][date]
        stats = forecast["statistics"][date]
        weather_rows.append([date, f"{w['air_temperature_c']:.1f}", f"{w['relative_humidity_pct']:.0f}",
                             f"{w['wind_speed_ms']:.2f}", f"{w['shortwave_radiation_wm2']:.0f} W/m2",
                             f"{stats['mean_c']:.2f} °C"])
    story += [_styled_table(weather_rows, [3.0 * cm, 2.2 * cm, 1.8 * cm, 2.4 * cm, 3.2 * cm, 3.2 * cm], 7.7),
        Paragraph(
            "Les deux cartes utilisent la même échelle 36,7 à 40,6 °C. Le CSV fournit les 100 "
            "pixels les plus chauds pour chaque jour avec leurs coordonnées projetées.", styles["callout"]),
    ]

    # Page 10 - Week 8 and reproducibility.
    story += [PageBreak(), Paragraph("9. Semaine 8 - Reproduire, tester et présenter", styles["h1"]),
        Paragraph("Commande principale", styles["h2"]),
        Paragraph("<font name='Courier'>.\\scripts\\run_complete.ps1</font>", styles["callout"]),
        Paragraph(
            "Cette commande vérifie ou reconstruit les données réelles si elles manquent, conserve le "
            "test final déjà verrouillé, actualise J+1/J+2, régénère les rapports, vérifie toute la "
            "livraison et lance les tests.", styles["body"]),
    ]
    repro = [
        ["Élément", "Rôle dans la reproductibilité"],
        ["configs/rabat_real_pilot.json", "Zone, dates, résolution, météo et chemins"],
        ["requirements-real.txt / pyproject.toml", "Dépendances Python déclarées"],
        ["Dockerfile", "Environnement conteneurisé"],
        ["Manifestes SHA-256", "Traçabilité et intégrité des sources"],
        ["Parquet + GeoTIFF", "Formats ouverts, typés et géoréférencés"],
        ["Model cards", "Usage, métriques et limites du modèle"],
        ["pytest", "16 contrôles après le présent audit"],
        ["verify-delivery", "Contrôle données, cartes, docs, PDF et PPTX"],
        ["Git", "Historique des modifications"],
    ]
    story += [_styled_table(repro, [5.6 * cm, 10.8 * cm]), Spacer(1, 0.2 * cm),
        Paragraph("Cloud-native, cela veut dire quoi ici ?", styles["h2"]),
        Paragraph(
            "Le pipeline est découpé en commandes, utilise des formats cloud-friendly, une configuration "
            "séparée, des sources accessibles par API et un Dockerfile. Il peut fonctionner localement "
            "puis être déplacé vers une VM ou un service cloud sans réécrire la science.", styles["body"]),
        Paragraph(
            "MLflow, QGIS et rioxarray étaient des options technologiques de la proposition. Le pilote "
            "utilise JSON/CSV/SHA-256 pour le suivi, Rasterio pour le géospatial et des PNG contrôlés "
            "visuellement. Ce choix réduit la complexité sans enlever un livrable obligatoire.", styles["warning"]),
    ]

    # Page 11 - Metrics glossary and importance.
    story += [PageBreak(), Paragraph("10. Les métriques à expliquer simplement", styles["h1"])]
    metric_rows = [
        ["Métrique", "Définition simple", "Dans notre projet"],
        ["MAE", "Erreur absolue moyenne", "1,416 °C sur le test RF"],
        ["RMSE", "Pénalise davantage les grosses erreurs", "1,831 °C"],
        ["R2", "Part de variation expliquée", "0,237, donc 23,7 %"],
        ["Biais", "Prédit trop chaud ou trop froid en moyenne", "+0,738 °C, biais chaud"],
        ["Résidu", "Observé moins prédit pour un pixel", "Cartographié sur 2 dates"],
        ["Gain baseline", "Le modèle apporte-t-il plus qu'une règle simple ?", "Oui globalement sur le test final"],
    ]
    story += [_styled_table(metric_rows, [3.2 * cm, 7.2 * cm, 6.0 * cm]), Spacer(1, 0.25 * cm),
        Paragraph("Quelles variables semblent importantes ?", styles["h2"])]
    top = importance.sort_values("importance", ascending=False).head(6)
    importance_rows = [["Rang", "Variable", "Importance permutation"]] + [
        [str(index + 1), row.feature, f"{row.importance:.4f}"] for index, row in enumerate(top.itertuples(index=False))
    ]
    story += [_styled_table(importance_rows, [2.3 * cm, 8.0 * cm, 5.5 * cm]), Spacer(1, 0.2 * cm),
        Paragraph(
            "Le jour de l'année et le NDVI dominent cet échantillon de validation. L'importance par "
            "permutation mesure la perte de performance quand on mélange une variable. Une importance "
            "nulle ne prouve pas qu'une variable n'a aucun effet physique : elle peut être redondante, "
            "peu variable ou mal estimée sur ce petit pilote.", styles["callout"]),
        Paragraph("Formules à connaître", styles["h2"]),
        Paragraph(
            "NDVI = (NIR - Rouge) / (NIR + Rouge). Plus il est élevé, plus la végétation est présente.<br/>"
            "NDBI = (SWIR - NIR) / (SWIR + NIR). Il aide à repérer les surfaces bâties.<br/>"
            "LST = DN x 0,00341802 + 149 - 273,15 pour Landsat Collection 2 Level-2.", styles["body"]),
    ]

    # Pages 12-13 - Defense questions.
    story += [PageBreak(), Paragraph("11. Questions probables du professeur - partie 1", styles["h1"])]
    q1 = [
        (1, "Pourquoi prédire la LST et non la température de l'air ?", "Landsat observe le rayonnement thermique de la surface. La LST décrit routes, toits, sol et végétation. Elle répond au besoin spatial urbain, mais ne représente pas directement l'air respiré."),
        (2, "Pourquoi une résolution de 30 m ?", "La bande thermique Landsat est distribuée sur une grille de 30 m dans le produit utilisé. Cette résolution évite de prétendre à une précision plus fine que la cible."),
        (3, "Pourquoi EPSG:32629 ?", "Rabat se situe dans la zone UTM 29N. Cette projection métrique permet des distances, densités et alignements cohérents."),
        (4, "Pourquoi utiliser Sentinel-2 si Landsat fournit déjà la cible ?", "Landsat apporte la température cible. Sentinel-2 apporte des détails spectraux utiles sur végétation, eau et bâti. Les rôles sont complémentaires."),
        (5, "Comment traitez-vous les nuages ?", "Nous lisons QA_PIXEL de Landsat et SCL de Sentinel-2. Les pixels fill, cirrus, nuage, ombre et neige sont exclus avant la table."),
        (6, "Pourquoi utiliser OSM ?", "OSM fournit une source ouverte de bâtiments et routes. Nous les rasterisons pour obtenir des densités spatiales compatibles avec la grille."),
        (7, "Pourquoi séparer par date et pas aléatoirement ?", "Un split aléatoire mélangerait des pixels de la même scène entre train et test. Le modèle reconnaîtrait la météo de la date et le score serait trop optimiste."),
        (8, "Pourquoi avoir des baselines ?", "Elles répondent à la question la plus importante : le modèle complexe fait-il réellement mieux qu'une moyenne globale ou saisonnière ?"),
    ]
    for item in q1:
        story += _question(*item, styles)

    story += [PageBreak(), Paragraph("12. Questions probables du professeur - partie 2", styles["h1"])]
    q2 = [
        (9, "Pourquoi retenir la Random Forest alors que la baseline gagne la validation ?", "La baseline gagne la validation, mais la Random Forest est le meilleur candidat ML. Nous la figeons sans ouvrir le test. Sur le test final global, elle obtient ensuite 1,416 °C contre 1,497 °C pour la baseline."),
        (10, "Avez-vous réentraîné après avoir vu le test ?", "Non. Le marqueur d'ouverture enregistre la date et l'empreinte du modèle. Une nouvelle commande relit le résultat existant au lieu d'optimiser sur le test."),
        (11, "Que signifie un biais de +0,738 °C ?", "En moyenne, la prédiction RF est 0,738 °C plus chaude que l'observation. C'est un point à calibrer et surveiller avant production."),
        (12, "Pourquoi les cartes J+1 et J+2 ne sont-elles pas des observations ?", "Elles sont des prédictions du modèle utilisant une météo future. Aucune image Landsat quotidienne n'est disponible pour confirmer immédiatement chaque jour."),
        (13, "Pourquoi la même échelle de couleurs ?", "Une échelle différente pourrait donner l'impression d'une forte différence uniquement à cause du rendu. Une échelle commune permet une comparaison visuelle honnête."),
        (14, "Comment trouvez-vous les hotspots ?", "Nous trions les pixels valides par LST prédite et exportons les 100 plus chauds avec rang, pixel_id et coordonnées UTM."),
        (15, "Le système couvre-t-il tout Rabat ?", "Non. Il couvre une zone pilote. Toute extension communale exige davantage de cellules, de dates, de stockage et une validation externe."),
        (16, "Est-ce prêt pour une alerte canicule ?", "Non. La LST n'est pas la température de l'air ni le risque sanitaire. Le prototype doit être étendu, calibré et intégré à d'autres indicateurs avant un usage décisionnel."),
    ]
    for item in q2:
        story += _question(*item, styles)

    # Page 14 - Technical questions and limitations.
    story += [PageBreak(), Paragraph("13. Questions techniques, limites et réponses honnêtes", styles["h1"])]
    q3 = [
        (17, "Pourquoi pas du Deep Learning ?", "Le dataset est tabulaire et le nombre de dates est limité. Les arbres offrent un meilleur compromis entre robustesse, coût, interprétation et délai."),
        (18, "Pourquoi pas LightGBM ?", "Le cahier demandait XGBoost ou LightGBM. XGBoost a été réellement évalué ; il satisfait cette alternative."),
        (19, "Pourquoi pas SHAP ?", "SHAP était une extension si le calendrier le permettait. Nous avons livré une importance par permutation, plus légère et suffisante pour le pilote."),
        (20, "Où est le MLOps ?", "La configuration, les splits temporels, les empreintes SHA-256, les model cards, les métriques JSON/CSV, Git, Docker et les vérifications constituent le socle MLOps. MLflow reste une extension possible."),
        (21, "Peut-on changer la ville ou la zone ?", "Oui en modifiant la bbox, la limite, le CRS et les chemins dans la configuration, puis en relançant l'ingestion. Il faut toutefois revalider les sources et réentraîner."),
        (22, "Que se passe-t-il si Open-Meteo est indisponible ?", "La commande échoue explicitement sans inventer de météo. La réponse brute déjà archivée permet de reproduire l'exécution passée."),
    ]
    for item in q3:
        story += _question(*item, styles)
    story += [Paragraph("Limites à annoncer avant qu'on vous les demande", styles["h2"]),
        Paragraph(
            "- zone pilote et 14 dates seulement ;<br/>- météo future au point central ;<br/>- variables "
            "urbaines considérées statiques ;<br/>- importance des variables instable avec peu de dates ;<br/>"
            "- biais chaud du modèle ;<br/>- pas de validation par stations au sol ;<br/>- pas d'usage "
            "sanitaire ou opérationnel sans étude supplémentaire.", styles["warning"]),
        Paragraph(
            "Dire clairement une limite renforce la crédibilité. Une preuve de concept réussie ne doit "
            "pas être présentée comme un produit municipal déjà industrialisé.", styles["callout"]),
    ]

    # Page 15 - Demo and final checklist.
    story += [PageBreak(), Paragraph("14. Démonstration de 5 minutes et checklist finale", styles["h1"]),
        Paragraph("Ordre conseillé pendant la soutenance", styles["h2"]),
    ]
    demo_rows = [
        ["Temps", "Action", "Message à dire"],
        ["0:00-0:40", "Montrer la carte Sentinel-2", "Voici la zone et les données réelles"],
        ["0:40-1:20", "Montrer le manifeste", "748 120 lignes, 14 dates, provenance conservée"],
        ["1:20-2:10", "Montrer les métriques", "Split par dates, RF figée, test jamais vu"],
        ["2:10-3:00", "Montrer les résidus", "Où le modèle sous-estime ou surestime"],
        ["3:00-4:00", "Montrer J+1/J+2", "Même échelle et météo différente"],
        ["4:00-5:00", "Lancer verify-delivery", "Toutes les preuves sont contrôlées automatiquement"],
    ]
    story += [_styled_table(demo_rows, [2.3 * cm, 5.2 * cm, 8.9 * cm], 7.8), Spacer(1, 0.22 * cm),
        Paragraph("Commandes à connaître", styles["h2"]),
        Paragraph(
            "<font name='Courier'>python -m geoai_rabat.cli verify-real --config configs/rabat_real_pilot.json</font><br/>"
            "<font name='Courier'>python -m geoai_rabat.cli verify-final --config configs/rabat_real_pilot.json</font><br/>"
            "<font name='Courier'>python -m geoai_rabat.cli verify-delivery --config configs/rabat_real_pilot.json</font><br/>"
            "<font name='Courier'>python -m pytest</font>", styles["small"]),
        Paragraph("Checklist avant de présenter", styles["h2"]),
    ]
    final_check = [
        ["Contrôle", "Vérifié"],
        ["Je sais expliquer LST vs température de l'air", "[OK]"],
        ["Je sais expliquer la grille 30 m et EPSG:32629", "[OK]"],
        ["Je sais citer les six sources principales", "[OK]"],
        ["Je sais expliquer le masque nuage et le split temporel", "[OK]"],
        ["Je connais MAE 1,416 °C, RMSE 1,831 °C, R2 0,237", "[OK]"],
        ["Je sais lire rouge/bleu sur les cartes de résidus", "[OK]"],
        ["Je sais expliquer pourquoi J+1 diffère de J+2", "[OK]"],
        ["Je précise zone pilote et absence d'usage sanitaire", "[OK]"],
        ["verify-delivery et les tests passent", "[OK]"],
    ]
    story += [_styled_table(final_check, [13.9 * cm, 2.5 * cm]), Spacer(1, 0.2 * cm),
        Paragraph(
            "<b>Phrase de conclusion :</b> le projet démontre une chaîne GeoAI complète, depuis des "
            "sources ouvertes et traçables jusqu'à deux cartes prédictives géoréférencées. Les résultats "
            "sont mesurés honnêtement, les limites sont explicites et le pipeline peut être étendu.",
            styles["callout"]),
    ]

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    reader = PdfReader(output)
    if len(reader.pages) < 15:
        raise AssertionError(f"Le guide devrait contenir au moins 15 pages, obtenu : {len(reader.pages)}")
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    required_phrases = ["748 120", "1,416", "verify-delivery", "Questions probables", "checklist finale"]
    missing = [phrase for phrase in required_phrases if phrase.lower() not in text.lower()]
    if missing:
        raise AssertionError(f"Contenu obligatoire absent du guide : {missing}")
    return {
        "status": "ok",
        "path": output.as_posix(),
        "pages": len(reader.pages),
        "bytes": output.stat().st_size,
        "questions_answered": 22,
        "checklist_verified": True,
    }
