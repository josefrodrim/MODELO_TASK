"""
Genera la presentacion PDF del caso practico Risk Data Scientist.
Uso: python3 scripts/build_presentation.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, white, black
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, NextPageTemplate,
    Paragraph, Spacer, Table, TableStyle, HRFlowable,
    PageBreak, Image, KeepTogether
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ---------------------------------------------------------------------------
# Fuentes con soporte Unicode completo
# ---------------------------------------------------------------------------

FONT_DIR = "/System/Library/Fonts/Supplemental"
pdfmetrics.registerFont(TTFont("Arial",        f"{FONT_DIR}/Arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold",   f"{FONT_DIR}/Arial Bold.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Italic", f"{FONT_DIR}/Arial Italic.ttf"))
pdfmetrics.registerFont(TTFont("Arial-BoldIt", f"{FONT_DIR}/Arial Bold Italic.ttf"))
from reportlab.pdfbase.pdfmetrics import registerFontFamily
registerFontFamily("Arial",
    normal="Arial", bold="Arial-Bold",
    italic="Arial-Italic", boldItalic="Arial-BoldIt")

# ---------------------------------------------------------------------------
# Paleta — fondo blanco, acentos rojo/marino
# ---------------------------------------------------------------------------

RED   = HexColor("#C8102E")
NAVY  = HexColor("#002D62")
BLUE  = HexColor("#0057A8")
GRAY  = HexColor("#5A5A5A")
LGRAY = HexColor("#F5F6F8")
MGRAY = HexColor("#D4D7DC")
DGRAY = HexColor("#333333")
WHITE = white

W, H  = A4
FIGURES = "reports/figures"
OUTPUT  = "reports/Presentacion_Caso_Practico_Risk_DS.pdf"

# ---------------------------------------------------------------------------
# Estilos
# ---------------------------------------------------------------------------

def mk(name, **kw):
    defaults = dict(fontName="Arial", fontSize=10, textColor=DGRAY,
                    leading=15, spaceAfter=4)
    defaults.update(kw)
    return ParagraphStyle(name, **defaults)

ST = {
    "title":   mk("title",  fontName="Arial-Bold", fontSize=26, textColor=NAVY,
                  leading=30, spaceAfter=6),
    "h1":      mk("h1",     fontName="Arial-Bold", fontSize=16, textColor=NAVY,
                  leading=20, spaceAfter=8, spaceBefore=4),
    "h2":      mk("h2",     fontName="Arial-Bold", fontSize=12, textColor=NAVY,
                  leading=15, spaceAfter=5, spaceBefore=6),
    "h3":      mk("h3",     fontName="Arial-Bold", fontSize=10, textColor=RED,
                  leading=13, spaceAfter=3),
    "body":    mk("body",   alignment=TA_JUSTIFY, leading=15),
    "bullet":  mk("bullet", leftIndent=12, firstLineIndent=-8, spaceAfter=3),
    "note":    mk("note",   fontName="Arial-Italic", fontSize=8.5, textColor=GRAY,
                  leading=11),
    "caption": mk("caption", fontName="Arial-Italic", fontSize=8, textColor=GRAY,
                  leading=10, alignment=TA_CENTER, spaceBefore=2, spaceAfter=6),
    "code":    mk("code",   fontName="Courier", fontSize=8.5, textColor=NAVY,
                  leftIndent=12, leading=12, backColor=LGRAY),
    "th":      mk("th",     fontName="Arial-Bold", fontSize=9, textColor=WHITE),
    "td":      mk("td",     fontSize=9, leading=12),
    "td_b":    mk("td_b",  fontName="Arial-Bold", fontSize=9, leading=12),
    "center":  mk("center", alignment=TA_CENTER),
    "cover_co":mk("cover_co", fontName="Arial", fontSize=9.5, textColor=GRAY,
                   leading=14),
}

def p(text, style="body", **kw):
    s = ST.get(style, ST["body"])
    if kw:
        s = ParagraphStyle(style + "_x", parent=s, **kw)
    return Paragraph(text, s)

def b(text):
    return Paragraph(f"•  {text}", ST["bullet"])

def hr(color=MGRAY, thick=0.5, before=2, after=8):
    return HRFlowable(width="100%", thickness=thick, color=color,
                      spaceBefore=before, spaceAfter=after)

def sp(h=0.3):
    return Spacer(1, h * cm)

# ---------------------------------------------------------------------------
# Dimensiones reales de cada figura (px) para calculo de altura exacta
# ---------------------------------------------------------------------------

IMG_DIMS = {
    "01_bivariate_analysis.png":          (2230, 1328),
    "01_correlation_matrix.png":          (1795, 1594),
    "01_missing_pattern.png":             (846,  753),
    "01_missing_values.png":              (1477, 733),
    "01_numeric_distributions.png":       (2382, 1612),
    "01_outlier_boxplots.png":            (2685, 889),
    "01_target_by_district.png":          (1785, 883),
    "01_target_distribution.png":         (2384, 581),
    "01_target_train_test.png":           (1484, 581),
    "01_x9_distribution.png":             (2385, 882),
    "02_outlier_capping.png":             (2085, 593),
    "02_preprocessing_comparison.png":    (2385, 1181),
    "02_sentinel_target_distribution.png":(2084, 740),
    "03_glm_assumptions.png":             (2084, 1520),
    "03_glm_coefficients.png":            (1485, 1539),
    "03_glm_decile_analysis.png":         (1484, 731),
    "03_glm_predictions.png":             (2084, 727),
    "03_vif_analysis.png":                (1184, 881),
    "04_feature_importance.png":          (1485, 1183),
    "04_lgbm_decile_analysis.png":        (1484, 731),
    "04_shap_importance.png":             (1185, 1108),
    "04_shap_summary.png":                (1276, 1409),
    "05_model_comparison.png":            (2385, 1480),
    "05_prediction_distributions.png":    (2084, 731),
    "05_psi_features.png":                (1484, 731),
    "05_psi_monitoring.png":              (1484, 731),
}

def _img(name, width_cm):
    path = f"{FIGURES}/{name}"
    w_pt = width_cm * cm
    if name in IMG_DIMS:
        iw, ih = IMG_DIMS[name]
        h_pt = w_pt * ih / iw
    else:
        h_pt = None
    return Image(path, width=w_pt, height=h_pt)

# ---------------------------------------------------------------------------
# Figuras de los notebooks
# ---------------------------------------------------------------------------

def fig(name, width_cm=14, caption=None):
    path = f"{FIGURES}/{name}"
    if not os.path.exists(path):
        return []
    items = [_img(name, width_cm)]
    if caption:
        items.append(p(caption, "caption"))
    return items

def fig_half(name_left, name_right, w=7.4, cap_l=None, cap_r=None):
    """Dos figuras lado a lado con alturas proporcionales."""
    row_imgs = []
    row_caps = []
    for name, cap in [(name_left, cap_l), (name_right, cap_r)]:
        path = f"{FIGURES}/{name}"
        if os.path.exists(path):
            row_imgs.append(_img(name, w))
            row_caps.append(p(cap or "", "caption"))
        else:
            row_imgs.append(Spacer(w * cm, 0.1 * cm))
            row_caps.append(p("", "caption"))
    t = Table([row_imgs, row_caps], colWidths=[w * cm, w * cm])
    t.setStyle(TableStyle([
        ("VALIGN",       (0,0), (-1,-1), "TOP"),
        ("ALIGN",        (0,0), (-1,-1), "CENTER"),
        ("LEFTPADDING",  (0,0), (-1,-1), 4),
        ("RIGHTPADDING", (0,0), (-1,-1), 4),
    ]))
    return [t]

# ---------------------------------------------------------------------------
# Tablas
# ---------------------------------------------------------------------------

TABLE_BASE = TableStyle([
    ("BACKGROUND",    (0, 0), (-1, 0), NAVY),
    ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
    ("FONTNAME",      (0, 0), (-1, 0), "Arial-Bold"),
    ("FONTSIZE",      (0, 0), (-1,-1), 9),
    ("ROWBACKGROUNDS",(0, 1), (-1,-1), [WHITE, LGRAY]),
    ("GRID",          (0, 0), (-1,-1), 0.3, MGRAY),
    ("VALIGN",        (0, 0), (-1,-1), "MIDDLE"),
    ("TOPPADDING",    (0, 0), (-1,-1), 5),
    ("BOTTOMPADDING", (0, 0), (-1,-1), 5),
    ("LEFTPADDING",   (0, 0), (-1,-1), 7),
    ("RIGHTPADDING",  (0, 0), (-1,-1), 7),
])

def tbl(data, col_widths, extra=None):
    t = Table(data, colWidths=col_widths)
    style = TableStyle(TABLE_BASE.getCommands())
    if extra:
        for cmd in extra:
            style.add(*cmd)
    t.setStyle(style)
    return t

# ---------------------------------------------------------------------------
# Header / footer
# ---------------------------------------------------------------------------

PAGE_N  = [0]
SECTION = [""]

def on_page(cv, doc):
    PAGE_N[0] += 1
    cv.saveState()

    # Linea roja superior
    cv.setFillColor(RED)
    cv.rect(0, H - 0.15*cm, W, 0.15*cm, fill=1, stroke=0)
    # Barra blanca de header
    cv.setFillColor(WHITE)
    cv.rect(0, H - 0.9*cm, W, 0.75*cm, fill=1, stroke=0)
    cv.setStrokeColor(MGRAY)
    cv.setLineWidth(0.4)
    cv.line(0, H - 0.9*cm, W, H - 0.9*cm)

    cv.setFillColor(NAVY)
    cv.setFont("Arial-Bold", 8)
    cv.drawString(1.8*cm, H - 0.62*cm, "Caso Practico Risk Data Scientist  |  Scotiabank Peru")
    cv.setFont("Arial", 8)
    cv.setFillColor(GRAY)
    cv.drawRightString(W - 1.8*cm, H - 0.62*cm, SECTION[0])

    # Footer
    cv.setStrokeColor(MGRAY)
    cv.setLineWidth(0.4)
    cv.line(1.8*cm, 0.85*cm, W - 1.8*cm, 0.85*cm)
    cv.setFont("Arial", 7.5)
    cv.setFillColor(GRAY)
    cv.drawString(1.8*cm, 0.42*cm, "Confidencial — uso interno")
    cv.setFont("Arial-Bold", 8.5)
    cv.setFillColor(NAVY)
    cv.drawCentredString(W / 2, 0.42*cm, str(PAGE_N[0]))

    cv.restoreState()

def on_cover(cv, doc):
    cv.saveState()

    cv.setFillColor(WHITE)
    cv.rect(0, 0, W, H, fill=1, stroke=0)

    # Barra roja superior
    cv.setFillColor(RED)
    cv.rect(0, H - 2*cm, W, 2*cm, fill=1, stroke=0)

    cv.setFont("Arial-Bold", 11)
    cv.setFillColor(WHITE)
    cv.drawString(2*cm, H - 1.3*cm, "SCOTIABANK PERU")
    cv.setFont("Arial", 9)
    cv.setFillColor(HexColor("#FFCDD2"))
    cv.drawString(2*cm, H - 1.72*cm, "Riesgo Crediticio — Evaluacion Data Scientist")

    # Linea decorativa vertical izquierda
    cv.setStrokeColor(RED)
    cv.setLineWidth(3)
    cv.line(2*cm, H * 0.30, 2*cm, H * 0.72)

    # Titulo
    cv.setFont("Arial-Bold", 32)
    cv.setFillColor(NAVY)
    cv.drawString(2.5*cm, H * 0.64, "Caso Practico")
    cv.setFont("Arial-Bold", 20)
    cv.setFillColor(RED)
    cv.drawString(2.5*cm, H * 0.58, "Risk Data Scientist")

    cv.setFont("Arial", 13)
    cv.setFillColor(DGRAY)
    cv.drawString(2.5*cm, H * 0.51, "Modelos Predictivos de Score de Riesgo Crediticio")
    cv.setFont("Arial-Italic", 11)
    cv.setFillColor(GRAY)
    cv.drawString(2.5*cm, H * 0.47, "GLM (OLS Log-Normal)  +  LightGBM (Gradient Boosting Tweedie)")

    cv.setStrokeColor(MGRAY)
    cv.setLineWidth(0.8)
    cv.line(2.5*cm, H * 0.43, W - 2*cm, H * 0.43)

    # Metricas en cajas ligeras
    metrics = [
        ("R2 Test",  "0.368", "LightGBM"),
        ("Gini",     "0.587", "2xAUC-1"),
        ("KS",       "0.444", "Kolmogorov-Smirnov"),
        ("RMSE",     "S/. 4,675", "Error prediccion"),
    ]
    box_w = (W - 4*cm) / 4
    y0 = H * 0.30
    for i, (label, val, sub) in enumerate(metrics):
        x = 2*cm + i * box_w
        cv.setStrokeColor(MGRAY)
        cv.setFillColor(LGRAY)
        cv.setLineWidth(0.6)
        cv.roundRect(x, y0, box_w - 0.25*cm, 2.5*cm, 4, fill=1, stroke=1)
        cv.setFont("Arial-Bold", 17)
        cv.setFillColor(RED)
        cv.drawCentredString(x + (box_w - 0.25*cm) / 2, y0 + 1.4*cm, val)
        cv.setFont("Arial-Bold", 8.5)
        cv.setFillColor(NAVY)
        cv.drawCentredString(x + (box_w - 0.25*cm) / 2, y0 + 0.85*cm, label)
        cv.setFont("Arial", 7)
        cv.setFillColor(GRAY)
        cv.drawCentredString(x + (box_w - 0.25*cm) / 2, y0 + 0.3*cm, sub)

    # Candidato
    cv.setFont("Arial-Bold", 13)
    cv.setFillColor(NAVY)
    cv.drawString(2.5*cm, H * 0.22, "Josef Rodriguez Mimbela")
    cv.setFont("Arial", 10)
    cv.setFillColor(GRAY)
    cv.drawString(2.5*cm, H * 0.18, "josef.rodrim@gmail.com    |    Junio 2026")

    cv.setStrokeColor(MGRAY)
    cv.setLineWidth(0.6)
    cv.line(2*cm, H * 0.13, W - 2*cm, H * 0.13)

    # Links
    cv.setFont("Arial", 8.5)
    cv.setFillColor(GRAY)
    cv.drawString(2.5*cm, H * 0.108, "Interfaz web:")
    cv.setFont("Arial-Bold", 8.5)
    cv.setFillColor(BLUE)
    cv.drawString(2.5*cm + 3.4*cm, H * 0.108, "modelo-task.vercel.app")

    cv.setFont("Arial", 8.5)
    cv.setFillColor(GRAY)
    cv.drawString(2.5*cm, H * 0.076, "API REST:")
    cv.setFont("Arial-Bold", 8.5)
    cv.setFillColor(BLUE)
    cv.drawString(2.5*cm + 3.4*cm, H * 0.076, "modelo-task.vercel.app/docs")

    cv.setFont("Arial", 8.5)
    cv.setFillColor(GRAY)
    cv.drawString(2.5*cm, H * 0.044, "Codigo fuente:")
    cv.setFont("Arial-Bold", 8.5)
    cv.setFillColor(BLUE)
    cv.drawString(2.5*cm + 3.4*cm, H * 0.044, "github.com/josefrodrim/MODELO_TASK")

    cv.setFont("Arial-Italic", 7.5)
    cv.setFillColor(HexColor("#AAAAAA"))
    cv.drawCentredString(W / 2, 0.5*cm, "Confidencial — elaborado para proceso de seleccion interno")

    cv.restoreState()

# ---------------------------------------------------------------------------
# Construccion del documento
# ---------------------------------------------------------------------------

def build():
    os.makedirs("reports", exist_ok=True)
    PAGE_N[0] = 0

    doc = BaseDocTemplate(
        OUTPUT, pagesize=A4,
        leftMargin=1.8*cm, rightMargin=1.8*cm,
        topMargin=1.3*cm, bottomMargin=1.3*cm,
    )

    cover_frame = Frame(0, 0, W, H, id="cover",
                        leftPadding=0, rightPadding=0,
                        topPadding=0, bottomPadding=0)
    main_frame = Frame(
        doc.leftMargin, doc.bottomMargin + 0.5*cm,
        W - doc.leftMargin - doc.rightMargin,
        H - doc.topMargin - doc.bottomMargin - 1.4*cm,
        id="main", showBoundary=0,
    )

    doc.addPageTemplates([
        PageTemplate(id="cover", frames=[cover_frame], onPage=on_cover),
        PageTemplate(id="main",  frames=[main_frame],  onPage=on_page),
    ])

    S = []
    A = S.append

    # ============================================================
    # PORTADA
    # ============================================================
    A(Spacer(W, H - 1*cm))
    A(NextPageTemplate("main"))
    A(PageBreak())

    # ============================================================
    # AGENDA — siguiendo los puntos a-h del enunciado
    # ============================================================
    SECTION[0] = "Contenido"
    A(p("Contenido", "h1"))
    A(hr(RED, thick=1.2, before=0, after=10))
    agenda = [
        ("-",  "Dataset",                   "50,000 observaciones, 12 variables predictoras, TARGET continuo"),
        ("a.", "Modelos: GLM + LightGBM",   "OLS Log-Normal (metodologia tradicional) + Gradient Boosting (ML)"),
        ("b.", "Supuestos de cada modelo",  "Evaluacion de supuestos estadisticos e impacto en resultados"),
        ("c.", "Preprocesamiento",          "Missing, outliers, transformaciones, imputacion y feature engineering"),
        ("d.", "Metricas y Comparacion",    "AUC, KS, Gini, R2, RMSE — comparacion en conjunto de test"),
        ("e.", "Interpretacion de Variables","Coherencia con el negocio/riesgo, SHAP values"),
        ("f.", "Seleccion del Modelo",      "Cual implementaria, trade-offs performance/interpretabilidad y riesgos"),
        ("g.", "Estabilidad y Monitoreo",   "PSI, cambios en distribucion poblacional, estrategia de alertas"),
        ("h.", "Modelo en Produccion",      "Entrenamiento, scoring, monitoreo — arquitectura y API desplegada"),
    ]
    for num, titulo, desc in agenda:
        row = [[
            p(f"<b>{num}</b>", textColor=RED, alignment=TA_CENTER),
            p(f"<b>{titulo}</b>"),
            p(desc, textColor=GRAY),
        ]]
        t = Table(row, colWidths=[0.7*cm, 5.5*cm, 8.8*cm])
        t.setStyle(TableStyle([
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING",    (0,0), (-1,-1), 7),
            ("BOTTOMPADDING", (0,0), (-1,-1), 7),
            ("LINEBELOW",     (0,0), (-1,-1), 0.3, MGRAY),
        ]))
        A(t)
    A(PageBreak())

    # ============================================================
    # DATASET (contexto, sin letra)
    # ============================================================
    SECTION[0] = "Dataset"
    A(p("Dataset", "h1"))
    A(hr(RED, thick=1.2, before=0, after=10))
    A(p(
        "El dataset contiene <b>50,000 observaciones</b> con 12 variables predictoras (X1-X12) "
        "y una variable objetivo continua (TARGET) que representa un score de credito "
        "en el rango <b>S/. 1,000-40,000</b> (media S/. 8,425). "
        "La distribucion es claramente asimetrica positiva (right-skewed), "
        "lo que justifica la transformacion logaritmica en el GLM "
        "y el objetivo Tweedie en LightGBM. "
        "Particion predefinida: <b>33,334 train</b> (BASE = TRAIN) y <b>16,667 test</b> (BASE = TEST)."
    ))
    A(sp(0.1))
    for item in fig_half("01_target_distribution.png", "01_target_train_test.png",
                          w=7.4,
                          cap_l="Distribucion del TARGET en train. Cola positiva tipica de scores crediticios.",
                          cap_r="Distribucion train vs. test. Alta similitud poblacional."):
        A(item)
    A(sp(0.1))
    for item in fig("01_missing_values.png", width_cm=13,
                     caption="Tasa de valores faltantes por variable. X3 y X4 contienen ademas centinelas (-9999998)."):
        A(item)
    A(sp(0.1))
    for item in fig("01_numeric_distributions.png", width_cm=15,
                     caption="Distribuciones de X1-X12. Se aprecian diferente escala y asimetria entre variables."):
        A(item)
    A(PageBreak())

    # ============================================================
    # a. MODELOS
    # ============================================================
    SECTION[0] = "a. Modelos"
    A(p("a. Modelos: GLM y LightGBM", "h1"))
    A(hr(RED, thick=1.2, before=0, after=10))
    A(p(
        "Se desarrollan dos modelos de regresion para estimar el TARGET: "
        "uno con metodologia tradicional (GLM) y otro con Machine Learning (LightGBM). "
        "Ambos reciben exactamente el mismo conjunto de 47 features procesadas por el pipeline <b>RiskPreprocessor</b>."
    ))
    A(sp(0.15))

    A(p("GLM — Modelo Lineal Generalizado (OLS Log-Normal)", "h2"))
    A(p(
        "Se estima OLS sobre <b>log(TARGET)</b>, equivalente a un GLM con distribucion "
        "log-normal y funcion link logaritmica. La prediccion en escala original es "
        "Y_hat = exp(X * beta). Las variables se seleccionaron por backward elimination "
        "(umbral p &gt; 0.05) y control de multicolinealidad (VIF &lt; 5). "
        "Resultado: <b>R2 test = 0.291, Gini = 0.545</b>."
    ))
    A(sp(0.1))
    for item in fig("03_glm_coefficients.png", width_cm=9,
                     caption="Figura a.1 — Coeficientes estandarizados del GLM con intervalos de confianza al 95%."):
        A(item)
    A(sp(0.15))

    A(p("LightGBM — Gradient Boosting con objetivo Tweedie (ML)", "h2"))
    A(p(
        "LightGBM se selecciona como modelo ML tras evaluar tambien XGBoost y CatBoost. "
        "El objetivo <b>Tweedie</b> (variance_power=1.64) modela directamente la distribucion "
        "right-skewed del TARGET. Los hiperparametros se buscan con <b>Optuna TPE</b> "
        "(30 trials, 3-fold CV) y el modelo se detiene con early stopping (50 rondas). "
        "Resultado: <b>R2 test = 0.368, Gini = 0.587</b>, 731 arboles finales."
    ))
    A(sp(0.1))

    config_rows = [
        [p("<b>Parametro</b>", "th"), p("<b>Valor</b>", "th"), p("<b>Justificacion</b>", "th")],
        [p("objective",      "td"), p("tweedie",         "td"), p("Distribucion continua positiva right-skewed", "td")],
        [p("variance_power", "td"), p("1.64 (Optuna)",   "td"), p("Optimizado en CV; entre Poisson (1.0) y Gamma (2.0)", "td")],
        [p("n_estimators",   "td"), p("731 (early stop)","td"), p("50 rondas sin mejora en validation loss", "td")],
        [p("num_leaves",     "td"), p("63",              "td"), p("Balanceo profundidad/overfitting en Optuna", "td")],
        [p("learning_rate",  "td"), p("0.047",           "td"), p("Encontrado por Optuna TPE Sampler (30 trials)", "td")],
    ]
    A(tbl(config_rows, [3.5*cm, 3.5*cm, 8*cm]))
    A(PageBreak())

    # ============================================================
    # b. SUPUESTOS
    # ============================================================
    SECTION[0] = "b. Supuestos"
    A(p("b. Supuestos de cada modelo", "h1"))
    A(hr(RED, thick=1.2, before=0, after=10))
    A(p(
        "El GLM (OLS) requiere cumplir supuestos estadisticos clasicos. "
        "LightGBM, al ser no parametrico, no los exige — pero tiene sus propios requisitos operativos. "
        "A continuacion se evalua cada supuesto y su impacto en los resultados."
    ))
    A(sp(0.15))

    A(p("GLM — Evaluacion de supuestos (OLS sobre log-TARGET)", "h2"))
    sup_rows = [
        [p("<b>Supuesto</b>", "th"), p("<b>Test</b>", "th"),
         p("<b>Resultado</b>", "th"), p("<b>Impacto</b>", "th")],
        [p("Normalidad de residuos", "td"), p("Jarque-Bera", "td"),
         p("<b>Violado</b>", "td", textColor=RED),
         p("Residuos leptocurticos. No invalida el modelo pero si los IC clasicos (requieren bootstrap).", "td")],
        [p("Homocedasticidad", "td"), p("Breusch-Pagan", "td"),
         p("<b>Violado</b>", "td", textColor=RED),
         p("Varianza de residuos crece con el nivel del score. Los estimadores OLS siguen siendo insesgados pero ineficientes.", "td")],
        [p("Autocorrelacion", "td"), p("Durbin-Watson = 2.00", "td"),
         p("<b>OK</b>", "td", textColor=HexColor("#1A7A3C")),
         p("Residuos no correlacionados en orden de registro. No hay patron temporal problematico.", "td")],
        [p("Multicolinealidad", "td"), p("VIF maximo &lt; 5", "td"),
         p("<b>OK</b>", "td", textColor=HexColor("#1A7A3C")),
         p("Variables independientes tras eliminar colineales y el target encoding de X9.", "td")],
        [p("Linealidad", "td"), p("Inspeccion visual", "td"),
         p("<b>Parcial</b>", "td", textColor=HexColor("#B07800")),
         p("Las transformaciones log y las interacciones capturan la mayoria de la no-linealidad residual.", "td")],
    ]
    A(tbl(sup_rows, [3.5*cm, 3.8*cm, 2.1*cm, 5.6*cm]))
    A(sp(0.1))
    for item in fig("03_glm_assumptions.png", width_cm=14,
                     caption="Figura b.1 — Diagnosticos de supuestos del GLM: residuos vs fitted, QQ-plot, escala-localizacion y leverage."):
        A(item)
    A(sp(0.15))

    A(p("LightGBM — Consideraciones en lugar de supuestos", "h2"))
    lgbm_sup_rows = [
        [p("<b>Aspecto</b>", "th"), p("<b>Evaluacion</b>", "th"), p("<b>Impacto</b>", "th")],
        [p("Distribucion de residuos", "td"),
         p("No requerida — no parametrico", "td"),
         p("Tweedie maneja asimetria directamente; no necesita transformacion del TARGET.", "td")],
        [p("Overfitting", "td"),
         p("Gap R2 train-test = 0.099", "td"),
         p("Moderado pero controlado. Early stopping + L1/L2 regularizacion + Optuna evitan sobreajuste severo.", "td")],
        [p("Missing values", "td"),
         p("Nativo en GBDT — aprende optimo para NaN", "td"),
         p("Puede aprender que la ausencia misma es informativa, sin necesidad de imputacion previa.", "td")],
        [p("Estabilidad de datos", "td"),
         p("PSI < 0.002 (muy estable)", "td"),
         p("Baja deriva entre train y test. El modelo generaliza bien sin ajustes adicionales.", "td")],
    ]
    A(tbl(lgbm_sup_rows, [3.5*cm, 4.5*cm, 7*cm]))
    A(sp(0.1))
    A(p(
        "Las violaciones de supuestos del GLM son esperables en datos financieros y no invalidan "
        "su uso como <b>baseline interpretable y referencia regulatoria</b>. "
        "LightGBM elude estos problemas pero requiere monitoreo activo de estabilidad.",
        "note"
    ))
    A(PageBreak())

    # ============================================================
    # c. PREPROCESAMIENTO
    # ============================================================
    SECTION[0] = "c. Preprocesamiento"
    A(p("c. Preprocesamiento y Feature Engineering", "h1"))
    A(hr(RED, thick=1.2, before=0, after=10))
    A(p(
        "Todas las transformaciones se encapsulan en la clase <b>RiskPreprocessor</b> "
        "con interfaz fit/transform, garantizando cero data leakage. "
        "El objeto serializado como <b>preprocessor.pkl</b> se reutiliza identicamente en produccion."
    ))
    A(sp(0.1))

    A(p("Tratamiento de missing values y outliers", "h2"))
    rows_miss = [
        [p("<b>Variable</b>", "th"), p("<b>Problema</b>", "th"), p("<b>Tratamiento</b>", "th")],
        [p("X1, X2, X5, X7, X11, X12", "td"),
         p("NaN aislados (786 a 16,398)", "td"),
         p("Imputacion por mediana de entrenamiento + flag binario (X_sin_dato)", "td")],
        [p("X3, X4", "td"),
         p("Centinelas -9999998 (15k-20k registros)", "td"),
         p("Remplazar por NaN, luego mediana + flag binario. El centinela no es dato numerico.", "td")],
        [p("X11 y X12", "td"),
         p("Siempre faltan juntos (1,720 casos)", "td"),
         p("Flag co_missing_X1_X2: ausencia conjunta es senal estructural de ausencia bancaria.", "td")],
        [p("X9 (distritos)", "td"),
         p("912 vacios + cardinalidad alta (26 valores)", "td"),
         p("Target encoding suavizado (k=10): media ponderada del TARGET por distrito.", "td")],
        [p("Outliers numericos", "td"),
         p("Valores extremos en X1, X7", "td"),
         p("Capping por percentil 99. Transformacion log para comprimir la escala.", "td")],
    ]
    A(tbl(rows_miss, [3.5*cm, 4.5*cm, 7*cm]))
    A(sp(0.1))

    A(p("Feature engineering aplicado", "h2"))
    rows_fe = [
        [p("<b>Feature creada</b>", "th"), p("<b>Formula</b>", "th"), p("<b>Motivo</b>", "th")],
        [p("X1_log", "td"), p("log(X1 + 1)", "td"),
         p("Normaliza ingreso right-skewed; mejora linealidad en GLM", "td")],
        [p("X7_log", "td"), p("log(X7 + 1)", "td"),
         p("Normaliza score buro; reduce influencia de outliers", "td")],
        [p("score_x_ingreso", "td"), p("X1_log x X7", "td"),
         p("Interaccion capacidad de pago x historial. Driver 1 con 38% SHAP.", "td")],
        [p("X6_x_score_ingreso_*", "td"), p("X6 (categorica) x score_x_ingreso", "td"),
         p("El segmento laboral modifica el efecto del ingreso sobre el riesgo.", "td")],
        [p("co_missing_X1_X2", "td"), p("1 si X1 Y X2 son NaN", "td"),
         p("Patron de ausencia conjunta: senal de ausencia total de historial bancario.", "td")],
        [p("X9_target_enc", "td"), p("E[TARGET | distrito] suavizado", "td"),
         p("Reemplaza 26 dummies por un unico ordinal; reduce overfitting geografico.", "td")],
    ]
    A(tbl(rows_fe, [3.5*cm, 4*cm, 7.5*cm]))
    A(sp(0.1))
    A(p("Pipeline produce <b>47 features finales</b> desde las 12 variables originales.", "note"))
    A(sp(0.1))
    for item in fig("02_preprocessing_comparison.png", width_cm=15,
                     caption="Figura c.1 — Comparacion antes/despues del preprocesamiento en features seleccionadas."):
        A(item)
    A(sp(0.1))
    for item in fig_half("02_outlier_capping.png", "02_sentinel_target_distribution.png",
                          cap_l="Figura c.2 — Tratamiento de outliers por capping en X1.",
                          cap_r="Figura c.3 — TARGET para registros con/sin centinela en X3."):
        A(item)
    A(PageBreak())

    # ============================================================
    # d. METRICAS Y COMPARACION
    # ============================================================
    SECTION[0] = "d. Metricas y Comparacion"
    A(p("d. Metricas de Rendimiento y Comparacion", "h1"))
    A(hr(RED, thick=1.2, before=0, after=10))
    A(p(
        "Se calculan metricas de regresion (R2, RMSE) y metricas de discriminacion bancaria "
        "(Gini = 2xAUC-1, KS) sobre el conjunto de test (16,667 obs). "
        "Se evaluan cuatro modelos: GLM, LightGBM, XGBoost y CatBoost."
    ))
    A(sp(0.1))
    for item in fig("05_model_comparison.png", width_cm=15.5,
                     caption="Figura d.1 — Comparacion de metricas clave. LightGBM domina en todas las dimensiones de test."):
        A(item)
    A(sp(0.1))

    comp_rows = [
        [p("<b>Modelo</b>", "th"), p("<b>R2 Train</b>", "th"), p("<b>R2 Test</b>", "th"),
         p("<b>RMSE Test (S/.)</b>", "th"), p("<b>KS Test</b>", "th"), p("<b>Gini Test</b>", "th")],
        [p("<b>LightGBM</b>", "td_b"), p("0.467","td"), p("<b>0.368</b>","td_b"),
         p("<b>4,675</b>","td_b"), p("<b>0.444</b>","td_b"), p("<b>0.587</b>","td_b")],
        [p("CatBoost","td"), p("0.441","td"), p("0.367","td"), p("4,678","td"), p("0.442","td"), p("0.586","td")],
        [p("XGBoost","td"),  p("0.436","td"), p("0.367","td"), p("4,680","td"), p("0.442","td"), p("0.585","td")],
        [p("GLM","td"),      p("0.302","td"), p("0.291","td"), p("4,951","td"), p("0.412","td"), p("0.545","td")],
    ]
    extra = [("BACKGROUND", (0,1), (-1,1), HexColor("#FFF5F5")),
             ("LINEBELOW",  (0,1), (-1,1), 1.2, RED)]
    A(tbl(comp_rows, [3.2*cm, 2.3*cm, 2.3*cm, 3.2*cm, 2.3*cm, 2.3*cm], extra))
    A(p(
        "Gini = 2 x AUC - 1, binarizando TARGET por la mediana. "
        "Estandar de discriminacion en scoring bancario (umbral industria: Gini &gt; 0.40 aceptable, &gt; 0.55 bueno).",
        "note"
    ))
    A(sp(0.1))
    A(p(
        "Los tres modelos GBDT convergen en R2 test aprox. 0.367, indicando que se ha alcanzado "
        "el <b>techo informacional</b> del dataset con las features disponibles. "
        "LightGBM lidera marginalmente en todas las metricas y es el que se selecciona."
    ))
    A(sp(0.1))
    for item in fig_half("04_lgbm_decile_analysis.png", "05_prediction_distributions.png",
                          cap_l="Figura d.2 — Deciles LightGBM: ordena correctamente (decil 10: S/.17,200 vs decil 1: S/.3,820).",
                          cap_r="Figura d.3 — Distribucion de predicciones vs. reales. LightGBM captura mejor la cola alta."):
        A(item)
    A(sp(0.1))
    for item in fig("03_glm_decile_analysis.png", width_cm=13,
                     caption="Figura d.4 — Deciles GLM: ordena bien los tramos medios pero subestima la cola alta (decil 10)."):
        A(item)
    A(PageBreak())

    # ============================================================
    # e. INTERPRETACION
    # ============================================================
    SECTION[0] = "e. Interpretacion de Variables"
    A(p("e. Interpretacion de Variables", "h1"))
    A(hr(RED, thick=1.2, before=0, after=10))
    A(p(
        "La interpretabilidad es un requisito regulatorio clave (directrices SBS, Marco de Riesgo de Modelos). "
        "LightGBM se explica mediante <b>SHAP values</b> (SHapley Additive exPlanations), "
        "que descomponen cada prediccion en contribuciones individuales por variable "
        "con garantia de consistencia matematica (axiomas de Shapley). "
        "El GLM ofrece coeficientes directos como referencia alternativa."
    ))
    A(sp(0.15))

    A(p("Importancia SHAP y coherencia con el negocio crediticio", "h2"))
    for item in fig_half("04_shap_importance.png", "04_shap_summary.png",
                          w=7.4,
                          cap_l="Figura e.1 — Importancia SHAP media por variable.",
                          cap_r="Figura e.2 — SHAP summary: cada punto es un cliente. Rojo = valor alto de la feature."):
        A(item)
    A(sp(0.1))

    interp_rows = [
        [p("<b>Variable</b>", "th"), p("<b>Efecto</b>", "th"), p("<b>Lectura de negocio / riesgo</b>", "th")],
        [p("score_x_ingreso", "td"), p("Positivo dominante (38%)", "td"),
         p("Ingreso alto + buen historial juntos: predictor mas robusto del score. Coherente con toda la teoria crediticia.", "td")],
        [p("X1_log (ingreso)", "td"), p("Positivo fuerte", "td"),
         p("Mayor capacidad de pago implica menor riesgo y mayor score. Signo esperado.", "td")],
        [p("X7 (score buro)", "td"), p("Positivo fuerte", "td"),
         p("Historial crediticio positivo predice calidad futura. Validado por GLM (coeficiente positivo significativo).", "td")],
        [p("X9_enc (distrito)", "td"), p("Positivo moderado", "td"),
         p("Zonas de alto ingreso medio (Miraflores, San Isidro) predicen scores mayores. Target encoding captura efecto geografico.", "td")],
        [p("co_missing_X1_X2", "td"), p("Negativo", "td"),
         p("Ausencia simultanea de ingreso y saldo bancario: senal de ausencia total de historial. Alto riesgo.", "td")],
        [p("X3_sin_dato", "td"), p("Negativo", "td"),
         p("Sin registro en X3: cliente sin relacion previa, mayor incertidumbre de scoring.", "td")],
        [p("Interacciones X6", "td"), p("Moderador", "td"),
         p("El tipo de empleo modifica el efecto del ingreso: independiente con igual sueldo tiene mayor variabilidad de score.", "td")],
    ]
    A(tbl(interp_rows, [3.5*cm, 3*cm, 8.5*cm]))
    A(sp(0.1))
    A(p(
        "Los signos SHAP del LightGBM son coherentes con los coeficientes del GLM, "
        "lo que da confianza en que los drivers son robustos y no artefactos del modelo de ML.",
        "note"
    ))
    A(PageBreak())

    # ============================================================
    # f. SELECCION DEL MODELO
    # ============================================================
    SECTION[0] = "f. Seleccion del Modelo"
    A(p("f. Seleccion del Modelo: ¿Cual implementaria?", "h1"))
    A(hr(RED, thick=1.2, before=0, after=10))
    A(p(
        "<b>Implementaria LightGBM.</b> "
        "Supera al GLM en todas las metricas relevantes (R2 +26%, RMSE -5.6%, Gini +7.7%) "
        "y ofrece interpretabilidad regulatoria via SHAP por cliente. "
        "El GLM se mantiene como <b>modelo de referencia y respaldo regulatorio</b> "
        "por su transparencia directa en coeficientes."
    ))
    A(sp(0.1))

    A(p("Trade-offs: performance vs. interpretabilidad", "h2"))
    trade_rows = [
        [p("<b>Dimension</b>", "th"), p("<b>LightGBM (recomendado)</b>", "th"), p("<b>GLM (referencia)</b>", "th")],
        [p("R2 / Gini / KS", "td"),  p("0.368 / 0.587 / 0.444", "td"), p("0.291 / 0.545 / 0.412", "td")],
        [p("RMSE (S/.)",       "td"), p("4,675 (-5.6%)", "td"),          p("4,951", "td")],
        [p("Supuestos",        "td"), p("No parametrico — no aplica", "td"), p("Heterocedasticidad y no-normalidad violadas", "td")],
        [p("Interpretabilidad","td"), p("SHAP por cliente (mas costoso)", "td"), p("Coeficientes directos (inmediata)", "td")],
        [p("Auditoria regulatoria","td"), p("Requiere explicacion post-hoc", "td"), p("Directamente auditable", "td")],
        [p("Latencia / tamano", "td"), p("&lt; 1ms / 1.2 MB (ONNX)", "td"), p("&lt; 1ms / 37 MB", "td")],
        [p("Retraining",        "td"), p("Mensual con Optuna o transfer learning", "td"), p("Mensual, mas rapido", "td")],
    ]
    A(tbl(trade_rows, [4.2*cm, 5.5*cm, 5.3*cm]))
    A(sp(0.15))

    A(p("Riesgos identificados y mitigaciones", "h2"))
    risk_rows = [
        [p("<b>Riesgo</b>", "th"), p("<b>Evidencia actual</b>", "th"), p("<b>Mitigacion</b>", "th")],
        [p("Overfitting moderado", "td"),
         p("Gap R2 train-test = 0.099", "td"),
         p("Early stopping + regularizacion L1/L2 + CV 3-fold en Optuna", "td")],
        [p("Deriva de features (drift)", "td"),
         p("PSI actual &lt; 0.002 — estable", "td"),
         p("Monitoreo mensual de PSI; retraining automatico si PSI &gt; 0.25", "td")],
        [p("Concentracion en un driver", "td"),
         p("score_x_ingreso = 38% SHAP", "td"),
         p("Alertas si la distribucion de X1 o X7 cambia &gt; 5% mensual", "td")],
        [p("Explicabilidad regulatoria", "td"),
         p("Caja gris — no transparente por defecto", "td"),
         p("SHAP por cliente disponible en API; GLM como benchmark publico de referencia", "td")],
        [p("Dependencia de ONNX Runtime", "td"),
         p("Nueva dependencia de infraestructura", "td"),
         p("Fallback al pkl de LightGBM en entornos con libgomp disponible", "td")],
    ]
    A(tbl(risk_rows, [3.5*cm, 4*cm, 7.5*cm]))
    A(PageBreak())

    # ============================================================
    # g. ESTABILIDAD Y MONITOREO
    # ============================================================
    SECTION[0] = "g. Estabilidad y Monitoreo"
    A(p("g. Estabilidad del Modelo y Metricas de Monitoreo", "h1"))
    A(hr(RED, thick=1.2, before=0, after=10))
    A(p(
        "El <b>Population Stability Index (PSI)</b> es el estandar bancario para detectar "
        "si la poblacion que ingresa al modelo en produccion difiere de la de entrenamiento, "
        "lo que indica que el modelo puede estar desactualizado. "
        "Se complementa con metricas de rendimiento reales (cuando hay etiquetas disponibles) "
        "y monitoreo de la distribucion del TARGET (concept drift)."
    ))
    A(sp(0.1))
    A(p(
        "Regla de decision: PSI &lt; 0.10 = sin cambio significativo | "
        "0.10-0.25 = cambio moderado, monitorear | &gt; 0.25 = cambio severo, reentrenar.",
        "note"
    ))
    A(sp(0.1))
    for item in fig_half("05_psi_features.png", "05_psi_monitoring.png",
                          cap_l="Figura g.1 — PSI por feature (train vs. test). Todos &lt; 0.002 — muy estables.",
                          cap_r="Figura g.2 — PSI del score predicho (train vs. test). Alta estabilidad poblacional."):
        A(item)
    A(sp(0.15))

    A(p("Estrategia completa de monitoreo", "h2"))
    mon_rows = [
        [p("<b>Metrica / Componente</b>", "th"), p("<b>Frecuencia</b>", "th"),
         p("<b>Umbral de alerta</b>", "th"), p("<b>Accion</b>", "th")],
        [p("PSI del score predicho",  "td"), p("Mensual",    "td"),
         p("&gt; 0.25", "td"), p("Reentrenar con datos del ultimo mes", "td")],
        [p("PSI de X1, X7, X9",      "td"), p("Mensual",    "td"),
         p("&gt; 0.10", "td"), p("Alertar al equipo; analizar causa raiz", "td")],
        [p("Distribucion del TARGET", "td"), p("Mensual",    "td"),
         p("Shift &gt; 10%", "td"), p("Analizar concept drift — cambio en el fenomeno subyacente", "td")],
        [p("R2 y Gini reales",        "td"), p("Trimestral", "td"),
         p("Degradacion &gt; 5%", "td"), p("Iniciar ciclo de reentrenamiento", "td")],
        [p("Cambios en poblacion",    "td"), p("Mensual",    "td"),
         p("Nuevo segmento &gt; 5%", "td"), p("Revisar si el modelo cubre el nuevo segmento", "td")],
        [p("Estado API (/health)",    "td"), p("Continuo",   "td"),
         p("model_ready = false", "td"), p("Alerta inmediata; rollback automatico", "td")],
    ]
    A(tbl(mon_rows, [4*cm, 2.2*cm, 3*cm, 5.8*cm]))
    A(PageBreak())

    # ============================================================
    # h. MODELO EN PRODUCCION
    # ============================================================
    SECTION[0] = "h. Modelo en Produccion"
    A(p("h. Modelo en Produccion", "h1"))
    A(hr(RED, thick=1.2, before=0, after=10))
    A(p(
        "El flujo de produccion tiene tres fases — entrenamiento, scoring y monitoreo — "
        "que funcionan de forma modular. La API REST ya esta desplegada en Vercel "
        "con el modelo LightGBM en formato ONNX."
    ))
    A(sp(0.1))

    pipeline = [
        (NAVY, "ENTRENAMIENTO",
         "Carga mensual de datos crudos → validacion de schema → "
         "RiskPreprocessor.fit(X_train, y_train) → "
         "Optuna HPO (30 trials, 3-fold CV) → early stopping → "
         "exportar lgbm_model.onnx y preprocessor.pkl → correr suite de 47 tests."),
        (HexColor("#1A5276"), "SCORING",
         "Ingesta de nuevo batch → RiskPreprocessor.transform(X) — mismo objeto fit → "
         "47 features → ONNX Runtime inference (lgbm_model.onnx, 1.2 MB) → "
         "score en S/. por cliente. Latencia &lt; 1ms, sin dependencia de libgomp."),
        (RED, "MONITOREO",
         "PSI mensual de features y score vs. ventana de referencia → "
         "R2/Gini trimestrales cuando hay etiquetas → "
         "trigger de reentrenamiento si PSI &gt; 0.25 o degradacion &gt; 5% → "
         "dashboard de drift + alertas al equipo de riesgo."),
    ]
    for color, nombre, desc in pipeline:
        row = [[
            p(f"<b>{nombre}</b>",
              ParagraphStyle(f"pn_{nombre}", fontName="Arial-Bold", fontSize=9, textColor=WHITE)),
            p(desc,
              ParagraphStyle(f"pd_{nombre}", fontName="Arial", fontSize=9,
                             textColor=DGRAY, leading=13)),
        ]]
        t = Table(row, colWidths=[3.8*cm, 11.2*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (0,-1), color),
            ("BACKGROUND",    (1,0), (1,-1), LGRAY),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING",    (0,0), (-1,-1), 10),
            ("BOTTOMPADDING", (0,0), (-1,-1), 10),
            ("LEFTPADDING",   (0,0), (-1,-1), 10),
            ("RIGHTPADDING",  (0,0), (-1,-1), 8),
            ("LINEBELOW",     (0,0), (-1,-1), 0.5, MGRAY),
        ]))
        A(t)
    A(sp(0.15))

    A(p("API REST — endpoints del servicio (ya en produccion en Vercel)", "h2"))
    api_rows = [
        [p("<b>Metodo</b>", "th"), p("<b>Endpoint</b>", "th"), p("<b>Descripcion</b>", "th")],
        [p("GET",  "td"), p("/health",        "td"), p("Estado del servicio y disponibilidad del modelo ONNX", "td")],
        [p("GET",  "td"), p("/model/info",    "td"), p("Metadatos: 47 features, version, tipo LightGBM ONNX", "td")],
        [p("POST", "td"), p("/predict",       "td"), p("Prediccion individual: recibe X1-X12, retorna score en S/. + request_id", "td")],
        [p("POST", "td"), p("/predict/batch", "td"), p("Prediccion en lote: hasta 10,000 registros por llamada", "td")],
    ]
    A(tbl(api_rows, [2*cm, 3.8*cm, 9.2*cm]))
    A(sp(0.1))
    A(p(
        'POST /predict  {"X1": 207137, "X7": 633, "X9": "SANTIAGO DE SURCO", ...}'
        '  ->  {"prediction": 9565.58, "request_id": "ca957d03-..."}',
        "code"
    ))
    A(sp(0.1))
    A(p(
        "Despliegue en <b>Vercel</b> (serverless, sin dependencia libgomp, ONNX Runtime) "
        "y <b>Docker</b> (python:3.11-slim, 405 MB) para entornos on-premise. "
        "CI/CD en GitHub Actions: 47 pruebas en cada push a main.",
        "note"
    ))
    A(PageBreak())

    # ============================================================
    # SLIDE FINAL
    # ============================================================
    SECTION[0] = "Acceso al Proyecto"
    A(sp(1.5))
    A(p("Acceso al Proyecto", "h1"))
    A(hr(RED, thick=1.2, before=0, after=12))
    A(p("El modelo esta desplegado y operativo. Puede ser consultado de forma inmediata:"))
    A(sp(0.5))

    web_data = [
        [p("INTERFAZ WEB INTERACTIVA — Frontend del Predictor",
           ParagraphStyle("wh", fontName="Arial-Bold", fontSize=10, textColor=WHITE, alignment=TA_CENTER))],
        [p("https://modelo-task.vercel.app",
           ParagraphStyle("wu", fontName="Arial-Bold", fontSize=16, textColor=RED, alignment=TA_CENTER))],
        [p("Perfiles de cliente preconfigurados  |  Score en tiempo real  |  Badge de nivel de riesgo",
           ParagraphStyle("we", fontName="Arial", fontSize=9, textColor=GRAY, alignment=TA_CENTER))],
    ]
    wt = Table(web_data, colWidths=[15*cm])
    wt.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (0,0), RED),
        ("BACKGROUND",    (0,1), (0,2), LGRAY),
        ("BOX",           (0,0), (-1,-1), 1, RED),
        ("TOPPADDING",    (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
    ]))
    A(wt)
    A(sp(0.35))

    api_data = [
        [p("API REST EN PRODUCCION — Vercel (Serverless)",
           ParagraphStyle("bh", fontName="Arial-Bold", fontSize=10, textColor=WHITE, alignment=TA_CENTER))],
        [p("https://modelo-task.vercel.app/docs",
           ParagraphStyle("bu", fontName="Arial-Bold", fontSize=16, textColor=BLUE, alignment=TA_CENTER))],
        [p("GET /health   |   GET /model/info   |   POST /predict   |   POST /predict/batch",
           ParagraphStyle("be", fontName="Arial", fontSize=9, textColor=GRAY, alignment=TA_CENTER))],
    ]
    at = Table(api_data, colWidths=[15*cm])
    at.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (0,0), NAVY),
        ("BACKGROUND",    (0,1), (0,2), LGRAY),
        ("BOX",           (0,0), (-1,-1), 1, NAVY),
        ("TOPPADDING",    (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
    ]))
    A(at)
    A(sp(0.35))

    gh_data = [
        [p("CODIGO FUENTE — GitHub",
           ParagraphStyle("gh", fontName="Arial-Bold", fontSize=10, textColor=WHITE, alignment=TA_CENTER))],
        [p("github.com/josefrodrim/MODELO_TASK",
           ParagraphStyle("gu", fontName="Arial-Bold", fontSize=16, textColor=BLUE, alignment=TA_CENTER))],
        [p("Notebooks ejecutados  |  Codigo modular (src/)  |  47 tests  |  CI/CD  |  Docker  |  Vercel",
           ParagraphStyle("ge", fontName="Arial", fontSize=9, textColor=GRAY, alignment=TA_CENTER))],
    ]
    gt = Table(gh_data, colWidths=[15*cm])
    gt.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (0,0), HexColor("#24292E")),
        ("BACKGROUND",    (0,1), (0,2), LGRAY),
        ("BOX",           (0,0), (-1,-1), 1, HexColor("#24292E")),
        ("TOPPADDING",    (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
    ]))
    A(gt)
    A(sp(0.45))

    A(p("Contenido del repositorio", "h2"))
    for item in [
        "<b>notebooks/</b>  Analisis exploratorio, preprocesamiento, GLM, LightGBM y comparacion (con outputs)",
        "<b>src/</b>         Codigo fuente modular — preprocessing, modelos, metricas, API REST",
        "<b>models/</b>      lgbm_model.onnx (1.2 MB) + preprocessor.pkl — listos para produccion",
        "<b>tests/</b>       47 pruebas pytest — metricas, preprocesamiento y endpoints API",
        "<b>scripts/</b>     train_all_models.py — entrena GLM + LightGBM + XGBoost + CatBoost",
        "<b>Dockerfile</b>   Imagen python:3.11-slim con healthcheck integrado",
        "<b>vercel.json</b>  Despliegue serverless — ONNX Runtime, sin libgomp",
    ]:
        A(b(item))

    doc.build(S)
    print(f"\nPDF generado: {OUTPUT}")
    sz = os.path.getsize(OUTPUT) / 1024
    print(f"Tamano: {sz:.0f} KB")


if __name__ == "__main__":
    build()
