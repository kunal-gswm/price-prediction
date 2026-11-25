"""Supplementary Source Code PDF Generator for Indian Stock Price Prediction Project.

Compiles complete verbatim source code across all project modules:
  - src/data/ (collector.py, preprocessing.py)
  - src/features/ (engineering.py)
  - src/models/ (linear_regression.py, evaluation.py, forecasting.py)
  - src/utils/ (config.py, currency.py, indian_calendar.py, logger.py)
  - src/visualization/ (charts.py)
  - scripts/ (train.py, generate_report.py)
  - app.py
Generates reports/Stock_Price_Prediction_Source_Code.pdf with syntax formatting,
numbered canvas headers, TrueType font embedding with Rupee (₹) support, and GitHub links.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Preformatted,
)
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from src.utils.config import REPORTS_DIR, CURRENCY_SYMBOL
from src.utils.logger import get_logger

logger = get_logger("source_code_pdf")


def setup_reportlab_fonts() -> tuple[str, str]:
    """Registers TrueType fonts that natively support the Indian Rupee symbol (₹, U+20B9)."""
    # 1. Register Monospace font for code first (Consolas has native Rupee support)
    code_font_name = "Courier"
    consolas_path = "C:/Windows/Fonts/consola.ttf"
    if os.path.exists(consolas_path):
        try:
            pdfmetrics.registerFont(TTFont("Consolas", consolas_path))
            pdfmetrics.registerFontFamily(
                "Consolas",
                normal="Consolas",
                bold="Consolas",
                italic="Consolas",
                boldItalic="Consolas",
            )
            code_font_name = "Consolas"
            logger.info("Registered TrueType font 'Consolas' for code listings with Rupee support.")
        except Exception as e:
            logger.warning(f"Failed to register Consolas: {e}")

    # 2. Register Proportional font family for text
    candidates = [
        (
            "SegoeUI",
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/segoeuib.ttf",
            "C:/Windows/Fonts/segoeuii.ttf",
            "C:/Windows/Fonts/segoeuiz.ttf",
        ),
        (
            "Arial",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/ariali.ttf",
            "C:/Windows/Fonts/arialbi.ttf",
        ),
        (
            "Calibri",
            "C:/Windows/Fonts/calibri.ttf",
            "C:/Windows/Fonts/calibrib.ttf",
            "C:/Windows/Fonts/calibrii.ttf",
            "C:/Windows/Fonts/calibriz.ttf",
        ),
    ]
    font_fam = "Helvetica"
    for fam_name, reg, bld, it, bit in candidates:
        if os.path.exists(reg) and os.path.exists(bld) and os.path.exists(it) and os.path.exists(bit):
            try:
                pdfmetrics.registerFont(TTFont(fam_name, reg))
                pdfmetrics.registerFont(TTFont(f"{fam_name}-Bold", bld))
                pdfmetrics.registerFont(TTFont(f"{fam_name}-Italic", it))
                pdfmetrics.registerFont(TTFont(f"{fam_name}-BoldItalic", bit))
                pdfmetrics.registerFontFamily(
                    fam_name,
                    normal=fam_name,
                    bold=f"{fam_name}-Bold",
                    italic=f"{fam_name}-Italic",
                    boldItalic=f"{fam_name}-BoldItalic",
                )
                logger.info(f"Registered TTFont family '{fam_name}' for full Indian Rupee (₹) glyph support.")
                font_fam = fam_name
                break
            except Exception as e:
                logger.warning(f"Failed to register font {fam_name}: {e}")

    return font_fam, code_font_name


class SourceCodeCanvas(canvas.Canvas):
    """Two-pass canvas for dynamic 'Page X of Y' numbering and running code headers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        font_name = "SegoeUI" if "SegoeUI" in pdfmetrics.getRegisteredFontNames() else "Helvetica"
        self.setFont(font_name, 8)
        self.setFillColor(colors.HexColor("#475569"))

        # Skip header and footer on cover page
        if self._pageNumber > 1:
            # Running Header
            self.drawString(54, 750, "Indian Stock Price Prediction | Complete Source Code Appendix")
            self.drawRightString(612 - 54, 750, "NSE India | Repository: github.com/kunal-gswm/price-prediction")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(54, 744, 612 - 54, 744)

            # Running Footer
            self.line(54, 45, 612 - 54, 45)
            page_text = f"Page {self._pageNumber} of {page_count}"
            self.drawString(54, 32, "Supplementary Implementation Document - Reliance Industries Limited (RELIANCE.NS)")
            self.drawRightString(612 - 54, 32, page_text)

        self.restoreState()


def build_source_code_pdf(code_fontSize: float = 6.5, code_leading: float = 8.2):
    """Generates the supplementary complete source code PDF document."""
    font_fam, code_font = setup_reportlab_fonts()
    font_bold = f"{font_fam}-Bold" if font_fam != "Helvetica" else "Helvetica-Bold"
    font_italic = f"{font_fam}-Italic" if font_fam != "Helvetica" else "Helvetica-Oblique"

    pdf_path = REPORTS_DIR / "Stock_Price_Prediction_Source_Code.pdf"
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=45,
        rightMargin=45,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CodeDocTitle",
        parent=styles["Normal"],
        fontName=font_bold,
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=10,
    )
    subtitle_style = ParagraphStyle(
        "CodeDocSubtitle",
        parent=styles["Normal"],
        fontName=font_fam,
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#334155"),
        spaceAfter=15,
    )
    meta_style = ParagraphStyle(
        "CodeDocMeta",
        parent=styles["Normal"],
        fontName=font_fam,
        fontSize=8.5,
        leading=12.5,
        textColor=colors.HexColor("#475569"),
    )
    section_style = ParagraphStyle(
        "ModuleHeader",
        parent=styles["Normal"],
        fontName=font_bold,
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=14,
        spaceAfter=4,
        keepWithNext=True,
    )
    desc_style = ParagraphStyle(
        "ModuleDesc",
        parent=styles["Normal"],
        fontName=font_italic,
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#475569"),
        spaceAfter=6,
        keepWithNext=True,
    )
    code_style = ParagraphStyle(
        "CodeBody",
        parent=styles["Normal"],
        fontName=code_font,
        fontSize=code_fontSize,
        leading=code_leading,
        textColor=colors.HexColor("#0f172a"),
    )

    story = []

    # Cover Page
    story.append(Paragraph("Indian Stock Price Prediction Project", title_style))
    story.append(
        Paragraph(
            "<b>Supplementary Document: Verbatim Source Code Appendix</b><br/>"
            "Full Machine Learning Pipeline, Indian Exchange Calendar Integration, "
            "Split Conformal Prediction Calibration, Statistical Testing, and Streamlit Dashboard Implementation",
            subtitle_style,
        )
    )

    cover_meta_text = (
        f"<b>Target Asset:</b> Reliance Industries Limited (RELIANCE.NS) on the National Stock Exchange of India (NSE)<br/>"
        f"<b>Primary Currency:</b> Indian Rupee (INR, {CURRENCY_SYMBOL}) | Timezone: Asia/Kolkata (IST)<br/>"
        f"<b>Repository:</b> https://github.com/kunal-gswm/price-prediction<br/>"
        f"<b>Companion Document:</b> Academic Technical Report (<code>Stock_Price_Prediction_Report.pdf</code>, 11 pages)<br/>"
        f"<b>Compiled:</b> {datetime.now().strftime('%B %d, %Y')}<br/>"
        f"<b>Scope:</b> Complete, verifiable Python implementation across 14 core modules."
    )
    story.append(Paragraph(cover_meta_text, meta_style))
    story.append(Spacer(1, 15))

    # Module Index Table
    modules_to_include = [
        ("src/utils/config.py", "Centralized Indian market configuration, ticker metadata, and file paths"),
        ("src/utils/currency.py", "Indian number formatting (Lakhs and Crores) and INR currency utilities"),
        ("src/utils/indian_calendar.py", "NSE official circular holiday schedule and XBOM secondary proxy calendar"),
        ("src/utils/logger.py", "Rotating production logger with ISO timestamps"),
        ("src/data/collector.py", "Historical market data collection from Yahoo Finance via yfinance for NSE equities"),
        ("src/data/preprocessing.py", "Data cleaning, deduplication, chronological 70/15/15 splitting, StandardScaler fitting"),
        ("src/features/engineering.py", "Zero-leakage technical indicators (SMA, EMA, lagged OHLCV, volatility) and next-day target"),
        ("src/models/linear_regression.py", "StockPriceModel wrapper supporting LinearRegression and Ridge with joblib serialization"),
        ("src/models/evaluation.py", "Split Conformal Prediction calibration, Diebold-Mariano testing, and directional metrics"),
        ("src/models/forecasting.py", "Next-trading-day point forecast and calendar roll inference engine"),
        ("src/visualization/charts.py", "Matplotlib academic static figure generator and Plotly interactive chart suite"),
        ("scripts/train.py", "End-to-end command-line training, calibration, figure generation, and evaluation pipeline"),
        ("scripts/generate_report.py", "Academic ReportLab PDF generator embedding verified metrics and figures"),
        ("app.py", "8-tab interactive Streamlit dashboard application for Indian stock prediction"),
    ]

    index_table_data = [["Module Path", "Functional Role in Project"]]
    for path_str, desc in modules_to_include:
        index_table_data.append([path_str, desc])

    idx_table = Table(index_table_data, colWidths=[180, 342])
    idx_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), font_bold),
            ("FONTNAME", (0, 1), (-1, -1), font_fam),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    story.append(idx_table)
    story.append(PageBreak())

    # Verbatim Modules
    for rel_path, role_desc in modules_to_include:
        full_path = PROJECT_ROOT / rel_path
        if not full_path.exists():
            continue

        story.append(Paragraph(f"Module: {rel_path}", section_style))
        story.append(Paragraph(f"<b>Purpose:</b> {role_desc}", desc_style))
        story.append(Spacer(1, 4))

        with open(full_path, "r", encoding="utf-8") as f:
            code_lines = f.readlines()

        # Format code lines with line numbers
        formatted_lines = []
        for line_num, line in enumerate(code_lines, start=1):
            clean_line = line.rstrip()
            # truncate overly long lines to prevent overflow
            if len(clean_line) > 115:
                clean_line = clean_line[:112] + "..."
            formatted_lines.append(f"{line_num:4d} | {clean_line}")

        code_text = "\n".join(formatted_lines)
        code_flowable = Preformatted(code_text, code_style)
        story.append(code_flowable)
        story.append(Spacer(1, 14))
        story.append(PageBreak())

    # Build the document
    doc.build(story, canvasmaker=SourceCodeCanvas)
    logger.info(f"Supplementary source code PDF successfully generated at: {pdf_path}")
    return pdf_path


if __name__ == "__main__":
    build_source_code_pdf()
