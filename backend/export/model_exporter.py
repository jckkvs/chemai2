"""
Model Exporter - chemai2/backend/export/model_exporter.py
Multi-format model export with Japanese PDF support
"""
import json
import os
import tempfile
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Literal
from datetime import datetime

import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import plotly.io as pio

from backend.core.config import settings
from backend.utils.logger import logger


class ModelExporter:
    """
    Unified model export engine
    
    Supports:
    - ONNX (with shape/type inference)
    - PMML (via sklearn2pmml)
    - Pickle (native)
    - PDF Report (Japanese NotoSansJP embedded)
    - Configuration JSON
    """
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or settings.EXPORT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._register_fonts()
    
    def _register_fonts(self):
        """Register NotoSansJP for Japanese PDF rendering"""
        font_paths = [
            "/usr/share/fonts/truetype/noto/NotoSansJP-Regular.otf",
            "/usr/share/fonts/noto-cjk/NotoSansJP-Regular.otf",
            os.path.join(os.path.dirname(__file__), "fonts", "NotoSansJP-Regular.otf"),
        ]
        for path in font_paths:
            if os.path.exists(path):
                pdfmetrics.registerFont(TTFont('NotoSansJP', path))
                logger.info(f"Registered Japanese font: {path}")
                self._font_name = 'NotoSansJP'
                return
        logger.warning("NotoSansJP not found. Falling back to default font (Japanese may not render).")
        self._font_name = 'Helvetica'
    
    def export_onnx(self, model, X_sample: pd.DataFrame, output_path: str = None) -> Path:
        """Export model to ONNX format"""
        try:
            import skl2onnx
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            
            initial_type = [('float_input', FloatTensorType([None, X_sample.shape[1]]))]
            onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=14)
            
            out_path = output_path or self.output_dir / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.onnx"
            with open(out_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            
            logger.info(f"ONNX model exported to {out_path}")
            return Path(out_path)
        except ImportError:
            logger.error("skl2onnx not installed. Run: pip install skl2onnx")
            raise
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            raise
    
    def export_pmml(self, model, X_sample: pd.DataFrame, output_path: str = None) -> Path:
        """Export model to PMML format"""
        try:
            from sklearn2pmml import sklearn2pmml
            from sklearn2pmml.pipeline import PMMLPipeline
            
            pipeline = PMMLPipeline([("estimator", model)])
            pipeline.fit(X_sample)  # Required for PMML schema generation
            
            out_path = output_path or self.output_dir / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pmml"
            sklearn2pmml(pipeline, str(out_path), with_repr=True)
            
            logger.info(f"PMML model exported to {out_path}")
            return Path(out_path)
        except ImportError:
            logger.error("sklearn2pmml not installed. Run: pip install sklearn2pmml")
            raise
        except Exception as e:
            logger.error(f"PMML export failed: {e}")
            raise
    
    def export_joblib(self, model, output_path: str = None) -> Path:
        """Export model using joblib"""
        import joblib
        out_path = output_path or self.output_dir / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        joblib.dump(model, out_path)
        logger.info(f"Model exported to {out_path} (joblib)")
        return Path(out_path)
    
    def export_pdf_report(self, model_info: Dict[str, Any], plots: Dict[str, Any] = None, 
                          constraints_report: Dict[str, Any] = None, output_path: str = None) -> Path:
        """Generate comprehensive PDF report with Japanese support"""
        out_path = output_path or self.output_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        doc = SimpleDocTemplate(str(out_path), pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
        elements = []
        
        # Styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('CustomTitle', parent=styles['Title'], fontName=self._font_name, fontSize=24, spaceAfter=12)
        heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading1'], fontName=self._font_name, fontSize=16, spaceBefore=18, spaceAfter=8)
        body_style = ParagraphStyle('CustomBody', parent=styles['Normal'], fontName=self._font_name, fontSize=11, leading=14, spaceAfter=6)
        
        # Title
        elements.append(Paragraph("ChemAI ML Studio - モデル解析レポート", title_style))
        elements.append(Spacer(1, 12))
        
        # Metadata
        elements.append(Paragraph(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body_style))
        elements.append(Paragraph(f"モデル: {model_info.get('estimator', 'N/A')}", body_style))
        elements.append(Paragraph(f"タスク: {model_info.get('task_type', 'N/A')}", body_style))
        elements.append(Spacer(1, 12))
        
        # Metrics Table
        if 'metrics' in model_info:
            elements.append(Paragraph("📊 評価指標", heading_style))
            metrics_data = [["指標", "値"]]
            for k, v in model_info['metrics'].items():
                metrics_data.append([k, str(round(v, 4))])
            
            table = Table(metrics_data, colWidths=[120*mm, 80*mm])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), '#4CAF50'),
                ('TEXTCOLOR', (0, 0), (-1, 0), '#FFFFFF'),
                ('FONTNAME', (0, 0), (-1, -1), self._font_name),
                ('FONTSIZE', (0, 0), (-1, -1), 11),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('GRID', (0, 0), (-1, -1), 1, '#CCCCCC'),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), ['#FFFFFF', '#F9F9F9'])
            ]))
            elements.append(table)
            elements.append(Spacer(1, 12))
        
        # Constraint Report
        if constraints_report:
            elements.append(Paragraph("🔒 制約検証結果", heading_style))
            for feat, eval_data in constraints_report.get('details', {}).items():
                status = "✅ 合格" if eval_data.get('passed') else "❌ 不合格"
                elements.append(Paragraph(f"<b>{feat}</b>: {status}", body_style))
            elements.append(Spacer(1, 12))
        
        # Plots
        if plots:
            elements.append(Paragraph("📈 可視化チャート", heading_style))
            for name, fig in plots.items():
                try:
                    img_bytes = pio.to_image(fig, format="png", width=800, height=400)
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        tmp.write(img_bytes)
                        tmp.flush()
                        elements.append(Image(tmp.name, width=160*mm, height=80*mm))
                        elements.append(Paragraph(name, body_style))
                except Exception as e:
                    elements.append(Paragraph(f"チャート生成エラー: {name} ({e})", body_style))
        
        # Build PDF
        doc.build(elements)
        logger.info(f"PDF report exported to {out_path}")
        return Path(out_path)
    
    def export_config(self, config: Dict[str, Any], output_path: str = None) -> Path:
        """Export pipeline configuration as JSON"""
        out_path = output_path or self.output_dir / f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Configuration exported to {out_path}")
        return Path(out_path)
