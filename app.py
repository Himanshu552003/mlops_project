import sys
import os
import shutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, render_template, request, send_file, url_for, abort, session
from werkzeug.utils import secure_filename
from src.models.model import predict_crop
from src.models.yield_model import predict_yield
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io
from datetime import datetime
import yaml
import logging
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow warnings
logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a')
logger = logging.getLogger(__name__)

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

app = Flask(__name__, static_url_path='/static', static_folder='static', template_folder='templates')
app.secret_key = "mysecret123"
UPLOAD_FOLDER = os.path.join(app.root_path, config["output"]["upload_folder"])  # static/uploads/
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create folder if it doesn’t exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    logger.info(f"Created folder: {UPLOAD_FOLDER}")

ALLOWED_EXTENSIONS = set(config["output"]["allowed_extensions"])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_numpy_types(data):
    """Convert NumPy types to native Python types for JSON serialization."""
    if isinstance(data, dict):
        return {k: convert_numpy_types(v) for k, v in data.items()}
    elif isinstance(data, (np.float32, np.float64)):
        return float(data)
    elif isinstance(data, (np.int32, np.int64)):
        return int(data)
    return data

def draw_background(canvas, doc):
    """Add farm background and ivory content overlay for all pages."""
    canvas.saveState()
    # Farm image background with 40% opacity
    try:
        farm_bg_path = os.path.join(app.root_path, 'static', 'farm_bg.jpg')
        if os.path.exists(farm_bg_path):
            canvas.drawImage(farm_bg_path, 0, 0, width=letter[0], height=letter[1], mask=[0, 255, 0, 255, 0, 255], preserveAspectRatio=True)
            canvas.setFillColor(colors.Color(0, 0, 0, alpha=0.6))  # 40% opacity overlay
            canvas.rect(0, 0, letter[0], letter[1], fill=1, stroke=0)
        else:
            logger.warning("Farm background image not found, using plain white")
            canvas.setFillColor(colors.white)
            canvas.rect(0, 0, letter[0], letter[1], fill=1, stroke=0)
    except Exception as e:
        logger.error(f"Error adding farm background: {str(e)}")
        canvas.setFillColor(colors.white)
        canvas.rect(0, 0, letter[0], letter[1], fill=1, stroke=0)
    # Ivory content overlay for readability
    canvas.setFillColor(colors.Color(1, 1, 0.96, alpha=0.9))  # Near-ivory with 90% opacity
    canvas.rect(20, 20, letter[0] - 40, letter[1] - 40, fill=1, stroke=0)
    # Green corner accents
    canvas.setStrokeColor(colors.Color(0.2, 0.5, 0.2, alpha=0.3))
    canvas.setLineWidth(2)
    canvas.arc(20, letter[1] - 40, 40, letter[1] - 20, startAng=0, extent=90)
    canvas.arc(letter[0] - 40, letter[1] - 40, letter[0] - 20, letter[1] - 20, startAng=90, extent=90)
    canvas.arc(20, 20, 40, 40, startAng=270, extent=90)
    canvas.arc(letter[0] - 40, 20, letter[0] - 20, 40, startAng=180, extent=90)
    canvas.restoreState()

def add_background_with_header(canvas, doc):
    """Add farm background, ivory content overlay, and header box for the first page."""
    draw_background(canvas, doc)
    canvas.saveState()
    # Draw header box with text
    header_width = 6.5 * inch
    header_height = 0.8 * inch
    header_x = (letter[0] - header_width) / 2  # Center horizontally
    header_y = letter[1] - 1 * inch - header_height  # Position below top margin
    canvas.setFillColor(colors.ivory)
    canvas.setStrokeColor(colors.darkgreen)
    canvas.setLineWidth(1)
    canvas.rect(header_x, header_y, header_width, header_height, fill=1, stroke=1)
    header_style = ParagraphStyle(
        name='HeaderStyle',
        fontSize=24,
        textColor=colors.darkgreen,
        alignment=1,  # Center
        leading=28,  # Line spacing
        fontName='Helvetica-Bold'
    )
    header_text = Paragraph("Plant Disease Detection and Yield Prediction Report", header_style)
    w, h = header_text.wrap(header_width - 30, header_height)  # Subtract padding from width
    header_text.drawOn(canvas, header_x + 15, header_y + (header_height - h) / 2)  # Center vertically with padding
    canvas.restoreState()

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    filename = None
    error = None
    yield_result = None
    yield_inputs = None

    if request.method == "POST":
        if 'image' not in request.files:
            error = "No file uploaded"
            logger.error("No file part in request")
            return render_template("index.html", error=error)
        
        file = request.files['image']
        if file.filename == '':
            error = "No file selected"
            logger.error("No file selected")
            return render_template("index.html", error=error)
        
        if not allowed_file(file.filename):
            error = "Invalid file type. Please upload a PNG, JPG, or JPEG image."
            logger.error(f"Invalid file type: {file.filename}")
            return render_template("index.html", error=error)

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        absolute_path = os.path.abspath(file_path)
        logger.info(f"Attempting to save file to: {absolute_path}")

        try:
            counter = 1
            base_name, extension = os.path.splitext(filename)
            while os.path.exists(file_path):
                filename = f"{base_name}_{counter}{extension}"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                counter += 1
            
            file.save(file_path)
            if not os.path.exists(file_path):
                logger.error(f"File not found after save: {absolute_path}")
                error = "Failed to save the uploaded image"
                return render_template("index.html", error=error)
            
            original_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"original_{filename}")
            shutil.copy(file_path, original_image_path)
            logger.info(f"Uploaded image saved: {absolute_path}")
            logger.info(f"Original image copy saved: {original_image_path}")
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            error = f"Error saving file: {str(e)}"
            return render_template("index.html", error=error)

        try:
            result = predict_crop(file_path)
            if 'error' in result:
                error = result['error']
                result = None
                logger.error(f"Disease prediction error: {error}")
            else:
                result = convert_numpy_types(result)
                result['original_img_path'] = original_image_path
                session['last_result'] = result
                session['last_filename'] = filename
                session['last_original_image_path'] = original_image_path
                logger.info(f"Disease prediction result: {result}")
        except Exception as e:
            error = f"Error analyzing image: {str(e)}"
            logger.error(f"Prediction error: {str(e)}")
            result = None

        if result and request.form.get("soil_type"):
            try:
                soil_type = request.form.get("soil_type")
                rainfall = float(request.form.get("rainfall"))
                fertilization = float(request.form.get("fertilization"))
                month = int(request.form.get("month"))
                crop_map = {"Cherry_(including_sour)": "Cherry", "Corn_(maize)": "Maize", "Pepper": "Pepper"}
                mapped_crop = crop_map.get(result['crop_type'], result['crop_type'].split("_")[0])
                yield_result = predict_yield(mapped_crop, result['disease_status'], soil_type, rainfall, fertilization, month)
                if 'error' in yield_result:
                    error = yield_result['error']
                    yield_result = None
                    logger.error(f"Yield prediction error: {error}")
                else:
                    yield_result = convert_numpy_types(yield_result)
                    yield_result['yield'] = round(yield_result['yield'], 2)
                    session['last_yield_result'] = yield_result
                    yield_inputs = {"soil_type": soil_type, "rainfall": rainfall, "fertilization": fertilization, "month": month}
                    session['last_yield_inputs'] = yield_inputs
                    logger.info(f"Yield prediction result: {yield_result}")
            except Exception as e:
                error = f"Error predicting yield: {str(e)}"
                logger.error(f"Yield prediction error: {str(e)}")
                yield_result = None

    return render_template("index.html", result=result, filename=filename, error=error, yield_result=yield_result, yield_inputs=yield_inputs)

@app.route("/download-report/<filename>")
def download_report(filename):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(file_path):
            logger.error(f"Uploaded image not found for report: {file_path}")
            abort(404)
        
        original_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"original_{filename}")
        if 'last_result' in session and session.get('last_filename') == filename:
            result = session['last_result']
            logger.info(f"Using cached result from session for {filename}: {result}")
        else:
            result = predict_crop(file_path)
            if 'error' in result:
                logger.error(f"Prediction error in report: {result['error']}")
                abort(500, description=result['error'])
            shutil.copy(file_path, original_image_path)
            result = convert_numpy_types(result)
            result['original_img_path'] = original_image_path
            session['last_result'] = result
            session['last_filename'] = filename
            session['last_original_image_path'] = original_image_path
            logger.info(f"Updated result: {result}")

        yield_result = session.get('last_yield_result') if 'last_yield_result' in session and session.get('last_filename') == filename else None
        yield_inputs = session.get('last_yield_inputs') if 'last_yield_inputs' in session and session.get('last_filename') == filename else None

        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            pdf_buffer,
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=1*inch + 0.8*inch,  # Adjusted for header on first page
            bottomMargin=1*inch
        )
        doc.onFirstPage = add_background_with_header
        doc.onLaterPages = draw_background  # Only background for later pages
        styles = getSampleStyleSheet()
        elements = []

        # Original Image
        try:
            logger.info(f"Adding original image to PDF: {original_image_path}")
            if not os.path.exists(original_image_path):
                logger.error(f"Original image file not found: {original_image_path}")
                raise FileNotFoundError(f"Original image missing: {original_image_path}")
            elements.append(Paragraph("<b>Uploaded Image:</b>", styles['Heading3']))
            elements.append(Image(original_image_path, width=3.5*inch, height=3.5*inch))
            elements.append(Spacer(1, 0.3*inch))
        except Exception as e:
            logger.error(f"Failed to add original image to report: {str(e)}")
            elements.append(Paragraph(f"Image not available: {str(e)}", styles['Normal']))

        # Disease Results
        normal_style = ParagraphStyle(name='NormalStyle', parent=styles['Normal'], fontSize=12, textColor=colors.black, spaceAfter=8, fontName='Helvetica')
        bold_style = ParagraphStyle(name='BoldStyle', parent=styles['Normal'], fontSize=14, textColor=colors.darkgreen, spaceAfter=10, fontName='Helvetica-Bold')
        disease_data = [
            [Paragraph("<b>Disease Detection Results</b>", bold_style)],
            [Paragraph(f"Crop Type: {result['crop_type']}", normal_style)],
            [Paragraph(f"Disease Status: {result['disease_status']}", normal_style)],
            [Paragraph(f"Disease Type: {result['disease_type']}", normal_style)],
            [Paragraph(f"Unified Prediction: {result['unified_prediction']}", normal_style)],
            [Paragraph(f"Confidence: {result['confidence']*100:.2f}%", normal_style)],
        ]
        disease_table = Table(disease_data, colWidths=[6*inch])
        disease_table.setStyle([
            ('BOX', (0, 0), (-1, -1), 1.5, colors.darkgreen),
            ('INNERGRID', (0, 1), (-1, -1), 0.25, colors.grey),
            ('BACKGROUND', (0, 0), (-1, 0), colors.palegreen),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 8),
        ])
        elements.append(disease_table)
        elements.append(Spacer(1, 0.4*inch))

        # Yield Prediction Results
        if yield_result and yield_inputs:
            yield_data = [
                [Paragraph("<b>Yield Prediction Results</b>", bold_style)],
                [Paragraph(f"Soil Type: {yield_inputs['soil_type']}", normal_style)],
                [Paragraph(f"Rainfall: {yield_inputs['rainfall']:.2f} mm", normal_style)],
                [Paragraph(f"Fertilization: {yield_inputs['fertilization']:.2f} kg/ha", normal_style)],
                [Paragraph(f"Month: {yield_inputs['month']}", normal_style)],
                [Paragraph(f"Predicted Yield: {yield_result['yield']:.2f} tonnes/ha", normal_style)],
            ]
            yield_table = Table(yield_data, colWidths=[6*inch])
            yield_table.setStyle([
                ('BOX', (0, 0), (-1, -1), 1.5, colors.darkgreen),
                ('INNERGRID', (0, 1), (-1, -1), 0.25, colors.grey),
                ('BACKGROUND', (0, 0), (-1, 0), colors.palegreen),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('PADDING', (0, 0), (-1, -1), 8),
            ])
            elements.append(yield_table)
            elements.append(Spacer(1, 0.4*inch))

        # Footer
        footer_style = ParagraphStyle(name='FooterStyle', parent=styles['Normal'], textColor=colors.darkgrey, fontSize=11, alignment=1, fontName='Helvetica-Oblique')
        elements.append(Spacer(1, 0.5*inch))
        elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", footer_style))
        elements.append(Paragraph("Developed by Suyash", footer_style))

        doc.build(elements)
        pdf_buffer.seek(0)

        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name=config["output"]["pdf_filename"],
            mimetype="application/pdf"
        )
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        abort(500, description=str(e))

if __name__ == "__main__":
    print(" * Running on http://localhost:5000 (Press CTRL+C to quit)")
    app.run(host="0.0.0.0", port=5000, debug=True)