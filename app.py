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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
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
UPLOAD_FOLDER = os.path.join(app.root_path, config["output"]["upload_folder"])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    logger.info(f"Created upload folder: {UPLOAD_FOLDER}")

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
            
            # Save the original uploaded image
            file.save(file_path)
            if not os.path.exists(file_path):
                logger.error(f"File not found after save: {absolute_path}")
                error = "Failed to save the uploaded image"
                return render_template("index.html", error=error)
            
            # Save a copy of the original image to prevent overwrite
            original_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"original_{filename}")
            shutil.copy(file_path, original_image_path)
            logger.info(f"Uploaded image saved successfully: {absolute_path}")
            logger.info(f"Original image copy saved: {original_image_path}")
            logger.info(f"Image URL for display: {url_for('static', filename='uploads/original_' + filename)}")
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
                # Remove heatmap_path, add original_img_path
                result.pop('heatmap_path', None)  # Remove heatmap if present
                result['original_img_path'] = original_image_path
                session['last_result'] = result
                session['last_filename'] = filename
                session['last_uploaded_image_path'] = original_image_path
                logger.info(f"Disease prediction result: {result}")

                soil_type = request.form.get("soil_type")
                rainfall = request.form.get("rainfall")
                fertilization = request.form.get("fertilization")
                month = request.form.get("month")

                if soil_type and rainfall and fertilization and month:
                    crop_map = {
                        "Cherry_(including_sour)": "Cherry",
                        "Corn_(maize)": "Maize",
                        "Pepper": "Pepper"
                    }
                    mapped_crop = crop_map.get(result['crop_type'], result['crop_type'].split("_")[0])
                    yield_result = predict_yield(
                        mapped_crop,
                        result['disease_status'],
                        soil_type,
                        rainfall,
                        fertilization,
                        month
                    )
                    if 'error' in yield_result:
                        error = yield_result['error']
                        yield_result = None
                        logger.error(f"Yield prediction error: {error}")
                    else:
                        yield_result = convert_numpy_types(yield_result)
                        yield_result['yield'] = round(yield_result['yield'], 2)
                        session['last_yield_result'] = yield_result
                        yield_inputs = {
                            "soil_type": soil_type,
                            "rainfall": float(rainfall),
                            "fertilization": float(fertilization),
                            "month": int(month)
                        }
                        session['last_yield_inputs'] = yield_inputs
                        logger.info(f"Yield prediction result: {yield_result}")
        except Exception as e:
            error = f"Error analyzing image or predicting yield: {str(e)}"
            logger.error(f"Prediction error: {str(e)}")
            result = None

    return render_template("index.html", result=result, filename=filename, error=error, yield_result=yield_result, yield_inputs=yield_inputs)

@app.route("/download-report/<filename>")
def download_report(filename):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(file_path):
            logger.error(f"Uploaded image not found for report: {file_path}")
            abort(404)
        
        if 'last_result' in session and session.get('last_filename') == filename:
            result = session['last_result']
            uploaded_image_path = session.get('last_uploaded_image_path', file_path)
            logger.info(f"Using cached result from session for {filename}: {result}")
            logger.info(f"Using uploaded image path from session: {uploaded_image_path}")
        else:
            result = predict_crop(file_path)
            if 'error' in result:
                logger.error(f"Prediction error in report: {result['error']}")
                abort(500, description=result['error'])
            original_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"original_{filename}")
            shutil.copy(file_path, original_image_path)
            result = convert_numpy_types(result)
            result.pop('heatmap_path', None)  # Remove heatmap if present
            result['original_img_path'] = original_image_path
            session['last_result'] = result
            session['last_filename'] = filename
            session['last_uploaded_image_path'] = original_image_path
            logger.info(f"New prediction result: {result}")
            logger.info(f"Stored uploaded image path: {original_image_path}")

        yield_result = session.get('last_yield_result') if 'last_yield_result' in session and session.get('last_filename') == filename else None
        yield_inputs = session.get('last_yield_inputs') if 'last_yield_inputs' in session and session.get('last_filename') == filename else None

        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
        styles = getSampleStyleSheet()
        elements = []

        # Title
        title_style = ParagraphStyle(name='TitleStyle', parent=styles['Heading1'], fontSize=20, textColor=colors.darkgreen, alignment=1)
        elements.append(Paragraph("Plant Disease Detection and Yield Prediction Report", title_style))
        elements.append(Spacer(1, 20))

        # Uploaded Image Only
        try:
            uploaded_image_path = result.get('original_img_path', file_path)
            logger.info(f"Adding uploaded image to PDF: {uploaded_image_path}")
            if not os.path.exists(uploaded_image_path):
                logger.error(f"Uploaded image file not found: {uploaded_image_path}")
                raise FileNotFoundError(f"Uploaded image missing: {uploaded_image_path}")
            elements.append(Paragraph("Uploaded Image:", styles['Normal']))
            elements.append(Image(uploaded_image_path, width=200, height=200))
            elements.append(Spacer(1, 20))
        except Exception as e:
            logger.error(f"Failed to add image to report: {str(e)}")
            elements.append(Paragraph(f"Image not available: {str(e)}", styles['Normal']))

        # Disease Results (Bold Header)
        normal_style = ParagraphStyle(name='NormalStyle', parent=styles['Normal'], fontSize=12, textColor=colors.black, spaceAfter=10)
        bold_style = ParagraphStyle(name='BoldStyle', parent=styles['Normal'], fontSize=12, textColor=colors.black, spaceAfter=10, fontName='Helvetica-Bold')
        elements.append(Paragraph("<b>Disease Detection Results:</b>", bold_style))
        elements.append(Paragraph(f"Crop Type: {result['crop_type']}", normal_style))
        elements.append(Paragraph(f"Disease Status: {result['disease_status']}", normal_style))
        elements.append(Paragraph(f"Disease Type: {result['disease_type']}", normal_style))
        elements.append(Paragraph(f"Unified Prediction: {result['unified_prediction']}", normal_style))
        elements.append(Paragraph(f"Confidence: {result['confidence']*100:.2f}%", normal_style))

        # Yield Prediction Results (Bold Header)
        if yield_result and yield_inputs:
            elements.append(Spacer(1, 20))
            elements.append(Paragraph("<b>Yield Prediction Results:</b>", bold_style))
            elements.append(Paragraph(f"Soil Type: {yield_inputs['soil_type']}", normal_style))
            elements.append(Paragraph(f"Rainfall: {yield_inputs['rainfall']:.2f} mm", normal_style))
            elements.append(Paragraph(f"Fertilization: {yield_inputs['fertilization']:.2f} kg/ha", normal_style))
            elements.append(Paragraph(f"Month: {yield_inputs['month']}", normal_style))
            elements.append(Paragraph(f"Predicted Yield: {yield_result['yield']:.2f} tonnes/ha", normal_style))

        # Footer
        footer_style = ParagraphStyle(name='FooterStyle', parent=styles['Normal'], textColor=colors.gray, fontSize=10, alignment=1)
        elements.append(Spacer(1, 20))
        elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", footer_style))

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