import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, render_template, request, send_file, url_for, abort, session
from werkzeug.utils import secure_filename
from src.models.model import predict_crop
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import io
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a')
logger = logging.getLogger(__name__)

app = Flask(__name__, static_url_path='/static', static_folder='static', template_folder='templates')
app.secret_key = "mysecret123"  # Fixed secret key
UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    logger.info(f"Created upload folder: {UPLOAD_FOLDER}")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    filename = None
    error = None

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
            logger.info(f"File saved successfully: {absolute_path}")
            logger.info(f"Image URL for display: {url_for('static', filename='uploads/' + filename)}")
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            error = f"Error saving file: {str(e)}"
            return render_template("index.html", error=error)

        try:
            result = predict_crop(file_path)
            if 'error' in result:
                error = result['error']
                result = None
                logger.error(f"Prediction error: {error}")
            else:
                session['last_result'] = result
                session['last_filename'] = filename
                logger.info(f"Prediction result: {result}")
        except Exception as e:
            error = f"Error analyzing image: {str(e)}"
            logger.error(f"Prediction error: {str(e)}")
            result = None

    return render_template("index.html", result=result, filename=filename, error=error)

@app.route("/download-report/<filename>")
def download_report(filename):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(file_path):
            logger.error(f"File not found for report: {file_path}")
            abort(404)
        
        if 'last_result' in session and session.get('last_filename') == filename:
            result = session['last_result']
            logger.info(f"Using cached result from session for {filename}: {result}")
        else:
            result = predict_crop(file_path)
            if 'error' in result:
                logger.error(f"Prediction error in report: {result['error']}")
                abort(500, description=result['error'])

        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
        styles = getSampleStyleSheet()
        elements = []

        title_style = ParagraphStyle(name='TitleStyle', parent=styles['Heading1'], fontSize=20, textColor=colors.darkgreen, alignment=1)
        elements.append(Paragraph("Plant Disease Detection Report", title_style))
        elements.append(Spacer(1, 20))

        try:
            elements.append(Image(file_path, width=200, height=200))
            elements.append(Spacer(1, 10))
            elements.append(Image(result['heatmap_path'], width=200, height=200))
            elements.append(Spacer(1, 20))
            logger.info(f"Images added to report: {file_path}, {result['heatmap_path']}")
        except Exception as e:
            logger.error(f"Failed to add images to report: {str(e)}")
            elements.append(Paragraph(f"Images not available: {str(e)}", styles['Normal']))

        normal_style = ParagraphStyle(name='NormalStyle', parent=styles['Normal'], fontSize=12, textColor=colors.black, spaceAfter=10)
        elements.append(Paragraph(f"Crop Type: {result['crop_type']}", normal_style))
        elements.append(Paragraph(f"Disease Status: {result['disease_status']}", normal_style))
        elements.append(Paragraph(f"Disease Type: {result['disease_type']}", normal_style))
        elements.append(Paragraph(f"Unified Prediction: {result['unified_prediction']}", normal_style))
        elements.append(Paragraph(f"Confidence: {result['confidence']*100:.2f}%", normal_style))
        elements.append(Paragraph("Heatmap Explanation: The heatmap highlights areas of the leaf that the model focused on to make its prediction. Red areas indicate high focus (likely disease spots), while blue/green areas indicate lower focus.", normal_style))

        footer_style = ParagraphStyle(name='FooterStyle', parent=styles['Normal'], textColor=colors.gray, fontSize=10, alignment=1)
        elements.append(Spacer(1, 20))
        elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", footer_style))

        doc.build(elements)
        pdf_buffer.seek(0)

        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name=f"{filename}_report.pdf",
            mimetype="application/pdf"
        )
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        abort(500, description=str(e))

if __name__ == "__main__":
    print(" * Running on http://localhost:5000 (Press CTRL+C to quit)")
    app.run(host="0.0.0.0", port=5000, debug=True)