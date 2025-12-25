import os
import json
import httpx
import numpy as np
import joblib
import re
from werkzeug.utils import secure_filename
from flask import Blueprint, request, render_template, flash, current_app
from PyPDF2 import PdfReader
from docx import Document
import pytesseract
from PIL import Image

cbc_bp = Blueprint('cbc_bp', __name__, template_folder='templates')

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'png', 'jpg', 'jpeg'}

cbc_model = joblib.load('models/cbcmodel.pkl')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        return ''.join([page.extract_text() for page in reader.pages])

def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    return '\n'.join(para.text for para in doc.paragraphs)

def extract_text_from_image(image_path):
    return pytesseract.image_to_string(Image.open(image_path))

@cbc_bp.route("/", methods=["GET", "POST"])
def index():
    cbc_data = None
    error = None
    result = None

    if request.method == "POST":
        file = request.files.get('file_input')
        
        # Handle file upload
        if file and file.filename != '':
            if allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)

                cbc_data = extract_cbc_data(filepath)

                if cbc_data:
                    input_data = np.array([[cbc_data[param] for param in ['WBC', 'RBC', 'HGB', 'MCV', 'MCH', 'MCHC', 'PLT']]])
                    prediction = cbc_model.predict(input_data)[0]

                    label_names = ["Bacterial_Infection", "Iron_Def_Anemia", "Thrombocytosis"]
                    flagged_conditions = [label_names[i] for i, val in enumerate(prediction[:3]) if val == 1]

                    if flagged_conditions:
                        result = f"Based on your blood values, there are potential indicators of: {', '.join(flagged_conditions)}. Please consult a healthcare professional."
                    else:
                        result = "There are no clear hematological flags. However, this does not replace a medical checkup."

                else:
                    error = "Could not extract required CBC values from the report."

        # Handle manual input
        else:
            try:
                wbc = float(request.form.get('WBC'))
                rbc = float(request.form.get('RBC'))
                hgb = float(request.form.get('HGB'))
                mcv = float(request.form.get('MCV'))
                mch = float(request.form.get('MCH'))
                mchc = float(request.form.get('MCHC'))
                plt = float(request.form.get('PLT'))

                cbc_data = {
                    'WBC': wbc,
                    'RBC': rbc,
                    'HGB': hgb,
                    'MCV': mcv,
                    'MCH': mch,
                    'MCHC': mchc,
                    'PLT': plt
                }

                input_data = np.array([[wbc, rbc, hgb, mcv, mch, mchc, plt]])
                prediction = cbc_model.predict(input_data)[0]

                label_names = ["Bacterial_Infection", "Iron_Def_Anemia", "Thrombocytosis"]
                flagged_conditions = [label_names[i] for i, val in enumerate(prediction[:3]) if val == 1]

                if flagged_conditions:
                    result = f"Based on your blood values, there are potential indicators of: {', '.join(flagged_conditions)}. Please consult a healthcare professional."
                else:
                    result = "There are no clear hematological flags. However, this does not replace a medical checkup."

            except Exception as e:
                error = f"Error in manual input: {str(e)}"

    return render_template('cbc.html', error=error, cbc_data=cbc_data, result=result)

def extract_cbc_data(file_path):
    ext = file_path.rsplit('.', 1)[-1].lower()
    if ext == 'pdf':
        text = extract_text_from_pdf(file_path)
    elif ext == 'docx':
        text = extract_text_from_docx(file_path)
    elif ext in ['jpg', 'jpeg', 'png']:
        text = extract_text_from_image(file_path)
    else:
        return None
    return call_llm_api(text)

def call_llm_api(text):
    try:
        api_key = os.getenv("GROQ_API_KEY")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        system_prompt = (
            "You are a medical data extractor. Extract the following CBC parameters from text: "
            "WBC (in /cumm), RBC (millions/cumm), HGB (g/dL), MCV (fL), MCH (pg), MCHC (g/dL), PLT (per cumm). "
            "Respond in JSON format with float values. Return null for any missing value."
        )

        user_prompt = f"Extract these CBC values from the report:\n{text}"

        payload = {
            "model": "llama3-8b-8192",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }

        response = httpx.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        reply = response.json()["choices"][0]["message"]["content"]
        match = re.search(r'\{[\s\S]*?\}', reply)
        data = json.loads(match.group(0))

        return {
            "WBC": float(data.get("WBC", 0)),
            "RBC": float(data.get("RBC", 0)),
            "HGB": float(data.get("HGB", 0)),
            "MCV": float(data.get("MCV", 0)),
            "MCH": float(data.get("MCH", 0)),
            "MCHC": float(data.get("MCHC", 0)),
            "PLT": float(data.get("PLT", 0)),
        }

    except Exception as e:
        print(f"LLM error: {e}")
        return None
