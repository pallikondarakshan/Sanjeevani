import os
import json
import httpx
import numpy as np
import joblib
import pandas as pd 
import re
from flask import Blueprint, request, render_template, current_app
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from docx import Document
import pytesseract
from PIL import Image

liver_bp = Blueprint("liver_bp", __name__)
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'png', 'jpg', 'jpeg'}

liver_model = joblib.load('models/livermodel.pkl')

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

@liver_bp.route("/", methods=["GET", "POST"])
def index():
    liver_data = None
    error = None
    result = None
    prediction_confidence = None

    if request.method == "POST":
        file = request.files.get('file_input')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            liver_data = extract_liver_data(filepath)
        else:
            try:
                liver_data = {
                    "Age": float(request.form.get("Age")),
                    "Gender": float(request.form.get("Gender")),
                    "Total_Bilirubin": float(request.form.get("Total_Bilirubin")),
                    "Direct_Bilirubin": float(request.form.get("Direct_Bilirubin")),
                    "ALP": float(request.form.get("ALP")),
                    "ALT": float(request.form.get("ALT")),
                    "AST": float(request.form.get("AST")),
                    "Total_Proteins": float(request.form.get("Total_Proteins")),
                    "Albumin": float(request.form.get("Albumin")),
                    "A_G_Ratio": float(request.form.get("A_G_Ratio")),
                }
            except (ValueError, TypeError):
                error = "⚠️ Please upload a valid file or enter all values correctly."
                return render_template("liver.html", error=error)

        if liver_data:
            input_data = pd.DataFrame([liver_data])

            prediction = liver_model.predict(input_data)[0]
            prediction_prob = liver_model.predict_proba(input_data)[0]
            confidence = prediction_prob[1] if prediction == 1 else prediction_prob[0]
            prediction_confidence = f"{confidence * 100:.2f}%"

            if prediction == 1:
                result = f"⚠️ The report suggests potential indicators of liver disease. Please consult a medical professional. Confidence: {prediction_confidence}."
            else:
                result = f"✅ The model predicts normal liver function. Confidence: {prediction_confidence}. Please consult a doctor for confirmation."

            return render_template("liver.html", liver_data=liver_data, result=result)

        else:
            error = "⚠️ Could not extract required liver function values."

    return render_template("liver.html", liver_data=liver_data, result=result, error=error)

def extract_liver_data(file_path):
    file_ext = file_path.rsplit('.', 1)[-1].lower()
    if file_ext == 'pdf':
        text = extract_text_from_pdf(file_path)
    elif file_ext == 'docx':
        text = extract_text_from_docx(file_path)
    elif file_ext in ['jpg', 'jpeg', 'png']:
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
            "You are a medical data extractor. Extract these liver-related values from the lab report: "
            "Age, Gender (1 for Male, 0 for Female), Total_Bilirubin, Direct_Bilirubin, ALP, ALT, AST, "
            "Total_Proteins, Albumin, A_G_Ratio. Return **only** a valid JSON object. Example: "
            '{ "Age": 45, "Gender": 1, "Total_Bilirubin": 0.9, "Direct_Bilirubin": 0.3, "ALP": 250, '
            '"ALT": 35, "AST": 42, "Total_Proteins": 6.9, "Albumin": 3.5, "A_G_Ratio": 1.1 }'
        )

        user_prompt = f"Extract values from:\n{text}"

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
        if not match:
            raise ValueError("No JSON object found in LLM response.")

        json_str = match.group(0)

        # Clean and standardize the LLM output
        json_str = json_str.replace("'", '"')
        json_str = re.sub(r",\s*}", "}", json_str)
        json_str = re.sub(r",\s*]", "]", json_str)

        data = json.loads(json_str)

        return {
            "Age": float(data.get("Age", 0)),
            "Gender": float(data.get("Gender", 0)),
            "Total_Bilirubin": float(data.get("Total_Bilirubin", 0)),
            "Direct_Bilirubin": float(data.get("Direct_Bilirubin", 0)),
            "ALP": float(data.get("ALP", 0)),
            "ALT": float(data.get("ALT", 0)),
            "AST": float(data.get("AST", 0)),
            "Total_Proteins": float(data.get("Total_Proteins", 0)),
            "Albumin": float(data.get("Albumin", 0)),
            "A_G_Ratio": float(data.get("A_G_Ratio", 0)),
        }

    except Exception as e:
        print(f"Error during LLM processing: {e}")
        return None
