# thyroid.py

from flask import Blueprint, request, render_template, current_app
import os
import json
import httpx
import numpy as np
import re
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from docx import Document
import pytesseract
from PIL import Image
import joblib

# Define the blueprint for thyroid
thyroid_bp = Blueprint('thyroid', __name__)

# Load model
rf_model = joblib.load('models/thyroidmodel.pkl')

# Text extractors
def extract_text_from_pdf(path):
    with open(path, 'rb') as f:
        reader = PdfReader(f)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

def extract_text_from_docx(path):
    doc = Document(path)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text_from_image(path):
    img = Image.open(path)
    return pytesseract.image_to_string(img)

# Full extraction logic
def extract_thyroid_data(path):
    ext = path.rsplit('.', 1)[-1].lower()
    if ext == 'pdf':
        text = extract_text_from_pdf(path)
    elif ext == 'docx':
        text = extract_text_from_docx(path)
    elif ext in ['jpg', 'jpeg', 'png']:
        text = extract_text_from_image(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return call_llm_api(text)

# LLM + JSON parsing
def call_llm_api(text):
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise Exception("GROQ_API_KEY not set.")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        system_prompt = (
            "You are a highly accurate and reliable medical data extraction AI. Your task is to extract specific thyroid-related information "
            "from a given medical report. The parameters you need to extract are:\n\n"
            "- T3 (Triiodothyronine)\n"
            "- TT4 (Thyroxine, Free T4, or FT4)\n"
            "- TSH (Thyroid Stimulating Hormone)\n"
            "- Age (Age of the patient in years)\n\n"
            "Return only the extracted data in a valid JSON format like:\n"
            "{ \"T3\": float, \"TT4\": float, \"TSH\": float, \"Age\": int }\n"
            "Do not include explanations. Use null if data is not present."
        )

        user_prompt = f"Here is a thyroid lab report:\n{text}"

        payload = {
            "model": "llama3-8b-8192",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }

        response = httpx.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        if response.status_code != 200:
            raise Exception(f"Groq API error: {response.status_code} - {response.text}")

        reply = response.json()["choices"][0]["message"]["content"]
        print(f"Raw LLM Response: {reply}")
        match = re.search(r'\{[\s\S]*?\}', reply)
        if not match:
            raise Exception("No valid JSON found in LLM output.")

        data = json.loads(match.group(0))

        def safe_float(val):
            return float(val) if val is not None else None

        def safe_int(val):
            return int(val) if val is not None else None

        return {
            "T3": safe_float(data.get("T3")),
            "TT4": safe_float(data.get("TT4")),
            "TSH": safe_float(data.get("TSH")),
            "Age": safe_int(data.get("Age"))
        }

    except Exception as e:
        print(f"Error while processing LLM response: {str(e)}")
        return None

# Main route for thyroid-related functionality
@thyroid_bp.route("/", methods=["GET", "POST"])
def index():
    thyroid_data = {}
    age = None
    result = None
    prediction_confidence = None
    error = None

    if request.method == "POST":
        file = request.files.get("file_input")

        # Extract from uploaded file
        if file and file.filename != "":
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)  # Use current_app.config
                file.save(filepath)

                thyroid_data = extract_thyroid_data(filepath)
                if thyroid_data and "Age" in thyroid_data:
                    thyroid_data["age"] = thyroid_data.pop("Age")  # Normalize key name
            except Exception as e:
                error = "Failed to extract values from the uploaded file."
                print(f"File extraction error: {str(e)}")

        # Fallback: Manual form input
        if not thyroid_data:
            try:
                thyroid_data["T3"] = float(request.form["T3"])
                thyroid_data["TT4"] = float(request.form["TT4"])
                thyroid_data["TSH"] = float(request.form["TSH"])
                thyroid_data["age"] = float(request.form["age"])
            except Exception as e:
                error = "Unable to extract or parse thyroid data."
                print(f"Manual input error: {str(e)}")

        # Run prediction if data was collected
        if thyroid_data and not error:
            try:
                features = np.array([[thyroid_data["T3"], thyroid_data["TT4"], thyroid_data["TSH"], thyroid_data["age"]]])
                pred = rf_model.predict(features)[0]
                prediction_confidence = max(rf_model.predict_proba(features)[0]) * 100  # Prediction confidence
                
                if pred == "Positive":
                    result = f"⚠ Model predicts possible thyroid disorder. Please consult a medical professional for a thorough assessment. Confidence:{prediction_confidence:.2f}%"
                else:
                    result = f"✅ Model predicts normal thyroid function. Please consult a medical professional for a thorough assessment. Confidence: {prediction_confidence:.2f}%"
            except Exception as e:
                error = "Prediction failed. Please check your input values."

    return render_template("thyroid.html", result=result, error=error, thyroid_data=thyroid_data, age=thyroid_data.get("age"), prediction_confidence=prediction_confidence)
