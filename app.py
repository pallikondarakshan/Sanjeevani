from flask import Flask, render_template
from liver import liver_bp
from cbc import cbc_bp
from thyroid import thyroid_bp 
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.secret_key = "Groq_Api_key"
app.register_blueprint(liver_bp, url_prefix='/liver')
app.register_blueprint(cbc_bp, url_prefix='/cbc')
app.register_blueprint(thyroid_bp, url_prefix='/thyroid') 
@app.route('/hospitals')
def hospitals():
    return render_template('hospitals.html')
@app.route('/yoga')
def yoga():
    return render_template('yoga.html')    
@app.route('/vaccine')
def vaccine():
    return render_template('vaccine.html')
@app.route("/")
def home():
    return render_template("index.html")
if __name__ == "__main__":
    app.run(debug=True, port=5000)







