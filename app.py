#imports
from flask import Flask, render_template, request, flash, redirect, send_file
from backend import predict, Transformer_deskew
from skimage.io import imread
import json 
import os

#file extensions allowed
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
#flask app secret code
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

def allowed_file(filename):
    """Function to check the file extension.

    Args:
        filename : The first parameter.

    Returns:
        bool: boolean of the check

    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#initialisation of the first route (home page)
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

#route which actives the prediction
@app.route('/api', methods=['GET', 'POST'])
def inference():
    """Function to realize the prediction on selected files.

    Args:
        None

    Returns:
        json file: predictions contained in the json file

    """
    if request.method == 'POST':
        #check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect('/')

        #get back the uploaded file
        uploaded_files = request.files.getlist("file")
        data = {}

        #check if the file is no empty or with a wrong extension and realize prediction by calling the model (function predict imported)
        for file in uploaded_files:
            if file.filename == '':
                flash('No selected file', 'error')
                return redirect('/')
            if file and allowed_file(file.filename):
                x = imread(file)
                y = predict(x)
                data[f"{file.filename}"] =  str(y[0])
            else:
                flash("Mauvaise extension de fichier (allowed : 'png', 'jpg', 'jpeg')", 'error')
                return redirect('/')

        #check if a prediction has already been created and delete it
        if os.path.isfile("./predictions/prediction.json"):
            os.remove("./predictions/prediction.json")

        #write the prediction in a new json file
        with open("./predictions/prediction.json", "a") as f:
            json.dump(data, f)
        return send_file("./predictions/prediction.json", as_attachment=True)
            

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
