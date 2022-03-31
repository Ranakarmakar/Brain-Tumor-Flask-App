from flask import Flask, render_template, request
import os
import classify
from PIL import Image
import numpy as np

app = Flask(__name__)
port = int(os.getenv('PORT', 8000))
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']


@app.route('/', methods=['GET'])
def hello_world():
    return render_template("home.html")


@app.route('/', methods=["POST"])
def predict():
    image_file = request.files["imagefile"]
    image_path = "./images/" + image_file.filename
    image_file.save(image_path)
    image = Image.open(image_file)
    prediction = classify.predict(image)
    result = " {} with a {:.2f}% Confidence.".format(class_names[np.argmax(prediction)], 100 * np.max(prediction))

    return render_template("home.html", prediction=result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)
    #app.run(debug=True)
