import base64

from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from flask_cors import CORS

app = Flask(__name__)
#CORS(app)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.debug = True

@app.route('/')
def index():
    return '''
    <form method="POST" action="/upload" enctype="multipart/form-data">
        <input type="file" name="file">
        <input type="submit" value="Upload">
    </form>
    '''

@app.route('/upload', methods=['POST','GET'])
def upload():
    data = request.get_json()
    if 'base64_image' not in data:
        return 'No base64 image provided'

    base64_image = data['base64_image']

    try:
        # Decode the base64 image data
        image_data = base64.b64decode(base64_image)

        # Specify the upload folder and file name
        upload_folder = app.config['UPLOAD_FOLDER']
        filename = os.path.join(upload_folder, 'uploaded_image.jpeg')

        # Save the image to the specified file
        with open(filename, 'wb') as f:
            f.write(image_data)

        return f'''
            <h2>Choose What to Do:</h2>
            <a href="/dominant_colors/{filename}">View Dominant Colors</a>
            <a href="/histogram/{filename}">View Histogram</a>
            '''

    except Exception as e:
        return f'Error: {str(e)}'


@app.route('/histogram/<filename>')
def histogram_route(filename):
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img = cv2.imread(img_path, 1)
    chans = cv2.split(img)
    colors = ("b", "g", "r")
    plt.figure()
    plt.title("Color Histogram")
    plt.xlabel("Bins")
    plt.ylabel("Number of Pixels")

    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])

    hist_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'hist_' + filename)
    plt.savefig(hist_image_path)
    hist_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'hist_' + filename)
    return send_file(hist_image_path, as_attachment=True)


@app.route('/dominant_colors/<filename>')
def dominant_colors_route(filename):
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img = cv2.imread(img_path, 1)

    # Resize the image
    width = 50  # Change this to your desired width
    ratio = img.shape[0] / img.shape[1]
    height = int(img.shape[1] * ratio)
    dim = (width, height)
    img = cv2.resize(img, dim)

    # Number of dominant colors
    nbreDominantColors = 10  # Change this to the desired number of dominant colors

    # Reshape the image
    examples = img.reshape((img.shape[0] * img.shape[1], 3))

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=nbreDominantColors)
    kmeans.fit(examples)

    # Get the cluster centers representing dominant colors
    colors = kmeans.cluster_centers_.astype(int)

    # Create an image to display the dominant colors
    barColorW = 75
    barColorH = 50
    imgr = np.zeros((barColorH, barColorW * nbreDominantColors, 3), dtype=np.uint8)

    for i in range(nbreDominantColors):
        cv2.rectangle(imgr, (i * barColorW, 0), ((i + 1) * barColorW, barColorH), [int(x) for x in colors[i]], -1)

    str_ = "Dominant Colors: " + str(nbreDominantColors)


    # Save the dominant color image
    dominant_colors_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'dominant_' + filename)
    cv2.imwrite(dominant_colors_image_path, imgr)
    dominant_colors_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'dominant_' +filename)
    return send_file(dominant_colors_image_path, as_attachment=True)

if __name__ == '__main__':
    app.run()
