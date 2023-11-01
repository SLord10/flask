from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

app = Flask(__name__)

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

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        return f'''
        <h2>Choose What to Do:</h2>
        <a href="/dominant_colors/{file.filename}">View Dominant Colors</a>
        <a href="/histogram/{file.filename}">View Histogram</a>
        '''


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

    return '''
    <h2>Color Histogram</h2>
    <img src="/uploads/{0}" alt="Histogram">
    <a href="/download_hist/{0}">Download Histogram Image</a>
    '''.format('hist_' + filename)


@app.route('/download_hist/<filename>')
def download_histogram(filename):
    hist_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
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

    return '''
    <h2>Dominant Colors</h2>
    <img src="/uploads/{0}" alt="Dominant Colors">
    <a href="/download_dominant/{0}">Download Dominant Colors Image</a>
    '''.format('dominant_' + filename)

@app.route('/download_dominant/<filename>')
def download_dominant_colors(filename):
    dominant_colors_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return send_file(dominant_colors_image_path, as_attachment=True)

if __name__ == '__main__':
    app.run()
