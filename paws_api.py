from flask import Flask, request, jsonify
from random import randint
from flask.ext.cors import CORS
from image_analyzer import analyze_image

app = Flask(__name__)
CORS(app)

@app.route("/api/v1/dog_image_categories")
def analysis():
    image_url = request.args.get('image')
    image_breed = analyze_image(image_url)
    return jsonify(breed=image_breed)

if __name__ == "__main__":
    app.run()
