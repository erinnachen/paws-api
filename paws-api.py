from flask import Flask, request, jsonify
from random import randint
app = Flask(__name__)

@app.route("/api/v1/dog_image_categories")
def analysis():
    breeds = ["German Shepherd", "Chihuahua", "Boston Terrier", "Siberian Husky", "Labrador Retriever", "Golden Retriever", "Poodle"]
    return jsonify(breed=breeds[randint(0, len(breeds)-1)])

if __name__ == "__main__":
    app.run()
