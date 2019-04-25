from flask import Flask, request, jsonify
import logging
from flask_cors import CORS
import sys
import os

sys.path.append(os.path.abspath("/src/"))
from utensor.predict import Model

app = Flask(__name__)
CORS(app)

logging.basicConfig(
    filename="/src/app/api.log",
    level=logging.INFO,
    format="%(levelname)s %(asctime)s - %(message)s",
)
log = logging.getLogger()

log.info("loading model ...")
model = Model()
model.load(checkpoint_path="/src/data/")
log.info("model loaded")


@app.route("/", methods=["GET"])
def home():
    return """ 
        <h1>UmayuxLabs transformer</h1>
        <p>This is the implementation of the transformes (attention is all you need from google)
        made by UmayuxLabs.</p>
        <p>This API has an endpoint where you can retrieve a response to a text input</p>
        <p>Depending on the model you will receive a different output. This is a generic API</p>
        <h3>USAGE</h3>
        <p>
            GET: /predict/<sentence> <br>
            RETURN: {message: "...", status: 1}
        </p>
    """


@app.route("/predict/<query>", methods=["GET"])
def predict(query):
    return jsonify({"prediction": model.query(query)})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
