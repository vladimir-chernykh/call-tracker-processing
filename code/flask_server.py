import os
import argparse

from flask import Flask
from flask import jsonify
from flask import request

import librosa
from werkzeug.utils import secure_filename


application = Flask(__name__)


@application.route("/get_duration", methods=["POST"])
def get_duration():

    if request.files.get("audio"):

        audio = request.files["audio"]
        filepath = os.path.join(application.config['UPLOAD_FOLDER'], secure_filename(audio.filename))
        audio.save(filepath)

        signal, sampling_rate = librosa.load(filepath, sr=None)
        duration = librosa.get_duration(y=signal, sr=sampling_rate)

        return jsonify({"status": "ok", "data": duration})
    else:
        return jsonify({"status": "error", "data": "No audio file was passed"})


if __name__ == "__main__":

    UPLOAD_FOLDER = "temp"
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    application.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

    # argument parser from command line
    parser = argparse.ArgumentParser(add_help=True)

    # set of arguments to parse
    parser.add_argument("--port", 
                        type=int, 
                        default=3000,
                        help="port to run flask server")

    # parse arguments
    args = parser.parse_args()

    # launch flask server accessible from all hosts
    application.run(port=args.port, host="0.0.0.0", processes=2)
