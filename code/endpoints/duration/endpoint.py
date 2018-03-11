import os
import librosa

from flask import jsonify
from flask import request
from flask.views import MethodView

from werkzeug.utils import secure_filename


class DurationEndpoint(MethodView):

    def __init__(self, config):

        super(DurationEndpoint, self).__init__()
        self.config = config

    def post(self):

        if "content_id" in request.form:

            filepath = os.path.join(self.config["upload_folder"],
                                    secure_filename(request.form["content_id"]))

            if os.path.exists(filepath):
                signal, sampling_rate = librosa.load(filepath, sr=None)
                duration = librosa.get_duration(y=signal, sr=sampling_rate)
                return jsonify({"status": "ok",
                                "msg": "Duration has been calculated",
                                "result": {"duration": duration}})
            else:
                return jsonify({"status": "error",
                                "msg": "No audio file with given 'content_id'",
                                "result": {}})

        elif "audio" in request.files:

            audio = request.files["audio"]
            filepath = os.path.join(self.config["upload_folder"],
                                    secure_filename(audio.filename))
            audio.save(filepath)
            signal, sampling_rate = librosa.load(filepath, sr=None)
            duration = librosa.get_duration(y=signal, sr=sampling_rate)

            os.remove(filepath)

            return jsonify({"status": "ok",
                            "msg": "Duration has been calculated",
                            "result": {"duration": duration}})

        else:

            return jsonify({"status": "error",
                            "msg": "No audio file was passed",
                            "result": {}})
