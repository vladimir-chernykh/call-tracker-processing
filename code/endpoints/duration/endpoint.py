import os
import librosa

from flask import jsonify
from flask import request
from flask.views import MethodView

from werkzeug.utils import secure_filename


class DurationEndpoint(MethodView):
    """
    This endpoint is responsible for calculating of the audio file duration.

    All the requests are handled via POST method.

    The class inherits from MethodView
    http://flask.pocoo.org/docs/views/
    """

    def __init__(self, config):
        """ Constructor

        Args:
            config(dict): dictionary with the common endpoints configurations
        """

        super(DurationEndpoint, self).__init__()
        self.config = config

    def post(self):
        """ Handler for the POST requests.

        Using this method users are able to POST the file for the duration calculation

        Users have two options of sending the file to the system:

        1) In the form field of the POST request. In this case the data are uploaded to the server and erased
        immediately after the calculation is done. Example:

        curl localhost:3000/duration -X POST -F audio=@./data/examples/c-dur.mp3

        2) Provide 'content_id' of the file previously submitted to the 'content' endpoint.
        In this case there is no file transmission over the network. File is not deleted after the request. Example:

        curl localhost:3000/duration -X POST -F content_id=<content_id>

        The answer to this request contains either the duration of the requested file or the error message.

        Return:
            response(flask.Response): web-response with json information about request wrapped inside
        """

        # if file is requested through the 'content_id'
        if "content_id" in request.form:

            # generate filepath to look up the file
            filepath = os.path.join(self.config["upload_folder"],
                                    secure_filename(request.form["content_id"]))

            if os.path.exists(filepath):
                # load the file if it exists
                signal, sampling_rate = librosa.load(filepath, sr=None)
                # calculate the duration
                duration = librosa.get_duration(y=signal, sr=sampling_rate)
                # send successful response with the result
                response = jsonify({"status": "ok",
                                    "msg": "Duration has been calculated",
                                    "result": {"duration": duration}})
                response.status_code = 200
                return response
            else:
                # send "file is not found" response
                response = jsonify({"status": "error",
                                    "msg": "No audio file with given 'content_id'",
                                    "result": {}})
                response.status_code = 400
                return response

        # if the file is sent via the form
        elif "audio" in request.files:

            # get file from the request
            audio = request.files["audio"]
            # create secure filepath where to temporary save file
            filepath = os.path.join(self.config["upload_folder"],
                                    secure_filename(audio.filename))
            # TODO get rid of the unnecessary IO op
            # save file temporarily
            audio.save(filepath)
            # load it back in the appropriate format
            signal, sampling_rate = librosa.load(filepath, sr=None)
            # calculate the duration
            duration = librosa.get_duration(y=signal, sr=sampling_rate)
            # remove temp file
            os.remove(filepath)
            # send successful response with the result
            response = jsonify({"status": "ok",
                                "msg": "Duration has been calculated",
                                "result": {"duration": duration}})
            response.status_code = 200
            return response

        # if file is not specified
        else:
            # send "no file" response
            response = jsonify({"status": "error",
                                "msg": "No audio file was passed",
                                "result": {}})
            response.status_code = 400
            return response
