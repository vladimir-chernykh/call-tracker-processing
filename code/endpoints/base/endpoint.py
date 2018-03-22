import os
from abc import ABC, ABCMeta, abstractmethod

from flask import jsonify
from flask import request
from flask.views import MethodView, MethodViewType
from werkzeug.utils import secure_filename

from scipy.io.wavfile import read as read_wav

from ..utils import allowed_file


class BaseEndpoint(MethodView, ABC,
                   metaclass=type("_MethodViewTypeABCMeta",
                                  (MethodViewType, ABCMeta),
                                  {})):
    """ This is the abstract base class for functional endpoints.
    One need to implement 'process' method which processes the signal and outputs the requires features.
    There is also a 'get_request_params' methods which allows to pass any additional arguments from the
    request to the 'process' method.

    All the requests are handled via POST method.

    The class inherits from MethodView
    http://flask.pocoo.org/docs/views/
    """

    def __init__(self, config):
        """ Constructor

        Args:
            config(dict): dictionary with the common endpoints configurations
        """

        super().__init__()
        self.config = config

    @abstractmethod
    def process(self, signal, framerate, request_params):
        """ Abstract method which implements core functionality of the endpoint.

        Args:
            signal(numpy.array): array with the raw audio waveform
            framerate(int): sampling frequency of the recording
            request_params(dict): additional parameters from the post request which
                                  comes from the 'get_request_params' method

        Return:
            _(tuple): tuple which contains two elements:
                result(dict): dictionary with the processing results
                msg(str): human-readable message about the processing
        """

        result = {"blank": 0.0}
        msg = "blank"
        return result, msg

    def get_request_params(self):
        """ Extracts additional parameters from the request for passing to the 'process' method.
        To perform the extraction one is to use request module of the flask framework.

        Return:
            request_params(dict): dictionary which contains additional parameters parsed from the request

        Example:

            if "desire" in request.files:
                request_params = {"desire": request.files["desire"].filename}
            return request_params
        """

        request_params = dict()
        return request_params

    def post(self):
        """ General handler for the POST requests.

        Using this method users are able to POST the file for the 'process' method.
        The answer to this request contains either the results of the processing or the error message.

        Users have two options of sending the file to the system:

        1) In the form field of the POST request. In this case the data are uploaded to the server and erased
        immediately after the calculation is done. Example:

        curl localhost:3000/duration -X POST -F audio=@data/examples/speech.wav

        2) Provide 'content_id' of the file previously submitted to the 'content' endpoint.
        In this case there is no file transmission over the network. File is not deleted after the request. Example:

        curl localhost:3000/duration -X POST -F content_id=<content_id>

        Return:
            response(flask.Response): web-response with json information about processing wrapped inside
        """

        # if file is requested through the 'content_id'
        if "content_id" in request.form:

            # generate filepath to look up the file
            filepath = os.path.join(self.config["upload_folder"],
                                    secure_filename(request.form["content_id"]))

            if os.path.exists(filepath):
                # read file if exists
                framerate, signal = read_wav(filepath)
                # process the data
                result, msg = self.process(signal, framerate, self.get_request_params())
                # send successful response with the result
                response = jsonify({"status": "ok",
                                    "msg": msg,
                                    "result": result})
                response.status_code = 200
            else:
                # send "file is not found" response
                response = jsonify({"status": "error",
                                    "msg": "No file with given 'content_id'",
                                    "result": {}})
                response.status_code = 400

        # if the file is sent via the form
        elif "audio" in request.files:

            # get file from the request
            audio = request.files["audio"]
            # if file has allowed extension
            if allowed_file(audio.filename):
                # read file
                try:
                    framerate, signal = read_wav(audio.stream)
                    # process data
                    result, msg = self.process(signal, framerate, self.get_request_params())
                    # send successful response with the result
                    response = jsonify({"status": "ok",
                                        "msg": msg,
                                        "result": result})
                    response.status_code = 200
                except ValueError:
                    # catch error from reading process; it might happen when passed file is not actually wav
                    # send "bad file" response
                    response = jsonify({"status": "error",
                                        "msg": "Unable to read file. Check that it is indeed one of " +
                                               str(self.config["allowed_extensions"]),
                                        "result": {}})
                    response.status_code = 400
            else:
                # send "wrong extension" response
                response = jsonify({"status": "error",
                                    "msg": "Only " + str(self.config["allowed_extensions"]) + " file are allowed",
                                    "result": {}})
                response.status_code = 400

        # if file is not specified
        else:
            # send "no file" response
            response = jsonify({"status": "error",
                                "msg": "No file was passed",
                                "result": {}})
            response.status_code = 400

        return response
