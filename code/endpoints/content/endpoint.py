import os
import uuid

from flask import jsonify
from flask import request
from flask.views import MethodView

from werkzeug.utils import secure_filename

from ..utils import allowed_file
from scipy.io.wavfile import read as read_wav
from scipy.io.wavfile import write as write_wav


class ContentEndpoint(MethodView):
    """ These class creates a 'content' endpoint which is responsible for working with data at the server.

    There are two main features available:
    1) Send the file to the server via the POST request
    2) Delete the file from the server

    The class inherits directly from MethodView and not BaseEndpoint.
    """

    def __init__(self, config):
        """ Constructor

        Args:
            config(dict): dictionary with the common endpoints configurations
        """

        super(ContentEndpoint, self).__init__()
        self.config = config

    def post(self):
        """ Handler for the POST requests.

        Using this method users sent files to the server. File should be attached in the form field. Example:

        curl localhost:3000/content -X POST -F audio=@data/examples/speech.wav

        The answer to this request contains the 'content_id' which is an uuid of the file at the server.
        One can address this file by given 'content_id' in all other endpoints.

        Return:
            response(flask.Response): web-response with json information about processing wrapped inside
        """

        if "audio" in request.files:
            # get file from the request
            audio = request.files["audio"]
            # if file has allowed extension
            if allowed_file(audio.filename):
                # generate random uuid
                content_id = str(uuid.uuid4()).replace("-", "")
                # create filepath where to save file
                filepath = os.path.join(self.config["upload_folder"], content_id)
                try:
                    # read file from buffer
                    framerate, signal = read_wav(audio.stream)
                    # save file to the disk
                    write_wav(filepath, framerate, signal)
                    # send successful response
                    response = jsonify({"status": "ok",
                                        "msg": "The file has been added",
                                        "result": {"content_id": content_id}})
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
                                    "msg": "Only " + str(self.config["allowed_extensions"]) + " files are allowed",
                                    "result": {}})
                response.status_code = 400
        else:
            # send "no file" response
            response = jsonify({"status": "error",
                                "msg": "No file was passed",
                                "result": {}})
            response.status_code = 400

        return response

    def delete(self, content_id):
        """ Handler for the DELETE requests.

        This method deletes the requested file from the server.
        To do that send the DELETE request to the content/<content_id> endpoint. Example:

        curl localhost:3000/content/<content_id> -X DELETE

        This command deletes the 'content_id' file from the server.

        Args:
            content_id(str): uuid of the file to delete

        Return:
            response(flask.Response): web-response with json information about request wrapped inside
        """

        # get content_id from the form and securitize it
        # securitization is needed to prevent leaks and unauthorized access to other files
        # http://flask.pocoo.org/docs/0.12/patterns/fileuploads/#a-gentle-introduction
        content_id = secure_filename(content_id)
        # create filepath where to look for the file
        filepath = os.path.join(self.config["upload_folder"], content_id)

        if os.path.exists(filepath):
            # delete if exists
            os.remove(filepath)
            # send successful response
            response = jsonify({"status": "ok",
                                "msg": "The file has been deleted",
                                "result": {"content_id": content_id}})
            response.status_code = 200
        else:
            # send error otherwise
            response = jsonify({"status": "error",
                                "msg": "No file with given 'content_id'",
                                "result": {}})
            response.status_code = 400

        return response
