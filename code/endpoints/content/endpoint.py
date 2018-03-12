import os
import uuid

from flask import jsonify
from flask import request
from flask.views import MethodView

from werkzeug.utils import secure_filename


class ContentEndpoint(MethodView):
    """ These class creates a 'content' endpoint which is responsible for working with data at the server.

    There are two main features available:
    1) Send the file to the server via the POST request
    2) Delete the file from the server

    The class inherits from MethodView
    http://flask.pocoo.org/docs/views/
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

        curl localhost:3000/content -X POST -F audio=@./data/examples/c-dur.mp3

        The answer to this request contains the 'content_id' which is an uuid of the file at the server.
        One can address this file by given 'content_id' in all other endpoints.

        Return:
            _(flask.Response): web-response with json information about request wrapped inside
        """

        # get file from the request
        audio = request.files["audio"]
        # generate random uuid
        content_id = str(uuid.uuid4()).replace("-", "")
        # create filepath where to save file
        filepath = os.path.join(self.config["upload_folder"], content_id)
        # save file
        audio.save(filepath)
        # send successful response
        return jsonify({"status": "ok",
                        "msg": "The file has been added",
                        "result": {"content_id": content_id}})

    def delete(self, content_id):
        """ Handler for the DELETE requests.

        This method deletes the requested file from the server.
        To do that send the DELETE request to the content/<content_id> endpoint. Example:

        curl localhost:3000/content/<content_id> -X DELETE

        This command deletes the 'content_id' file from the server.

        Args:
            content_id(str): uuid of the file to delete

        Return:
            _(flask.Response): web-response with json information about request wrapped inside
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
            return jsonify({"status": "ok",
                            "msg": "The file has been deleted",
                            "result": {"content_id": content_id}})
        else:
            # send error otherwise
            return jsonify({"status": "error",
                            "msg": "No file with given 'content_id'",
                            "result": {}})
