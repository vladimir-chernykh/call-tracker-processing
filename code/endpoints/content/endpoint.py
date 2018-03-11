import os
import uuid

from flask import jsonify
from flask import request
from flask.views import MethodView

from werkzeug.utils import secure_filename


class ContentEndpoint(MethodView):

    def __init__(self, config):

        super(ContentEndpoint, self).__init__()
        self.config = config

    def post(self):
        audio = request.files["audio"]
        content_id = str(uuid.uuid4()).replace("-", "")
        filepath = os.path.join(self.config["upload_folder"], content_id)
        audio.save(filepath)
        return jsonify({"status": "ok",
                        "msg": "The file has been added",
                        "result": {"content_id": content_id}})

    def delete(self, content_id):
        content_id = secure_filename(content_id)
        filepath = os.path.join(self.config["upload_folder"], content_id)
        if os.path.exists(filepath):
            os.remove(filepath)
            return jsonify({"status": "ok",
                            "msg": "The file has been deleted",
                            "result": {"content_id": content_id}})
        else:
            return jsonify({"status": "error",
                            "msg": "No file with given 'content_id'",
                            "result": {}})
