from flask import Blueprint

from ..config import config
from .endpoint import ContentEndpoint

# create blueprint which is a standalone application
# it will be later included into the main flask server
# read here in more details
# http://flask.pocoo.org/docs/blueprints/
content_blueprint = Blueprint("content", __name__)

# create endpoint
content_view = ContentEndpoint.as_view("content",
                                       config=config)

# specify urls at which to serve the endpoint
content_blueprint.add_url_rule("/content",
                               view_func=content_view,
                               methods=["POST"])
content_blueprint.add_url_rule("/content/<string:content_id>",
                               view_func=content_view,
                               methods=["DELETE"])
