from flask import Blueprint

from ..config import config
from .endpoint import DurationEndpoint

# create blueprint which is a standalone application
# it will be later included into the main flask server
# read here in more details
# http://flask.pocoo.org/docs/blueprints/
duration_blueprint = Blueprint("duration", __name__)

# create endpoint
duration_view = DurationEndpoint.as_view("duration",
                                         config=config)

# specify urls at which to serve the endpoint
duration_blueprint.add_url_rule("/duration",
                                view_func=duration_view,
                                methods=["POST"])
