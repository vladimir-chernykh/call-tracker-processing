from flask import Blueprint

from ..config import config
from .endpoint import PaceEndpoint

# create blueprint which is a standalone application
# it will be later included into the main flask server
# read here in more details
# http://flask.pocoo.org/docs/blueprints/
pace_blueprint = Blueprint("pace", __name__)

# create endpoint
pace_view = PaceEndpoint.as_view("pace",
                                 config=config)

# specify urls at which to serve the endpoint
pace_blueprint.add_url_rule("/pace",
                            view_func=pace_view,
                            methods=["POST"])
