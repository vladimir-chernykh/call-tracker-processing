from flask import Blueprint

from ..config import config
from .endpoint import EmotionEndpoint

# create blueprint which is a standalone application
# it will be later included into the main flask server
# read here in more details
# http://flask.pocoo.org/docs/blueprints/
emotion_blueprint = Blueprint("emotion", __name__)

# create endpoint
emotion_view = EmotionEndpoint.as_view("emotion",
                                       config=config)

# specify urls at which to serve the endpoint
emotion_blueprint.add_url_rule("/emotion",
                               view_func=emotion_view,
                               methods=["POST"])
