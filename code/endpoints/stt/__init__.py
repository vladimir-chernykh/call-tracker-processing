from flask import Blueprint

from ..config import config
from .endpoint import SpeechToTextEndpoint

# create blueprint which is a standalone application
# it will be later included into the main flask server
# read here in more details
# http://flask.pocoo.org/docs/blueprints/
stt_blueprint = Blueprint("stt", __name__)

# create endpoint
stt_view = SpeechToTextEndpoint.as_view("stt",
                                        config=config)

# specify urls at which to serve the endpoint
stt_blueprint.add_url_rule("/stt",
                           view_func=stt_view,
                           methods=["POST"])
