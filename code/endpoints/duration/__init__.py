from flask import Blueprint

from ..config import config
from .endpoint import DurationEndpoint

duration_blueprint = Blueprint("duration", __name__)

duration_view = DurationEndpoint.as_view("duration",
                                         config=config)
duration_blueprint.add_url_rule("/duration",
                                view_func=duration_view,
                                methods=["POST"])
