from flask import Blueprint

from ..config import config
from .endpoint import ContentEndpoint

content_blueprint = Blueprint("content", __name__)

content_view = ContentEndpoint.as_view("content",
                                       config=config)
content_blueprint.add_url_rule("/content",
                               view_func=content_view,
                               methods=["POST"])
content_blueprint.add_url_rule("/content/<string:content_id>",
                               view_func=content_view,
                               methods=["DELETE"])
