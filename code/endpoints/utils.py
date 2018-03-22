from .config import config


def allowed_file(filename):
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in config["allowed_extensions"]
