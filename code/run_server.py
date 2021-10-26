import argparse

from flask import Flask

from endpoints.stt import stt_blueprint
from endpoints.pace import pace_blueprint
from endpoints.emotion import emotion_blueprint
from endpoints.content import content_blueprint
from endpoints.duration import duration_blueprint

from error_handlers import error_404, error_500


if __name__ == "__main__":

    # argument parser from command line
    parser = argparse.ArgumentParser(add_help=True)

    # set of arguments to parse
    parser.add_argument("--port",
                        type=int,
                        default=3000,
                        help="port to run flask server")
    parser.add_argument("--processes",
                        type=int,
                        default=1,
                        help="maximum number of concurrent processes to run flask server")

    # parse arguments
    args = parser.parse_args()

    # create Flask server
    application = Flask(__name__)

    # register endpoints in the flask 'blueprint' format
    # which is a standalone building block
    # read here in more details
    # http://flask.pocoo.org/docs/blueprints/
    application.register_blueprint(content_blueprint)
    application.register_blueprint(duration_blueprint)
    application.register_blueprint(stt_blueprint)
    application.register_blueprint(pace_blueprint)
    application.register_blueprint(emotion_blueprint)

    # register custom error handlers
    application.register_error_handler(404, error_404)
    application.register_error_handler(500, error_500)

    # launch flask server accessible from all hosts
    application.run(port=args.port, host="0.0.0.0", processes=args.processes)
