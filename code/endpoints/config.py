import os
import json

config = json.load(open(os.path.join(os.path.dirname(__file__), "config.json")))


def verify_config():

    # upload_folder
    os.makedirs(config["upload_folder"], exist_ok=True)


verify_config()
