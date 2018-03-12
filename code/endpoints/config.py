import os
import json

# load common endpoint configuration file
config = json.load(open(os.path.join(os.path.dirname(__file__), "config.json")))


def verify_config():
    """ Do all the necessary sanity checks for the configuration file

    Return:
         None
    """

    # ------------------- upload_folder ------------------- #
    # check if the upload folder exists and create if not
    os.makedirs(config["upload_folder"], exist_ok=True)


# do verification
verify_config()
