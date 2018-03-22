import os
import json

# load common endpoint configuration file
config = json.load(open(os.path.join(os.path.dirname(__file__), "config.json")))


def verify_config():
    """ Do all the necessary sanity checks for the configuration file

    Return:
         None
    """

    # -------------------------------------- upload_folder -------------------------------------- #
    # check if the upload folder exists and create if not
    os.makedirs(config["upload_folder"], exist_ok=True)

    # --------------------------------------- wit_api_key --------------------------------------- #
    # check if the API key is a string
    assert (type(config["wit_api_key"]) == str), "WIT API key should be a string!"

    # ----------------------------------- allowed_extensions ------------------------------------ #
    # check that allowed_extensions is a list which contains only wav
    assert (type(config["allowed_extensions"]) == list), "Allowed extensions should be a list of strings"
    assert (len(config["allowed_extensions"]) == 1), "Only wav is supported by now"
    assert (config["allowed_extensions"][0] == "wav"), "Only wav is supported by now"


# do verification
verify_config()
