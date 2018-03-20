from flask import jsonify


def error_404(error):
    response = jsonify({"status": "error",
                        "msg": "The requested URL was not found on the server",
                        "result": {}})
    response.status_code = 404
    return response


def error_500(error):
    response = jsonify({"status": "error",
                        "msg": "The server encountered an internal error and was unable to complete your request. " +
                               "Either the server is overloaded or there is an error in the application.",
                        "result": {}})
    response.status_code = 500
    return response
