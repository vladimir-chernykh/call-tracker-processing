import os

import speech_recognition as sr

from flask import jsonify
from flask import request
from flask.views import MethodView

from werkzeug.utils import secure_filename


class SpeechToTextEndpoint(MethodView):
    """
    This endpoint is responsible for transcribing the audio file into the text.

    All the requests are handled via POST method.

    The class inherits from MethodView
    http://flask.pocoo.org/docs/views/
    """

    def __init__(self, config):
        """ Constructor

        Args:
            config(dict): dictionary with the common endpoints configurations
        """

        super(SpeechToTextEndpoint, self).__init__()
        self.config = config

    def _recognize(self, file):
        """ Perform speech-to-text transcription from the audio sample.
        For now it uses https://wit.ai API. In future the transition to the custom recognizer will be done.
        Note that the output may contain <ERROR> parts if some chunks of the audio file were not recognized.

        Args:
            file(str or file-like object): if file is a str, then it is interpreted as a path to an audio file.
                                           Otherwise, file should be a file-like object such as io.BytesIO or similar.

        Return:
            text(str): recognized text
        """

        # create recognizer instance
        r = sr.Recognizer()
        # open file
        with sr.AudioFile(file) as source:
            # read the entire audio file
            audio = r.record(source)

        # calculate the duration
        duration = len(audio.frame_data) / audio.sample_width / audio.sample_rate

        # array to store chunk-by-chunk recognized text
        text = []

        # TODO: parallelize using multiprocessing.Pool

        # split the input audio file into the chunks of 15 seconds length
        # it is the restriction of wit.ai (having no more than 255 symbols in an output)
        for start_ms in range(0, int(duration * 1000), 15000):
            # crop the segment
            _audio_part = audio.get_segment(start_ms=start_ms, end_ms=start_ms + 15000)
            # send a request
            try:
                _text_part = r.recognize_wit(_audio_part, key=self.config["wit_api_key"])
            except sr.UnknownValueError:
                _text_part = "<RECOGNITION ERROR>"
            except sr.RequestError as e:
                _text_part = "<REQUEST ERROR: {0}>".format(e)
            # add recognized text
            text.append(_text_part)
        # join texts of all chunks
        text = " ".join(text)

        return text

    def post(self):
        """ Handler for the POST requests.

        Using this method users are able to POST the file for the speech to text transcription.

        Users have two options of sending the file to the system:

        1) In the form field of the POST request. In this case the data are uploaded to the server and erased
        immediately after the calculation is done. Example:

        curl localhost:3000/duration -X POST -F audio=@data/examples/speech.wav

        2) Provide 'content_id' of the file previously submitted to the 'content' endpoint.
        In this case there is no file transmission over the network. File is not deleted after the request. Example:

        curl localhost:3000/duration -X POST -F content_id=<content_id>

        The answer to this request contains either the recognized text of the requested file or the error message.

        Return:
            response(flask.Response): web-response with json information about request wrapped inside
        """

        # if file is requested through the 'content_id'
        if "content_id" in request.form:

            # generate filepath to look up the file
            filepath = os.path.join(self.config["upload_folder"],
                                    secure_filename(request.form["content_id"]))

            if os.path.exists(filepath):
                # perform speech-to-text if file exists
                text = self._recognize(filepath)
                # send successful response
                response = jsonify({"status": "ok",
                                    "msg": "Transcription has successfully been done",
                                    "result": {"text": text}})
                response.status_code = 200
                return response
            else:
                # send "file is not found" response
                response = jsonify({"status": "error",
                                    "msg": "No audio file with given 'content_id'",
                                    "result": {}})
                response.status_code = 400
                return response

        # if the file is sent via the form
        elif "audio" in request.files:

            # get file from the request
            audio = request.files["audio"]

            # perform speech-to-text
            text = self._recognize(audio)
            # send successful response
            response = jsonify({"status": "ok",
                                "msg": "Transcription has successfully been done",
                                "result": {"text": text}})
            response.status_code = 200
            return response

        # if file is not specified
        else:
            # send "no file" response
            response = jsonify({"status": "error",
                                "msg": "No audio file was passed",
                                "result": {}})
            response.status_code = 400
            return response
