from flask import request

from ..base.endpoint import BaseEndpoint
from ..stt.endpoint import SpeechToTextEndpoint
from ..duration.endpoint import DurationEndpoint


class PaceEndpoint(BaseEndpoint):
    """ This endpoint is responsible for transcribing the audio file into the text.
    """

    def get_request_params(self):

        request_params = dict()
        if "text" in request.form:
            request_params["text"] = request.form["text"]

        return request_params

    def process(self, signal, framerate, request_params):
        """ Perform speech-to-text transcription from the audio sample.
        For now it uses https://wit.ai API. In future the transition to the custom recognizer will be done.
        Note that the output may contain <ERROR> parts if some chunks of the audio file were not recognized.

        Args:
            signal(numpy.array): array with the raw audio waveform
            framerate(int): sampling frequency of the recording
            request_params(dict): additional parameters from the post request which
                                  comes from the 'get_request_params' method

        Return:
            _(tuple): tuple which contains two elements:
                result(dict): dictionary with the processing results
                msg(str): human-readable message about the processing
        """

        duration = DurationEndpoint(self.config).process(signal, framerate, request_params)[0]["duration"]
        if "text" in request_params:
            text = request_params["text"]
        else:
            text = SpeechToTextEndpoint(self.config).process(signal, framerate, request_params)[0]["text"]

        pace = len(text.split(" ")) * 60.0 / duration

        result = {"pace": pace}
        msg = "Pace of speech has successfully been calculated"
        return result, msg
