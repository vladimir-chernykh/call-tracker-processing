import numpy as np

from ..base.endpoint import BaseEndpoint

from .utilities.BlstmCTC import BlstmCTC
from .utilities.calculate_features import calculate_features


class EmotionEndpoint(BaseEndpoint):
    """ This endpoint is responsible for transcribing the audio file into the text.
    """

    def __init__(self, config):

        super().__init__(config)
        self.model = BlstmCTC(modelname="ctc", modelpath="./code/endpoints/emotion/models")

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

        _features = calculate_features((signal - signal.mean()) / signal.std(), framerate, None).T

        _class = self.model.predict(np.array([_features]))[0][0]
        if _class == 0:
            emotion = "neutral"
        else:
            emotion = "angry"

        result = {"emotion": emotion}
        msg = "Emotional background of speech has successfully been detected"
        return result, msg
