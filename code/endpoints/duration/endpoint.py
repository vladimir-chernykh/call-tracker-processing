from ..base.endpoint import BaseEndpoint


class DurationEndpoint(BaseEndpoint):
    """ This endpoint is responsible for audio file duration calculation.
    """

    def process(self, signal, framerate, request_params):
        """ Calculates the duration of the input signal in seconds.

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

        # core processing
        duration = len(signal) * 1.0 / framerate
        # return results
        result = {"duration": duration}
        msg = "Duration has been calculated"
        return result, msg
