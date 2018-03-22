import io

import speech_recognition as sr
from scipy.io.wavfile import write as write_wav

from ..base.endpoint import BaseEndpoint


class SpeechToTextEndpoint(BaseEndpoint):
    """ This endpoint is responsible for transcribing the audio file into the text.
    """

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

        # create recognizer instance
        r = sr.Recognizer()
        # create binary stream for writing raw waveform data
        with io.BytesIO() as file:
            # write raw waveform into the format of wav file
            write_wav(file, framerate, signal)
            # open wav file
            with sr.AudioFile(file) as source:
                # read the entire wav file into appropriate library format
                audio = r.record(source)

        # calculate the duration
        duration = len(signal) / framerate

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

        # return results
        result = {"text": text}
        msg = "Transcription has successfully been done"
        return result, msg
