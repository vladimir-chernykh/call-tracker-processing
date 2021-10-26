import os
import sys
import csv
import json
import wave
import progressbar

import numpy as np

from sklearn.preprocessing import label_binarize

from .calculate_features import calculate_features


########################################################################################################################
#                                           data reading and processing                                                #
########################################################################################################################


def get_audio(path_to_wav, filename):
    wav = wave.open(os.path.join(path_to_wav, filename), mode="r")
    (nchannels, sampwidth, framerate, nframes, comptype, compname) = wav.getparams()
    content = wav.readframes(nframes)
    samples = np.fromstring(content, dtype="int" + str(sampwidth * 8))
    return (nchannels, sampwidth, framerate, nframes, comptype, compname), samples


def get_transcriptions(path_to_transcriptions, filename):
    f = open(os.path.join(path_to_transcriptions, filename), "r").read()
    f = np.array(f.split("\n"))
    transcription = {}
    for i in range(len(f) - 1):
        g = f[i]
        i1 = g.find(": ")
        i0 = g.find(" [")
        ind_id = g[:i0]
        ind_ts = g[i1 + 2:]
        transcription[ind_id] = ind_ts
    return transcription


def get_emotions(path_to_emotions, filename):
    f = open(os.path.join(path_to_emotions, filename), "r").read()
    f = np.array(f.split("\n"))
    idx = f == ""
    idx_n = np.arange(len(f))[idx]
    emotion = []
    for i in range(len(idx_n) - 2):
        g = f[idx_n[i]+1:idx_n[i+1]]
        head = g[0]
        i0 = head.find(" - ")
        start_time = float(head[head.find("[") + 1:head.find(" - ")])
        end_time = float(head[head.find(" - ") + 3:head.find("]")])
        actor_id = head[head.find(filename[:-4]) + len(filename[:-4]) + 1:
                        head.find(filename[:-4]) + len(filename[:-4]) + 5]
        emo = head[head.find("\t[") - 3:head.find("\t[")]
        vad = head[head.find("\t[") + 1:]

        v = float(vad[1:7])
        a = float(vad[9:15])
        d = float(vad[17:23])
        
        j = 1
        emos = []
        while g[j][0] == "C":
            head = g[j]
            start_idx = head.find("\t") + 1
            evoluator_emo = []
            idx = head.find(";", start_idx)
            while idx != -1:
                evoluator_emo.append(head[start_idx:idx].strip().lower()[:3])
                start_idx = idx + 1
                idx = head.find(";", start_idx)
            emos.append(evoluator_emo)
            j += 1

        emotion.append({"start": start_time,
                        "end": end_time,
                        "id": filename[:-4] + "_" + actor_id,
                        "v": v,
                        "a": a,
                        "d": d,
                        "emotion": emo,
                        "emo_evo": emos})
    return emotion


def split_wav(wav, emotions):
    (nchannels, sampwidth, framerate, nframes, comptype, compname), samples = wav

    left = samples[0::nchannels]
    right = samples[1::nchannels]

    frames = []
    for ie, e in enumerate(emotions):
        start = e["start"]
        end = e["end"]

        e["right"] = right[int(start * framerate):int(end * framerate)]
        e["left"] = left[int(start * framerate):int(end * framerate)]

        frames.append({"left": e["left"], "right": e["right"]})
    
    return frames


def read_iemocap_data(params):
    data = []
    sessions = os.listdir(params["path_to_sessions"])
    for session in sessions:
        path_to_wav = os.path.join(params["path_to_sessions"], session, "dialog/wav/")
        path_to_emotions = os.path.join(params["path_to_sessions"], session, "dialog/EmoEvaluation/")
        path_to_transcriptions = os.path.join(params["path_to_sessions"], session, "dialog/transcriptions/")

        files = os.listdir(path_to_wav)
        files = [f[:-4] for f in files if f.endswith(".wav") and (f[7] == params["session_type"][0] or
                                                                  params["session_type"] == "all")]
        for f in files:           
            wav = get_audio(path_to_wav, f + ".wav")
            transcriptions = get_transcriptions(path_to_transcriptions, f + ".txt")
            emotions = get_emotions(path_to_emotions, f + ".txt")
            sample = split_wav(wav, emotions)

            for ie, e in enumerate(emotions):
                e["signal"] = sample[ie]["left"]
                e.pop("left", None)
                e.pop("right", None)
                e["transcription"] = transcriptions[e["id"]]
                e["framerate"] = wav[0][2]
                if e["emotion"] in params["emotions"]:
                    data.append(e)
    sort_key = get_field(data, "id")
    return np.array(data)[np.argsort(sort_key)]


def get_field(data, key):
    return np.array([e[key] for e in data])


########################################################################################################################
#                                          features generation and processing                                          #
########################################################################################################################


def get_features(data, params):
    frames_excluded = 0
    frames_included = 0
    bar = progressbar.ProgressBar()
    for d in bar(data):
        st_features = calculate_features(d["signal"], d["framerate"], None).T
        x = []
        y = []
        for f in st_features:
            if f[1] > 1.e-4:
                frames_included += 1
                x.append(f)
                y.append(d["emotion"])
            else:
                frames_excluded += 1
        x = np.array(x, dtype=float)
        y = np.array(y)
        save_sample(x, y, os.path.join(params["path_to_features"], d["id"] + ".csv"))
    with open(os.path.join(params["path_to_features"], "emotions.info"), "w") as f:
        json.dump({"emotions": params["emotions"], 
                   "session_type": params["session_type"], 
                   "frames_included": frames_included, 
                   "frames_excluded": frames_excluded}, 
                  f)
    return frames_included, frames_excluded


def save_sample(x, y, name):
    with open(name, "w") as csvfile:
        w = csv.writer(csvfile, delimiter=",")
        for i in range(x.shape[0]):
            row = x[i, :].tolist()
            row.append(y[i])
            w.writerow(row)


def load_sample(name):
    with open(name, "r") as csvfile:
        r = csv.reader(csvfile, delimiter=",")
        x = []
        y = []
        for row in r:
            x.append(row[:-1])
            y.append(row[-1])
    return np.array(x, dtype=float), np.array(y)


def get_sample(params):
    files = os.listdir(params["path_to_features"])
    ids = np.sort([f[:-4] for f in files if f.endswith(".csv")])

    tx = []
    ty = []
    proper_ids = []

    for i in ids:
        x, y = load_sample(os.path.join(params["path_to_features"], i + ".csv"))
        if len(x) > 0:
            tx.append(np.array(x, dtype=float))
            ty.append(y[0])
            proper_ids.append(i)

    return np.array(tx), np.array(ty), np.array(proper_ids)


########################################################################################################################
#                                                 general helpers                                                      #
########################################################################################################################


def to_categorical(y, classes):
    binarized = label_binarize(y, classes)
    if len(classes) == 2:
        return np.hstack((binarized, 1 - binarized))
    else:
        return binarized


def pad_sequence_into_array(Xs, maxlen=None, truncating="post", padding="pre", value=0.):
    """
    Padding sequence (list of numpy arrays) into an numpy array
    :param Xs: list of numpy arrays. The arrays must have the same shape except the first dimension.
    :param maxlen: the allowed maximum of the first dimension of Xs's arrays. Any array longer than maxlen is truncated to maxlen
    :param truncating: = 'pre'/'post', indicating whether the truncation happens at either the beginning or the end of the array (default)
    :param padding: = 'pre'/'post',indicating whether the padding happens at either the beginning or the end of the array (default)
    :param value: scalar, the padding value, default = 0.0
    :return: Xout, the padded sequence (now an augmented array with shape (Narrays, N1stdim, N2nddim, ...)
    :return: mask, the corresponding mask, binary array, with shape (Narray, N1stdim)
    """
    Nsamples = len(Xs)
    if maxlen is None:
        lengths = [s.shape[0] for s in Xs]    # 'sequences' must be list, 's' must be numpy array, len(s) return the first dimension of s
        maxlen = np.max(lengths)

    Xout = np.ones(shape=[Nsamples, maxlen] + list(Xs[0].shape[1:]), dtype=Xs[0].dtype) * np.asarray(value, dtype=Xs[0].dtype)
    Mask = np.zeros(shape=[Nsamples, maxlen], dtype=Xout.dtype)
    for i in range(Nsamples):
        x = Xs[i]
        if truncating == "pre":
            trunc = x[-maxlen:]
        elif truncating == "post":
            trunc = x[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % truncating)
        if padding == "post":
            Xout[i, :len(trunc)] = trunc
            Mask[i, :len(trunc)] = 1
        elif padding == "pre":
            Xout[i, -len(trunc):] = trunc
            Mask[i, -len(trunc):] = 1
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return Xout, Mask

