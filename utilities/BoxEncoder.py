import numpy as np
import pandas as pd
import config as cfg
from dcase_util.data import DecisionEncoder
from dcase_util.data import ProbabilityEncoder

class BoxEncoder:
    """"
        Adapted after DecisionEncoder.find_contiguous_regions method in
        https://github.com/DCASE-REPO/dcase_util/blob/master/dcase_util/data/decisions.py

        Encode labels into numpy arrays where 1 correspond to presence of the class and 0 absence.
        Multiple 1 can appear on the same line, it is for multi label problem.
    Args:
        labels: list, the classes which will be encoded
        n_frames: int, (Default value = None) only useful for strong labels. The number of frames of a segment.
    Attributes:
        labels: list, the classes which will be encoded
        n_frames: int, only useful for strong labels. The number of frames of a segment.
    """

    def __init__(self, labels,  seconds, generate_patch=False):
        if type(labels) in [np.ndarray, np.array]:
            labels = labels.tolist()
        self.labels = labels
        self.seconds = seconds
        self.generate_patch = generate_patch

    def encode_unlabel(self, boxes):
        """
        Args:
            labels:  (c_list, l_list)
        Returns:

        """
        y = {}
        y["labels"] = np.asarray([0]*len(boxes))
        y["boxes"] = np.asarray(boxes)
        y["orig_size"] = np.asarray(self.seconds)
        y["patches"] = []
        return y


    def encode_weak(self, labels):
        """ Encode a list of weak labels into a numpy array

        Args:
            labels: list, list of labels to encode (to a vector of 0 and 1)

        Returns:
            numpy.array
            A vector containing 1 for each label, and 0 everywhere else
        """
        # useful for tensor empty labels
        y = {"labels": [], "boxes": [], "orig_size": []}
        if type(labels) is str:
            if labels == "empty":
                return y
            else:
                labels = labels.split(",")
        if type(labels) is pd.DataFrame:
            if labels.empty:
                labels = []
            elif "event_label" in labels.columns:
                labels = labels["event_label"]
        if isinstance(self.labels, int):
            y[labels] = len(labels) * [0]
        else:
            for label in labels:
                if not pd.isna(label):
                    i = int(self.labels.index(label))
                    y["labels"].append(i)
        y["labels"] = np.asarray(y["labels"])
        y["boxes"] = np.asarray(y["boxes"])
        y["orig_size"] = np.asarray(self.seconds)
        if self.generate_patch:
            y["patches"] = []
        return y

    def encode_strong_df(self, label_df):
        """Encode a list (or pandas Dataframe or Serie) of strong labels, they correspond to a given filename

        Args:
            label_df: pandas DataFrame or Series, contains filename, onset (in frames) and offset (in frames)
                If only filename (no onset offset) is specified, it will return the event on all the frames
                onset and offset should be in frames
        Returns:
            numpy.array
            Encoded labels, 1 where the label is present, 0 otherwise
        """
        y = {"labels": [], "boxes": [], "orig_size": []}
        assert self.seconds is not None, "n_seconds need to be specified when using strong encoder"
        if type(label_df) is str:
            if label_df == 'empty':
                pass
        elif type(label_df) is pd.DataFrame:
            if {"onset", "offset", "event_label"}.issubset(label_df.columns):
                for _, row in label_df.iterrows():
                    if not pd.isna(row["event_label"]):
                        if isinstance(self.labels, int):
                            i = 0
                        else:
                            i = int(self.labels.index(row["event_label"]))
                        y["labels"].append(i)
                        onset = float(row["onset"]) / self.seconds
                        offset = float(row["offset"]) / self.seconds
                        y["boxes"].append([(onset + offset) / 2, offset - onset])

        elif type(label_df) in [pd.Series, list, np.ndarray]:  # list of list or list of strings
            if type(label_df) is pd.Series:
                if {"onset", "offset", "event_label"}.issubset(label_df.index):  # means only one value
                    if not pd.isna(label_df["event_label"]):
                        if isinstance(self.labels, int):
                            i = 0
                        else:
                            i = int(self.labels.index(label_df["event_label"]))
                        onset = float(label_df["onset"]) / self.seconds
                        offset = float(label_df["offset"]) / self.seconds
                        y["labels"].append(i)
                        y["boxes"].append([(onset + offset) / 2, offset - onset])
                    return y

            for event_label in label_df:
                # List of string, so weak labels to be encoded in strong
                if type(event_label) is str:
                    if event_label is not "":
                        if isinstance(self.labels, int):
                            i = 0
                        else:
                            i = int(self.labels.index(event_label))
                        y["labels"].append(i)

                # List of list, with [label, onset, offset]
                elif len(event_label) == 3:
                    if event_label[0] is not "":
                        if isinstance(self.labels, int):
                            i = 0
                        else:
                            i = int(self.labels.index(event_label[0]))
                        onset = float(event_label[1]) / self.seconds
                        offset = float(event_label[2]) / self.seconds
                        y["labels"].append(i)
                        y["boxes"].append([(onset + offset) / 2, offset - onset])

                else:
                    raise NotImplementedError("cannot encode strong, type mismatch: {}".format(type(event_label)))

        else:
            raise NotImplementedError("To encode_strong, type is pandas.Dataframe with onset, offset and event_label"
                                      "columns, or it is a list or pandas Series of event labels, "
                                      "type given: {}".format(type(label_df)))
        # put events with the same class label together
        y["labels"] = np.asarray(y["labels"])
        y["boxes"] = np.asarray(y["boxes"])
        # index = y["labels"].argsort()
        # y["labels"] = y["labels"][index]
        # y["boxes"] = y["boxes"][index]
        y["orig_size"] = np.asarray(self.seconds)
        if self.generate_patch:
            y["patches"] = []
        return y

    def decode_weak(self, labels):
        """ Decode the encoded weak labels
        Args:
            labels: numpy.array, the encoded labels to be decoded

        Returns:
            list
            Decoded labels, list of string

        """
        result_labels = []
        for i, value in enumerate(labels):
            if value == 1:
                result_labels.append(self.labels[i])
        return result_labels

    def decode_strong(self, labels, threshold=0.5, del_overlap = True):
        """ Decode the encoded strong labels
        Args:
            labels: numpy.array, the encoded labels to be decoded
        Returns:
            list
            Decoded labels, list of list: [[label, onset offset], ...]

        """
        result_labels = []
        num_queries = len(labels["scores"])
        event_dict = {}
        if not del_overlap:
            for i in range(num_queries):
                if labels["scores"][i] > threshold :
                    # ignore result with duration less than 0.2s
                    onset, offset = labels['boxes'][i]
                    if offset - onset >= 0.2:
                        onset, offset = labels['boxes'][i]
                        result_labels.append([self.labels[labels["labels"][i]], onset, offset, labels["scores"][i]])
        else:
            assert not isinstance(self.labels, int), "Don't support del-overlap under self-supervision mode"
            for i in range(num_queries):
                if labels["scores"][i] >= threshold :
                    onset, offset = labels['boxes'][i]
                    # ignore result with duration less than 0.2s
                    if offset - onset >= 0.2:
                        class_index = labels["labels"][i]
                        event_dict.setdefault(self.labels[class_index], []).append(
                            np.asarray([labels['scores'][i], onset, offset]))

            # del overlap box of same class according to score
            for event in event_dict:
                event_dict[event] = np.vstack(event_dict[event])
                index = np.argsort(event_dict[event], axis=0)[:, 1]
                event_dict[event] = event_dict[event][index]
                i = 1
                while i < len(event_dict[event]):
                    if event_dict[event][i][1] < event_dict[event][i - 1][2]:
                        if event_dict[event][i][0] > event_dict[event][i - 1][0]:
                            event_dict[event] = np.delete(event_dict[event], i - 1, axis=0)
                        else:
                            event_dict[event] = np.delete(event_dict[event], i, axis=0)
                        continue
                    i += 1
                for i in event_dict[event]:
                    result_labels.append([event, i[1], i[2], i[0]])
        return result_labels

    def state_dict(self):
        return {"labels": self.labels,
                "n_frames": self.seconds}

    @classmethod
    def load_state_dict(cls, state_dict):
        labels = state_dict["labels"]
        n_frames = state_dict["n_frames"]
        return cls(labels, n_frames)
