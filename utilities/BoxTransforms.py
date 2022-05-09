import warnings

import librosa
import numpy as np
import torch
from torchvision import transforms
from PIL import ImageFilter
import random

class Transform:
    def transform_data(self, data):
        # Mandatory to be defined by subclasses
        raise NotImplementedError("Abstract object")

    def transform_label(self, label):
        # Do nothing, to be changed in subclasses if needed
        return label

    def _apply_transform(self, sample_no_index):
        data, label = sample_no_index
        if type(data) is tuple:  # meaning there is more than one data_input (could be duet, triplet...)
            data = list(data)
            for k in range(len(data)):
                if (type(self).__name__ == "TimeMask"):
                    if (k == 0):
                        continue
                data[k] = self.transform_data(data[k])
            data = tuple(data)
        else:
            data = self.transform_data(data)
        if self.__class__.__name__ == 'Query':
            data, label = self.transform_label(sample_no_index)
        else:
            label = self.transform_label(label)
        return data, label

    def __call__(self, sample):
        """ Apply the transformation
        Args:
            sample: tuple, a sample defined by a DataLoad class

        Returns:
            tuple
            The transformed tuple
        """
        if type(sample[1]) is int:  # Means there is an index, may be another way to make it cleaner
            sample_data, index = sample
            sample_data = self._apply_transform(sample_data)
            sample = sample_data, index
        else:
            sample = self._apply_transform(sample)
        return sample


class ApplyLog(Transform):
    """Convert ndarrays in sample to Tensors."""

    def transform_data(self, data):
        """ Apply the transformation on data
        Args:
            data: np.array, the data to be modified

        Returns:
            np.array
            The transformed data
        """
        return librosa.amplitude_to_db(data.T).T


def pad_trunc_seq(x, max_len):
    """Pad or truncate a sequence data to a fixed length.
    The sequence should be on axis -2.

    Args:
      x: ndarray, input sequence data.
      max_len: integer, length of sequence to be padded or truncated.

    Returns:
      ndarray, Padded or truncated input sequence data.
    """
    shape = x.shape
    if shape[-2] <= max_len:
        padded = max_len - shape[-2]
        padded_shape = ((0, 0),) * len(shape[:-2]) + ((0, padded), (0, 0))
        x = np.pad(x, padded_shape, mode="constant")
    else:
        x = x[..., :max_len, :]
    return x


class PadOrTrunc(Transform):
    """ Pad or truncate a sequence given a number of frames
    Args:
        nb_frames: int, the number of frames to match
    Attributes:
        nb_frames: int, the number of frames to match
    """

    def __init__(self, nb_frames, apply_to_label=False):
        self.nb_frames = nb_frames
        self.apply_to_label = apply_to_label

    def transform_label(self, label):
        if self.apply_to_label:
            return pad_trunc_seq(label, self.nb_frames)
        else:
            return label

    def transform_data(self, data):
        """ Apply the transformation on data
        Args:
            data: np.array, the data to be modified

        Returns:
            np.array
            The transformed data
        """
        return pad_trunc_seq(data, self.nb_frames)


class AugmentGaussianNoise(Transform):
    """ Pad or truncate a sequence given a number of frames
           Args:
               mean: float, mean of the Gaussian noise to add
           Attributes:
               std: float, std of the Gaussian noise to add
           """

    def __init__(self, mean=0., std=None, snr=None, p=0.5):
        self.mean = mean
        self.std = std
        self.snr = snr
        self.p = p

    @staticmethod
    def gaussian_noise(features, snr):
        """Apply gaussian noise on each point of the data

            Args:
                features: numpy.array, features to be modified
                snr: float, average snr to be used for data augmentation
            Returns:
                numpy.ndarray
                Modified features
                """
        # If using source separation, using only the first audio (the mixture) to compute the gaussian noise,
        # Otherwise it just removes the first axis if it was an extended one
        if len(features.shape) == 3:
            feat_used = features[0]
        else:
            feat_used = features
        std = np.sqrt(np.mean((feat_used ** 2) * (10 ** (-snr / 10)), axis=-2))
        try:
            noise = np.random.normal(0, std, features.shape)
        except Exception as e:
            warnings.warn(f"the computed noise did not work std: {std}, using 0.5 for std instead")
            noise = np.random.normal(0, 0.5, features.shape)

        return features + noise

    def transform_data(self, data):
        """ Apply the transformation on data
            Args:
                data: np.array, the data to be modified

            Returns:
                (np.array, np.array)
                (original data, noisy_data (data + noise))
        """
        random_num = np.random.uniform(0, 1)
        if random_num < self.p:
            if self.std is not None:
                noisy_data = data + np.abs(np.random.normal(0, 0.5 ** 2, data.shape))
            elif self.snr is not None:
                noisy_data = self.gaussian_noise(data, self.snr)
            else:
                raise NotImplementedError("Only (mean, std) or snr can be given")
            return data, noisy_data
        else:
            return data, data


class ToTensor(Transform):
    """Convert ndarrays in sample to Tensors.
    Args:
        unsqueeze_axis: int, (Default value = None) add an dimension to the axis mentioned.
            Useful to add a channel axis to use CNN.
    Attributes:
        unsqueeze_axis: int, add an dimension to the axis mentioned.
            Useful to add a channel axis to use CNN.
    """

    def __init__(self, unsqueeze_axis=None):
        self.unsqueeze_axis = unsqueeze_axis

    def transform_data(self, data):
        """ Apply the transformation on data
            Args:
                data: np.array, the data to be modified

            Returns:
                np.array
                The transformed data
        """
        res_data = torch.from_numpy(data).float()
        if self.unsqueeze_axis is not None:
            res_data = res_data.unsqueeze(self.unsqueeze_axis)
        return res_data

    def transform_label(self, label):
        label["labels"] = torch.from_numpy(label["labels"]).long()
        label["boxes"] = torch.from_numpy(label["boxes"]).float()
        label["orig_size"] = torch.from_numpy(label["orig_size"])
        return label  # float otherwise error


class Normalize(Transform):
    """Normalize inputs
    Args:
        scaler: Scaler object, the scaler to be used to normalize the data
    Attributes:
        scaler : Scaler object, the scaler to be used to normalize the data
    """

    def __init__(self, scaler):
        self.scaler = scaler

    def transform_data(self, data):
        """ Apply the transformation on data
            Args:
                data: np.array, the data to be modified

            Returns:
                np.array
                The transformed data
        """
        return self.scaler.normalize(data)


class CombineChannels(Transform):
    """ Combine channels when using source separation (to remove the channels with low intensity)
       Args:
           combine_on: str, in {"max", "min"}, the channel in which to combine the channels with the smallest energy
           n_channel_mix: int, the number of lowest energy channel to combine in another one
   """

    def __init__(self, combine_on="max", n_channel_mix=2):
        self.combine_on = combine_on
        self.n_channel_mix = n_channel_mix

    def transform_data(self, data):
        """ Apply the transformation on data
            Args:
                data: np.array, the data to be modified, assuming the first values are the mixture,
                    and the other channels the sources

            Returns:
                np.array
                The transformed data
        """
        mix = data[:1]  # :1 is just to keep the first axis
        sources = data[1:]
        channels_en = (sources ** 2).sum(-1).sum(-1)  # Get the energy per channel
        indexes_sorted = channels_en.argsort()
        sources_to_add = sources[indexes_sorted[:2]].sum(0)
        if self.combine_on == "min":
            sources[indexes_sorted[2]] += sources_to_add
        elif self.combine_on == "max":
            sources[indexes_sorted[-1]] += sources_to_add
        return np.concatenate((mix, sources[indexes_sorted[2:]]))


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms: list of ``Transform`` objects, list of transforms to compose.
        Example of transform: ToTensor()
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def add_transform(self, transform):
        t = self.transforms.copy()
        t.append(transform)
        return Compose(t)

    def __call__(self, audio):
        for t in self.transforms:
            audio = t(audio)
        return audio

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'

        return format_string


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Query(Transform):
    def __init__(self, fixed_patch_size=False):
        self.fixed_patch_size = fixed_patch_size
        self.transformer = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 64)),
            # transforms.RandomApply([
            #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            # ], p=0.8),
            # transforms.RandomGrayscale(p=0.2),
            # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.ToTensor()
        ])

    def transform_data(self, data):
        return data

    def transform_label(self, sample):
        data, label = sample
        if "patches" not in label:
            return data, label
        c, t, f = data.shape
        assert "boxes" in label, "there are no 'boxes' in label, please check your data"
        patches = []
        for box in label['boxes']:
            c, l = box.numpy()
            s, e = c - l / 2, c + l / 2
            s_idx, e_idx = int(s * t), int(e * t)
            if self.fixed_patch_size:
                e_idx = min(t, s_idx + 128)
                s_idx = e_idx - 128
                patch_t = data[:, s_idx:e_idx, :]
            else:
                # make sure patch is not empty
                if s_idx >= e_idx:
                    s_idx = max(0, s_idx - 1)
                    e_idx = min(t, e_idx + 1)
                patch_ori = data[:, s_idx:e_idx, :]
                # map to [0,1]
                min_v, max_v = patch_ori.min(), patch_ori.max()
                patch_norm = (patch_ori - min_v) / (max_v - min_v)
                patch_norm_t = self.transformer(patch_norm)
                patch_t = patch_norm_t * (max_v - min_v) + min_v
            patches.append(patch_t)
        label["patches"] = torch.stack(patches, dim=0)
        return data, label


class TimeMask(Transform):
    def __init__(self, min_band_part=0.0, max_band_part=0.1, fade=False, p=0.2):
        """
        :param min_band_part: Minimum length of the silent part as a fraction of the
            total sound length. Float.
        :param max_band_part: Maximum length of the silent part as a fraction of the
            total sound length. Float.
        :param fade: Bool, Add linear fade in and fade out of the silent part.
        :param p: The probability of applying this transform
        """
        self.min_band_part = min_band_part
        self.max_band_part = max_band_part
        self.fade = fade
        self.p = p
        self.parameters = {}

    def randomize_parameters(self):
        self.parameters["apply"] = np.random.uniform(0, 1) < self.p
        self.parameters["t"] = np.random.uniform(self.min_band_part, self.max_band_part)
        self.parameters["t0"] = np.random.uniform(0, 1 - self.parameters["t"])

    def transform_data(self, data):
        self.randomize_parameters()
        if self.parameters["apply"]:
            nframes, nfreq = data.shape
            t = int(self.parameters["t"] * nframes)
            t0 = int(self.parameters["t0"] * nframes)
            mask = np.zeros((t, nfreq))
            if self.fade:
                fade_length = int(t * 0.1)
                mask[0:fade_length, :] = np.linspace(1, 0, num=fade_length)
                mask[-fade_length:, :] = np.linspace(0, 1, num=fade_length)
            data[t0:t0 + t, :] *= mask
        return data


class FreqMask(Transform):
    def __init__(self, min_mask_fraction=0.03, max_mask_fraction=0.4, fill_mode="constant", fill_constant=0, p=0.5):
        self.min_mask_fraction = min_mask_fraction
        self.max_mask_fraction = max_mask_fraction
        assert fill_mode in ("mean", "constant")
        self.fill_mode = fill_mode
        self.constant = fill_constant
        self.p = p
        self.parameters = {}

    def randomize_parameters(self):
        self.parameters["apply"] = np.random.uniform(0, 1) < self.p
        self.parameters["f"] = np.random.uniform(self.min_mask_fraction, self.max_mask_fraction)
        self.parameters["f0"] = np.random.uniform(0, 1 - self.parameters["f"])

    def transform_data(self, data):
        self.randomize_parameters()
        if self.parameters["apply"]:
            nframe, nmel = data.shape
            f = int(self.parameters["f"] * nmel)
            f0 = int(self.parameters["f0"] * nmel)
            if self.fill_mode == "mean":
                fill_value = np.mean(data[:, f0:f0 + f])
            else:
                fill_value = self.constant
            data[:, f0:f + f0] = fill_value
        return data


class FreqShift(Transform):
    def __init__(self, p=0.5, max_band=4, mean=0, std=2):
        self.p = p
        self.max_band = max_band
        self.mean = mean
        self.std = std
        self.parameters = {}

    def randomize_parameters(self):
        self.parameters["apply"] = np.random.uniform(0, 1) < self.p
        shift_size  = int(np.random.normal(self.mean, self.std))
        while abs(shift_size) > self.max_band:
            shift_size = int(np.random.normal(self.mean, self.std))
        self.parameters["shift_size"] = shift_size

    def transform_data(self, data):
        self.randomize_parameters()
        if self.parameters["apply"]:
            data = np.roll(data, self.parameters["shift_size"], axis=1)
            if self.parameters["shift_size"] >= 0:
                data[:, :self.parameters["shift_size"]] = 0
            else:
                data[:, self.parameters["shift_size"]:] = 0
        return data


def get_transforms(frames=None, scaler=None, add_axis=0, noise_dict_params=None, combine_channels_args=None,
                   crop_patch=False, fixed_patch_size=False, freq_mask=False, freq_shift=False, time_mask=False):
    transf = []
    unsqueeze_axis = None
    if add_axis is not None:
        unsqueeze_axis = add_axis

    if combine_channels_args is not None:
        transf.append(CombineChannels(*combine_channels_args))

    if noise_dict_params is not None:
        transf.append(AugmentGaussianNoise(**noise_dict_params))

    transf.append(ApplyLog())

    if frames is not None:
        transf.append(PadOrTrunc(nb_frames=frames))

    if time_mask:
        transf.append(TimeMask())

    if freq_mask:
        transf.append(FreqMask(fill_mode="mean"))


    if freq_shift:
        transf.append(FreqShift())

    transf.append(ToTensor(unsqueeze_axis=unsqueeze_axis))

    if scaler is not None:
        transf.append(Normalize(scaler=scaler))

    if crop_patch:
        transf.append(Query(fixed_patch_size))

    return Compose(transf)
