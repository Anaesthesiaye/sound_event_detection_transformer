import bisect
import numpy as np
import pandas as pd
import torch
import random
import warnings
from PIL import ImageFilter
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from utilities.utils import to_cuda_if_available
from utilities.Logger import create_logger
import config as cfg
from utilities.BoxTransforms import Compose

torch.manual_seed(0)
random.seed(0)


class DataLoadDf(Dataset):
    """ Class derived from pytorch DESED
    Prepare the data to be use in a batch mode

    Args:
        df: pandas.DataFrame, the dataframe containing the set infromation (feat_filenames, labels),
            it should contain these columns :
            "feature_filename"
            "feature_filename", "event_labels"
            "feature_filename", "onset", "offset", "event_label"
        encode_function: function(), function which encode labels
        transform: function(), (Default value = None), function to be applied to the sample (pytorch transformations)
        return_indexes: bool, (Default value = False) whether or not to return indexes when use __getitem__

    Attributes:
        df: pandas.DataFrame, the dataframe containing the set information (feat_filenames, labels, ...)
        encode_function: function(), function which encode labels
        transform : function(), function to be applied to the sample (pytorch transformations)
        return_indexes: bool, whether or not to return indexes when use __getitem__
    """

    def __init__(self, df, encode_function=None, transform=None, return_indexes=False, in_memory=False,
                 num_patches=None, sigma=0.26, mu= 0.2, fixed_patch_size=False):
        self.df = df
        self.encode_function = encode_function
        self.transform = transform
        self.return_indexes = return_indexes
        self.feat_filenames = df.feature_filename.drop_duplicates()
        self.filenames = df.filename.drop_duplicates()
        self.in_memory = in_memory
        self.num_patches = num_patches
        self.sigma = sigma
        self.mu = mu
        self.fixed_patch_size=fixed_patch_size
        self.logger = create_logger(__name__, terminal_level=cfg.terminal_level)
        if self.in_memory:
            self.features = {}

    def get_random_patch(self, feature):

        def get_random_center(i):
            return np.random.randint(int(t*i/2)+1, int(t*(1-i/2)))/t
        t, f = feature.shape

        if self.fixed_patch_size:
            l = np.asarray([128/t] * self.num_patches)
        else:
            l = self.mu + self.sigma * np.random.randn(5*self.num_patches)
            idx = [ i >= 0.05 and i < 0.8 for i in l]
            l = l[idx][:self.num_patches]
        c= [ get_random_center(i) for i in l]
        s, e = (c-l/2)*t, (c+l/2)*t
        s = [int(i) for i in s]
        if self.fixed_patch_size:
            e = [i+128 for i in s]
        else:
            e = [int(i) for i in e]
        boxes = [[(i+j)/(2*t), (j-i)/t] for i, j in zip(s, e)]
        return boxes


    def set_return_indexes(self, val):
        """ Set the value of self.return_indexes
        Args:
            val : bool, whether or not to return indexes when use __getitem__
        """
        self.return_indexes = val

    def get_feature_file_func(self, filename):
        """Get a feature file from a filename
        Args:
            filename:  str, name of the file to get the feature

        Returns:
            numpy.array
            containing the features computed previously
        """
        if not self.in_memory:
            data = np.load(filename).astype(np.float32)
        else:
            if self.features.get(filename) is None:
                data = np.load(filename).astype(np.float32)
                self.features[filename] = data
            else:
                data = self.features[filename]
        return data

    def __len__(self):
        """
        Returns:
            int
                Length of the object
        """
        length = len(self.feat_filenames)
        return length

    def get_sample(self, index):
        """From an index, get the features and the labels to create a sample

        Args:
            index: int, Index of the sample desired

        Returns:
            tuple
            Tuple containing the features and the labels (numpy.array, numpy.array)

        """
        features = self.get_feature_file_func(self.feat_filenames.iloc[index])

        # event_labels means weak labels, event_label means strong labels
        if "event_labels" in self.df.columns or {"onset", "offset", "event_label"}.issubset(self.df.columns):
            if "event_labels" in self.df.columns:
                label = self.df.iloc[index]["event_labels"]
                if pd.isna(label):
                    label = []
                if type(label) is str:
                    if label == "":
                        label = []
                    else:
                        label = label.split(",")
            else:
                cols = ["onset", "offset", "event_label"]
                label = self.df[self.df.filename == self.filenames.iloc[index]][cols]
                if label.empty:
                    label = []
        else:
            if self.num_patches :
                label = self.get_random_patch(features)
            else:
                label = "empty"

        if index == 0:
            self.logger.debug("label to encode: {}".format(label))
        if self.encode_function is not None:
            # labels are a list of string or list of list [[label, onset, offset]]
            y = self.encode_function(label)
        else:
            y = label
        sample = features, y
        return sample

    def __getitem__(self, index):
        """ Get a sample and transform it to be used in a ss_model, use the transformations

        Args:
            index : int, index of the sample desired

        Returns:
            tuple
            Tuple containing the features and the labels (numpy.array, numpy.array) or
            Tuple containing the features, the labels and the index (numpy.array, numpy.array, int)

        """
        sample = self.get_sample(index)

        if self.transform:
            sample = self.transform(sample)

        if self.return_indexes:
            sample = (sample, index)

        return sample

    def set_transform(self, transform):
        """Set the transformations used on a sample

        Args:
            transform: function(), the new transformations
        """
        self.transform = transform

    def add_transform(self, transform):
        if type(self.transform) is not Compose:
            raise TypeError("To add transform, the transform should already be a compose of transforms")
        transforms = self.transform.add_transform(transform)
        return DataLoadDf(self.df, self.encode_function, transforms, self.return_indexes, self.in_memory)


class ConcatDataset(Dataset):
    """
    DESED to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.

    Args:
        datasets : sequence, list of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    @property
    def cluster_indices(self):
        cluster_ind = []
        prec = 0
        for size in self.cumulative_sizes:
            cluster_ind.append(range(prec, size))
            prec = size
        return cluster_ind

    def __init__(self, datasets):
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes

    @property
    def df(self):
        df = self.datasets[0].df
        for dataset in self.datasets[1:]:
            df = pd.concat([df, dataset.df], axis=0, ignore_index=True, sort=False)
        return df


class MultiStreamBatchSampler(Sampler):
    """Takes a dataset with cluster_indices property, cuts it into batch-sized chunks
    Drops the extra items, not fitting into exact batches
    Args:
        data_source : DESED, a DESED to sample from. Should have a cluster_indices property
        batch_size : int, a batch size that you would like to use later with Dataloader class
        shuffle : bool, whether to shuffle the data or not
    Attributes:
        data_source : DESED, a DESED to sample from. Should have a cluster_indices property
        batch_size : int, a batch size that you would like to use later with Dataloader class
        shuffle : bool, whether to shuffle the data or not
    """

    def __init__(self, data_source, batch_sizes, shuffle=True):
        super(MultiStreamBatchSampler, self).__init__(data_source)
        self.data_source = data_source
        self.batch_sizes = batch_sizes
        l_bs = len(batch_sizes)
        nb_dataset = len(self.data_source.cluster_indices)
        assert l_bs == nb_dataset, "batch_sizes must be the same length as the number of datasets in " \
                                   "the source {} != {}".format(l_bs, nb_dataset)
        self.shuffle = shuffle

    def __iter__(self):
        indices = self.data_source.cluster_indices
        if self.shuffle:
            for i in range(len(self.batch_sizes)):
                indices[i] = np.random.permutation(indices[i])
        iterators = []
        for i in range(len(self.batch_sizes)):
            iterators.append(grouper(indices[i], self.batch_sizes[i]))

        return (sum(subbatch_ind, ()) for subbatch_ind in zip(*iterators))

    def __len__(self):
        val = np.inf
        for i in range(len(self.batch_sizes)):
            val = min(val, len(self.data_source.cluster_indices[i]) // self.batch_sizes[i])
        return val


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


class data_prefetcher():
    def __init__(self, loader, return_indexes=False):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.return_index = return_indexes
        self.preload()

    def preload(self):
        try:
            if self.return_index:
                (self.next_input, self.next_target), self.next_index = next(self.loader)
            else:
                self.next_input, self.next_target = next(self.loader)
                self.next_index = None
        except StopIteration:
            self.next_input = None
            self.next_target = None
            self.next_index = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = to_cuda_if_available(self.next_input)
            self.next_target = to_cuda_if_available(self.next_target)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        index = self.next_index
        self.preload()
        if not self.return_index:
            return input, target
        else:
            return (input, target), index
