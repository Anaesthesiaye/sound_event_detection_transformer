# -*- coding: utf-8 -*-
from __future__ import print_function

import glob

import numpy as np
import os
import os.path as osp
import librosa
import time
import pandas as pd
import config as cfg
from utilities.Logger import create_logger
import soundfile
import scipy
eps = np.spacing(1, dtype=np.float64)

class SedData:
    """
    Data are organized in `audio/` and corresponding `metadata/` folders.
    audio folder contains wav files, and metadata folder contains .tsv files.

    The organisation should always be the same in the audio and metadata folders. (See example)
    If there are multiple metadata files for a single audio files, add the name in the list of `merged_folders_name`.
    (See validation folder example). Be careful, it works only for one level of folder.

    tab separated value metadata files (.tsv) contains columns:
        - filename                                  (unlabeled data)
        - filename  event_labels                    (weakly labeled data)
        - filename  onset   offset  event_label     (strongly labeled data)

    Example:
    - dataset
        - metadata
            - train
                - synthetic20
                    - soundscapes.tsv   (audio_dir associated: audio/train/synthetic20/soundscapes)
                - unlabel_in_domain.tsv (audio_dir associated: audio/train/unlabel_in_domain)
                - weak.tsv              (audio_dir associated: audio/train/weak)
            - validation
                - validation.tsv        (audio_dir associated: audio/validation) --> so audio_dir has to be declared
                - test_dcase2018.tsv    (audio_dir associated: audio/validation)
                - eval_dcase2018.tsv    (audio_dir associated: audio/validation)
            -eval
                - public.tsv            (audio_dir associated: audio/eval/public)
        - audio
            - train
                - synthetic20           (synthetic data generated for dcase 2020, you can create your own)
                    - soundscapes
                    - separated_sources (optional, only using source separation)
                - unlabel_in_domain
                - unlabel_in_domain_ss  (optional, only using source separation)
                - weak
                - weak_ss               (optional, only using source separation)
            - validation
            - validation_ss             (optional, only using source separation)

    Args:
        base_feature_dir: str, optional, base directory to store the features
        recompute_features: bool, optional, whether or not to recompute features
        compute_log: bool, optional, whether or not saving the logarithm of the feature or not
            (particularly useful to put False to apply some data augmentation)

    Attributes:
        base_feature_dir: str, base directory to store the features
        recompute_features: bool, whether or not to recompute features
        compute_log: bool, whether or not saving the logarithm of the feature or not
            (particularly useful to put False to apply some data augmentation)
        feature_dir : str, directory to store the features

    """
    def __init__(self, dataname="urbansed",  recompute_features=False, compute_log=True):
        # Parameters, they're kept if we need to reproduce the dataset
        ext_freq = ''
        self.compute_log = compute_log
        if dataname == "urbansed":
            self.sample_rate = cfg.usample_rate
            self.n_fft = cfg.un_fft
            self.n_window = cfg.un_window
            self.hop_size = cfg.uhop_size
            self.n_mels = cfg.un_mels
            base_feature_dir = osp.join(cfg.urbansed_dir, "features")
        else:
            self.sample_rate = cfg.sample_rate
            self.n_fft = cfg.n_fft
            self.n_window = cfg.n_window
            self.hop_size = cfg.hop_size
            self.n_mels = cfg.n_mels
            base_feature_dir = osp.join(cfg.dcase_dir, "features")
        if not self.compute_log:
            ext_freq = "_nolog"
        # Defined parameters
        self.dataname = dataname
        self.recompute_features = recompute_features


        # Feature dir to not have the same name with different parameters
        feature_dir = osp.join(base_feature_dir, f"sr{self.sample_rate}_win{self.n_window}_hop{self.hop_size}"
                                                 f"_mels{self.n_mels}{ext_freq}")

        self.feature_dir = osp.join(feature_dir, "features")
        self.meta_feat_dir = osp.join(feature_dir, "metadata")
        # create folder if not exist
        os.makedirs(self.feature_dir, exist_ok=True)
        os.makedirs(self.meta_feat_dir, exist_ok=True)
        self.logger = create_logger(__name__, terminal_level=cfg.terminal_level)

    def state_dict(self):
        """ get the important parameters to save for the class
        Returns:
            dict
        """
        parameters = {
            "feature_dir": self.feature_dir,
            "meta_feat_dir": self.meta_feat_dir,
            "compute_log": self.compute_log,
            "sample_rate": self.sample_rate,
            "n_window": self.n_window,
            "hop_size": self.hop_size,
            "n_mels": self.n_mels,
        }
        return parameters

    @classmethod
    def load_state_dict(cls, state_dict):
        """ load the dataset from previously saved parameters
        Args:
            state_dict: dict, parameter saved with state_dict function
        Returns:
            DESED class object with the right parameters
        """
        desed_obj = cls()
        desed_obj.feature_dir = state_dict["feature_dir"]
        desed_obj.meta_feat_dir = state_dict["meta_feat_dir"]
        desed_obj.compute_log = state_dict["compute_log"]
        desed_obj.sample_rate = state_dict["sample_rate"]
        desed_obj.n_window = state_dict["n_window"]
        desed_obj.hop_size = state_dict["hop_size"]
        desed_obj.n_mels = state_dict["n_mels"]
        desed_obj.mel_min_max_freq = state_dict["mel_min_max_freq"]
        return desed_obj

    def initialize_and_get_df(self, tsv_path, audio_dir=None, nb_files=None):
        """ Initialize the dataset, extract the features dataframes
        Args:
            tsv_path: str, tsv path in the initial dataset
            audio_dir: str, the path where to search the filename of the df
            
            nb_files: int, optional, the number of file to take in the dataframe if taking a small part of the dataset.

        Returns:
            pd.DataFrame
            The dataframe containing the right features and labels
        """
        # Check parameters
        if audio_dir is None:
            audio_dir = meta_path_to_audio_dir(tsv_path)
        assert osp.exists(audio_dir), f"the directory {audio_dir} does not exist"

        # Path to save features, subdir, otherwise could have duplicate paths for synthetic data
        fdir = audio_dir
        fdir = fdir[:-1] if fdir.endswith(osp.sep) else fdir
        subdir = osp.sep.join(fdir.split(osp.sep)[-2:])
        meta_feat_dir = osp.join(self.meta_feat_dir, subdir)
        feature_dir = osp.join(self.feature_dir, subdir)
        self.logger.debug(feature_dir)
        os.makedirs(meta_feat_dir, exist_ok=True)
        os.makedirs(feature_dir, exist_ok=True)

        df_meta = self.get_df_from_meta(tsv_path, nb_files)
        self.logger.info(f"{tsv_path} Total file number: {len(df_meta.filename.unique())}")

        # Meta filename
        ext_tsv_feature = ""
        fname, ext = osp.splitext(osp.basename(tsv_path))
        feat_fname = fname + ext_tsv_feature + ext
        if nb_files is not None:
            feat_fname = f"{nb_files}_{feat_fname}"
        features_tsv = osp.join(meta_feat_dir, feat_fname)

        t = time.time()
        if not osp.exists(features_tsv):
            self.logger.info(f"Getting features ...")
            df_features = self.extract_features_from_df(df_meta, audio_dir, feature_dir)
            if len(df_features) != 0:
                df_features.to_csv(features_tsv, sep="\t", index=False)
                self.logger.info(f"features created/retrieved in {time.time() - t:.2f}s, metadata: {features_tsv}")
            else:
                raise IndexError(f"Empty features DataFrames {features_tsv}")
        else:
            df_features = pd.read_csv(features_tsv, sep="\t")
        return df_features


    def load_and_compute_mel_spec(self, wav_path):
        (audio, _) = read_audio(wav_path, self.sample_rate)
        ham_win = np.hamming(self.n_window)
        spec = librosa.stft(
            audio,
            n_fft=self.n_fft,
            win_length=self.n_window,
            hop_length=self.hop_size,
            window=ham_win,
            center=True,
            pad_mode='reflect'
        )
        mel_spec = librosa.feature.melspectrogram(
            S=np.abs(spec),  # amplitude, for energy: spec**2 but don't forget to change amplitude_to_db.
            sr=self.sample_rate,
            n_mels=self.n_mels,
            htk=False, norm=None)
        if self.compute_log:
            mel_spec = librosa.amplitude_to_db(mel_spec)  # 10 * log10(S**2 / ref), ref default is 1
        mel_spec = mel_spec.T
        mel_spec = mel_spec.astype(np.float32)

        return mel_spec

    def _extract_features(self, wav_path, out_path):
        if not osp.exists(out_path):
            try:
                mel_spec = self.load_and_compute_mel_spec(wav_path)
                os.makedirs(osp.dirname(out_path), exist_ok=True)
                np.save(out_path, mel_spec)
            except IOError as e:
                self.logger.error(e)

    def _extract_features_ss(self, wav_path, wav_paths_ss, out_path):
        try:
            features = np.expand_dims(self.load_and_compute_mel_spec(wav_path), axis=0)
            for wav_path_ss in wav_paths_ss:
                sep_features = np.expand_dims(self.load_and_compute_mel_spec(wav_path_ss), axis=0)
                features = np.concatenate((features, sep_features))
            os.makedirs(osp.dirname(out_path), exist_ok=True)
            np.save(out_path, features)
        except IOError as e:
            self.logger.error(e)

    def _extract_features_file(self, filename, audio_dir, feature_dir, audio_dir_ss=None, pattern_ss=None,
                               ext_ss_feature_file="_ss", keep_sources=None):
        wav_path = osp.join(audio_dir, filename)
        if not osp.isfile(wav_path):
            self.logger.error("File %s is in the tsv file but the feature is not extracted because "
                         "file do not exist!" % wav_path)
            out_path = None
        else:
            if audio_dir_ss is None:
                out_filename = osp.join(osp.splitext(filename)[0] + ".npy")
                out_path = osp.join(feature_dir, out_filename)
                self._extract_features(wav_path, out_path)
            else:
                # To be changed if you have new separated sounds from the same mixture
                out_filename = osp.join(osp.splitext(filename)[0] + ext_ss_feature_file + ".npy")
                out_path = osp.join(feature_dir, out_filename)
                bname, ext = osp.splitext(filename)
                if keep_sources is None:
                    wav_paths_ss = glob.glob(osp.join(audio_dir_ss, bname + pattern_ss, "*" + ext))
                else:
                    wav_paths_ss = []
                    for s_ind in keep_sources:
                        audio_file = osp.join(audio_dir_ss, bname + pattern_ss, s_ind + ext)
                        assert osp.exists(audio_file), f"Audio file does not exists: {audio_file}"
                        wav_paths_ss.append(audio_file)
                if not osp.exists(out_path):
                    self._extract_features_ss(wav_path, wav_paths_ss, out_path)

        return filename, out_path

    def extract_features_from_df(self, df_meta, audio_dir, feature_dir):
        """Extract log mel spectrogram features.

        Args:
            df_meta : pd.DataFrame, containing at least column "filename" with name of the wav to compute features
            audio_dir: str, the path where to find the wav files specified by the dataframe
            feature_dir: str, the path where to search and save the features.

        Returns:
            pd.DataFrame containing the initial meta + column with the "feature_filename"
        """

        df_features = pd.DataFrame()
        fpaths = df_meta["filename"]
        events_num = []
        uniq_fpaths = fpaths.drop_duplicates().to_list()
        for filename in uniq_fpaths:  # tqdm(uniq_fpaths)
            filename, out_path = self._extract_features_file(filename, audio_dir = audio_dir, feature_dir=feature_dir)
            if out_path is not None:
                row_features = df_meta[df_meta.filename == filename]
                row_features.loc[:, "feature_filename"] = out_path
                df_features = df_features.append(row_features, ignore_index=True)
                events_num.append(len(row_features))
        print("=" * 50)
        print("event numbers in {}".format(audio_dir.split('/')[-1]))
        events_num = np.asarray(events_num)
        events_num_pd = pd.DataFrame([[events_num.min(), events_num.max(), events_num.mean(), events_num.std()]],
                                     columns=["min", "max", "mean", "std"])
        print(events_num_pd)

        return df_features.reset_index(drop=True)

    @staticmethod
    def get_classes(list_dfs):
        """ Get the different classes of the dataset
        Returns:
            A list containing the classes
        """
        classes = []
        for df in list_dfs:
            if "event_label" in df.columns:
                classes.extend(df["event_label"].dropna().unique())  # dropna avoid the issue between string and float
            elif "event_labels" in df.columns:
                classes.extend(df.event_labels.str.split(',', expand=True).unstack().dropna().unique())
        return list(set(classes))

    @staticmethod
    def get_subpart_data(df, nb_files, pattern_ss=None):
        """Get a subpart of a dataframe (only the number of files specified)
        Args:
            df : pd.DataFrame, the dataframe to extract a subpart of it (nb of filenames)
            nb_files: int, the number of file to take in the dataframe if taking a small part of the dataset.
            pattern_ss: str, if nb_files is not None, the pattern is needed to get same ss than soundscapes
        Returns:
            pd.DataFrame containing the only the number of files specified
        """
        column = "filename"
        if not nb_files > len(df[column].unique()):
            if pattern_ss is not None:
                filenames = df[column].apply(lambda x: x.split(pattern_ss)[0])
                filenames = filenames.drop_duplicates()
                # sort_values and random_state are used to have the same filenames each time (also for normal and ss)
                filenames_kept = filenames.sort_values().sample(nb_files, random_state=10)
                df_kept = df[df[column].apply(lambda x: x.split(pattern_ss)[0]).isin(filenames_kept)].reset_index(
                    drop=True)
            else:
                filenames = df[column].drop_duplicates()
                # sort_values and random_state are used to have the same filenames each time (also for normal and ss)
                filenames_kept = filenames.sort_values().sample(nb_files, random_state=10)
                df_kept = df[df[column].isin(filenames_kept)].reset_index(drop=True)
        else:
            df_kept = df
        return df_kept

    @staticmethod
    def get_df_from_meta(meta_name, nb_files=None, pattern_ss=None):
        """
        Extract a pandas dataframe from a tsv file

        Args:
            meta_name : str, path of the tsv file to extract the df
            nb_files: int, the number of file to take in the dataframe if taking a small part of the dataset.
            pattern_ss: str, if nb_files is not None, the pattern is needed to get same ss than soundscapes
        Returns:
            dataframe
        """
        df = pd.read_csv(meta_name, header=0, sep="\t")
        if nb_files is not None:
            df = SedData.get_subpart_data(df, nb_files, pattern_ss=pattern_ss)
        return df


def read_audio(path, target_fs=None, **kwargs):
    """ Read a wav file
    Args:
        path: str, path of the audio file
        target_fs: int, (Default value = None) sampling rate of the returned audio file, if not specified, the sampling
            rate of the audio file is taken

    Returns:
        tuple
        (numpy.array, sampling rate), array containing the audio at the sampling rate given

    """
    (audio, fs) = soundfile.read(path, **kwargs)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio, fs

def meta_path_to_audio_dir(tsv_path):
    return os.path.splitext(tsv_path.replace("metadata", "audio"))[0]


def audio_dir_to_meta_path(audio_dir):
    return audio_dir.replace("audio", "metadata") + ".tsv"


def get_durations_df(gtruth_path, audio_dir=None):
    if audio_dir is None:
        audio_dir = meta_path_to_audio_dir(gtruth_path)
    path, ext = os.path.splitext(gtruth_path)
    path_durations_synth = path + "_durations" + ext
    if not os.path.exists(path_durations_synth):
        durations_df = generate_tsv_wav_durations(audio_dir, path_durations_synth)
    else:
        durations_df = pd.read_csv(path_durations_synth, sep="\t")
    return durations_df


def generate_tsv_wav_durations(audio_dir, out_tsv):
    """ Generate a dataframe with filename and duration of the file
    Args:
        audio_dir: str, the path of the folder where audio files are (used by glob.glob)
        out_tsv: str, the path of the output tsv file

    Returns:
        pd.DataFrame, the dataframe containing filenames and durations
    """
    meta_list = []
    for file in glob.glob(os.path.join(audio_dir, "*.wav")):
        d = soundfile.info(file).duration
        meta_list.append([os.path.basename(file), d])
    meta_df = pd.DataFrame(meta_list, columns=["filename", "duration"])
    if out_tsv is not None:
        meta_df.to_csv(out_tsv, sep="\t", index=False, float_format="%.1f")
    return meta_df


def get_dfs(desed_dataset, dataname, unlabel_data=False):
    if dataname == 'urbansed':
        train_df = desed_dataset.initialize_and_get_df(cfg.urban_train_tsv)
        valid_df = desed_dataset.initialize_and_get_df(cfg.urban_valid_tsv)
        eval_df = desed_dataset.initialize_and_get_df(cfg.urban_eval_tsv)

        data_dfs = {
            "train": train_df,
            "validation": valid_df,
            "eval": eval_df
        }
    else:
        weak_df = desed_dataset.initialize_and_get_df(cfg.weak)
        synthetic_df = desed_dataset.initialize_and_get_df(cfg.synthetic)
        validation_df = desed_dataset.initialize_and_get_df(cfg.validation, audio_dir=cfg.audio_validation_dir)
        eval_df = desed_dataset.initialize_and_get_df(cfg.eval_desed)
        data_dfs = {
            "weak": weak_df,
            "synthetic": synthetic_df,
            "validation": validation_df,
            "eval": eval_df
        }
        if unlabel_data:
            unlabel_df = desed_dataset.initialize_and_get_df(cfg.unlabel)
            data_dfs["unlabel"] = unlabel_df
    return data_dfs



