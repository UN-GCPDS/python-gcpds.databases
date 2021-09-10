from .google_drive_downloader import GoogleDriveDownloader as gdd
from scipy.io import loadmat
import os
from abc import ABCMeta, abstractmethod
from typing import Union, Optional
import numpy as np
# from .databases import databases
import json
import mne
import sys
import tables
import logging

ALL = 'all'
mne.set_log_level('CRITICAL')


# ----------------------------------------------------------------------
def drive_mounted():
    """"""
    return '/content' in sys.path and '/env/python' in sys.path and os.path.exists('/content/drive/Shareddrives/GCPDS')


# ----------------------------------------------------------------------
def load_fids(file):
    """"""
    return json.load(open(file, 'rb'))


# ----------------------------------------------------------------------
def load_mat(path: str, mat: str, fid: str, size: Optional[int] = None, overwrite: Optional[bool] = False, loop: Optional[int] = 0) -> np.ndarray:
    """Get the raw data for one individual file.

    If the file does not exist in the specified path then tries to download it
    from Google Drive.
    """

    filepath = os.path.join(path, mat)

    if os.path.exists(filepath) and not overwrite:

        if filepath.endswith('.mat'):
            try:
                return loadmat(filepath)
            except ValueError:
                try:
                    return tables.open_file(filepath, driver="H5FD_CORE")
                except:
                    pass
                    # logging.warning('Corrupt database!!\n, overwriting...')
                    # return load_mat(path, mat, fid, size, overwrite=True)

        elif filepath.endswith('.edf'):
            try:
                return mne.io.read_raw_edf(filepath)
            except:
                pass

        elif filepath.endswith('.npy'):
            try:
                return np.load(filepath)
            except:
                pass
        elif filepath.endswith('.bdf'):
            try:
                return mne.io.read_raw_bdf(filepath)
            except:
                pass
        elif filepath.endswith('.gdf'):
            try:
                return mne.io.read_raw_gdf(filepath)
            except:
                pass

        if loop > 2:
            logging.warning(
                'Several unsuccessful attempts, the data access quota could be compromised.')
            logging.warning(
                'Many read and write tasks over Google Drive databases could block the background access system almost 24 hours.')
            sys.exit()

        if drive_mounted():
            logging.warning('Corrupt database!!')
            return
        else:
            logging.warning('Corrupt database!!\noverwriting...')
            return load_mat(path, mat, fid, size, overwrite=True, loop=loop + 1)

    else:
        logging.warning('Database not found!')
        logging.warning('downloading...')

        if drive_mounted():
            logging.warning('Write on the shared drive has been disabled.')
            logging.warning(
                f'The directory name is optional for Google Drive mounted environment')
            sys.exit()

        os.makedirs(path, exist_ok=True)
        gdd.download_file_from_google_drive(file_id=fid,
                                            dest_path=filepath,
                                            unzip=False,
                                            overwrite=overwrite,
                                            size=size)
        return load_mat(path, mat, fid, size, loop=loop + 1)


# ----------------------------------------------------------------------
def download_metadata(path, metadata):
    """"""
    os.makedirs(path, exist_ok=True)
    for file in metadata:
        fid, size = metadata[file]
        gdd.download_file_from_google_drive(file_id=fid,
                                            dest_path=os.path.join(
                                                path, file),
                                            unzip=False,
                                            overwrite=True,
                                            size=size)


# # ----------------------------------------------------------------------
# def get_menmap_filename():
    # """"""
    # filename = ''.join([random.choice(string.ascii_lowercase)
        # for i in range(16)])
    # return f'{filename}.menmap'

########################################################################
class DatabaseBase(metaclass=ABCMeta):
    """"""

    # ----------------------------------------------------------------------
    def __init__(self, path: Optional[str] = None) -> None:
        """Constructor"""

        if path and drive_mounted():
            logging.warning(
                'The directory folder is optional for Google Drive mounted environment.')

        if not path and drive_mounted():
            logging.info('Using the Google Drive environment')

        self.path = path
        # self.usemenmap = usemenmap

    # ----------------------------------------------------------------------
    def __repr__(self):
        """"""
        lines = []
        lines.append('#' * 50)
        lines.append(f'{self.__class__.__name__}')
        lines.append('-' * 50)
        lines.append(f"Channels: {self.metadata['channels']}")
        lines.append(f"Sampling rate: {self.metadata['sampling_rate']} Hz")
        lines.append(f"Montage: {self.metadata['montage']}")
        lines.append(f"Subjects: {self.metadata['subjects']}")
        lines.append(f"Trials duration: {self.metadata['duration']}")
        lines.append(f"Trials tmin: {self.metadata['tmin']}")
        lines.append(f"Classes: {self.metadata['classes']}")
        if 'non_task_classes' in self.metadata:
            lines.append(
                f"Non-task classes: {self.metadata['non_task_classes']}")
        lines.append('#' * 50)
        return '\n'.join(lines)

    # ----------------------------------------------------------------------
    @abstractmethod
    def load_subject(self, subject: int, mode: str) -> None:
        """"""
        if not mode in ['training', 'evaluation']:
            raise Exception(
                f"No mode {mode} available, only 'training', 'evaluation'")

        filename_subject = self.metadata[f'subject_{mode}_pattern'](subject)

        if self.path is None:
            self.path = self.metadata['directory']

        if os.path.split(filename_subject)[-1] not in self.metadata[f'subject_{mode}_files'].keys():
            raise Exception(f"Subject {subject} not in list of subjects.")

        fid, size = self.metadata[f'subject_{mode}_files'][os.path.split(
            filename_subject)[-1]]

        self.subject = subject
        self.mode = mode

        self.runs = self.metadata[f'runs_{mode}'][subject - 1]
        # self.data = load_mat(self.path, filename_subject, fid)['eeg'][0][0]
        return load_mat(self.path, filename_subject, fid, size)

    # # ----------------------------------------------------------------------
    # def to_menmap(self, array):
        # """"""
        # array = np.array(array.tolist())

        # filename = os.path.join(self.path, get_menmap_filename())
        # fp = np.memmap(filename, dtype=array.dtype,
        # mode='w+', shape=array.shape)
        # fp[:] = array[:]
        # del array, fp
        # mmap = np.memmap(filename, mode='r')
        # return mmap

    # ----------------------------------------------------------------------
    @ abstractmethod
    def get_run(self, run: int, classes: Union[int, str], channels=Union[int, str], reject_bad_trials: Optional[bool] = True) -> np.ndarray:
        """"""
        if run > self.runs:
            raise Exception(f'The current user only have {self.runs} runs.')

        if isinstance(channels, (list, tuple)) and -1 in channels:
            raise Exception('The channels are 1-based arrays')

        if isinstance(classes, (list, tuple)) and np.max(classes) >= len(self.metadata['classes']):
            raise Exception(
                f"The class index {np.max(classes)} is out of range.")

    # ----------------------------------------------------------------------
    def get_data(self, classes: Optional[list] = ALL, channels: Optional[list] = ALL, reject_bad_trials: Optional[bool] = True, keep_runs_separated: bool = False):
        """Return all runs."""

        if keep_runs_separated:
            return [self.get_run(run, classes=classes, channels=channels, reject_bad_trials=reject_bad_trials) for run in range(self.runs)]

        else:
            r, c = self.get_run(
                0, classes=classes, channels=channels, reject_bad_trials=reject_bad_trials)
            for run in range(1, self.runs):
                r_, c_ = self.get_run(
                    run, classes=classes, channels=channels, reject_bad_trials=reject_bad_trials)
                if not r_ is None:
                    r = np.concatenate([r, r_], axis=0)
                    c = np.concatenate([c, c_])

            return r, c

            # start = 0
            # for _ in range(self.runs):
            # r, c = self.get_run(start, classes=classes, channels=channels,
            # reject_bad_trials=reject_bad_trials)
            # if not r is None:
            # break
            # else:
            # start += 1

            # for run in range(start + 1, self.runs):
            # r_, c_ = self.get_run(
            # run, classes=classes, channels=channels, reject_bad_trials=reject_bad_trials)
            # if not r_ is None:
            # r = np.concatenate([r, r_], axis=0)
            # c = np.concatenate([c, c_])

            # return r, c

    # ----------------------------------------------------------------------
    def format_channels_selectors(self, channels=None):
        """Generate the channels vector.

        If no selector then all channels will be used, the channels cant be
        indicated with the name or the index. If index is used this must be
        1-based array.
        """

        if channels != ALL:
            channels = [(list(map(str.lower, self.metadata['channels'])).index(
                ch.lower()) + 1) if isinstance(ch, str) else (ch) for ch in channels]
        else:
            channels = list(
                range(1, len(self.metadata['channels']) + 1))

        return np.array(channels)

    # ----------------------------------------------------------------------
    def format_class_selector(self, classes):
        """"""
        if classes != ALL:
            classes = [self.metadata['classes'].index(
                cls) if isinstance(cls, str) else cls for cls in classes]
        else:
            classes = range(len(self.metadata['classes']))

        return classes

    # ----------------------------------------------------------------------
    def format_non_class_selector(self, classes):
        """"""
        if classes != ALL:
            classes = [self.metadata['non_task_classes'].index(
                cls) if isinstance(cls, str) else cls for cls in classes]
        else:
            classes = range(len(self.metadata['non_task_classes']))

        return classes

    # ----------------------------------------------------------------------
    def format_runs(self, runs):
        """"""
        if runs != ALL:
            return self.metadata[runs][self.subject]
        else:
            return range(self.runs)

    # ----------------------------------------------------------------------
    def get_epochs(self, run=ALL, classes=ALL, channels=ALL, kwargs_run={}, **kwargs):
        """"""
        # # Remove channels that not correspond with the montage
        # montage = mne.channels.make_standard_montage(self.metadata['montage'])
        # channels_names = set(self.metadata['channels']).intersection(
        # set(montage.ch_names))
        # channels_missings = set(self.metadata['channels']).difference(
        # set(montage.ch_names))

        # if channels_missings:
        # logging.warning(
        # f"Missing {channels_missings} channels in {self.metadata['montage']} montage.\n"
        # f"Missing channels will be removed from MNE Epochs")

        montage = mne.channels.make_standard_montage(
            self.metadata['montage'])

        # Channels names with the MNE standard capitalization

        if channels == ALL:
            source = self.metadata['channels']
        else:
            source = channels

        target = montage.ch_names
        channels_names = []
        for ch_s in source:
            for ch_t in target:
                if ch_s.lower().strip() == ch_t.lower().strip():
                    channels_names.append(ch_t)

        # Missing channels
        channels_missings = set(channels_names).difference(
            set(montage.ch_names))
        if channels_missings:
            print(f"Missing {channels_missings} channels in {montage_name} montage.\n"
                  f"Missing channels will be removed from MNE Epochs")

        info = mne.create_info(list(channels_names),
                               sfreq=self.metadata['sampling_rate'], ch_types=["eeg"] * len(
                                   channels))
        info.set_montage(self.metadata['montage'])

        if run != ALL:
            data, classes_ = self.get_run(
                run, classes, channels=list(channels_names), **kwargs_run)
        else:
            data, classes_ = self.get_data(
                classes, channels=list(channels_names), **kwargs_run)

        events = [[i, 1, cls] for i, cls in enumerate(classes_)]
        event_id = {e: i for i, e in enumerate(
            self.metadata['classes']) if i in classes_}

        return mne.EpochsArray(data, info, events=events, tmin=self.metadata['tmin'], event_id=event_id, **kwargs)

    # ----------------------------------------------------------------------
    def get_metadata(self):
        """"""
        download_metadata(os.path.join(self.path, 'metadata'),
                          self.metadata['metadata'])

    # ----------------------------------------------------------------------
    def non_task(self, *args, **kwargs):
        """"""
        logging.warning("This database has not non-task data")

    # ----------------------------------------------------------------------
    def test_integrity(self):
        """"""
        import logging

        if 'runs_training' in self.metadata:
            if self.metadata['subjects'] != len(self.metadata['runs_training']):
                logging.error(
                    "Number of 'subjects' not correspond with the number of 'runs' for training.")

            try:
                for subj in range(1, self.metadata['subjects'] + 1):
                    if not self.metadata['subject_training_pattern'](subj) in self.metadata['subject_training_files']:
                        logging.error(
                            f"'{self.metadata['subject_training_pattern'](subj)}' not in 'fids.json'")
            except:
                logging.error('Pattern for trainning files not working!')
                pass

        if 'runs_evaluation' in self.metadata:
            if self.metadata['subjects'] != len(self.metadata['runs_evaluation']):
                logging.error(
                    "Number of 'subjects' not correspond with the number of 'runs' for evaluation.")

            try:
                for subj in range(1, self.metadata['subjects'] + 1):
                    if not self.metadata['subject_evaluation_pattern'](subj) in self.metadata['subject_evaluation_files']:
                        logging.error(
                            f"'{self.metadata['subject_evaluation_pattern'](subj)}' not in 'fids.json'")
            except:
                logging.error('Pattern for trainning files not working!')
                pass

        self.load_subject(1)

        gdata = self.get_data()

        if len(gdata) != 2:
            logging.error(
                "The method 'get_data' MUST return a tuple of 2 elements (data, classes).")
        else:
            data, class_ = gdata

            if data.ndim != 3:
                logging.error(
                    "The first element of the method 'get_data' MUST be a 3 dimencional array (trials, channels, time)")

            if data.shape[0] != len(class_):
                logging.error(
                    "The method 'get_data' MUST have the same number of trials.")

            if data.shape[1] != len(self.metadata['channels']):
                logging.error(
                    f"The second dimension of the first element of the method 'get_data' MUST have the same number of channels ({len(self.metadata['channels'])})")

        ndata = self.non_task()

        if ndata and len(ndata) != len(self.metadata['non_task_classes']):
            logging.error(
                "The method 'non_task' must return a tuple of the same size of 'non_task_classes' classes.")
