import os
import sys
import mne
from types import ModuleType
from typing import Optional, Tuple

import numpy as np

from ..base import DatabaseBase, ALL, load_fids


########################################################################
class Database(DatabaseBase):
    """"""
    fids = load_fids(os.path.join(os.path.dirname(__file__), 'fids.json'))

    metadata = {
        'channels': ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1',
                     'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7',
                     'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz',
                     'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8',
                     'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2', 'EXG1',
                     'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8', 'Status'],


        'classes': ['low memory load',
                    'medium memory load',
                    'high memory load'
                    ],

        'non_task_classes': [],

        'sampling_rate': 2048,
        'montage': 'standard_1020',
        'tmin': -0.2,
        'duration': 3.2,
        'reference': '',
        'subjects': 23,
        'runs_training': [1] * 23,

        'subject_training_files': fids['EEG_data_raw subjects'],
        'subject_training_pattern': lambda subject: f'S{str(subject).rjust(2, "0")}.bdf',

        # 'subject_evaluation_files': fids['DUMMY evaluation'],
        # 'subject_evaluation_pattern': lambda subject: f'dummy_data-{str(subject).rjust(2, "0")}.npy',

        'metadata': fids['EEG_data_raw Metadata'],
        'directory': 'databases/EEG_data_raw',
    }

    # ----------------------------------------------------------------------
    def load_subject(self,
                     subject: int,
                     mode: str = 'training',
                     ) -> None:
        """Load subject

        Search for the file and load their content.

        Parameters
        ----------
        subject
            The integer `ID` for some subject.
        mode
            Some databases contain different data labels, by default is
            `training`. Literal ['training', 'evaluation']
        """
        #  This method work with a variety of data types
        self.data = super().load_subject(subject, mode)

    # ----------------------------------------------------------------------
    def get_run(self,
                run: int,
                classes: list = ALL,
                channels: Optional[list] = ALL,
                reject_bad_trials: Optional[bool] = True,
                ) -> Tuple[np.ndarray, np.ndarray]:
        """Return EEG data for specific run.

        This method automatically format the classes and the channels into a
        single format in order to standarize the scripts.

        Parameters
        ----------
        run
            The number of the desired run (0-based index).
        classes
            A list with the target classes, can be strings like in metadata or
            integers with the class position in the metadata. Default all.
        channels
            A list of channels to filter the database, channels can be strings
            (from metadata) or 1-based index.
        reject_bad_trials
            Boolean to show if bad trials should be removed or not.

        Returns
        -------
        run
            Tuple of Trials and Classes. Trials are numpy arrays of shape
            (trial, channels, time). Classes are one-dimensional  arrays with
            integers that show the class according to the metadata.
        """
        classes = self.format_class_selector(classes)
        channels = self.format_channels_selectors(channels)
        super().get_run(run, classes, channels)

        one_item = [10, 11, 12, 13]
        two_item = [20, 21, 22, 23]
        four_item = [40, 41, 42, 43]
        # The stimulus was presented on the right visual hemifield
        right_hemifield_keys = [10, 13, 20, 23, 40, 43]
        # The stimulus was presented on the left visual hemifield
        left_hemifield_keys = [11, 12, 21, 22, 41, 42]
        Ntrials = 96
        fs = 2048
        Xdata = self.data.get_data()
        events = mne.find_events(self.data)
        trial_range = np.array([-200, 3000]) / 1000
        trial_ind = np.zeros((Ntrials, 2))
        labels = np.zeros(Ntrials, dtype = int)
        count_1 = 0

        for i in range(len(events)):
            if ((events[i][2] in right_hemifield_keys) or ((events[i][2] in left_hemifield_keys))):
                # Starting point of interest for each trial
                trial_ind[count_1][0] = (
                    events[i][0] + np.floor(fs * trial_range[0]))
                # Final point of interest of each trial
                trial_ind[count_1][1] = (
                    events[i][0] + np.ceil(fs * trial_range[1]))
                count_1 += 1
                if (events[i][2] in one_item):
                    labels[count_1 - 1] = 0
                elif(events[i][2] in two_item):
                    labels[count_1 - 1] = 1
                elif(events[i][2] in four_item):
                    labels[count_1 - 1] = 2

        data__ = []
        for i in trial_ind:
            data__.append(Xdata[:, int(i[0]):int(i[1])])

        data__ = np.array(data__)[:, channels - 1, :]

        return data__, labels

# ------------------------------------------------------------------------------
#  Keep this code
########################################################################


class CallableModule(ModuleType):
    # ----------------------------------------------------------------------
    def __call__(self, *args, **kwargs):
        """"""
        return Database(*args, **kwargs)


sys.modules[__name__].__class__ = CallableModule

# ------------------------------------------------------------------------------
