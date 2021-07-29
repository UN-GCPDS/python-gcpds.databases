import os
import sys
from types import ModuleType
from typing import Optional, Tuple

import numpy as np

from ..base import DatabaseBase, ALL, load_fids


########################################################################
class Database(DatabaseBase):
    """"""
    fids = load_fids(os.path.join(os.path.dirname(__file__), 'fids.json'))

    metadata = {
        'channels': [f'CH-{ch}' for ch in range(1, 17)],
        'classes': ['class 0',
                    'class 1',
                    'class 4',
                    ],
        'non_task_classes': ['non-task 0',
                             'non-task 1',
                             ],
        'sampling_rate': 1000,
        'montage': 'standard_1005',
        'tmin': -3,
        'duration': 7,
        'reference': '',
        'subjects': 7,
        'runs_training': [1] * 7,

        'subject_training_files': fids['DUMMY training'],
        'subject_training_pattern': lambda subject: f'dummy_data-{str(subject).rjust(2, "0")}.npy',

        # 'subject_evaluation_files': fids['DUMMY evaluation'],
        # 'subject_evaluation_pattern': lambda subject: f'dummy_data-{str(subject).rjust(2, "0")}.npy',

        'metadata': fids['DUMMY metadata'],
        'directory': 'databases/DUMMY',
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

        data = self.data.reshape(-1, 16, 1000)
        cls = [0] * 30 + [1] * 30

        return data, cls

    # ----------------------------------------------------------------------
    def non_task(self,
                 non_task_classes: Optional[list] = ALL,
                 runs: Optional[list] = ALL,
                 channels: Optional[list] = ALL,
                 ) -> np.ndarray:
        """Non task trials.

        Must return a list of list of trials (run, non-task, trial, channel, time)

        Parameters
        ----------

        non_task_classes
            Similarly to the classes for `get_run` method. Default all.
        runs
            List of runs. Default all.
        channels
            List of channels. Default all.

        Returns
        -------
        non_task
            List of trials for each `non-task` and `run`, a 5-dimensional
            indexable object.
        """

        channels = self.format_channels_selectors(channels)
        non_task_classes = self.format_non_class_selector(non_task_classes)

        data = [np.random.random(size=(10, 16, 1000)) for i in range(
            len(self.metadata['non_task_classes']))]

        return data


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
