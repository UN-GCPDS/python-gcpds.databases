import os
import sys
from types import ModuleType
from typing import Optional, Tuple

import numpy as np

from ...base import DatabaseBase, ALL, load_fids


########################################################################
class Database(DatabaseBase):
    """"""
    fids = load_fids(os.path.join(os.path.dirname(__file__), 'fids.json'))

    metadata = {
        'channels': ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1',
                     'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
                     'P1', 'Pz', 'P2', 'POz'],
        'classes': ['left hand', 'right hand', 'feet', 'tongue'],
        'sampling_rate': 250,
        'montage': 'standard_1020',
        'tmin': -2,
        'duration': 7,
        'reference': '',
        'subjects': 9,
        'runs_training': [6, 6, 6, 6, 6, 6, 6, 6, 6],
        'runs_evaluation': [6, 6, 6, 6, 6, 6, 6, 6, 6],

        'subject_training_files': fids['BCI2a training'],
        'subject_training_pattern': lambda subject: f'A{str(subject).rjust(2, "0")}T.mat',

        'subject_evaluation_files': fids['BCI2a evaluation'],
        'subject_evaluation_pattern': lambda subject: f'A{str(subject).rjust(2, "0")}E.mat',

        'metadata': fids['BCI2a metadata'],
        'directory': 'databases/BCI_Competition_IV/dataset_2a',
    }

    # ----------------------------------------------------------------------
    def load_subject(self, subject: int, mode: str = 'training') -> None:
        """"""
        data = super().load_subject(subject, mode)
        self.data = data['data'][0]

    # ----------------------------------------------------------------------
    def get_run(self, run: int, classes: Optional[list] = ALL, channels: Optional[list] = ALL, reject_bad_trials: Optional[bool] = True) -> Tuple[np.ndarray, np.ndarray]:
        """"""
        classes = self.format_class_selector(classes)
        channels = self.format_channels_selectors(channels)
        super().get_run(run, classes, channels, reject_bad_trials)

        # # A04T contains only the eye movement condition
        if self.subject == 4 and self.mode == 'training':
            run = run - 2

        artifacts = self.data[3 + run][0][0][5].T[0] == 1
        classes_list = np.array([i[0] for i in self.data[3 + run][0][0][2]])
        starts = [s[0] for s in self.data[3 + run][0][0][1]]
        end = int(self.metadata['sampling_rate'] * self.metadata['duration'])

        run = np.array([self.data[3 + run][0][0][0][start:start + end]
                        for start in starts])

        # Remove EOG
        run = run[:, :, : 22]

        # Select channels
        run = run[:, :, channels - 1]

        # trial x channel x time
        run = np.moveaxis(run, 2, 1)

        # reject bad trials
        if reject_bad_trials:
            run = run[~artifacts]
            classes_list = classes_list[~artifacts]

        idx = []
        c = []
        for cls in classes:
            idx.append(np.where(np.array(classes_list) == cls + 1)[0])
            c.append([cls] * len(idx[-1]))

        return run[np.concatenate(idx), :, :], np.concatenate(c)


########################################################################
class CallableModule(ModuleType):
    # ----------------------------------------------------------------------
    def __call__(self, *args, **kwargs):
        """"""
        return Database(*args, **kwargs)


sys.modules[__name__].__class__ = CallableModule
