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
        'channels': ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3',
                     'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2'],
        'classes': ['noise-50',
                    'noise-100',
                    'noise-150',
                    'noise-200',
                    'music-50',
                    'music-100',
                    'music-150',
                    'music-200',
                    ],
        'non_task_classes': ['resting'],
        'sampling_rate': 1000,
        'montage': 'standard_1020',
        'tmin': 0,
        'duration': 9.5,
        'reference': '',
        'subjects': 21,
        'runs_training': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1],
        'subject_training_files': fids['AuditoryProcessing training'],
        'subject_training_pattern': lambda subject: f'P{subject}_BCMI_frontHN_2017.mat',

        'metadata': fids['AuditoryProcessing metadata'],
        'directory': 'databases/Auditory_processing',
    }

    # ----------------------------------------------------------------------

    def load_subject(self, subject: int, mode: 'str' = 'training', classes: Optional[list] = ALL) -> None:
        """"""
        self.data = super().load_subject(subject, mode)

    # ----------------------------------------------------------------------
    def get_run(self, run: int, classes: Optional[list] = ALL, channels: Optional[list] = ALL, reject_bad_trials: Optional[bool] = True) -> Tuple[np.ndarray, np.ndarray]:
        """"""
        classes = self.format_class_selector(classes)
        channels = self.format_channels_selectors(channels)
        super().get_run(run, classes, channels, reject_bad_trials)

        data = []
        classes_out = []

        for class_ in classes:
            # self.metadata['classes'][class_]

            cls_base = self.metadata['classes'][class_][:self.metadata['classes'][class_].find(
                '-')]
            cls = self.metadata['classes'][class_]
            all_ = np.array(
                [f'{cls_base}-{tempo}' for tempo in self.data[cls_base][0][0][1][0]])

            data.append(self.data[cls_base][0][0][0].T[all_ == cls])
            classes_out.extend([class_] * self.data[cls_base]
                               [0][0][0].T[all_ == cls].shape[0])

        run = np.concatenate(data)

        return run[:, channels - 1, :], np.array(classes_out)

    # ----------------------------------------------------------------------
    def non_task(self, non_task_classes: Optional[list] = ALL, runs: Optional[list] = ALL, channels: Optional[list] = ALL) -> np.ndarray:
        """"""
        channels = self.format_channels_selectors(channels)
        non_task_classes = self.format_non_class_selector(non_task_classes)
        runs = self.format_runs(runs)
        data = self.data['base'].T

        return [[data[channels - 1, :]]]


########################################################################
class CallableModule(ModuleType):
    # ----------------------------------------------------------------------
    def __call__(self, *args, **kwargs):
        """"""
        return Database(*args, **kwargs)


sys.modules[__name__].__class__ = CallableModule


