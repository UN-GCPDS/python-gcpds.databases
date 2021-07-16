import os
import sys
from types import ModuleType
from typing import Optional, Tuple

import numpy as np

from ..base import DatabaseBase, ALL, load_fids, load_mat


########################################################################
class Database(DatabaseBase):
    """"""
    fids = load_fids(os.path.join(os.path.dirname(__file__), 'fids.json'))

    metadata = {
        'channels': ['Fc5', 'Fc3', 'Fc1', 'Fcz', 'Fc2', 'Fc4', 'Fc6', 'C5',
                     'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'Cp5', 'Cp3', 'Cp1',
                     'Cpz', 'Cp2', 'Cp4', 'Cp6', 'Fp1', 'Fpz', 'Fp2', 'Af7',
                     'Af3', 'Afz', 'Af4', 'Af8', 'F7', 'F5', 'F3', 'F1', 'Fz',
                     'F2', 'F4', 'F6', 'F8', 'Ft7', 'Ft8', 'T7', 'T8', 'T9',
                     'T10', 'Tp7', 'Tp8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2',
                     'P4', 'P6', 'P8', 'Po7', 'Po3', 'Poz', 'Po4', 'Po8', 'O1',
                     'Oz', 'O2', 'Iz',
                     ],
        'classes': ['right fist mi',
                    'left fist mi',
                    'both fist mi',
                    'both feet mi',
                    'right fist mm',
                    'left fist mm',
                    'both fist mm',
                    'both feet mm',
                    ],
        'sampling_rate': 160,
        'montage': 'standard_1005',
        'tmin': -4,
        'duration': 8,
        'reference': '',
        'subjects': 109,
        'runs': [3] * 109,

        'subject_files': fids['PhysionetMMI training'],
        'subject_pattern': lambda subject, run: os.path.join(f'S{str(subject).rjust(3, "0")}', f'S{str(subject).rjust(3, "0")}R{str(run).rjust(2, "0")}.edf'),

        'metadata': fids['PhysionetMMI metadata'],
        'directory': 'databases/PhysioNet-MI_ME',

    }

    # ----------------------------------------------------------------------
    def __init__(self, path: Optional[str] = None) -> None:
        """Constructor"""
        self.path = path

        self.classes = {
            'right fist mi': ([4, 8, 12], 'T1'),
            'left fist mi': ([4, 8, 12], 'T2'),

            'both fist mi': ([6, 10, 14], 'T1'),
            'both feet mi': ([6, 10, 14], 'T2'),

            'right fist mm': ([3, 7, 11], 'T1'),
            'left fist mm': ([3, 7, 11], 'T2'),

            'both fist mm': ([5, 9, 13], 'T1'),
            'both feet mm': ([5, 9, 13], 'T2'),
        }

    # ----------------------------------------------------------------------
    def load_subject(self, subject: int, mode: str = 'training', classes: list = ALL) -> None:
        """"""
        if not mode in ['training', 'evaluation']:
            raise Exception(
                f"No mode {mode} available, only 'training', 'evaluation'")

        self.runs = self.metadata[f'runs'][subject - 1]

        if self.path is None:
            self.path = self.metadata['directory']

        if classes != ALL:
            classes_runs = set(np.concatenate(
                [self.classes[cls][0] for cls in classes]).tolist())

        sessions = []
        for run in range(1, 15):

            if classes != ALL and (run not in classes_runs):
                sessions.append([])
                continue

            filename_subject = self.metadata[f'subject_pattern'](
                subject, run)

            if os.path.split(filename_subject)[-1] not in self.metadata[f'subject_files'].keys():
                raise Exception(
                    f"Subject {subject} not in list of subjects.")

            fid, size = self.metadata[f'subject_files'][os.path.split(
                filename_subject)[-1]]

            self.subject = subject
            self.mode = mode

            sessions.append(load_mat(self.path, filename_subject, fid, size))

        self.data_ = sessions

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
            runs, desc = self.classes[self.metadata['classes'][class_]]

            if self.data_[runs[run] - 1]:
                raw_data = self.data_[runs[run] - 1].get_data()
                eeg = np.array([raw_data[:, int((cl - 4) * 160):int((cl - 4) * 160) + (160 * 8)] for cl in self.data_[
                               runs[run] - 1].annotations.onset[self.data_[runs[run] - 1].annotations.description == desc]])
                data.append(eeg)
                classes_out.extend([class_] * eeg.shape[0])

        run = np.concatenate(data)

        return run[:, channels - 1, :], np.array(classes_out)


########################################################################
class CallableModule(ModuleType):
    # ----------------------------------------------------------------------
    def __call__(self, *args, **kwargs):
        """"""
        return Database(*args, **kwargs)


sys.modules[__name__].__class__ = CallableModule

