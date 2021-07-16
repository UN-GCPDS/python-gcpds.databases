import os
import sys
import warnings
from types import ModuleType
from typing import Optional, Tuple

import numpy as np

from ..base import DatabaseBase, ALL, load_fids


########################################################################
class Database(DatabaseBase):
    """"""
    fids = load_fids(os.path.join(os.path.dirname(__file__), 'fids.json'))

    metadata = {
        'channels': ['Fp1', 'Fp2', 'Fpz', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5',
                     'FC1', 'FC2', 'FC6', 'M1', 'T7', 'C3', 'Cz', 'C4', 'T8',
                     'M2', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4',
                     'P8', 'POz', 'O1', 'Oz', 'O2', 'EOGh', 'EOGv', 'EMG_RH',
                     'EMG_LH', 'EMG_RF', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1',
                     'F2', 'F6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2', 'C6',
                     'CP3', 'CPz', 'CP4', 'P5', 'P1', 'P2', 'P6', 'PO5', 'PO3',
                     'PO4', 'PO6', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8',
                     'FT9', 'FT10', 'TPP9h', 'TPP10h', 'PO9', 'PO10', 'P9',
                     'P10', 'AFF1', 'AFz', 'AFF2', 'FFC5h', 'FFC3h', 'FFC4h',
                     'FFC6h', 'FCC5h', 'FCC3h', 'FCC4h', 'FCC6h', 'CCP5h',
                     'CCP3h', 'CCP4h', 'CCP6h', 'CPP5h', 'CPP3h', 'CPP4h',
                     'CPP6h', 'PPO1', 'PPO2', 'I1', 'Iz', 'I2', 'AFp3h',
                     'AFp4h', 'AFF5h', 'AFF6h', 'FFT7h', 'FFC1h', 'FFC2h',
                     'FFT8h', 'FTT9h', 'FTT7h', 'FCC1h', 'FCC2h', 'FTT8h',
                     'FTT10h', 'TTP7h', 'CCP1h', 'CCP2h', 'TTP8h', 'TPP7h',
                     'CPP1h', 'CPP2h', 'TPP8h', 'PPO9h', 'PPO5h', 'PPO6h',
                     'PPO10h', 'POO9h', 'POO3h', 'POO4h', 'POO10h', 'OI1h',
                     'OI2h',
                     ],
        'classes': ['right hand',
                    'left hand',
                    'rest',
                    'feet',
                    ],
        'sampling_rate': 500,
        'montage': 'standard_1005',
        'tmin': 0,
        'duration': 4,
        'reference': '',
        'subjects': 14,
        'runs_training': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'runs_evaluation': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],

        'subject_training_files': fids['HighGamma training'],
        'subject_training_pattern': lambda subject: os.path.join('train', f'{subject}.mat'),

        'subject_evaluation_files': fids['HighGamma evaluation'],
        'subject_evaluation_pattern': lambda subject: os.path.join('test', f'{subject}.mat'),

        'metadata': fids['HighGamma metadata'],
        'directory': 'databases/HighGamma-ME',
    }

    # ----------------------------------------------------------------------
    def load_subject(self, subject: int, mode: str = 'training') -> None:
        """"""
        data = super().load_subject(subject, mode)
        # self.runs = self.metadata[f'runs_{mode}'][subject - 1]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.data = data.root
            # DataTypeWarning: Unsupported type for attribute 'MATLAB_fields' in node 'mrk'
            self.data_mrk = data.root.mrk

    # ----------------------------------------------------------------------
    def get_run(self, run: int, classes: Optional[list] = ALL, channels: Optional[list] = ALL, reject_bad_trials: Optional[bool] = True) -> Tuple[np.ndarray, np.ndarray]:
        """"""
        classes = self.format_class_selector(classes)
        channels = self.format_channels_selectors(channels)
        super().get_run(run, classes, channels, reject_bad_trials)

        classes_list = self.data_mrk.event.desc.read()[0]
        starts = ((self.data_mrk.time.read() / 1000) * 500).T[0].astype(int)
        end = int(self.metadata['sampling_rate'] * self.metadata['duration'])

        data = np.concatenate(
            [getattr(self.data, f"ch{ch}") for ch in channels])

        run = np.array([data[:, start:start + end] for start in starts])

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

