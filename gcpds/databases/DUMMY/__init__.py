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
                    ],
        'non_task_classes': ['non-task 0',
                             'non-task 1',
                             ],
        'sampling_rate': 1000,
        'montage': 'standard_1005',
        'tmin': -2,
        'duration': 7,
        'reference': '',
        'subjects': 7,
        'runs_training': [1] * 7,

        'subject_training_files': fids['DUMMY training'],
        'subject_training_pattern': lambda subject: f'dummy_data-{str(subject).rjust(2, "0")}.npy',

        'subject_evaluation_files': {},
        'subject_evaluation_pattern': lambda subject: f'dummy_data-{str(subject).rjust(2, "0")}.npy',

        'metadata': fids['DUMMY metadata'],
        'directory': 'databases/DUMMY',
    }

    # ----------------------------------------------------------------------
    def load_subject(self,
                     subject: int,
                     mode: str = 'training',
                     ) -> None:
        """"""
        self.data = super().load_subject(subject, mode)

    # ----------------------------------------------------------------------
    def get_run(self,
                run: int,
                classes: list = ALL,
                channels: Optional[list] = ALL,
                reject_bad_trials: Optional[bool] = True,
                ) -> Tuple[np.ndarray, np.ndarray]:
        """"""
        classes = self.format_class_selector(classes)
        channels = self.format_channels_selectors(channels)

    # ----------------------------------------------------------------------
    def get_data(self,
                 classes: Optional[list] = ALL,
                 channels: Optional[list] = ALL,
                 reject_bad_trials: Optional[bool] = True,
                 keep_runs_separated: bool = False,
                 ) -> list:
        """Return all runs."""
        classes = self.format_class_selector(classes)

    # ----------------------------------------------------------------------
    def non_task(self,
                 non_task_classes: Optional[list] = ALL,
                 runs: Optional = None,
                 channels: Optional[list] = ALL,
                 ) -> np.ndarray:
        """"""
        channels = self.format_channels_selectors(channels)
        non_task_classes = self.format_non_class_selector(non_task_classes)


########################################################################
class CallableModule(ModuleType):
    # ----------------------------------------------------------------------
    def __call__(self, *args, **kwargs):
        """"""
        return Database(*args, **kwargs)


sys.modules[__name__].__class__ = CallableModule

