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

    metadata = {'channels': ['Fp1', 'AF7' , 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7'
                             , 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7',
                             'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7',
                             'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz',
                             'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz',
                             'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2',
                             'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6',
                             'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8',
                             'PO4', 'O2',
                             ],
    'classes': ['left hand mi',
                'right hand mi',
                'left hand mm',
                'right hand mm',
                ],
    'non_task_classes': ['resting',
                         'eye_blinking',
                         'eye_up_down',
                         'eye_left_right',
                         'jaw_clenching',
                         'head_left_right',
                         ],
    'sampling_rate': 512,
    'montage': 'standard_1005',
    'tmin': -2,
    'duration': 7,
    'reference': '',
    'subjects': 52,
    'runs_training': [5, 5, 5, 5, 5, 5, 6, 5, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                      5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                      5, 5, 5, 5, 5, 5, 5, 6, 5, 5, 5, 5, 5, 5],

    'subject_training_files': fids['GIGA training'],
    'subject_training_pattern': lambda subject: f's{str(subject).rjust(2, "0")}.mat',

    'subject_evaluation_files': {},
    'subject_evaluation_pattern': lambda subject: f's{str(subject).rjust(2, "0")}.mat',

    'metadata': fids['GIGA metadata'],
    'directory': 'databases/GIGA-MI_ME',
    }


    # ----------------------------------------------------------------------
    def load_subject(self, subject: int, mode: str = 'training') -> None:
        """"""
        data = super().load_subject(subject, mode)
        self.data = data['eeg'][0][0]

    # ----------------------------------------------------------------------
    def get_run(self, run: int, classes: list = ALL, channels: Optional[list] = ALL, reject_bad_trials: Optional[bool] = True) -> Tuple[np.ndarray, np.ndarray]:
        """"""
        classes = self.format_class_selector(classes)
        channels = self.format_channels_selectors(channels)
        # super().get_run(run, classes, channels, reject_bad_trials)

        if (0 in classes or 1 in classes) and (2 in classes or 3 in classes):
            raise Exception(
                '`get_run()` not support merged classes, use only `mm` or `mi`')

        if classes[0] in [2, 3]:  # mm
            if run > 0:
                raise Exception('There is only 1 run for motor movement.')
            reject_bad_trials = False
            COUNT = 6
            CUE = 5
            TRIAL = 3

        elif classes[0] in [0, 1]:  # mi
            super().get_run(run, classes, channels, reject_bad_trials)
            COUNT = 9
            CUE = 11
            TRIAL = 7

        BAD = 14
        # Index of starts of all cues
        all_cues = np.where(self.data[CUE][0] == 1)[0]

        # Split in runs and select the specified run
        trials_count = self.data[COUNT][0][0]
        cues = np.array([all_cues[i:i + 20]
                         for i in range(0, trials_count, 20)][run])

        start = (self.metadata['sampling_rate'] * 2) - 1
        end = int(self.metadata['sampling_rate']
                  * (self.metadata['duration']+self.metadata['tmin'])) + 1

        # reject bad trial
        if reject_bad_trials:
            bad_trials = {}
            for cls in classes:
                trials_runs = np.ones((trials_count,), dtype=bool)
                # 14 bad trials--bad trials MI
                tmp = self.data[BAD][0][0][1][0][cls]
                if len(tmp) != 0:
                    trials_runs[tmp - 1] = 0
                bad_trials[cls] = [trials_runs[i:i + 20]
                                   for i in range(0, trials_count, 20)]
        #
        trials = []
        classes_out = []
        for cls in classes:

            if cls > 1:
                # classes starts in index 7 for mm
                data = self.data[TRIAL + cls - 2]
            else:
                # classes starts in index 3 for mi
                data = self.data[TRIAL + cls]

            if reject_bad_trials:
                # cls*max_runs -- run-1
                # x = bad_trials[(cls * self.runs) + run]
                x = bad_trials[cls][run]
                cues_r = cues[x]
                if len(cues_r):
                    trials.extend([data[:, cue - start:cue + end]
                                   for cue in cues_r])
                    classes_out.extend([cls] * len(cues_r))
            else:
                trials.extend([data[:, cue - start:cue + end] for cue in cues])
                classes_out.extend([cls] * len(cues))

        if not len(trials):
            logging.warning(
                f'The subject {self.subject} in the run {run} has no data.')
            return None, None

        # Select only EEG channels
        run = np.array(trials)[:, :len(self.metadata['channels']), :]

        # Select channels
        run = run[:, channels - 1, :]

        return run, np.array(classes_out)

    # ----------------------------------------------------------------------
    def get_data(self, classes: Optional[list] = ALL, channels: Optional[list] = ALL, reject_bad_trials: Optional[bool] = True, keep_runs_separated: bool = False) -> list:
        """Return all runs."""
        classes = self.format_class_selector(classes)

        runs_copy = self.runs
        runs = []
        classes_out = []

        for cls in classes:
            if cls in [2, 3]:  # mm
                self.runs = 1
                r, c = super().get_data([cls], channels, reject_bad_trials)
            elif cls in [0, 1]:  # mi
                self.runs = runs_copy
                r, c = super().get_data([cls], channels, reject_bad_trials)

            runs.append(r)
            classes_out.append(c)

        self.runs = runs_copy

        if keep_runs_separated:
            return list(zip(runs, classes_out))

        else:
            return np.concatenate(runs), np.concatenate(classes_out)

    # ----------------------------------------------------------------------
    def non_task(self, non_task_classes: Optional[list] = ALL, runs: Optional[list] = None, channels: Optional[list] = ALL) -> np.ndarray:
        """"""
        channels = self.format_channels_selectors(channels)
        non_task_classes = self.format_non_class_selector(non_task_classes)
        all_non_task = (self.data[1], *[_[0] for _ in self.data[0]])
        non_task = []
        for index in non_task_classes:
            non_task.append(all_non_task[index][channels - 1])
        return [non_task]


########################################################################
class CallableModule(ModuleType):
    # ----------------------------------------------------------------------
    def __call__(self, *args, **kwargs):
        """"""
        return Database(*args, **kwargs)

sys.modules[__name__].__class__ = CallableModule
