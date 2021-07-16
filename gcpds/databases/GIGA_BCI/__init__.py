import os
from typing import Optional, Tuple

import numpy as np

from ..base import DatabaseBase, ALL, load_fids, load_mat


fids = load_fids(os.path.join(os.path.dirname(__file__), 'fids.json'))


########################################################################
class GIGA_BCI(DatabaseBase):
    """"""

    metadata = {
        'channels': ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1',
                     'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5',
                     'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8',
                     'PO9', 'O1', 'Oz', 'O2', 'PO10', 'FC3', 'FC4', 'C5', 'C1',
                     'C2', 'C6', 'CP3', 'CPz', 'CP4', 'P1', 'P2', 'POz', 'FT9',
                     'FTT9h', 'TTP7h', 'TP7', 'TPP9h', 'FT10', 'FTT10h',
                     'TPP8h', 'TP8', 'TPP10h', 'F9', 'F10', 'AF7', 'AF3', 'AF4',
                     'AF8', 'PO3', 'PO4',
                     ],
        'classes': ['right',
                    'left',
                    ],
        'non_task_classes': ['pre task',
                             'post task',
                             'eyemovement_blinking',
                             'eyemovement_horizen',
                             'eyemovement_vertical',
                             'teeth',
                             'EMG_bothHandGrasping',
                             ],
        'sampling_rate': 1000,
        'montage': 'standard_1020',
        'tmin': -3,
        'duration': 7,
        'reference': '',
        'subjects': 54,
        'runs_training': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                          2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                          2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                          ],
        'runs': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],

        # 'subject_files': list(filter(lambda f: 'MI' in f, fids['BCIilliteracy'])),
        'subject_files': {key: fids['BCIilliteracy'][key] for key in fids['BCIilliteracy'] if ('MI' in key or 'Artifact' in key)},
        'subject_pattern': lambda subject, run: os.path.join(f'session{run}', f'sess{str(run).rjust(2, "0")}_subj{str(subject).rjust(2, "0")}_EEG_MI.mat'),
        'artifact_pattern': lambda subject, run: os.path.join(f'session{run}', f'sess{str(run).rjust(2, "0")}_subj{str(subject).rjust(2, "0")}_EEG_Artifact.mat'),

        'metadata': fids['BCIilliteracy metadata'],
        'directory': 'databases/GIGA-BCI',

    }

    # ----------------------------------------------------------------------
    def load_subject(self, subject: int, mode: str) -> None:
        """"""
        if not mode in ['training', 'evaluation']:
            raise Exception(
                f"No mode {mode} available, only 'training', 'evaluation'")

        self.runs = self.metadata[f'runs'][subject - 1]

        if self.path is None:
            self.path = self.metadata['directory']

        sessions = []
        for run in range(self.runs):
            filename_subject = self.metadata[f'subject_pattern'](
                subject, run + 1)

            if os.path.split(filename_subject)[-1] not in self.metadata[f'subject_files'].keys():
                raise Exception(
                    f"Subject {subject} not in list of subjects.")

            fid, size = self.metadata[f'subject_files'][os.path.split(
                filename_subject)[-1]]

            self.subject = subject
            self.mode = mode

            sessions.append(load_mat(self.path, filename_subject, fid, size))

        artifacts = []
        for run in range(self.runs):
            filename_artifact = self.metadata[f'artifact_pattern'](
                subject, run + 1)

            fid, size = self.metadata[f'subject_files'][os.path.split(
                filename_artifact)[-1]]

            artifacts.append(
                load_mat(self.path, filename_artifact, fid, size))

        return sessions, artifacts

    # ----------------------------------------------------------------------
    def get_run(self, run: int, classes: Optional[list] = ALL, channels: Optional[list] = ALL, reject_bad_trials: Optional[bool] = True) -> Tuple[np.ndarray, np.ndarray]:
        """"""
        classes = self.format_class_selector(classes)
        channels = self.format_channels_selectors(channels)
        super().get_run(run, classes, channels, reject_bad_trials)

        data = self.data_[run]

        classes_list = data[4][0]
        starts = data[2][0]
        end = int(self.metadata['sampling_rate'] * self.metadata['duration'])

        run = np.array([data[1][start:start + end] for start in starts])

        # trial x channel x time
        run = np.moveaxis(run, 2, 1)

        # Select channels
        run = run[:, channels - 1, :]

        idx = []
        c = []
        for cls in classes:
            idx.append(np.where(np.array(classes_list) == cls + 1)[0])
            c.append([cls] * len(idx[-1]))

        return run[np.concatenate(idx), :, :], np.concatenate(c)

    # ----------------------------------------------------------------------
    def non_task(self, non_task_classes: Optional[list] = ALL, runs: Optional[list] = ALL, channels: Optional[list] = ALL) -> np.ndarray:
        """"""
        channels = self.format_channels_selectors(channels)
        non_task_classes = self.format_non_class_selector(non_task_classes)
        runs = self.format_runs(runs)

        rst = []
        for session in runs:
            # 13: pre_rest
            # 14: post_rest
            rst_rn = []
            nt = [self.data_[session][13].T, self.data_[session][14].T,
                  *np.moveaxis(self.artifacts_[session], 0, 1)]

            for index in non_task_classes:
                rst_rn.append(nt[index][channels - 1])
            rst.append(rst_rn)

        return rst


########################################################################
class MI(GIGA_BCI):
    """"""

    # ----------------------------------------------------------------------
    def load_subject(self, subject: int, mode: str = 'training') -> None:
        """"""
        data_, artifacts_ = super().load_subject(subject, mode)

        if mode == 'training':
            self.data_ = [d['EEG_MI_train'][0][0] for d in data_]
        elif mode == 'evaluation':
            self.data_ = [d['EEG_MI_test'][0][0] for d in data_]

        self.artifacts_ = [a['EEG_Artifact'][0][0][0] for a in artifacts_]


########################################################################
class ERP(GIGA_BCI):
    """"""

    metadata = GIGA_BCI.metadata.copy()
    metadata.update({
        'classes': ['target',
                    'nontarget',
                    ],
        'tmin': -4.5,
        'duration': 14.5,
        # 'subject_files': list(filter(lambda f: 'ERP' in f, fids['BCIilliteracy'])),
        'subject_files': {key: fids['BCIilliteracy'][key] for key in fids['BCIilliteracy'] if ('ERP' in key or 'Artifact' in key)},
        'subject_pattern': lambda subject, run: os.path.join(f'session{run}', f'sess{str(run).rjust(2, "0")}_subj{str(subject).rjust(2, "0")}_EEG_ERP.mat'),
    })

    # ----------------------------------------------------------------------

    def load_subject(self, subject: int, mode: str = 'training') -> None:
        """"""
        data_, artifacts_ = super().load_subject(subject, mode)

        if mode == 'training':
            self.data_ = [d['EEG_ERP_train'][0][0] for d in data_]
        elif mode == 'evaluation':
            self.data_ = [d['EEG_ERP_test'][0][0] for d in data_]

        self.artifacts_ = [a['EEG_Artifact'][0][0][0] for a in artifacts_]


########################################################################
class SSVEP(GIGA_BCI):
    """"""
    metadata = GIGA_BCI.metadata.copy()
    metadata.update({
        'classes': ['up', 'left', 'right', 'down'],
        'tmin': -4,
        'duration': 8,
        # 'subject_files': list(filter(lambda f: 'SSVEP' in f, fids['BCIilliteracy'])),
        'subject_files': {key: fids['BCIilliteracy'][key] for key in fids['BCIilliteracy'] if ('SSVEP' in key or 'Artifact' in key)},
        'subject_pattern': lambda subject, run: os.path.join(f'session{run}', f'sess{str(run).rjust(2, "0")}_subj{str(subject).rjust(2, "0")}_EEG_SSVEP.mat'),
    })

    # ----------------------------------------------------------------------
    def load_subject(self, subject: int, mode: str = 'training') -> None:
        """"""
        data_, artifacts_ = super().load_subject(subject, mode)

        if mode == 'training':
            self.data_ = [d['EEG_SSVEP_train'][0][0] for d in data_]
        elif mode == 'evaluation':
            self.data_ = [d['EEG_SSVEP_test'][0][0] for d in data_]

        self.artifacts_ = [a['EEG_Artifact'][0][0][0] for a in artifacts_]
