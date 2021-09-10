import os
import sys
from types import ModuleType
from typing import Optional, Tuple

import mne
import numpy as np

from ...base import DatabaseBase, ALL, load_fids


########################################################################
class Database(DatabaseBase):
    """"""
    fids = load_fids(os.path.join(os.path.dirname(__file__), 'fids.json'))

    metadata = {
        'channels': ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1',
                     'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
                     'P1', 'Pz', 'P2', 'POz',
                     'EOG-left', 'EOG-central', 'EOG-right'],
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
        'subject_training_pattern': lambda subject: f'A{str(subject).rjust(2, "0")}T.gdf',

        'subject_evaluation_files': fids['BCI2a evaluation'],
        'subject_evaluation_pattern': lambda subject: f'A{str(subject).rjust(2, "0")}E.gdf',

        'metadata': fids['BCI2a metadata'],
        'directory': 'databases/BCI_Competition_IV/dataset_2a',
    }

    # ----------------------------------------------------------------------
    def load_subject(self, subject: int, mode: str = 'training') -> None:
        """"""
        self.data = super().load_subject(subject, mode)

    # ----------------------------------------------------------------------
    def get_run(self, run: int, classes: Optional[list] = ALL, channels: Optional[list] = ALL, reject_bad_trials: Optional[bool] = True) -> Tuple[np.ndarray, np.ndarray]:
        """"""
        classes = self.format_class_selector(classes)
        channels = self.format_channels_selectors(channels)
        super().get_run(run, classes, channels, reject_bad_trials)

        # Renombrar anotaciones para que sea más fácil de manipular
        ann = self.data.annotations
        ann.description = [d.replace('769', 'left hand')
                           for d in ann.description]
        ann.description = [d.replace('770', 'right hand')
                           for d in ann.description]
        ann.description = [d.replace('771', 'feet') for d in ann.description]
        ann.description = [d.replace('772', 'tongue')
                           for d in ann.description]
        ann.description = [d.replace('32766', 'run')
                           for d in ann.description]

        # Los bad trial DEBEN iniciar con la palabra "bad" para que MNE los elimine de forma automática
        if reject_bad_trials:
            ann.description = [d.replace('1023', 'bad trial')
                               for d in ann.description]
        self.data.set_annotations(ann)

        events, event_id = mne.events_from_annotations(self.data)
        sesions = events[events[:, 2] == event_id['run']][:, 0]
        sesions = np.concatenate([sesions, [events[-1][0]]])

        split_events = []
        for i in range(len(sesions) - 1):
            sub_events = events[(events[:, 0] >= sesions[i]) & (
                events[:, 0] <= sesions[i + 1])]
            if sub_events.shape[0] > 10:
                split_events.append(sub_events)

        event_id = {k: event_id[k]
                    for k in event_id if k in self.metadata['classes']}

        epochs = mne.Epochs(self.data, split_events[run], event_id, tmin=self.metadata['tmin'], preload=True,
                            tmax=self.metadata['duration'] + self.metadata['tmin'], reject_by_annotation=True, event_repeated='merge')
        epochs.rename_channels({old: new for old, new in zip(
            epochs.ch_names, self.metadata['channels'])})
        epochs.pick_channels(np.array(self.metadata['channels'])[channels])

        data = []
        classes_ = []
        for cls in classes:

            data.append(epochs[self.metadata['classes'][cls]].get_data())
            classes_.append([cls] * data[-1].shape[0])

        return np.concatenate(data), np.concatenate(classes_)


########################################################################
class CallableModule(ModuleType):
    # ----------------------------------------------------------------------
    def __call__(self, *args, **kwargs):
        """"""
        return Database(*args, **kwargs)


sys.modules[__name__].__class__ = CallableModule
