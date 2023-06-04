"""
This is a file script used for loading the dataset
"""

import datetime
import pathlib
from typing import Tuple, List, Union, Callable, Optional
import pdb
import os
import pdb

import numpy as np
import pandas as pd

import dv_processing as dv
from torch.utils.data import Dataset


class AedatProcessorBase:
    """
    Base class processing aedat4 files.

    Manages basic bookkeeping which is reused between multiple implementations.
    """

    def __init__(self, path, filter_noise):
        # Aedat4 recording file
        self.path = path
        self.recording = dv.io.MonoCameraRecording(str(path))
        self.lowest_ts, self.highest_ts = self.recording.getTimeRange()

        # Filter chain removing noise
        self.filter_noise = filter_noise
        self.filter_chain = dv.EventFilterChain()
        if filter_noise:
            self.filter_chain.addFilter(dv.RefractoryPeriodFilter(self.recording.getEventResolution(), refractoryPeriod=datetime.timedelta(microseconds=2000)))
            self.filter_chain.addFilter(dv.noise.BackgroundActivityNoiseFilter(self.recording.getEventResolution()))

        # Bookkeeping
        self.current_ts = self.lowest_ts

    def restore_filter_chain(self):
        # Filter chain removing noise
        self.filter_chain = dv.EventFilterChain()
        if self.filter_noise:
            self.filter_chain.addFilter(dv.RefractoryPeriodFilter(self.recording.getEventResolution(), refractoryPeriod=datetime.timedelta(microseconds=2000)))
            self.filter_chain.addFilter(dv.noise.BackgroundActivityNoiseFilter(self.recording.getEventResolution()))

    def get_recording_time_range(self):
        """Get the time range of the aedat4 file recording."""
        return self.lowest_ts, self.highest_ts

    def get_current_ts(self):
        """Get the most recent readout timestamp."""
        return self.current_ts

    def __read_raw_events_until(self, timestamp):
        assert timestamp >= self.current_ts
        assert timestamp >= self.lowest_ts
        assert timestamp <= self.highest_ts

        events = self.recording.getEventsTimeRange(int(self.current_ts), int(timestamp))
        self.current_ts = timestamp

        return events

    def read_events_until(self, timestamp):
        """Read event from aedat4 file until the given timestamp."""
        events = self.__read_raw_events_until(timestamp)
        self.filter_chain.accept(events)
        return self.filter_chain.generateEvents()

    def generate_frame(self, timestamp):
        """Generate an image frame at the given timestamp."""
        raise NotImplementedError


class AedatProcessorLinear(AedatProcessorBase):
    """Aedat file processor using accumulator with linear decay."""

    def __init__(self, path, contribution, decay, neutral_val, ignore_polarity=False, filter_noise=False):
        """
        Constructor.

        :param path: path to an aedat4 file to read
        :param contribution: event contribution
        :param decay: accumulator decay (linear)
        :param neutral_val:
        :param ignore_polarity:
        :param filter_noise: if true, noise pixels will be filtered out
        """
        super().__init__(path, filter_noise)

        # Accumulator drawing the events on images
        self.accumulator = dv.Accumulator(self.recording.getEventResolution(),
                                          decayFunction=dv.Accumulator.Decay.LINEAR,
                                          decayParam=decay,
                                          synchronousDecay=True,
                                          eventContribution=contribution,
                                          maxPotential=1.0,
                                          neutralPotential=neutral_val,
                                          minPotential=0.0,
                                          rectifyPolarity=ignore_polarity)

    def collect_events(self, start_timestamp, end_timestamp)-> np.array:
        
        # slice the event array
        events = self.read_events_until(end_timestamp)
        return events.sliceTime(start_timestamp)


    def generate_frame(self, timestamp) -> np.ndarray:
        """
        Generate a 1D frame from events
        """
        events = self.read_events_until(timestamp)
        self.accumulator.accept(events)
        image = self.accumulator.generateFrame().image
        assert image.dtype == np.uint8

        return image


def read_csv(path, is_with_ellipsis, is_with_coords):
    """
    Read a csv file and reatain all columns with the listed column names.
    Depending on the configuation, a different set of columns from the file is retained
    """
    header_items = ['timestamp', 'possible']
    if is_with_coords is True:
        header_items.append('center_x')
        header_items.append('center_y')
    if is_with_ellipsis is True:
        header_items.append('axis_x')
        header_items.append('axis_y')
        header_items.append('angle')

    label_file_df = pd.read_csv(path)
    label_file_df = label_file_df[header_items]

    return label_file_df



class EyeTrackingInivationDataset():
    'Initialize by creating a time ordered stack of frames and events'
    def __init__(
        self, 
        data_dir: str, 
        save_to: str,
        list_experiments: List, 
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        
        # Transforms
        self.transform = transform
        self.target_transform = target_transform

        self.y = pd.read_csv(os.path.join(data_dir, "silver.csv"))
        self.data_dir = data_dir
        experiments = np.unique(self.y["exp_name"]).tolist()

        filter_values = [experiments[item] for item in list_experiments]
        self.y = self.y[self.y['exp_name'].isin(filter_values)]

    def __len__(self):
        return len(self.y)
    
    def __repr__(self):
        return self.__class__.__name__
    
    def __getitem__(self, index):

        row = self.y.iloc[index]
        label =  (row["x_coord"], row["y_coord"])

        aedat_path = pathlib.Path(os.path.join(self.data_dir, row["exp_name"], "events.aedat4"))
        aedat_processor = AedatProcessorLinear(aedat_path, 0.25, 1e-7, 0.5)

        events = aedat_processor.read_events_until(row["t_end"])
        coord = events.coordinates()
        timestamp = events.timestamps()
        features = events.polarities().astype(np.byte)

        data = {
            'coords': coord,
            'ts': timestamp,
            'polarity': features
        }


        if self.transform:
            data = self.transform(data)
        
        if self.target_transform:
            label = self.target_transform(label)

        return data, label, 

