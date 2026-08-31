"""
Time feature utilities.

This file follows the time feature encoding used in the official
Autoformer implementation.

Each time feature is normalized in the range [-0.5, 0.5].
"""

from typing import List

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


class TimeFeature:

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__ + "()"



class SecondOfMinute(TimeFeature):

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5



class MinuteOfHour(TimeFeature):

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5



class HourOfDay(TimeFeature):

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:      # 23pm is the last hour of the day -> 0.5; 0am is the first hour of the day -> -0.5, 12pm is the middle hour of the day -> 0
        return index.hour / 23.0 - 0.5



class DayOfWeek(TimeFeature):

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5



class DayOfMonth(TimeFeature):

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5



class DayOfYear(TimeFeature):

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5



class MonthOfYear(TimeFeature):

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5



class WeekOfYear(TimeFeature):

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week.astype(float) - 1) / 52.0 - 0.5




def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Return the time features appropriate for the given frequency string.

    Examples of supported frequencies:
    - "h" or "H": hourly
    - "t" or "T": minutely
    - "d" or "D": daily
    """

    features_by_offsets = {         # the more data is fine, the more features are extracted
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Second: [SecondOfMinute, MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
    }

    freq_str = freq_str.lower()
    freq_map = {
        "y": "YE",
        "a": "YE",
        "m": "ME",
        "w": "W",
        "d": "D",
        "b": "B",
        "h": "h",
        "t": "min",
        "min": "min",
        "s": "s",
    }
    if freq_str in freq_map:
        freq_str = freq_map[freq_str]

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
Unsupported frequency {freq_str}

Supported frequencies:
    Y   - yearly
    M   - monthly
    W   - weekly
    D   - daily
    B   - business day
    H   - hourly
    T   - minutely
    S   - secondly
"""
    raise RuntimeError(supported_freq_msg)


def time_features(dates, freq="h"):
    """
    Build normalized time features.

    Parameters
    ----------
    dates:
        Pandas Series, DatetimeIndex, or array-like object containing dates.
    freq:
        Frequency string.

    Returns
    -------
    data_stamp:
        Array with shape [num_timesteps, num_time_features].
    """

    dates = pd.to_datetime(dates)       # transform the input dates into a pandas DatetimeIndex object, e.g. ('2023-01-01 00:00:00', '2023-01-01 01:00:00', ...)

    if not isinstance(dates, pd.DatetimeIndex):
        dates = pd.DatetimeIndex(dates)

    data_stamp = np.vstack(
        [
            feat(dates)
            for feat in time_features_from_frequency_str(freq)
        ]
    )

    return data_stamp.transpose(1, 0)