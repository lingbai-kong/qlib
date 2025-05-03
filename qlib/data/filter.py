# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import print_function
from abc import abstractmethod

import re
import pandas as pd
import akshare as ak
import numpy as np
import abc

from .data import Cal, DatasetD


class BaseDFilter(abc.ABC):
    """Dynamic Instruments Filter Abstract class

    Users can override this class to construct their own filter

    Override __init__ to input filter regulations

    Override filter_main to use the regulations to filter instruments
    """

    def __init__(self):
        pass

    @staticmethod
    def from_config(config):
        """Construct an instance from config dict.

        Parameters
        ----------
        config : dict
            dict of config parameters.
        """
        raise NotImplementedError("Subclass of BaseDFilter must reimplement `from_config` method")

    @abstractmethod
    def to_config(self):
        """Construct an instance from config dict.

        Returns
        ----------
        dict
            return the dict of config parameters.
        """
        raise NotImplementedError("Subclass of BaseDFilter must reimplement `to_config` method")


class SeriesDFilter(BaseDFilter):
    """Dynamic Instruments Filter Abstract class to filter a series of certain features

    Filters should provide parameters:

    - filter start time
    - filter end time
    - filter rule

    Override __init__ to assign a certain rule to filter the series.

    Override _getFilterSeries to use the rule to filter the series and get a dict of {inst => series}, or override filter_main for more advanced series filter rule
    """

    def __init__(self, fstart_time=None, fend_time=None, keep=False):
        """Init function for filter base class.
            Filter a set of instruments based on a certain rule within a certain period assigned by fstart_time and fend_time.

        Parameters
        ----------
        fstart_time: str
            the time for the filter rule to start filter the instruments.
        fend_time: str
            the time for the filter rule to stop filter the instruments.
        keep: bool
            whether to keep the instruments of which features don't exist in the filter time span.
        """
        super(SeriesDFilter, self).__init__()
        self.filter_start_time = pd.Timestamp(fstart_time) if fstart_time else None
        self.filter_end_time = pd.Timestamp(fend_time) if fend_time else None
        self.keep = keep

    def _getTimeBound(self, instruments):
        """Get time bound for all instruments.

        Parameters
        ----------
        instruments: dict
            the dict of instruments in the form {instrument_name => list of timestamp tuple}.

        Returns
        ----------
        pd.Timestamp, pd.Timestamp
            the lower time bound and upper time bound of all the instruments.
        """
        trange = Cal.calendar(freq=self.filter_freq)
        ubound, lbound = trange[0], trange[-1]
        for _, timestamp in instruments.items():
            if timestamp:
                lbound = timestamp[0][0] if timestamp[0][0] < lbound else lbound
                ubound = timestamp[-1][-1] if timestamp[-1][-1] > ubound else ubound
        return lbound, ubound

    def _toSeries(self, time_range, target_timestamp):
        """Convert the target timestamp to a pandas series of bool value within a time range.
            Make the time inside the target_timestamp range TRUE, others FALSE.

        Parameters
        ----------
        time_range : D.calendar
            the time range of the instruments.
        target_timestamp : list
            the list of tuple (timestamp, timestamp).

        Returns
        ----------
        pd.Series
            the series of bool value for an instrument.
        """
        # Construct a whole dict of {date => bool}
        timestamp_series = {timestamp: False for timestamp in time_range}
        # Convert to pd.Series
        timestamp_series = pd.Series(timestamp_series)
        # Fill the date within target_timestamp with TRUE
        for start, end in target_timestamp:
            timestamp_series[Cal.calendar(start_time=start, end_time=end, freq=self.filter_freq)] = True
        return timestamp_series

    def _filterSeries(self, timestamp_series, filter_series):
        """Filter the timestamp series with filter series by using element-wise AND operation of the two series.

        Parameters
        ----------
        timestamp_series : pd.Series
            the series of bool value indicating existing time.
        filter_series : pd.Series
            the series of bool value indicating filter feature.

        Returns
        ----------
        pd.Series
            the series of bool value indicating whether the date satisfies the filter condition and exists in target timestamp.
        """
        fstart, fend = list(filter_series.keys())[0], list(filter_series.keys())[-1]
        filter_series = filter_series.astype("bool")  # Make sure the filter_series is boolean
        timestamp_series[fstart:fend] = timestamp_series[fstart:fend] & filter_series
        return timestamp_series

    def _toTimestamp(self, timestamp_series):
        """Convert the timestamp series to a list of tuple (timestamp, timestamp) indicating a continuous range of TRUE.

        Parameters
        ----------
        timestamp_series: pd.Series
            the series of bool value after being filtered.

        Returns
        ----------
        list
            the list of tuple (timestamp, timestamp).
        """
        # sort the timestamp_series according to the timestamps
        timestamp_series.sort_index()
        timestamp = []
        _lbool = None
        _ltime = None
        _cur_start = None
        for _ts, _bool in timestamp_series.items():
            # there is likely to be NAN when the filter series don't have the
            # bool value, so we just change the NAN into False
            if _bool == np.nan:
                _bool = False
            if _lbool is None:
                _cur_start = _ts
                _lbool = _bool
                _ltime = _ts
                continue
            if (_lbool, _bool) == (True, False):
                if _cur_start:
                    timestamp.append((_cur_start, _ltime))
            elif (_lbool, _bool) == (False, True):
                _cur_start = _ts
            _lbool = _bool
            _ltime = _ts
        if _lbool:
            timestamp.append((_cur_start, _ltime))
        return timestamp

    def __call__(self, instruments, start_time=None, end_time=None, freq="day"):
        """Call this filter to get filtered instruments list"""
        self.filter_freq = freq
        return self.filter_main(instruments, start_time, end_time)

    @abstractmethod
    def _getFilterSeries(self, instruments, fstart, fend):
        """Get filter series based on the rules assigned during the initialization and the input time range.

        Parameters
        ----------
        instruments : dict
            the dict of instruments to be filtered.
        fstart : pd.Timestamp
            start time of filter.
        fend : pd.Timestamp
            end time of filter.

        .. note:: fstart/fend indicates the intersection of instruments start/end time and filter start/end time.

        Returns
        ----------
        pd.Dataframe
            a series of {pd.Timestamp => bool}.
        """
        raise NotImplementedError("Subclass of SeriesDFilter must reimplement `getFilterSeries` method")

    def filter_main(self, instruments, start_time=None, end_time=None):
        """Implement this method to filter the instruments.

        Parameters
        ----------
        instruments: dict
            input instruments to be filtered.
        start_time: str
            start of the time range.
        end_time: str
            end of the time range.

        Returns
        ----------
        dict
            filtered instruments, same structure as input instruments.
        """
        lbound, ubound = self._getTimeBound(instruments)
        start_time = pd.Timestamp(start_time or lbound)
        end_time = pd.Timestamp(end_time or ubound)
        _instruments_filtered = {}
        _all_calendar = Cal.calendar(start_time=start_time, end_time=end_time, freq=self.filter_freq)
        _filter_calendar = Cal.calendar(
            start_time=self.filter_start_time and max(self.filter_start_time, _all_calendar[0]) or _all_calendar[0],
            end_time=self.filter_end_time and min(self.filter_end_time, _all_calendar[-1]) or _all_calendar[-1],
            freq=self.filter_freq,
        )
        _all_filter_series = self._getFilterSeries(instruments, _filter_calendar[0], _filter_calendar[-1])
        for inst, timestamp in instruments.items():
            # Construct a whole map of date
            _timestamp_series = self._toSeries(_all_calendar, timestamp)
            # Get filter series
            if inst in _all_filter_series:
                _filter_series = _all_filter_series[inst]
            else:
                if self.keep:
                    _filter_series = pd.Series({timestamp: True for timestamp in _filter_calendar})
                else:
                    _filter_series = pd.Series({timestamp: False for timestamp in _filter_calendar})
            # Calculate bool value within the range of filter
            _timestamp_series = self._filterSeries(_timestamp_series, _filter_series)
            # Reform the map to (start_timestamp, end_timestamp) format
            _timestamp = self._toTimestamp(_timestamp_series)
            # Remove empty timestamp
            if _timestamp:
                _instruments_filtered[inst] = _timestamp
        return _instruments_filtered


class NameDFilter(SeriesDFilter):
    """Name dynamic instrument filter

    Filter the instruments based on a regulated name format.

    A name rule regular expression is required.
    """

    def __init__(self, name_rule_re, fstart_time=None, fend_time=None):
        """Init function for name filter class

        Parameters
        ----------
        name_rule_re: str
            regular expression for the name rule.
        """
        super(NameDFilter, self).__init__(fstart_time, fend_time)
        self.name_rule_re = name_rule_re

    def _getFilterSeries(self, instruments, fstart, fend):
        all_filter_series = {}
        filter_calendar = Cal.calendar(start_time=fstart, end_time=fend, freq=self.filter_freq)
        for inst, timestamp in instruments.items():
            if re.match(self.name_rule_re, inst):
                _filter_series = pd.Series({timestamp: True for timestamp in filter_calendar})
            else:
                _filter_series = pd.Series({timestamp: False for timestamp in filter_calendar})
            all_filter_series[inst] = _filter_series
        return all_filter_series

    @staticmethod
    def from_config(config):
        return NameDFilter(
            name_rule_re=config["name_rule_re"],
            fstart_time=config["filter_start_time"],
            fend_time=config["filter_end_time"],
        )

    def to_config(self):
        return {
            "filter_type": "NameDFilter",
            "name_rule_re": self.name_rule_re,
            "filter_start_time": str(self.filter_start_time) if self.filter_start_time else self.filter_start_time,
            "filter_end_time": str(self.filter_end_time) if self.filter_end_time else self.filter_end_time,
        }


class ExpressionDFilter(SeriesDFilter):
    """Expression dynamic instrument filter

    Filter the instruments based on a certain expression.

    An expression rule indicating a certain feature field is required.

    Examples
    ----------
    - *basic features filter* : rule_expression = '$close/$open>5'
    - *cross-sectional features filter* : rule_expression = '$rank($close)<10'
    - *time-sequence features filter* : rule_expression = '$Ref($close, 3)>100'
    """

    def __init__(self, rule_expression, fstart_time=None, fend_time=None, keep=False):
        """Init function for expression filter class

        Parameters
        ----------
        fstart_time: str
            filter the feature starting from this time.
        fend_time: str
            filter the feature ending by this time.
        rule_expression: str
            an input expression for the rule.
        """
        super(ExpressionDFilter, self).__init__(fstart_time, fend_time, keep=keep)
        self.rule_expression = rule_expression

    def _getFilterSeries(self, instruments, fstart, fend):
        # do not use dataset cache
        try:
            _features = DatasetD.dataset(
                instruments,
                [self.rule_expression],
                fstart,
                fend,
                freq=self.filter_freq,
                disk_cache=0,
            )
        except TypeError:
            # use LocalDatasetProvider
            _features = DatasetD.dataset(instruments, [self.rule_expression], fstart, fend, freq=self.filter_freq)
        rule_expression_field_name = list(_features.keys())[0]
        all_filter_series = _features[rule_expression_field_name]
        return all_filter_series

    @staticmethod
    def from_config(config):
        return ExpressionDFilter(
            rule_expression=config["rule_expression"],
            fstart_time=config["filter_start_time"],
            fend_time=config["filter_end_time"],
            keep=config["keep"],
        )

    def to_config(self):
        return {
            "filter_type": "ExpressionDFilter",
            "rule_expression": self.rule_expression,
            "filter_start_time": str(self.filter_start_time) if self.filter_start_time else self.filter_start_time,
            "filter_end_time": str(self.filter_end_time) if self.filter_end_time else self.filter_end_time,
            "keep": self.keep,
        }

class STAKFilter(SeriesDFilter):
    """Non-ST dynamic instrument filter
    
    Filter the instruments which are not ST or *ST.
    """

    def __init__(self, fstart_time=None, fend_time=None):
        """Initialize ST stock filter
        
        Parameters
        ----------
        fstart_time : str, optional
            Start time for filtering, defaults to None
        fend_time : str, optional
            End time for filtering, defaults to None
        """
        super(STAKFilter, self).__init__(fstart_time, fend_time)


    def _getSTStockCodeList(self):
        stock_info_a_code_name_df = ak.stock_info_a_code_name()
        st_stock_code = stock_info_a_code_name_df[
            stock_info_a_code_name_df['name'].str.contains('ST')
        ]['code'].tolist()
        return st_stock_code

    def _getFilterSeries(self, instruments, fstart, fend):
        all_filter_series = {}
        filter_calendar = Cal.calendar(start_time=fstart, end_time=fend, freq=self.filter_freq)
        st_stock_code = self._getSTStockCodeList()
        for inst, timestamp in instruments.items():
            if inst[2:] in st_stock_code:
                _filter_series = pd.Series({timestamp: False for timestamp in filter_calendar})
            else:
                _filter_series = pd.Series({timestamp: True for timestamp in filter_calendar})
            all_filter_series[inst] = _filter_series
        return all_filter_series

    @staticmethod
    def from_config(config):
        return STAKFilter(
            fstart_time=config["filter_start_time"],
            fend_time=config["filter_end_time"],
        )

    def to_config(self):
        return {
            "filter_type": "STAKFilter",
            "filter_start_time": str(self.filter_start_time) if self.filter_start_time else self.filter_start_time,
            "filter_end_time": str(self.filter_end_time) if self.filter_end_time else self.filter_end_time,
        }

class NewStockAKFilter(SeriesDFilter):
    """Filter for newly listed stocks
    
    Filter stocks by comparing their listing date with the filter time range
    """
    
    def __init__(self, days_threshold=30, fstart_time=None, fend_time=None, keep=False):
        """Initialize new stock filter
        
        Parameters
        ----------
        days_threshold : int
            Stocks listed within this number of days are considered new
        fstart_time : str
            Filter start time 
        fend_time : str
            Filter end time
        keep : bool
            Whether to keep stocks with missing data
        """
        super(NewStockAKFilter, self).__init__(fstart_time, fend_time)
        self.days_threshold = days_threshold
        
    def _getStockDateList(self):
        """Get listing dates for all stocks"""
        sh_stock = ak.stock_info_sh_name_code("主板A股")[["证券代码", "上市日期"]].rename(columns={"证券代码": "code", "上市日期": "list_date"})
        sz_stock = ak.stock_info_sz_name_code("A股列表")[["A股代码", "A股上市日期"]].rename(columns={"A股代码": "code", "A股上市日期": "list_date"})
        bj_stock = ak.stock_info_bj_name_code()[["证券代码", "上市日期"]].rename(columns={"证券代码": "code", "上市日期": "list_date"})
        stock = pd.concat([sh_stock, sz_stock, bj_stock], axis=0)
        return stock.set_index('code')['list_date'].to_dict()
        
    def _getFilterSeries(self, instruments, fstart, fend):
        all_filter_series = {}
        filter_calendar = Cal.calendar(start_time=fstart, end_time=fend, freq=self.filter_freq)
        stock_list_date = self._getStockDateList()
        
        for inst, timestamp in instruments.items():
            code = inst[2:]
            try:
                list_date = pd.Timestamp(stock_list_date[code])
                # Calculate days since listing
                days_since_list = (fend - list_date).days
                if days_since_list <= self.days_threshold:
                    _filter_series = pd.Series({timestamp: False for timestamp in filter_calendar})
                else:
                    _filter_series = pd.Series({timestamp: True for timestamp in filter_calendar})
            except KeyError:
                # Handle case when stock code is not found in listing dates
                _filter_series = pd.Series({timestamp: self.keep for timestamp in filter_calendar})
                
            all_filter_series[inst] = _filter_series
        return all_filter_series

    @staticmethod
    def from_config(config):
        return NewStockAKFilter(
            days_threshold=config.get("days_threshold", 30),
            fstart_time=config.get("filter_start_time"),
            fend_time=config.get("filter_end_time"),
            keep=config.get("keep", False)
        )

    def to_config(self):
        return {
            "filter_type": "NewStockAKFilter",
            "days_threshold": self.days_threshold,
            "filter_start_time": str(self.filter_start_time) if self.filter_start_time else self.filter_start_time,
            "filter_end_time": str(self.filter_end_time) if self.filter_end_time else self.filter_end_time,
            "keep": self.keep
        }
    
# FixMe: Too many requests to akshare API, need to be optimized
class PEFilter(SeriesDFilter):
    """PE ratio dynamic filter
    
    Filter stocks based on PE ratio indicator
    """
    
    def __init__(self, pe_min=None, pe_max=None, fstart_time=None, fend_time=None, keep=False):
        """Initialize PE ratio filter
        
        Parameters
        ----------
        pe_min : float, optional
            Minimum allowed PE ratio
        pe_max : float, optional
            Maximum allowed PE ratio
        fstart_time : str, optional
            Filter start time
        fend_time : str, optional
            Filter end time
        keep : bool, optional
            Whether to keep stocks with missing data
        """
        super(PEFilter, self).__init__(fstart_time, fend_time, keep)
        self.pe_min = pe_min
        self.pe_max = pe_max
        
    def _getFilterSeries(self, instruments, fstart, fend):
        """Get filter series based on PE ratio criteria
        
        Parameters
        ----------
        instruments : dict
            Dictionary of instruments to filter
        fstart : pd.Timestamp
            Start time of filter period
        fend : pd.Timestamp
            End time of filter period
            
        Returns
        -------
        dict
            Dictionary of {instrument: filter_series}
        """
        from tqdm import tqdm
        from concurrent.futures import ThreadPoolExecutor, as_completed

        all_filter_series = {}
        filter_calendar = Cal.calendar(start_time=fstart, end_time=fend, freq=self.filter_freq)
        
        def fetch_data(code):
            try:
                df = ak.stock_a_indicator_lg(symbol=code)
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.set_index('trade_date')
                return code, df[(df.index >= fstart) & (df.index <= fend)]
            except Exception:
                return code, None
        
        with tqdm(total=len(instruments), desc="Processing PE filter") as pbar:
            with ThreadPoolExecutor(max_workers=200) as executor:
                future_to_code = {
                    executor.submit(fetch_data, inst[2:]): inst 
                    for inst in instruments.keys()
                }
                
                for future in as_completed(future_to_code):
                    inst = future_to_code[future]
                    code, df = future.result()
                    
                    try:
                        filter_series = pd.Series(True, index=filter_calendar)
                        if self.pe_min is not None:
                            filter_series &= df['pe_ttm'] >= self.pe_min
                        if self.pe_max is not None:
                            filter_series &= df['pe_ttm'] <= self.pe_max
                    except Exception:
                        filter_series = pd.Series(self.keep, index=filter_calendar)
                        
                    all_filter_series[inst] = filter_series
                    pbar.update(1)  # 更新进度条
                    
        return all_filter_series
# FixMe: Too many requests to akshare API, need to be optimized
class PBFilter(SeriesDFilter):
    """PB ratio dynamic filter
    
    Filter stocks based on PB ratio indicator
    """
    
    def __init__(self, pb_min=None, pb_max=None, fstart_time=None, fend_time=None, keep=False):
        """Initialize PB ratio filter
        
        Parameters
        ----------
        pb_min : float, optional
            Minimum allowed PB ratio
        pb_max : float, optional
            Maximum allowed PB ratio
        fstart_time : str, optional
            Filter start time
        fend_time : str, optional
            Filter end time
        keep : bool, optional
            Whether to keep stocks with missing data
        """
        super(PBFilter, self).__init__(fstart_time, fend_time, keep)
        self.pb_min = pb_min
        self.pb_max = pb_max
        
    def _getFilterSeries(self, instruments, fstart, fend):
        """Get filter series based on PB ratio criteria
        
        Parameters
        ----------
        instruments : dict
            Dictionary of instruments to filter
        fstart : pd.Timestamp
            Start time of filter period
        fend : pd.Timestamp
            End time of filter period
            
        Returns
        -------
        dict
            Dictionary of {instrument: filter_series}
        """
        from tqdm import tqdm
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        all_filter_series = {}
        filter_calendar = Cal.calendar(start_time=fstart, end_time=fend, freq=self.filter_freq)
        
        def fetch_data(code):
            try:
                df = ak.stock_a_indicator_lg(symbol=code)
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.set_index('trade_date')
                return code, df[(df.index >= fstart) & (df.index <= fend)]
            except Exception:
                return code, None
        
        # 添加进度条
        with tqdm(total=len(instruments), desc="Processing PB filter") as pbar:
            with ThreadPoolExecutor(max_workers=200) as executor:
                future_to_code = {
                    executor.submit(fetch_data, inst[2:]): inst 
                    for inst in instruments.keys()
                }
                
                for future in as_completed(future_to_code):
                    inst = future_to_code[future]
                    code, df = future.result()
                    
                    try:
                        filter_series = pd.Series(True, index=filter_calendar)
                        if self.pb_min is not None:
                            filter_series &= df['pb'] >= self.pb_min
                        if self.pb_max is not None:
                            filter_series &= df['pb'] <= self.pb_max
                    except Exception:
                        filter_series = pd.Series(self.keep, index=filter_calendar)
                        
                    all_filter_series[inst] = filter_series
                    pbar.update(1)  # 更新进度条
                    
        return all_filter_series

    @staticmethod
    def from_config(config):
        """Create filter instance from config dictionary
        
        Parameters
        ----------
        config : dict
            Configuration dictionary
            
        Returns
        -------
        PBFilter
            PB filter instance
        """
        return PBFilter(
            pb_min=config.get("pb_min"),
            pb_max=config.get("pb_max"),
            fstart_time=config.get("filter_start_time"),
            fend_time=config.get("filter_end_time"),
            keep=config.get("keep", False)
        )

    def to_config(self):
        """Convert filter instance to config dictionary
        
        Returns
        -------
        dict
            Configuration dictionary
        """
        return {
            "filter_type": "PBFilter",
            "pb_min": self.pb_min,
            "pb_max": self.pb_max,
            "filter_start_time": str(self.filter_start_time) if self.filter_start_time else self.filter_start_time,
            "filter_end_time": str(self.filter_end_time) if self.filter_end_time else self.filter_end_time,
            "keep": self.keep,
        }