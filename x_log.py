"""
simple loggers
"""
import os
import os.path as osp
import numpy as np
import pandas as pd

class PLog:
    """ simple logger based in pandas
    Examples:
        log = PDLog(name)

        log.write(new_frame=True, Epoch=1, Iter=120, Loss=0.2, Acc=34)
    # or
        ...
        log.collect(new_frame=True, Epoch=1)
        log.collect(Iter=120)
        ...
        log.write() # write flushes .frame
    """
    def __init__(self, name, **kwargs):
        """ init reads csv if exists
        Args
            name     (str) csvfile
            iloc        (int[-1]) returns location as frame

        """
        self.name = osp.abspath(osp.expanduser(name))
        self.columns = None
        self.values = {}
        self.frame = {}
        self.len = 0

        self._log_interval = -1 if "log_interval" not in kwargs else kwargs["log_interval"]
        self._end = {"end":"\r"} if "end" not in kwargs else kwargs["end"]
        self._allow_missing = True if "allow_missing" not in kwargs else kwargs["allow_missing"]
        _iloc = -1 if "iloc" not in kwargs else kwargs["iloc"]
        self.read(iloc=_iloc, init=True)

    def read(self, iloc=-1, init=False):
        """
        Args
            iloc: fills self.values with dfl.iloc[iloc]
        """
        if not osp.isfile(self.name):
            os.makedirs(osp.split(self.name)[0], exist_ok=True)
            return None

        dfl = pd.read_csv(self.name)
        self.len = len(dfl)
        self.columns = list(dfl.columns)
        if len(dfl) > abs(iloc):
            self.values = dict(dfl.iloc[iloc])
        if init:
            print(f"{self.name} found with len {self.len}\n{self.values}")
        return dfl

    def extend_keys(self, new_keys):
        """ adds new nan values for all rows of new keys
            overwrites stored file with new with kesy
        """
        dfl = self.read()
        for key in new_keys:
            if key not in dfl:
                dfl[key] = [np.nan for i in range(self.len)]
        self.columns = list(dfl.columns)
        dfl.to_csv(self.name, index=False)

    def _check_for_armaggeddon(self, **values):
        if self.columns is not None:
            _bad = [key for key in values if key not in self.columns]
            assert not _bad, f"keys {_bad} not in columns {self.columns}, to add new key run, self.extend_keys({_bad})"

    def collect(self, new_frame=False, **values):
        """collect key values to make dataframe
            values need to be key:[valyue]
        """
        if new_frame:
            self.frame = {}
        self._check_for_armaggeddon(**values)
        self.frame.update(**values)

    def _fix_columns(self):
        if self.columns is not None:
            _frame = {}
            for col in self.columns:
                if col not in self.frame:
                    assert self._allow_missing, "missing columns not allowed, pass [np.nan] to .write() or PLog(allow_missing=True)"
                    _frame[col] = [np.nan]
                else:
                    _frame[col] = self.frame[col]
            self.frame = _frame
        else:
            self.columns = list(self.frame.keys())

        _lens = []
        for col in self.frame:
            if not isinstance(self.frame[col], list):
                self.frame[col] = [self.frame[col]]
            _len = len(self.frame[col])
            if _len not in _lens:
                _lens.append(_len)
        assert len(_lens) == 1, f"multiple column lengths found, {_lens}"

    def write(self, new_frame=False, **values):
        """collect key values to make dataframe
        """
        # build dict
        self.collect(new_frame=new_frame, **values)
        self._fix_columns()

        # write to csv
        dfl = pd.DataFrame(self.frame)
        dfl.to_csv(self.name, index=False, mode='a',
                   header=not osp.isfile(self.name))
        # cleanup
        self.len += 1
        self.values = {**self.frame}
        self.frame = {}
        # log
        if not self.len%self._log_interval:
            if self.len == 1:
                print("\t".join(list(self.values.keys())))
            print("\t".join([str(l[0]).replace("nan", "") for l in self.values.values()]),
                  **self._end)
