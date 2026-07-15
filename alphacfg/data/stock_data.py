from enum import IntEnum
import os
from pathlib import Path
from typing import List, Union, Optional, Tuple

import numpy as np
import pandas as pd
import torch


class FeatureType(IntEnum):
    OPEN = 0
    CLOSE = 1
    HIGH = 2
    LOW = 3
    VOLUME = 4
    VWAP = 5


class StockData:
    _qlib_initialized: bool = False
    _provider_uri: Optional[str] = None
    _region: Optional[str] = None

    def __init__(self,
                 instrument: Union[str, List[str]],
                 start_time: str,
                 end_time: str,
                 max_backtrack_days: int = 100,
                 max_future_days: int = 30,
                 features: Optional[List[FeatureType]] = None,
                 device: torch.device = torch.device('cuda:0')) -> None:
        self._init_qlib()

        self._instrument = instrument
        self.max_backtrack_days = max_backtrack_days
        self.max_future_days = max_future_days
        self._start_time = start_time
        self._end_time = end_time
        self._features = features if features is not None else list(FeatureType)
        self.device = device
        self.data, self._dates, self._stock_ids = self._get_data()

    @classmethod
    def _init_qlib(cls) -> None:
        if cls._qlib_initialized:
            return
        import qlib
        from qlib.config import REG_CN, REG_US
        provider_uri = cls._resolve_provider_uri()
        region = cls._resolve_region(provider_uri)
        qlib.init(provider_uri=provider_uri, region=REG_US if region == "us" else REG_CN)
        cls._qlib_initialized = True

    @classmethod
    def _resolve_provider_uri(cls) -> str:
        if cls._provider_uri is not None:
            return cls._provider_uri

        env_provider_uri = os.environ.get("ALPHACFG_QLIB_PROVIDER_URI")
        if env_provider_uri:
            cls._provider_uri = str(Path(env_provider_uri).expanduser())
            return cls._provider_uri

        project_provider_uri = Path(__file__).resolve().parents[2] / "data" / "cn_data_rolling"
        if project_provider_uri.exists():
            cls._provider_uri = str(project_provider_uri)
            return cls._provider_uri

        cls._provider_uri = str(Path("~/.qlib/qlib_data/cn_data_rolling").expanduser())
        return cls._provider_uri

    @classmethod
    def _resolve_region(cls, provider_uri: str) -> str:
        """Resolve the Qlib market region from the environment or data path."""
        if cls._region is not None:
            return cls._region

        region = os.environ.get("ALPHACFG_QLIB_REGION", "").strip().lower()
        if not region:
            region = "us" if "us_data" in Path(provider_uri).name.lower() else "cn"
        if region not in {"cn", "us"}:
            raise ValueError("ALPHACFG_QLIB_REGION must be 'cn' or 'us'")
        cls._region = region
        return region

    def _load_exprs(self, exprs: Union[str, List[str]]) -> pd.DataFrame:
        # This evaluates an expression on the data and returns the dataframe
        # It might throw on illegal expressions like "Ref(constant, dtime)"
        from qlib.data.dataset.loader import QlibDataLoader
        from qlib.data import D
        if not isinstance(exprs, list):
            exprs = [exprs]
        cal: np.ndarray = D.calendar()
        start_index = int(cal.searchsorted(pd.Timestamp(self._start_time), side="left"))  # type: ignore
        end_index = int(cal.searchsorted(pd.Timestamp(self._end_time), side="right")) - 1  # type: ignore
        if start_index - self.max_backtrack_days < 0:
            raise ValueError(
                f"start_time={self._start_time} does not have enough "
                f"backtrack days ({self.max_backtrack_days})"
            )
        if end_index < start_index:
            raise ValueError(f"empty calendar range: {self._start_time} ~ {self._end_time}")
        if end_index + self.max_future_days >= len(cal):
            raise ValueError(
                f"end_time={self._end_time} does not have enough "
                f"future days ({self.max_future_days}); latest calendar is {cal[-1]}"
            )
        real_start_time = cal[start_index - self.max_backtrack_days]
        real_end_time = cal[end_index + self.max_future_days]
        return (QlibDataLoader(config=exprs)  # type: ignore
                .load(self._instrument, real_start_time, real_end_time))

    def _get_data(self) -> Tuple[torch.Tensor, pd.Index, pd.Index]:
        features = ['$' + f.name.lower() for f in self._features]
        df = self._load_exprs(features)
        dates = pd.Index(sorted(df.index.get_level_values("datetime").unique()))
        stock_ids = pd.Index(sorted(df.index.get_level_values("instrument").unique()))
        values = []
        for feature in features:
            feature_df = df[feature].unstack("instrument")
            feature_df = feature_df.reindex(index=dates, columns=stock_ids)
            values.append(feature_df.to_numpy(dtype=np.float32))
        values = np.stack(values, axis=1)

        return torch.tensor(values, dtype=torch.float, device=self.device), dates, stock_ids

    @property
    def n_features(self) -> int:
        return len(self._features)

    @property
    def n_stocks(self) -> int:
        return self.data.shape[-1]

    @property
    def n_days(self) -> int:
        return self.data.shape[0] - self.max_backtrack_days - self.max_future_days

    def make_dataframe(self, data: Union[torch.Tensor, List[torch.Tensor]], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Parameters:
        - `data`: a tensor of size `(n_days, n_stocks[, n_columns])`, or a list of tensors of size `(n_days, n_stocks)`
        - `columns`: an optional list of column names
        """
        if isinstance(data, list):
            data = torch.stack(data, dim=2)
        if len(data.shape) == 2:
            data = data.unsqueeze(2)
        if columns is None:
            columns = [str(i) for i in range(data.shape[2])]
        n_days, n_stocks, n_columns = data.shape
        if self.n_days != n_days:
            raise ValueError(f"number of days in the provided tensor ({n_days}) doesn't "
                             f"match that of the current StockData ({self.n_days})")
        if self.n_stocks != n_stocks:
            raise ValueError(f"number of stocks in the provided tensor ({n_stocks}) doesn't "
                             f"match that of the current StockData ({self.n_stocks})")
        if len(columns) != n_columns:
            raise ValueError(f"size of columns ({len(columns)}) doesn't match with "
                             f"tensor feature count ({data.shape[2]})")
        if self.max_future_days == 0:
            date_index = self._dates[self.max_backtrack_days:]
        else:
            date_index = self._dates[self.max_backtrack_days:-self.max_future_days]
        index = pd.MultiIndex.from_product([date_index, self._stock_ids])
        data = data.reshape(-1, n_columns)
        return pd.DataFrame(data.detach().cpu().numpy(), index=index, columns=columns)
