from typing import List, Tuple, Optional, Sequence
from torch import Tensor
import torch
from alphacfg.pytorch_utils import normalize_by_day

from .calculator_ import AlphaCalculator
from .correlation import batch_pearsonr, batch_spearmanr
from .expression import Expression
from .stock_data import StockData


class QLibStockDataCalculator(AlphaCalculator):
    def __init__(self, data: StockData, target: Optional[Expression]):
        self.data = data

        if target is None: # Combination-only mode
            self.target_value = None
        else:
            self.target_value = normalize_by_day(target.evaluate(self.data))

    def evaluate_alpha(self, expr: Expression) -> Tensor:
            return normalize_by_day(expr.evaluate(self.data))
    def _calc_alpha(self, expr: Expression) -> Tensor:
        return normalize_by_day(expr.evaluate(self.data))

    def _calc_IC(self, value1: Tensor, value2: Tensor) -> float:
        return batch_pearsonr(value1, value2).mean().item()

    def _calc_rIC(self, value1: Tensor, value2: Tensor) -> float:
        return batch_spearmanr(value1, value2).mean().item()

    def make_ensemble_alpha(self, exprs: Sequence[Expression], weights: Sequence[float]) -> Tensor:
        n = len(exprs)
        factors = [self.evaluate_alpha(exprs[i]) * weights[i] for i in range(n)]
        return torch.sum(torch.stack(factors, dim=0), dim=0)

    def calc_single_IC_ret(self, expr: Expression) -> float:
        value = self._calc_alpha(expr)
        return self._calc_IC(value, self.target_value)

    def calc_single_rIC_ret(self, expr: Expression) -> float:
        value = self._calc_alpha(expr)
        return self._calc_rIC(value, self.target_value)

    def calc_single_all_ret(self, expr: Expression) -> Tuple[float, float]:
        value = self._calc_alpha(expr)
        return self._calc_IC(value, self.target_value), self._calc_rIC(value, self.target_value)

    def calc_mutual_IC(self, expr1: Expression, expr2: Expression) -> float:
        value1, value2 = self._calc_alpha(expr1), self._calc_alpha(expr2)
        return self._calc_IC(value1, value2)

    def calc_pool_IC_ret(self, exprs: List[Expression], weights: List[float]) -> float:
        with torch.no_grad():
            ensemble_value = self.make_ensemble_alpha(exprs, weights)
            return self._calc_IC(ensemble_value, self.target_value)

    def calc_pool_rIC_ret(self, exprs: List[Expression], weights: List[float]) -> float:
        with torch.no_grad():
            ensemble_value = self.make_ensemble_alpha(exprs, weights)
            return self._calc_rIC(ensemble_value, self.target_value)

    def calc_pool_all_ret(self, exprs: List[Expression], weights: List[float]) -> Tuple[float, float]:
        with torch.no_grad():
            ensemble_value = self.make_ensemble_alpha(exprs, weights)
            return self._calc_IC(ensemble_value, self.target_value), self._calc_rIC(ensemble_value, self.target_value)


class TensorAlphaCalculator(AlphaCalculator):
    def __init__(self, target: Optional[Tensor], data: StockData) -> None:
        self._target = target
        self.data = data

    def evaluate_alpha(self, expr: Expression) -> Tensor:
        return normalize_by_day(expr.evaluate(self.data))

    @property
    def n_days(self) -> int:
        return self.data.n_days

    @property
    def target(self) -> Tensor:
        if self._target is None:
            raise ValueError("A target must be set before calculating non-mutual IC.")
        return self._target

    def make_ensemble_alpha(self, exprs: Sequence[Expression], weights: Sequence[float]) -> Tensor:
        n = len(exprs)
        factors = [self.evaluate_alpha(exprs[i]) * weights[i] for i in range(n)]
        return torch.sum(torch.stack(factors, dim=0), dim=0)

    def _calc_IC(self, value1: Tensor, value2: Tensor) -> float:
        return batch_pearsonr(value1, value2).mean().item()

    def _calc_rIC(self, value1: Tensor, value2: Tensor) -> float:
        return batch_spearmanr(value1, value2).mean().item()

    def _IR_from_batch(self, batch: Tensor) -> float:
        mean, std = batch.mean(), batch.std()
        return (mean / std).item()

    def _calc_ICIR(self, value1: Tensor, value2: Tensor) -> float:
        return self._IR_from_batch(batch_pearsonr(value1, value2))

    def _calc_rICIR(self, value1: Tensor, value2: Tensor) -> float:
        return self._IR_from_batch(batch_spearmanr(value1, value2))

    def calc_single_IC_ret(self, expr: Expression) -> float:
        return self._calc_IC(self.evaluate_alpha(expr), self.target)

    def calc_single_IC_ret_daily(self, expr: Expression) -> Tensor:
        return batch_pearsonr(self.evaluate_alpha(expr), self.target)

    def calc_single_rIC_ret(self, expr: Expression) -> float:
        return self._calc_rIC(self.evaluate_alpha(expr), self.target)

    def calc_single_all_ret(self, expr: Expression) -> Tuple[float, float]:
        value = self.evaluate_alpha(expr)
        target = self.target
        return self._calc_IC(value, target), self._calc_rIC(value, target)

    def calc_mutual_IC(self, expr1: Expression, expr2: Expression) -> float:
        return self._calc_IC(self.evaluate_alpha(expr1), self.evaluate_alpha(expr2))

    def calc_mutual_IC_daily(self, expr1: Expression, expr2: Expression) -> Tensor:
        return batch_pearsonr(self.evaluate_alpha(expr1), self.evaluate_alpha(expr2))

    def calc_pool_IC_ret(self, exprs: Sequence[Expression], weights: Sequence[float]) -> float:
        with torch.no_grad():
            value = self.make_ensemble_alpha(exprs, weights)
            return self._calc_IC(value, self.target)

    def calc_pool_rIC_ret(self, exprs: Sequence[Expression], weights: Sequence[float]) -> float:
        with torch.no_grad():
            value = self.make_ensemble_alpha(exprs, weights)
            return self._calc_rIC(value, self.target)

    def calc_pool_all_ret(self, exprs: Sequence[Expression], weights: Sequence[float]) -> Tuple[float, float]:
        with torch.no_grad():
            value = self.make_ensemble_alpha(exprs, weights)
            target = self.target
            return self._calc_IC(value, target), self._calc_rIC(value, target)

    def calc_pool_all_ret_with_ir(self, exprs: Sequence[Expression], weights: Sequence[float]) -> Tuple[float, float, float, float]:
        "Returns IC, ICIR, Rank IC, Rank ICIR"
        with torch.no_grad():
            value = self.make_ensemble_alpha(exprs, weights)
            target = self.target
            ics = batch_pearsonr(value, target)
            rics = batch_spearmanr(value, target)
            ic_mean, ic_std = ics.mean().item(), ics.std().item()
            ric_mean, ric_std = rics.mean().item(), rics.std().item()
            return ic_mean, ic_mean / ic_std, ric_mean, ric_mean / ric_std
