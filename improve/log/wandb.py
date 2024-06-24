from pprint import pprint
from typing import (Any, Dict, List, Mapping, Optional, Sequence, TextIO,
                    Tuple, Union)

import wandb
from stable_baselines3.common.logger import KVWriter, Logger


class WandbLogger(Logger):

    def __init__(self, folder: Optional[str], output_formats: List[KVWriter]):
        super().__init__(folder, output_formats)

    def dump(self, step: int = 0) -> None:
        """Write all of the diagnostics from the current iteration"""
        wandb.log(self.name_to_value, step=step)
        super().dump(step)

    def record(
        self,
        key: str,
        value: Any,
        exclude: Optional[Union[str, Tuple[str, ...]]] = None,
    ) -> None:
        """
        Log a value of some diagnostic
        Call this once for each diagnostic quantity, each iteration
        If called many times, last value will be used.

        :param key: save to log this key
        :param value: save to log this value
        :param exclude: outputs to be excluded
        """
        self.name_to_value[key] = value
        self.name_to_excluded[key] = self.to_tuple(exclude)

    def record_mean(
        self,
        key: str,
        value: Optional[float],
        exclude: Optional[Union[str, Tuple[str, ...]]] = None,
    ) -> None:
        """
        The same as record(), but if called many times, values averaged.

        :param key: save to log this key
        :param value: save to log this value
        :param exclude: outputs to be excluded
        """
        if value is None:
            return
        old_val, count = self.name_to_value[key], self.name_to_count[key]
        self.name_to_value[key] = old_val * count / (count + 1) + value / (count + 1)
        self.name_to_count[key] = count + 1
        self.name_to_excluded[key] = self.to_tuple(exclude)

