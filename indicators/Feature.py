from pandas._typing import Axes, Dtype
from VolumePriceIndicator import VolumePriceIndicator


class Feature(VolumePriceIndicator):

    def __init__(
        self,
        data=None,
        index: Axes = None,
        columns: Axes = None,
        dtype: Dtype = None,
        copy: bool = None,
    ) -> None:
        super().__init__(data, index, columns, dtype, copy)
