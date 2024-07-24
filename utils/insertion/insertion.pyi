from typing import Tuple

import numpy as np
import numpy.typing as npt

def random(
    instance: npt.NDArray[np.float32],
    order: npt.NDArray[np.uint32],
    is_euclidean: bool,
    out: npt.NDArray[np.uint32],
) -> float: ...

def cvrp_random(
    customerpos: npt.NDArray[np.float32],
    depotx: float,
    depoty: float,
    demands: npt.NDArray[np.uint32],
    capacity: int,
    order: npt.NDArray[np.uint32],
    exploration: float,
) -> Tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]: ...
