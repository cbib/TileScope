"""Component-grain data structures.

`Tile` and `Tessellation` are frozen with read-only arrays / tuple
tile collections; consumers reference them but must never mutate.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Union, List, Optional, Tuple

try:
    from shapely.geometry import Polygon, MultiPolygon
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    Polygon = None
    MultiPolygon = None

from tilescope.cols import SOURCE_ROW_ID


def require_integer_point_indices(values, *, upper_bound=None) -> np.ndarray:
    """Validate positional membership; never interpret masks as integer positions."""
    indices = np.asarray(values)
    if indices.ndim != 1:
        raise ValueError("point_indices must be a 1-D integer array.")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("point_indices must contain integers, not Boolean masks or floats.")
    if indices.size and (
        indices.min() < 0 or indices.max() > np.iinfo(np.int64).max
        or (upper_bound is not None and indices.max() >= upper_bound)
    ):
        raise ValueError("point_indices contain an out of range position.")
    return indices.astype(np.int64, copy=False)


@dataclass(frozen=True)
class Tile:
    """One Voronoi tile: geometry + component-local point membership.

    `domain_region_id` is `None` only at the engine boundary; the region
    stage sets it before any analyzer consumes the record.
    Production membership is a sorted integer array, with one entry per
    member, rather than a Boolean array as long as the whole component.
    """
    tile_id: Union[int, str]
    polygon: Union[Polygon, MultiPolygon]
    point_indices: np.ndarray
    n_spots: int
    area: float
    tessellation_id: Optional[str] = None
    domain_region_id: Optional[int] = None

    def __post_init__(self) -> None:
        arr = require_integer_point_indices(self.point_indices)
        if arr is not self.point_indices:
            object.__setattr__(self, "point_indices", arr)
        try:
            self.point_indices.flags.writeable = False
        except ValueError:
            # Already read-only (shared buffer, etc.); that's fine.
            pass


@dataclass(frozen=True)
class Tessellation:
    """Complete tessellation of one component; tiles + metadata."""
    tessellation_id: str
    component_id: str
    tiles: Tuple[Tile, ...]
    method: str
    tile_min_points: int
    fraction: float
    rng_seed: Optional[int] = None
    hex_radius: Optional[float] = None
    # Engine `set_precision` value; downstream tolerances derive from
    # it so they mirror the engine's own snap.
    precision_grid: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.tiles, tuple):
            object.__setattr__(self, "tiles", tuple(self.tiles))

    @property
    def n_tiles(self) -> int:
        return len(self.tiles)

    @property
    def mean_tile_area(self) -> float:
        if not self.tiles:
            return 0.0
        return float(np.mean([t.area for t in self.tiles]))

    def get_tile_polygons(self) -> List[Polygon]:
        return [t.polygon for t in self.tiles]


@dataclass
class ComponentData:
    """Component-grain data bundle: spatial support + attached feature values.

    Spatial-only callers should prefer `SampleSupportBank.components[cid]`.
    """
    component_id: str
    points: np.ndarray                                  # (N, 2) world units
    canvas: object                                      # Canvas: object to avoid circular import
    polygon: Union[Polygon, MultiPolygon]
    expression_data: pd.DataFrame                       # Source-row attachment

    def __post_init__(self) -> None:
        if not SHAPELY_AVAILABLE:
            raise ImportError("Shapely required for ComponentData. Install with: pip install shapely")
        object.__setattr__(self, 'component_id', str(self.component_id))
        if not isinstance(self.expression_data, pd.DataFrame):
            raise TypeError(
                "ComponentData.expression_data must be a pandas DataFrame."
            )
        if SOURCE_ROW_ID not in self.expression_data.columns:
            raise ValueError(
                f"ComponentData(component_id={self.component_id!r}): "
                f"expression_data missing required {SOURCE_ROW_ID!r}."
            )
        if len(self.expression_data) != len(self.points):
            raise ValueError(
                f"ComponentData(component_id={self.component_id!r}): "
                f"expression_data rows ({len(self.expression_data)}) != "
                f"points ({len(self.points)})."
            )
