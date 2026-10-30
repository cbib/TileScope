"""
Visualization module for TileScope.

Pure visualization functions following the architecture pattern:
- painters: Vector-based rendering (pure matplotlib, no rasterization)
- roi_overlays: ROI rendering primitives (matplotlib operations)
- expression: High-level composition
"""

from tilescope.spatial.canvas import Canvas, make_canvas

from tilescope.viz.painters import draw_polygon_boundaries

from tilescope.viz.expression import (
    plot_expression_summary
)

from tilescope.viz.tiles import (
    plot_tile_densities
)

from tilescope.viz.consensus import (
    plot_consensus_grids
)

from tilescope.viz.visualization_runner import VisualizationRunner

__all__ = [
    'Canvas',
    'make_canvas',
    'draw_polygon_boundaries',
    'plot_expression_summary',
    'plot_tile_densities',
    'plot_consensus_grids',
    'VisualizationRunner',
]
