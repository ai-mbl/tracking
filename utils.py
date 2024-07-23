from __future__ import annotations

from typing import TYPE_CHECKING

import ilpy

from motile.constraints import Constraint
from motile.variables import EdgeSelected, NodeAppear
import numpy as np


if TYPE_CHECKING:
    from motile.solver import Solver


def normalize(x, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
    """Percentile-based image normalization."""

    mi = np.percentile(x,pmin,axis=axis,keepdims=True)
    ma = np.percentile(x,pmax,axis=axis,keepdims=True)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)

def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    if dtype is not None:
        x   = x.astype(dtype,copy=False)
        mi  = dtype(mi) if np.isscalar(mi) else mi.astype(dtype,copy=False)
        ma  = dtype(ma) if np.isscalar(ma) else ma.astype(dtype,copy=False)
        eps = dtype(eps)

    x = (x - mi) / ( ma - mi + eps )

    if clip:
        x = np.clip(x,0,1)

    return x

class InOutSymmetry(Constraint):
    r"""Ensures that all nodes, apart from the ones in the first and last
    frame, have as many incoming edges as outgoing edges.

    Adds the following linear constraint for nodes :math:`v` not in first or
    last frame:

    .. math::
          \sum_{e \in \\text{in_edges}(v)} x_e = \sum{e \in \\text{out_edges}(v)} x_e
    """

    def instantiate(self, solver: Solver) -> list[ilpy.Constraint]:
        edge_indicators = solver.get_variables(EdgeSelected)
        start, end = solver.graph.get_frames()

        constraints = []
        for node, attrs in solver.graph.nodes.items():
            constraint = ilpy.Constraint()

            if solver.graph.frame_attribute in attrs and attrs[
                solver.graph.frame_attribute
            ] not in (
                start,
                end - 1,  # type: ignore
            ):
                for prev_edge in solver.graph.prev_edges[node]:
                    ind_e = edge_indicators[prev_edge]
                    constraint.set_coefficient(ind_e, 1)
                for next_edge in solver.graph.next_edges[node]:
                    ind_e = edge_indicators[next_edge]
                    constraint.set_coefficient(ind_e, -1)
                constraint.set_relation(ilpy.Relation.Equal)
                constraint.set_value(0)

                constraints.append(constraint)

        return constraints


class MinTrackLength(Constraint):
    r"""Ensures that each appearing track consists of at least ``min_edges``
    edges.

    Currently only supports ``min_edges = 1``.

    Args:

        min_edges: The minimum number of edges per track.
    """

    def __init__(self, min_edges: int) -> None:
        if min_edges != 1:
            raise NotImplementedError(
                "Can only enforce minimum track length of 1 edge."
            )
        self.min_edges = min_edges

    def instantiate(self, solver):
        appear_indicators = solver.get_variables(NodeAppear)
        edge_indicators = solver.get_variables(EdgeSelected)
        for node in solver.graph.nodes:
            constraint = ilpy.Constraint()
            constraint.set_coefficient(appear_indicators[node], 1)
            for edge in solver.graph.next_edges[node]:
                constraint.set_coefficient(edge_indicators[edge], -1)
            constraint.set_relation(ilpy.Relation.LessEqual)
            constraint.set_value(0)
            yield constraint


