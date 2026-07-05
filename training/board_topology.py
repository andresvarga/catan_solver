"""Static board topology for the GNN encoder (roadmap phase 6).

The hex/vertex/edge *graph structure* (which vertex touches which hex, which
edges connect which vertices) is identical across every game -- Catan's
board shape never changes, only which terrain/number ends up on each hex and
who owns what. That means the adjacency used for message passing can be
precomputed once from a throwaway reference board rather than rebuilt every
forward pass, which is both simpler and faster than treating this as a
general variable-topology graph problem.

Each relation is stored as a pair of index arrays `(src, dst)` suitable for
`index_add_`-based scatter aggregation: message `i` flows from node `src[i]`
(in its own node-type space) to node `dst[i]` (in the destination type's
space).
"""
from __future__ import annotations

import numpy as np

from env.board import generate_board
from env.pettingzoo_env import NUM_EDGES, NUM_HEXES, NUM_VERTICES

_board = generate_board(randomize=False, seed=0)  # topology only; terrain/numbers unused here


def _edge_list(pairs: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    if not pairs:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    src, dst = zip(*pairs)
    return np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64)


_hex_to_vertex_pairs = []
_vertex_to_hex_pairs = []
_vertex_to_edge_pairs = []
_edge_to_vertex_pairs = []
_vertex_to_vertex_pairs = []

for hx in _board.hexes.values():
    for vid in hx.vertex_ids:
        _hex_to_vertex_pairs.append((hx.id, vid))
        _vertex_to_hex_pairs.append((vid, hx.id))

for v in _board.vertices.values():
    for eid in v.edge_ids:
        _vertex_to_edge_pairs.append((v.id, eid))
    for adj in v.adjacent_vertex_ids:
        _vertex_to_vertex_pairs.append((v.id, adj))

for e in _board.edges.values():
    for vid in e.vertex_ids:
        _edge_to_vertex_pairs.append((e.id, vid))

HEX_TO_VERTEX = _edge_list(_hex_to_vertex_pairs)
VERTEX_TO_HEX = _edge_list(_vertex_to_hex_pairs)
VERTEX_TO_EDGE = _edge_list(_vertex_to_edge_pairs)
EDGE_TO_VERTEX = _edge_list(_edge_to_vertex_pairs)
VERTEX_TO_VERTEX = _edge_list(_vertex_to_vertex_pairs)

assert HEX_TO_VERTEX[0].shape[0] == NUM_HEXES * 6
assert VERTEX_TO_EDGE[0].shape[0] == sum(len(v.edge_ids) for v in _board.vertices.values())
assert EDGE_TO_VERTEX[0].shape[0] == NUM_EDGES * 2

__all__ = [
    "NUM_HEXES", "NUM_VERTICES", "NUM_EDGES",
    "HEX_TO_VERTEX", "VERTEX_TO_HEX", "VERTEX_TO_EDGE", "EDGE_TO_VERTEX", "VERTEX_TO_VERTEX",
]
