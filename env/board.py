"""Board graph generation for standard 4-player Catan.

Hexes are laid out on an axial coordinate grid (radius-2 hexagon => 19 tiles).
Vertices and edges are derived geometrically from hex corners and deduplicated
by rounded pixel position, then re-expressed as a pure graph (adjacency lists)
so the rest of the engine never touches pixel coordinates again.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum


class Resource(Enum):
    WOOD = "wood"
    BRICK = "brick"
    SHEEP = "sheep"
    WHEAT = "wheat"
    ORE = "ore"


class HexType(Enum):
    WOOD = "wood"
    BRICK = "brick"
    SHEEP = "sheep"
    WHEAT = "wheat"
    ORE = "ore"
    DESERT = "desert"


HEX_TO_RESOURCE = {
    HexType.WOOD: Resource.WOOD,
    HexType.BRICK: Resource.BRICK,
    HexType.SHEEP: Resource.SHEEP,
    HexType.WHEAT: Resource.WHEAT,
    HexType.ORE: Resource.ORE,
}

# Standard base-game tile and number-token multisets.
STANDARD_HEX_COUNTS = {
    HexType.WOOD: 4,
    HexType.BRICK: 3,
    HexType.SHEEP: 4,
    HexType.WHEAT: 4,
    HexType.ORE: 3,
    HexType.DESERT: 1,
}
STANDARD_NUMBER_TOKENS = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12]

PIP_COUNT = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}

# Axial coordinates for a radius-2 hexagon (19 tiles).
AXIAL_COORDS = [
    (q, r)
    for q in range(-2, 3)
    for r in range(-2, 3)
    if -2 <= -q - r <= 2
]
assert len(AXIAL_COORDS) == 19

_HEX_SIZE = 10.0  # arbitrary unit; only used to dedupe shared corners

_AXIAL_DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
_HEX_NEIGHBORS = [
    [AXIAL_COORDS.index((q + dq, r + dr)) for dq, dr in _AXIAL_DIRECTIONS
     if (q + dq, r + dr) in AXIAL_COORDS]
    for q, r in AXIAL_COORDS
]


def _red_numbers_adjacent(terrains: list, numbers: list[int]) -> bool:
    """True if a 6 or 8 token would sit next to another 6 or 8 when
    `numbers` are dealt in hex order, skipping the desert."""
    it = iter(numbers)
    per_hex = [None if t == HexType.DESERT else next(it) for t in terrains]
    return any(per_hex[i] in (6, 8) and any(per_hex[j] in (6, 8) for j in _HEX_NEIGHBORS[i])
               for i in range(len(per_hex)))


def _axial_to_pixel(q: int, r: int) -> tuple[float, float]:
    x = _HEX_SIZE * (math.sqrt(3) * q + math.sqrt(3) / 2 * r)
    y = _HEX_SIZE * (1.5 * r)
    return x, y


def _hex_corners(q: int, r: int) -> list[tuple[float, float]]:
    cx, cy = _axial_to_pixel(q, r)
    corners = []
    for i in range(6):
        angle = math.radians(60 * i - 30)
        corners.append((round(cx + _HEX_SIZE * math.cos(angle), 3),
                         round(cy + _HEX_SIZE * math.sin(angle), 3)))
    return corners


@dataclass
class Hex:
    id: int
    q: int
    r: int
    terrain: HexType
    number: int | None  # None for desert
    vertex_ids: list[int] = field(default_factory=list)  # 6, in corner order
    edge_ids: list[int] = field(default_factory=list)  # 6, in corner order


@dataclass
class Vertex:
    id: int
    hex_ids: list[int] = field(default_factory=list)  # up to 3
    edge_ids: list[int] = field(default_factory=list)  # up to 3
    adjacent_vertex_ids: list[int] = field(default_factory=list)  # up to 3
    port: Resource | None = None  # None = no port
    port_generic: bool = False  # True = 3:1 "any" port


@dataclass
class Edge:
    id: int
    vertex_ids: tuple[int, int] = None  # exactly 2
    hex_ids: list[int] = field(default_factory=list)  # 1 or 2


@dataclass
class Board:
    hexes: dict[int, Hex]
    vertices: dict[int, Vertex]
    edges: dict[int, Edge]
    robber_hex: int

    def hex_by_axial(self, q: int, r: int) -> Hex | None:
        for h in self.hexes.values():
            if h.q == q and h.r == r:
                return h
        return None


def _dedup_key(pt: tuple[float, float]) -> tuple[float, float]:
    return (round(pt[0], 2), round(pt[1], 2))


def generate_board(randomize: bool = True, seed: int | None = None) -> Board:
    """Build a standard 19-hex Catan board.

    randomize=False still shuffles resources/numbers, but deterministically
    from `seed` (or 0 if unset) so curriculum stage 1's "fixed board" is a
    fixed *choice* of layout, not a hardcoded map of one specific real board.
    randomize=True with no seed gets real OS entropy instead -- falling back
    to the same fixed seed there would silently make every "random" board
    identical, which every real training entrypoint avoids by always passing
    an explicit seed, but is exactly the footgun an ad-hoc script/notebook
    that omits `seed=` could otherwise hit with no warning.
    """
    if seed is not None:
        rng = random.Random(seed)
    elif randomize:
        rng = random.Random()
    else:
        rng = random.Random(0)

    terrains: list[HexType] = []
    for terrain, count in STANDARD_HEX_COUNTS.items():
        terrains.extend([terrain] * count)
    numbers = list(STANDARD_NUMBER_TOKENS)
    if randomize:
        rng.shuffle(terrains)
        rng.shuffle(numbers)
    # Official variable set-up: red numbers (6, 8) must not be on adjacent
    # hexes. Re-deal the tokens (same seeded rng, so still deterministic per
    # seed -- this applies to the fixed curriculum board too) until they aren't.
    while _red_numbers_adjacent(terrains, numbers):
        rng.shuffle(numbers)

    # --- geometry pass: build hexes, dedup vertices/edges by pixel position ---
    vertex_lookup: dict[tuple[float, float], int] = {}
    edge_lookup: dict[frozenset[int], int] = {}
    vertices: dict[int, Vertex] = {}
    edges: dict[int, Edge] = {}
    hexes: dict[int, Hex] = {}

    next_vertex_id = 0
    next_edge_id = 0
    number_iter = iter(numbers)
    desert_hex_id = None

    for hex_id, (q, r) in enumerate(AXIAL_COORDS):
        terrain = terrains[hex_id]
        number = None if terrain == HexType.DESERT else next(number_iter)
        if terrain == HexType.DESERT:
            desert_hex_id = hex_id

        corners = _hex_corners(q, r)
        corner_vertex_ids = []
        for pt in corners:
            key = _dedup_key(pt)
            if key not in vertex_lookup:
                vertex_lookup[key] = next_vertex_id
                vertices[next_vertex_id] = Vertex(id=next_vertex_id)
                next_vertex_id += 1
            vid = vertex_lookup[key]
            corner_vertex_ids.append(vid)
            if hex_id not in vertices[vid].hex_ids:
                vertices[vid].hex_ids.append(hex_id)

        corner_edge_ids = []
        for i in range(6):
            v_a = corner_vertex_ids[i]
            v_b = corner_vertex_ids[(i + 1) % 6]
            ekey = frozenset((v_a, v_b))
            if ekey not in edge_lookup:
                edge_lookup[ekey] = next_edge_id
                edges[next_edge_id] = Edge(id=next_edge_id, vertex_ids=(v_a, v_b))
                next_edge_id += 1
            eid = edge_lookup[ekey]
            corner_edge_ids.append(eid)
            if hex_id not in edges[eid].hex_ids:
                edges[eid].hex_ids.append(hex_id)

        hexes[hex_id] = Hex(
            id=hex_id, q=q, r=r, terrain=terrain, number=number,
            vertex_ids=corner_vertex_ids, edge_ids=corner_edge_ids,
        )

    # vertex-edge and vertex-vertex adjacency
    for eid, e in edges.items():
        a, b = e.vertex_ids
        if eid not in vertices[a].edge_ids:
            vertices[a].edge_ids.append(eid)
        if eid not in vertices[b].edge_ids:
            vertices[b].edge_ids.append(eid)
        if b not in vertices[a].adjacent_vertex_ids:
            vertices[a].adjacent_vertex_ids.append(b)
        if a not in vertices[b].adjacent_vertex_ids:
            vertices[b].adjacent_vertex_ids.append(a)

    assert len(vertices) == 54, f"expected 54 vertices, got {len(vertices)}"
    assert len(edges) == 72, f"expected 72 edges, got {len(edges)}"

    _assign_ports(vertices, edges, rng, randomize)

    return Board(hexes=hexes, vertices=vertices, edges=edges, robber_hex=desert_hex_id)


def _boundary_edge_ring(vertices: dict[int, Vertex], edges: dict[int, Edge]) -> list[int]:
    """Return boundary edge ids (edges touching only 1 hex) in ring order."""
    boundary_edges = [e for e in edges.values() if len(e.hex_ids) == 1]
    boundary_vertex_set = set()
    for e in boundary_edges:
        boundary_vertex_set.update(e.vertex_ids)

    # walk the ring: start at any boundary vertex, repeatedly follow the
    # unique next boundary edge that hasn't been used yet.
    adjacency: dict[int, list[int]] = {v: [] for v in boundary_vertex_set}
    for e in boundary_edges:
        a, b = e.vertex_ids
        adjacency[a].append(e.id)
        adjacency[b].append(e.id)

    start_edge = boundary_edges[0]
    ordered = [start_edge.id]
    used = {start_edge.id}
    current_vertex = start_edge.vertex_ids[1]
    prev_vertex = start_edge.vertex_ids[0]
    while True:
        candidates = [eid for eid in adjacency[current_vertex] if eid not in used]
        if not candidates:
            break
        eid = candidates[0]
        ordered.append(eid)
        used.add(eid)
        e = edges[eid]
        next_vertex = e.vertex_ids[0] if e.vertex_ids[1] == current_vertex else e.vertex_ids[1]
        prev_vertex, current_vertex = current_vertex, next_vertex
        if len(ordered) == len(boundary_edges):
            break
    return ordered


def _assign_ports(vertices: dict[int, Vertex], edges: dict[int, Edge], rng: random.Random,
                   randomize: bool) -> None:
    """Place the standard 9 ports (4 generic 3:1, 5 specific 2:1) evenly around
    the perimeter, one every other boundary edge."""
    ring = _boundary_edge_ring(vertices, edges)
    port_types: list[tuple[Resource | None, bool]] = [(None, True)] * 4 + [
        (Resource.WOOD, False), (Resource.BRICK, False), (Resource.SHEEP, False),
        (Resource.WHEAT, False), (Resource.ORE, False),
    ]
    if randomize:
        rng.shuffle(port_types)

    n = len(ring)
    spacing = n // len(port_types)
    for i, (resource, generic) in enumerate(port_types):
        eid = ring[(i * spacing) % n]
        e = edges[eid]
        for vid in e.vertex_ids:
            vertices[vid].port = resource
            vertices[vid].port_generic = generic
