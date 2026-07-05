from env.board import generate_board, HexType, PIP_COUNT, STANDARD_HEX_COUNTS, STANDARD_NUMBER_TOKENS


def test_randomize_with_no_seed_gives_different_boards_across_calls():
    """Regression test: generate_board(randomize=True, seed=None) used to fall
    back to a fixed seed (0), so every "random" board without an explicit
    seed was silently identical. Real training entrypoints always pass an
    explicit seed, so this never showed up there, but any future ad-hoc
    script/notebook that omits seed= would otherwise train/evaluate on one
    fixed map forever with no warning."""
    terrains = set()
    for _ in range(8):
        b = generate_board(randomize=True, seed=None)
        terrains.add(tuple(h.terrain for h in sorted(b.hexes.values(), key=lambda h: h.id)))
    assert len(terrains) > 1, "expected different terrain layouts across repeated seed=None calls"


def test_hex_vertex_edge_counts():
    b = generate_board(randomize=True, seed=1)
    assert len(b.hexes) == 19
    assert len(b.vertices) == 54
    assert len(b.edges) == 72


def test_resource_and_number_multiset():
    b = generate_board(randomize=True, seed=2)
    from collections import Counter
    terrain_counts = Counter(h.terrain for h in b.hexes.values())
    assert dict(terrain_counts) == STANDARD_HEX_COUNTS
    numbers = sorted(h.number for h in b.hexes.values() if h.number is not None)
    assert numbers == sorted(STANDARD_NUMBER_TOKENS)


def test_desert_has_no_number_and_holds_robber():
    b = generate_board(randomize=True, seed=3)
    desert = [h for h in b.hexes.values() if h.terrain == HexType.DESERT][0]
    assert desert.number is None
    assert b.robber_hex == desert.id


def test_every_hex_has_six_unique_vertices_and_edges():
    b = generate_board(randomize=True, seed=4)
    for h in b.hexes.values():
        assert len(set(h.vertex_ids)) == 6
        assert len(set(h.edge_ids)) == 6


def test_vertex_and_edge_adjacency_bounds():
    b = generate_board(randomize=True, seed=5)
    for v in b.vertices.values():
        assert 1 <= len(v.hex_ids) <= 3
        assert 2 <= len(v.edge_ids) <= 3
        assert 2 <= len(v.adjacent_vertex_ids) <= 3
    for e in b.edges.values():
        assert len(e.vertex_ids) == 2
        assert 1 <= len(e.hex_ids) <= 2


def test_edges_are_symmetric_with_vertices():
    b = generate_board(randomize=True, seed=6)
    for e in b.edges.values():
        a, bb = e.vertex_ids
        assert e.id in b.vertices[a].edge_ids
        assert e.id in b.vertices[bb].edge_ids
        assert bb in b.vertices[a].adjacent_vertex_ids
        assert a in b.vertices[bb].adjacent_vertex_ids


def test_port_count_and_ratios():
    b = generate_board(randomize=True, seed=7)
    port_vertices = [v for v in b.vertices.values() if v.port is not None or v.port_generic]
    # 9 ports * 2 vertices each, minus any accidental overlap at shared corners
    assert 12 <= len(port_vertices) <= 18
    generic = sum(1 for v in port_vertices if v.port_generic)
    specific = sum(1 for v in port_vertices if v.port is not None and not v.port_generic)
    assert generic > 0 and specific > 0


def test_deterministic_seed_reproducible():
    b1 = generate_board(randomize=True, seed=99)
    b2 = generate_board(randomize=True, seed=99)
    assert [h.terrain for h in b1.hexes.values()] == [h.terrain for h in b2.hexes.values()]
    assert [h.number for h in b1.hexes.values()] == [h.number for h in b2.hexes.values()]


def test_fixed_board_flag_still_deterministic_across_calls():
    b1 = generate_board(randomize=False, seed=5)
    b2 = generate_board(randomize=False, seed=5)
    assert [h.terrain for h in b1.hexes.values()] == [h.terrain for h in b2.hexes.values()]


def test_pip_counts_match_probability_ordering():
    assert PIP_COUNT[6] == PIP_COUNT[8] == 5
    assert PIP_COUNT[2] == PIP_COUNT[12] == 1
