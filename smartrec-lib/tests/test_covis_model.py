from smartrec_lib.models import CoVisModel


def _internal(dataset, external_id):
    return int(dataset.item_id_map.convert_to_internal([external_id])[0])


def test_fit_builds_symmetric_neighbors(dataset):
    model = CoVisModel(min_cooc=2, top_k=100)
    model.fit(dataset)
    m1, m2 = _internal(dataset, "m1"), _internal(dataset, "m2")
    m1_neighbors = dict(model.neighbors[m1])
    m2_neighbors = dict(model.neighbors[m2])
    assert m1_neighbors[m2] == 2.0
    assert m2_neighbors[m1] == 2.0


def test_min_cooc_filters_weak_edges(dataset):
    model = CoVisModel(min_cooc=2, top_k=100)
    model.fit(dataset)
    m1, m3 = _internal(dataset, "m1"), _internal(dataset, "m3")
    # (m1, m3) co-occur only once (u2) -> filtered out
    assert m3 not in dict(model.neighbors.get(m1, []))


def test_top_k_truncates(dataset):
    model = CoVisModel(min_cooc=1, top_k=1)
    model.fit(dataset)
    assert all(len(nbrs) == 1 for nbrs in model.neighbors.values())
