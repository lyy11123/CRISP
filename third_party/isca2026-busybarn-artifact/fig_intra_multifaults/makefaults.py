from collections import defaultdict, deque
from typing import List, Tuple, Set, Dict, Optional
import random
import math

Node = Tuple[int, int]
Edge = Tuple[Node, Node]


def coord_to_id(coord: Node, M: int) -> int:
    """Map coordinate ``(i, j)`` to a node ID."""
    i, j = coord
    return i * M + j

def build_grid(N: int, M: int):
    """Build an N x M 4-neighbor grid and return vertices and sorted undirected edges."""
    V = [(i, j) for i in range(N) for j in range(M)]
    E = []
    for i in range(N):
        for j in range(M):
            if i + 1 < N:
                e = tuple(sorted(((i, j), (i + 1, j))))
                E.append(e)
            if j + 1 < M:
                e = tuple(sorted(((i, j), (i, j + 1))))
                E.append(e)
    return V, E

def _adj_index_edges(E: List[Edge]) -> Dict[Node, List[Edge]]:
    """Build an index from node to incident edges."""
    adjE = defaultdict(list)
    for u, v in E:
        adjE[u].append((u, v))
        adjE[v].append((u, v))
    return adjE

def _neighbors_4(n: Node, N: int, M: int):
    """Yield 4-neighbor adjacent coordinates."""
    i, j = n
    if i - 1 >= 0: yield (i - 1, j)
    if i + 1 < N:  yield (i + 1, j)
    if j - 1 >= 0: yield (i, j - 1)
    if j + 1 < M:  yield (i, j + 1)

def clustered_growth_failure(
    N: int,
    M: int,
    *,
    V: Optional[List[Node]] = None,
    E: Optional[List[Edge]] = None,
    node_rate: Optional[float] = None,
    link_rate: Optional[float] = None,
    B_node: Optional[int] = None,
    B_link: Optional[int] = None,
    K_seed: Optional[int] = None,
    p_grow: float = 0.45,
    p_link_in: float = 0.90,
    p_link_border: float = 0.40,
    seed: Optional[int] = None,
    boundary_perturb: bool = True,
    allow_edge_anywhere: bool = True
) -> Tuple[Set[Node], Set[Edge], int, int]:
    """
    Generate clustered failures on an N x M grid using a growth phase
    followed by a cleanup phase.

    Returns ``(fail_nodes, fail_edges, Bn, Bl)``, where ``Bn`` and ``Bl`` are
    the final target counts.

    Key points
    ----
    - Node and link failure counts are controlled independently via either
      ``node_rate/link_rate`` or ``B_node/B_link``.
    - The growth phase may miss the target counts; the cleanup phase pads or
      trims results so ``|Fv| = Bn`` and ``|Fe| = Bl``.
    - ``allow_edge_anywhere=True`` treats link failures as mostly independent
      from node failures. Set it to ``False`` to keep failed links near the
      cluster interior or boundary.

    Parameter checks
    ----
    - Provide either ``(node_rate, link_rate)`` or ``(B_node, B_link)``.
      If both are set, ``B_*`` takes priority.
    - ``node_rate`` and ``link_rate`` are rounded to integer targets and then
      clamped to ``[0, |V|]`` and ``[0, |E|]``.
    """
    rng = random.Random(seed)

    if V is None or E is None:
        V, E = build_grid(N, M)
    Vset = set(V)
    nV, nE = len(V), len(E)

    if B_node is None or B_link is None:
        if node_rate is None or link_rate is None:
            raise ValueError("Provide either (B_node, B_link) or (node_rate, link_rate).")
        B_node = int(round(max(0.0, min(1.0, node_rate)) * nV))
        B_link = int(round(max(0.0, min(1.0, link_rate)) * nE))
    B_node = max(0, min(B_node, nV))
    B_link = max(0, min(B_link, nE))

    K = K_seed if K_seed is not None else max(1, round(0.02 * min(N, M)))
    seeds = rng.sample(V, k=min(K, nV))
    adjE = _adj_index_edges(E)

    fail_nodes: Set[Node] = set()
    fail_edges: Set[Edge] = set()

    Q = deque(seeds)
    for s in seeds:
        if len(fail_nodes) < B_node:
            fail_nodes.add(s)
        for e in adjE[s]:
            if len(fail_edges) >= B_link:
                break
            if rng.random() < p_link_in:
                fail_edges.add(tuple(sorted(e)))

    def add_edges_for_node(nb: Node):
        """Add failed edges for a newly included node based on adjacency."""
        for e in adjE[nb]:
            u, v = e
            other = v if u == nb else u
            if other in fail_nodes:
                if len(fail_edges) < B_link and rng.random() < p_link_in:
                    fail_edges.add(tuple(sorted(e)))
            else:
                if len(fail_edges) < B_link and rng.random() < p_link_border:
                    fail_edges.add(tuple(sorted(e)))

    while Q and (len(fail_nodes) < B_node or len(fail_edges) < B_link):
        cur = Q.popleft()

        for nb in _neighbors_4(cur, N, M):
            if nb not in Vset or nb in fail_nodes:
                continue

            if len(fail_nodes) < B_node and rng.random() < p_grow:
                fail_nodes.add(nb)
                Q.append(nb)
                add_edges_for_node(nb)
            elif boundary_perturb and len(fail_edges) < B_link:
                for e in adjE[cur]:
                    if len(fail_edges) >= B_link:
                        break
                    if rng.random() < p_link_border:
                        fail_edges.add(tuple(sorted(e)))

        if len(fail_nodes) >= B_node and len(fail_edges) >= B_link:
            break

    if len(fail_nodes) < B_node:
        remaining_nodes = [n for n in V if n not in fail_nodes]
        need = min(B_node - len(fail_nodes), len(remaining_nodes))
        if need > 0:
            fail_nodes.update(rng.sample(remaining_nodes, need))

    if len(fail_edges) < B_link:
        if allow_edge_anywhere:
            remaining_edges = [e for e in E if e not in fail_edges]
        else:
            allowed = set()
            for n in fail_nodes:
                for e in adjE[n]:
                    allowed.add(tuple(sorted(e)))
            remaining_edges = [e for e in allowed if e not in fail_edges]
            if len(remaining_edges) < (B_link - len(fail_edges)):
                remaining_edges = [e for e in E if e not in fail_edges]
        need = min(B_link - len(fail_edges), len(remaining_edges))
        if need > 0:
            fail_edges.update(rng.sample(remaining_edges, need))

    if len(fail_nodes) > B_node:
        fail_nodes = set(rng.sample(list(fail_nodes), B_node))
    if len(fail_edges) > B_link:
        fail_edges = set(rng.sample(list(fail_edges), B_link))

    fail_edges = set(tuple(sorted(e)) for e in fail_edges)

    return fail_nodes, fail_edges, B_node, B_link


def generate_cluster_failures(
    N: int = 10,
    M: int = 10,
    error_rate: float = 0.15,
    sample_num: int = 10,
    *,
    K_seed: Optional[int] = 3,
    p_grow: float = 0.5,
    p_link_in: float = 0.85,
    p_link_border: float = 0.35,
    boundary_perturb: bool = True,
    allow_edge_anywhere: bool = False,
    base_seed: int = 0,
    bidirectional_edges: bool = True,
) -> List[Tuple[List[Tuple[int, int]], List[Tuple[Tuple[int,int], Tuple[int,int]]]]]:
    """
    Generate two batches of clustered failure samples:
      1) link-only failures (``B_node=0``, ``B_link=error_rate * |E|``)
      2) node-only failures (``B_node=error_rate * |V|``, ``B_link=0``)

    Returns
    ----
    failures: List[(failed_nodes_list, failed_links_list)]
      - ``failed_nodes_list``: ``List[Node]`` where nodes are ``(i, j)``
      - ``failed_links_list``: ``List[(u, v)]``; when
        ``bidirectional_edges=True`` both directions are included

    Tune the growth parameters to change cluster shape. ``error_rate`` is
    applied separately to ``|V|`` and ``|E|`` for the two sample sets.
    """
    V, E = build_grid(N, M)
    failures = []

    B_node = int(round(0.0 * len(V)))
    B_link = int(round(error_rate * len(E)))

    for sample_idx in range(sample_num):
        fn, fe, _, _ = clustered_growth_failure(
            N, M, V=V, E=E,
            B_node=B_node, B_link=B_link,
            K_seed=K_seed, p_grow=p_grow,
            p_link_in=p_link_in, p_link_border=p_link_border,
            seed=base_seed + sample_idx,
            boundary_perturb=boundary_perturb,
            allow_edge_anywhere=allow_edge_anywhere
        )
        failed_links = []
        if bidirectional_edges:
            for e in fe:
                failed_links.append(((0, coord_to_id(e[0], M)), (0, coord_to_id(e[1], M))))
                failed_links.append(((0, coord_to_id(e[1], M)), (0, coord_to_id(e[0], M))))
        else:
            for e in fe:
                failed_links.append(((0, coord_to_id(e[0], M)), (0, coord_to_id(e[1], M))))

        failed_nodes = []
        for n in fn:
            failed_nodes.append((0, coord_to_id(n, M)))

        failures.append((failed_nodes, failed_links))

    B_node = int(round(error_rate * len(V)))
    B_link = int(round(0.0 * len(E)))

    for sample_idx in range(sample_num):
        fn, fe, _, _ = clustered_growth_failure(
            N, M, V=V, E=E,
            B_node=B_node, B_link=B_link,
            K_seed=K_seed, p_grow=p_grow,
            p_link_in=p_link_in, p_link_border=p_link_border,
            seed=base_seed + sample_idx,
            boundary_perturb=boundary_perturb,
            allow_edge_anywhere=allow_edge_anywhere
        )
        failed_links = []
        if bidirectional_edges:
            for e in fe:
                failed_links.append(((0, coord_to_id(e[0], M)), (0, coord_to_id(e[1], M))))
                failed_links.append(((0, coord_to_id(e[1], M)), (0, coord_to_id(e[0], M))))
        else:
            for e in fe:
                failed_links.append(((0, coord_to_id(e[0], M)), (0, coord_to_id(e[1], M))))

        failed_nodes = []
        for n in fn:
            failed_nodes.append((0, coord_to_id(n, M)))

        failures.append((failed_nodes, failed_links))

    return failures


if __name__ == "__main__":
    N, M = 10, 10
    fn, fe, Bn, Bl = clustered_growth_failure(
        N, M,
        node_rate=0.15,
        link_rate=0.15,
        K_seed=None,
        p_grow=0.45,
        p_link_in=0.90,
        p_link_border=0.40,
        seed=0,
        boundary_perturb=True,
        allow_edge_anywhere=True
    )
    print(f"[By rate] nodes={len(fn)}/{Bn}, links={len(fe)}/{Bl}")

    V, E = build_grid(N, M)
    B_node = int(round(0.15 * len(V)))
    B_link = int(round(0.0 * len(E)))
    fn2, fe2, Bn2, Bl2 = clustered_growth_failure(
        N, M, V=V, E=E,
        B_node=B_node, B_link=B_link,
        K_seed=3, p_grow=0.5, p_link_in=0.85, p_link_border=0.35,
        seed=42, boundary_perturb=True, allow_edge_anywhere=False
    )
    print(f"[By budget] nodes={len(fn2)}/{Bn2}, links={len(fe2)}/{Bl2}")
    print("Failed nodes:", sorted(fn2))
    print("Failed links:", sorted(fe2))
