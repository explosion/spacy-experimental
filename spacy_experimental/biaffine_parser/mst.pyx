# cython: infer_types=True, profile=True, binding=True

# Copyright 2018-2020 Daniël de Kok and Tobias Pütz
# Copyright 2021 ExplosionAI GmbH
#
# Licensed under the Apache License, Version 2.0 or the MIT license, at your
# option.
#
# The implementation follows the description of:
#
# https://en.wikipedia.org/wiki/Edmonds%27_algorithm
#
# There are several differences compared to the above
# description:
#
# - We want to compute the maximum spanning tree. So, we find
#   incoming edges with maximum scores. This means that we have to
#   change the calculation scores of incoming edges of contracted
#   cycles. Here we follow Kübler et al., 2009, pp. 47.
# - Since the input is a score matrix, there are no parallel edges
#   in the input graph.
# - Since we use (a copy of) the score matrix to store weights of
#   incoming/outgoing contraction edges, we cannot store parallel
#   edges. So, we only store the highest scoring parallel edge
#   when computing the edges of the contraction. This does not change
#   the main algorithm, since the next recursion of Chu-Lui-Edmonds
#   would discard the lower-scoring edges anyway.

from cython.operator cimport dereference as deref, preincrement as inc
from libc.math cimport INFINITY, isfinite
from libcpp cimport bool
from libcpp.unordered_set cimport unordered_set
from libcpp.utility cimport pair
from libcpp.vector cimport vector
import numpy as np

cdef int NO_PARENT = -1

def mst_decode(scores):
    """Apply MST decoding to the pairwise attachment scores. Returns
    for each vertex the head in the maximum spanning tree"""

    # Within spacy, a root is encoded as a token that attaches to itself
    # (relative offset 0). However, the decoder uses a specific vertex,
    # typically 0. So, we stub an additional root vertex to accomodate
    # this.

    # We expect a biaffine attention matrix.
    if scores.shape[0] != scores.shape[1]:
        raise ValueError(f"Edge weight matrix with shape ({scores.shape[0]}, {scores.shape[1]}) is not a square matrix")

    seq_len = scores.shape[0]

    # The MST decoder expects float32, but the input could e.g. be float16.
    scores = scores.astype(np.float32)

    # Create score matrix with root row/column.
    with_root = np.full((seq_len + 1, seq_len + 1), -10000, dtype=scores.dtype)
    with_root[1:, 1:] = scores

    with_root[1:, 0] = scores.diagonal()
    with_root[np.diag_indices(with_root.shape[0])] = -10000

    heads = chu_liu_edmonds(with_root.T, 0)

    # Remove root vertex
    heads = heads[1:]

    for idx, head in enumerate(heads):
        if head == 0:
            heads[idx] = idx
        else:
            heads[idx] = head - 1

    return heads

cpdef chu_liu_edmonds(const float [:, :] scores, int root_vertex):
    """Chu-Liu-Edmonds maximum spanning tree for dense graphs

    This function returns the parent of each vertex in the maximum
    spanning tree of the `scores` square matrix, rooted at
    `root_vertex`. Each row in the matrix represents outgoing edge
    scores of the corresponding vertex, each column incoming edge
    scores. Thus, `scores[(parent, child)]` should give the weight of
    the edge from `parent` to child.

    Returns vertex parents. The length of the returned array equals
    the number of rows/columns of the scores matrix.
    """
    if scores.shape[0] != scores.shape[1]:
        raise ValueError(f"Edge weight matrix with shape ({scores.shape[0]}, {scores.shape[1]}) is not a square matrix")

    if root_vertex < 0 or root_vertex >= scores.shape[0]:
        raise IndexError(f"Root vertex {root_vertex} is out of bounds for edge weight matrix with shape ({scores.shape[0]}, {scores.shape[1]})")

    check_all_finite(scores)

    # We use this `Vec` to keep track of which vertices are 'active'. Vertices
    # that are part of a contracted cycle become inactive.
    cdef vector[bool] active_vertices = vector[bool](scores.shape[0], True)

    # The chu_liu_edmonds implementation mutates the scoring matrix, so
    # copy it to avoid modifying the caller's matrix.
    mst = _chu_liu_edmonds(scores.copy(), root_vertex, active_vertices)

    # Vertices with no parent (normally only the root vertex) are encoded
    # using the vertex -1, replace by None to make the result more Pythonic.
    return [None if vertex == -1 else vertex for vertex in mst]


cdef vector[int] _chu_liu_edmonds(float [:, :] scores, int root_vertex, vector[bool] &active_vertices) nogil:
    # For each vertex, find the parent with the highest incoming edge score.
    cdef vector[int] max_parents = find_max_parents(scores, root_vertex, active_vertices)

    # Base case: if the resulting graph does not contain a cycle, we
    # have found the MST of the (possibly contracted) graph.
    cdef vector[int] cycle = find_cycle(max_parents)
    if cycle.empty():
        return max_parents

    # Contract the cycle into a single vertex. We use the first vertex of
    # the cycle to represent the cycle.
    cdef pair[replacement_map, replacement_map] replacements = \
        contract_cycle(scores, max_parents, active_vertices, cycle)

    cdef replacement_map incoming_replacements = replacements.first
    cdef replacement_map outgoing_replacements = replacements.second

    # Recursively apply Chu-Liu-Edmonds to the graph with the contracted
    # cycle, until we hit the base case.
    cdef vector[int] contracted_mst = _chu_liu_edmonds(scores, root_vertex, active_vertices)

    # Expand the contracted cycle in the MST.
    return expand_cycle(max_parents, contracted_mst, cycle, incoming_replacements, outgoing_replacements)

cdef check_all_finite(const float [:, :] scores):
    cdef int i, j

    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            if not isfinite(scores[i, j]):
                raise ValueError(f"Edge weight matrix contains non-finite score: {scores[i, j]}",)

cdef pair[replacement_map, replacement_map] contract_cycle(
        float [:, :] scores, const vector[int] &max_parents, vector[bool] &active_vertices,
        const vector[int] &cycle) nogil:
    """Contract the given cycle. Updates the score matrix and active vertices.
       Returns a mapping of replaced edges."""
    # The first vertex of the cycle is used to represent the contraction.
    cdef int first_in_cycle = cycle[0]

    # Get the sum of edge scores in the cycle. See Kübler et al., 2009,
    # pp. 47.
    cdef float cycle_sum = 0.
    cdef int parent
    cdef int vertex
    cdef vector[int].const_iterator vertex_iter = cycle.const_begin()
    while vertex_iter != cycle.end():
        vertex = deref(vertex_iter)
        parent = max_parents[vertex]
        cycle_sum += scores[parent, vertex]
        inc(vertex_iter)

    # Mark the cycle vertices as inactive.
    cdef size_t i
    for i in range(1, cycle.size()):
        active_vertices[cycle[i]] = False

    # Convert the cycle to a set for constant-time lookups. Cython does
    # not seem to support the constructor that allows us to create the
    # set from an iterator pair.
    cdef unordered_set[int] cycle_vertices
    cycle_vertices.insert(cycle.const_begin(), cycle.const_end())

    cdef replacement_map incoming_replacements
    cdef replacement_map outgoing_replacements

    cdef float best_incoming
    cdef float best_outgoing
    cdef int best_incoming_vertex
    cdef int best_outgoing_vertex
    cdef int cycle_vertex
    cdef int best_parent
    cdef float best_weight
    cdef float incoming_score
    for vertex in range(scores.shape[0]):
        if not active_vertices[vertex] or cycle_vertices.find(vertex) != cycle_vertices.end():
            continue

        best_incoming = -INFINITY
        best_outgoing = -INFINITY
        best_incoming_vertex = -1
        best_outgoing_vertex = -1

        vertex_iter = cycle.const_begin()
        while vertex_iter != cycle.end():
            cycle_vertex = deref(vertex_iter)

            # Replace (v, w) by (v_cycle, w)
            if scores[cycle_vertex, vertex] > best_outgoing:
                best_outgoing = scores[cycle_vertex, vertex]
                best_outgoing_vertex = cycle_vertex

            best_parent = max_parents[cycle_vertex]
            best_weight = scores[best_parent, cycle_vertex]
            incoming_score = cycle_sum + scores[vertex, cycle_vertex] - best_weight

            # Replace (u, v) by (u, v_cycle)
            if incoming_score > best_incoming:
                best_incoming = incoming_score
                best_incoming_vertex = cycle_vertex

            inc(vertex_iter)

        # Save max incoming edge(u, v_cyle) and max outgoing edge (v_cycle, w).
        scores[vertex, first_in_cycle] = best_incoming;
        scores[first_in_cycle, vertex] = best_outgoing;

        incoming_replacements[pair[int, int](vertex, first_in_cycle)] = best_incoming_vertex
        outgoing_replacements[pair[int, int](first_in_cycle, vertex)] = best_outgoing_vertex

    return pair[replacement_map, replacement_map](incoming_replacements, outgoing_replacements)

cdef vector[int] expand_cycle(vector[int] max_parents, vector[int] &mst, vector[int] cycle,
        replacement_map incoming_replacements, replacement_map outgoing_replacements) nogil:
    cdef int cycle_vertex = cycle[0]

    # Find out which edge was replaced by the incoming edge of the cycle vertex...
    cdef int kicked_out = incoming_replacements[pair[int, int](mst[cycle_vertex], cycle_vertex)]

    # ...v of the kicked-out edge (u, v) becomes the root of the to-be-broken cycle.
    mst[kicked_out] = mst[cycle_vertex]

    # Copy all other edges from the cycle
    for cycle_vertex in cycle:
        if cycle_vertex == kicked_out:
            continue
        mst[cycle_vertex] = max_parents[cycle_vertex]

    # Restore original outgoing edges, replacing (v_cycle, w) by (v, w).
    cdef pair[pair[int, int], int] contracted_edge
    cdef pair[int, int] edge_pair
    cdef int orig_edge
    for contracted_edge in outgoing_replacements:
        edge_pair = contracted_edge.first
        orig_edge = contracted_edge.second
        if mst[edge_pair.second] == edge_pair.first:
            mst[edge_pair.second] = orig_edge

    return mst

cdef vector[int] find_cycle(const vector[int] &parents) nogil:
    cdef vector[bool] visited = vector[bool](parents.size(), False)
    cdef vector[bool] on_stack = vector[bool](parents.size(), False)
    cdef vector[int] edge_to = vector[int](parents.size(), 0)
    cdef vector[int] cycle
    cdef size_t start

    for start in range(0, parents.size()):
        cycle = _find_cycle(parents, visited, edge_to, on_stack, start)
        if not cycle.empty():
            return cycle

    return vector[int]()

cdef vector[int] _find_cycle(const vector[int] &parents, vector[bool] &visited, vector[int] &edge_to, vector[bool] &on_stack, size_t vertex) nogil:
    cdef vector[int] cycle
    cdef int cycle_vertex

    visited[vertex] = True
    on_stack[vertex] = True

    parent = parents[vertex]
    if parent >= 0:
        if not visited[parent]:
            edge_to[parent] = vertex
            cycle = _find_cycle(parents, visited, edge_to, on_stack, parent)
            if not cycle.empty():
                return cycle
        elif on_stack[parent]:
            # We have a cycle if we encounter a vertex that is already on the
            # call stack.
            cycle_vertex = vertex

            while cycle_vertex != parent:
                cycle.push_back(cycle_vertex)
                cycle_vertex = edge_to[cycle_vertex]
            cycle.push_back(parent)

            return cycle

    on_stack[vertex] = False

cdef vector[int] find_max_parents(const float [:, :] scores, int root_vertex,
                                  const vector[bool] &active_vertices) nogil:
    cdef vector[int] max_parents = vector[int](active_vertices.size(), NO_PARENT)
    cdef const float [:] child_incoming
    cdef int child, parent, best_parent
    cdef float score, best_score

    for child in range(scores.shape[1]):
        if child == root_vertex or not active_vertices[child]:
            continue

        child_incoming = scores[:, child]
        best_parent = root_vertex
        best_score = child_incoming[root_vertex]
        for parent in range(child_incoming.shape[0]):
            score = child_incoming[parent]
            if parent != child and score > best_score and active_vertices[parent]:
                best_parent = parent
                best_score = score
        max_parents[child] = best_parent

    return max_parents
