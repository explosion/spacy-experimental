// Copyright 2018-2020 Daniël de Kok and Tobias Pütz
//
// Licensed under the Apache License, Version 2.0 or the MIT license, at your
// option.

//! Find the maximum spanning tree using Chu Liu Edmonds

use std::collections::{HashMap, HashSet};
use std::f32;

use ndarray::{ArrayView2, ArrayViewMut2, Axis};
use ordered_float::OrderedFloat;

// The implementation follows the description of:
//
// https://en.wikipedia.org/wiki/Edmonds%27_algorithm
//
// There are several differences compared to the above
// description:
//
// - We want to compute the maximum spanning tree. So, we find
//   incoming edges with maximum scores. This means that we have to
//   change the calculation scores of incoming edges of contracted
//   cycles. Here we follow Kübler et al., 2009, pp. 47.
// - Since the input is a score matrix, there are no parallel edges
//   in the input graph.
// - Since we use (a copy of) the score matrix to store weights of
//   incoming/outgoing contraction edges, we cannot store parallel
//   edges. So, we only store the highest scoring parallel edge
//   when computing the edges of the contraction. This does not change
//   the main algorithm, since the next recursion of Chu-Lui-Edmonds
//   would discard the lower-scoring edges anyway.

/// Chu-Liu-Edmonds maximum spanning tree for dense graphs
///
/// This function returns the parent of each vertex in the maximum
/// spanning tree of the `scores` square matrix, rooted at
/// `root_vertex`.  Each row in the matrix represents outgoing edge
/// scores of the corresponding vertex, each column incoming edge
/// scores. Thus, `scores[(parent, child)]` should give the weight of
/// the edge from `parent` to child.
///
/// Returns vertex parents. The length of the returned `Vec` equals
/// the number of rows/columns of the scores matrix.
pub fn chu_liu_edmonds(scores: ArrayView2<f32>, root_vertex: usize) -> Vec<Option<usize>> {
    assert_eq!(
        scores.nrows(),
        scores.ncols(),
        "Score matrix must be a square matrix, has shape: ({}, {})",
        scores.nrows(),
        scores.ncols()
    );

    // We use this `Vec` to keep track of which vertices are 'active'.
    // Vertices that are part of a contracted cycle become inactive.
    let mut active_vertices = vec![true; scores.nrows()];

    chu_liu_edmonds_(
        scores.to_owned().view_mut(),
        root_vertex,
        &mut active_vertices,
    )
}

fn chu_liu_edmonds_(
    mut scores: ArrayViewMut2<f32>,
    root_vertex: usize,
    active_vertices: &mut [bool],
) -> Vec<Option<usize>> {
    // For each vertex, find the parent with the highest incoming edge
    // score.
    let max_parents = find_max_parents(scores.view(), root_vertex, active_vertices);

    // Base case: if the resulting graph does not contain a cycle, we
    // have found the MST of the (possibly contracted) graph.
    let cycle = match find_cycle(&max_parents) {
        Some(cycle) => cycle,
        None => return max_parents,
    };

    // Contract the cycle into a single vertex. We use the first
    // vertex of the cycle to represent the cycle.
    let (incoming_replacements, outgoing_replacements) =
        contract_cycle(scores.view_mut(), &max_parents, active_vertices, &cycle);

    // Recursively apply Chu-Liu-Edmonds to the graph with the
    // contracted cycle, until we hit the base case.
    let contracted_mst = chu_liu_edmonds_(scores, root_vertex, active_vertices);

    // Expand the contracted cycle in the MST.
    expand_cycle(
        max_parents,
        contracted_mst,
        cycle,
        incoming_replacements,
        outgoing_replacements,
    )
}

/// Contract the given cycle.
///
/// This updates the score matrix and active vertices.
///
/// Returns a mapping of replaced edges.
#[allow(clippy::type_complexity)]
fn contract_cycle(
    mut scores: ArrayViewMut2<f32>,
    max_parents: &[Option<usize>],
    active_vertices: &mut [bool],
    cycle: &[usize],
) -> (
    HashMap<(usize, usize), usize>,
    HashMap<(usize, usize), usize>,
) {
    // We will use the first vertex of the cycle to represent the
    // contraction.
    let first_in_cycle = cycle[0];

    // Get the sum of edge scores in the cycle. See Kübler et al.,
    // 2009, pp. 47.
    let cycle_sum = cycle
        .iter()
        .map(|&vertex| {
            let parent = max_parents[vertex].unwrap();
            scores[(parent, vertex)]
        })
        .sum::<f32>();

    // Mark cycle vertices as inactive.
    for &vertex in &cycle[1..] {
        active_vertices[vertex] = false;
    }

    // Convert the cycle to a set for constant-time
    // lookups. Constructing and using a set has a negative
    // performance impact on small graphs, but we are willing to trade
    // off a small loss for better runtime properties on large graphs.
    let cycle = cycle.iter().map(ToOwned::to_owned).collect::<HashSet<_>>();

    let mut incoming_replacements = HashMap::new();
    let mut outgoing_replacements = HashMap::new();
    for vertex in 0..scores.nrows() {
        // Skip inactive vertices and vertices that are in the cycle.
        if !active_vertices[vertex] || cycle.contains(&vertex) {
            continue;
        }

        let mut best_incoming = -f32::INFINITY;
        let mut best_outgoing = -f32::INFINITY;

        let mut best_incoming_vertex = None;
        let mut best_outgoing_vertex = None;

        for &cycle_vertex in &cycle {
            // Replace (v, w) by (v_cycle, w)
            if scores[(cycle_vertex, vertex)] > best_outgoing {
                best_outgoing = scores[(cycle_vertex, vertex)];
                best_outgoing_vertex = Some(cycle_vertex);
            }

            let best_parent = max_parents[cycle_vertex].unwrap();
            let best_weight = scores[(best_parent, cycle_vertex)];
            let incoming_score = cycle_sum + scores[(vertex, cycle_vertex)] - best_weight;

            // Replace (u, v) by (u, v_cycle)
            if incoming_score > best_incoming {
                best_incoming = incoming_score;
                best_incoming_vertex = Some(cycle_vertex);
            }
        }

        // Save max incoming edge (u, v_cyle) and max outgoing edge
        // (v_cycle, w).
        scores[(vertex, first_in_cycle)] = best_incoming;
        scores[(first_in_cycle, vertex)] = best_outgoing;

        incoming_replacements.insert(
            (vertex, first_in_cycle),
            best_incoming_vertex.expect("No edge improves over -INF"),
        );
        outgoing_replacements.insert(
            (first_in_cycle, vertex),
            best_outgoing_vertex.expect("No edge improves over -INF"),
        );
    }

    (incoming_replacements, outgoing_replacements)
}

/// Expand contracted cycles.
fn expand_cycle(
    max_parents: Vec<Option<usize>>,
    mut mst: Vec<Option<usize>>,
    cycle: Vec<usize>,
    incoming_replacements: HashMap<(usize, usize), usize>,
    outgoing_replacements: HashMap<(usize, usize), usize>,
) -> Vec<Option<usize>> {
    let cycle_vertex = cycle[0];

    // Find out which edge was replaced by the incoming edge of
    // the cycle vertex...
    let kicked_out = incoming_replacements[&(mst[cycle_vertex].unwrap(), cycle_vertex)];

    // ...v of the kicked-out edge (u, v) becomes the root of the
    // to-be-broken cycle.
    mst[kicked_out] = mst[cycle_vertex];

    // Copy all other edges from the cycle.
    for cycle_vertex in cycle {
        if cycle_vertex == kicked_out {
            continue;
        }

        mst[cycle_vertex] = max_parents[cycle_vertex];
    }

    // Restore original outgoing edges, replacing (v_cycle, w) by
    // (v, w).
    for (contracted_edge, orig_edge) in outgoing_replacements {
        if mst[contracted_edge.1] == Some(contracted_edge.0) {
            mst[contracted_edge.1] = Some(orig_edge);
        }
    }

    mst
}

/// Find the parent vertex with the highest edge score for every
/// active vertex.
fn find_max_parents(
    scores: ArrayView2<f32>,
    root_vertex: usize,
    active_vertices: &[bool],
) -> Vec<Option<usize>> {
    let mut max_parents = vec![None; active_vertices.len()];

    for child in 0..scores.ncols() {
        // Do not search for parents of root.
        if child == root_vertex {
            continue;
        }

        // Skip inactive vertices.
        if !active_vertices[child] {
            continue;
        }

        // Edge scores are indexed as (parent, child).
        let parent = scores
            .index_axis(Axis(1), child)
            .iter()
            .enumerate()
            // Ignore self-loops and inactive vertices.
            .filter(|v| v.0 != child && active_vertices[v.0])
            // Find the source (parent) with the largest score.
            .max_by_key(|v| OrderedFloat(*v.1))
            // Return the index of the largest parent.
            .map(|v| v.0);

        max_parents[child] = parent;
    }

    max_parents
}

fn find_cycle(parents: &[Option<usize>]) -> Option<Vec<usize>> {
    let mut visited = vec![false; parents.len()];
    let mut on_stack = vec![false; parents.len()];
    let mut edge_to = vec![0; parents.len()];

    for start in 0..parents.len() {
        if let cycle @ Some(_) =
            find_cycle_(parents, &mut visited, &mut edge_to, &mut on_stack, start)
        {
            return cycle;
        }
    }

    None
}

fn find_cycle_(
    parents: &[Option<usize>],
    visited: &mut [bool],
    edge_to: &mut [usize],
    on_stack: &mut [bool],
    vertex: usize,
) -> Option<Vec<usize>> {
    visited[vertex] = true;
    on_stack[vertex] = true;

    // Add the vertex to the stack.
    if let Some(parent) = parents[vertex] {
        // Don't perform DFS when the vertex was already visited.
        if !visited[parent] {
            edge_to[parent] = vertex;
            if let cycle @ Some(_) = find_cycle_(parents, visited, edge_to, on_stack, parent) {
                return cycle;
            }
        } else if on_stack[parent] {
            let mut cycle = Vec::new();
            let mut cycle_vertex = vertex;

            while cycle_vertex != parent {
                cycle.push(cycle_vertex);
                cycle_vertex = edge_to[cycle_vertex];
            }
            cycle.push(parent);

            return Some(cycle);
        }
    }

    on_stack[vertex] = false;
    visited[vertex] = true;

    None
}

#[cfg(test)]
mod tests {
    use ndarray::{array, Array};
    use ndarray_rand::rand::SeedableRng;
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;
    use rand_xorshift::XorShiftRng;

    use super::{chu_liu_edmonds, find_cycle, find_max_parents};

    fn assert_tree(parents: &[Option<usize>], root: usize) {
        for (vertex, &parent) in parents.iter().enumerate() {
            if vertex == root {
                assert_eq!(
                    parent, None,
                    "Root vertex {} has a parent in graph {:?}",
                    root, parents
                )
            } else {
                assert!(
                    parent.is_some(),
                    "Non-root vertex {} does not have a parent in the graph {:?}",
                    vertex,
                    parents
                )
            }
        }

        let cycle = find_cycle(parents);
        assert_eq!(
            find_cycle(parents),
            None,
            "Graph {:?} contains a cycle: {:?}",
            parents,
            cycle.unwrap()
        );
    }

    #[test]
    pub fn finds_max_parents() {
        let distances = Array::range(0f32, 25f32, 1f32).into_shape((5, 5)).unwrap();
        let max_parents = find_max_parents(distances.view(), 0, &[true; 5]);
        assert_eq!(max_parents, vec![None, Some(4), Some(4), Some(4), Some(3)]);
    }

    #[test]
    pub fn finds_max_parents_with_inactive_vertices() {
        let distances = Array::range(0f32, 25f32, 1f32).into_shape((5, 5)).unwrap();
        let max_parents = find_max_parents(distances.view(), 0, &[true, false, true, false, true]);
        assert_eq!(max_parents, vec![None, None, Some(4), None, Some(2)]);
    }

    #[test]
    pub fn finds_trees_in_random_graphs() {
        // We should probably use quickcheck or proptest for this, but
        // then I have to figure out how to do proper shrinkage, since
        // we require square matrices. For now, we are just happy to know
        // if we produce proper trees.

        const NUM_TEST_ITERATIONS: usize = 1000;

        let mut rng = XorShiftRng::seed_from_u64(42);
        for _ in 0..NUM_TEST_ITERATIONS {
            let scores = Array::random_using((10, 10), Uniform::new(0f32, 1f32), &mut rng);
            let mst = chu_liu_edmonds(scores.view(), 0);
            assert_tree(&mst, 0);
        }
    }

    #[test]
    pub fn finds_cycle() {
        // No cycle.
        assert_eq!(
            find_cycle(&[None, Some(0), Some(1), Some(2), Some(3)]),
            None,
        );

        // No cycle.
        assert_eq!(
            find_cycle(&[None, Some(0), Some(0), Some(0), Some(0)]),
            None,
        );

        // Short cycle: 3 -> 4 -> 3
        assert_eq!(
            find_cycle(&[None, Some(4), Some(4), Some(4), Some(3)]),
            Some(vec![3, 4])
        );

        // Long cycle: 1 -> 2 -> 3 -> 4 -> 1
        assert_eq!(
            find_cycle(&[None, Some(4), Some(1), Some(2), Some(3)]),
            Some(vec![2, 3, 4, 1])
        );

        // Self-cycle
        assert_eq!(find_cycle(&[Some(0)]), Some(vec![0]));
    }

    #[test]
    fn correctly_decodes_toy_matrices() {
        let scores = Array::zeros((1, 1));
        let parents = chu_liu_edmonds(scores.view(), 0);
        assert_eq!(parents, vec![None]);

        let scores = Array::range(1f32, 10f32, 1f32).into_shape((3, 3)).unwrap();
        let parents = chu_liu_edmonds(scores.view(), 0);
        assert_eq!(parents, vec![None, Some(2), Some(0)]);

        let scores = Array::range(1f32, 17f32, 1f32).into_shape((4, 4)).unwrap();
        let parents = chu_liu_edmonds(scores.view(), 0);
        assert_eq!(parents, vec![None, Some(3), Some(3), Some(0)]);
    }

    #[test]
    fn correctly_decodes_random_large_matrices() {
        // This unit test checks the output for five random matrices
        // against the output of the AllenNLP implementation of
        // Chu-Lui-Edmonds.

        let check1 = array![
            [
                0.15154335, 0.21364425, 0.02926004, 0.24640401, 0.05929783, 0.98366485, 0.53015432,
                0.07778964, 0.00989446, 0.17998191
            ],
            [
                0.68921352, 0.33551225, 0.91974265, 0.08476561, 0.48800752, 0.87661821, 0.31723634,
                0.51386131, 0.97963044, 0.36960274
            ],
            [
                0.13969799, 0.46092784, 0.75821582, 0.78823102, 0.63945137, 0.42556879, 0.81997744,
                0.12978648, 0.40536874, 0.4744205
            ],
            [
                0.40688978, 0.25514681, 0.59851297, 0.82950985, 0.46627791, 0.05888491, 0.97450763,
                0.90287058, 0.35996474, 0.6448661
            ],
            [
                0.30530523, 0.76566773, 0.64714425, 0.1424588, 0.14283951, 0.00153444, 0.9688441,
                0.87582559, 0.63371798, 0.67004456
            ],
            [
                0.88822529, 0.26780501, 0.61901697, 0.35049028, 0.06430303, 0.44334551, 0.15308377,
                0.42145127, 0.87420229, 0.3309963
            ],
            [
                0.31808055, 0.35399265, 0.31438455, 0.63534316, 0.36917357, 0.7707749, 0.1686939,
                0.66622048, 0.67872444, 0.28663183
            ],
            [
                0.82167446, 0.15910145, 0.6654594, 0.54279563, 0.19068867, 0.17368633, 0.07199292,
                0.29239669, 0.60002772, 0.75121407
            ],
            [
                0.74016819, 0.28619099, 0.71608573, 0.64490596, 0.05975497, 0.8792097, 0.85888953,
                0.90590799, 0.62783992, 0.12660846
            ],
            [
                0.80810707, 0.10910174, 0.11777376, 0.36885688, 0.88732921, 0.82053854, 0.84096041,
                0.53546477, 0.49554398, 0.21705035
            ]
        ];

        assert_eq!(
            chu_liu_edmonds(check1.view(), 0),
            [
                None,
                Some(4),
                Some(1),
                Some(2),
                Some(9),
                Some(0),
                Some(3),
                Some(8),
                Some(5),
                Some(7)
            ]
        );

        let check2 = array![
            [
                0.63699522, 0.87615555, 0.45236657, 0.5188734, 0.13080447, 0.30954603, 0.70385654,
                0.00940039, 0.99012901, 0.91048303
            ],
            [
                0.6110081, 0.11629512, 0.91845679, 0.55938488, 0.45709085, 0.16727591, 0.3338458,
                0.87262039, 0.26543677, 0.78429413
            ],
            [
                0.06226577, 0.3509711, 0.8738929, 0.77723445, 0.83439156, 0.72800083, 0.70465176,
                0.9323746, 0.01803918, 0.50092784
            ],
            [
                0.30294811, 0.65599656, 0.23342294, 0.01840916, 0.78500845, 0.78103093, 0.82584077,
                0.72756822, 0.60326683, 0.44574654
            ],
            [
                0.75513096, 0.06980882, 0.72330091, 0.94334981, 0.262673, 0.84566782, 0.6318016,
                0.0442728, 0.2669838, 0.59781991
            ],
            [
                0.27443631, 0.33890352, 0.83353679, 0.88552379, 0.89789705, 0.00165288, 0.17836232,
                0.59181986, 0.426987, 0.91632828
            ],
            [
                0.55585136, 0.87230681, 0.10995064, 0.65543565, 0.96603594, 0.34425304, 0.07438735,
                0.21991817, 0.53278602, 0.46460502
            ],
            [
                0.78368679, 0.55949995, 0.42268737, 0.1681499, 0.62903574, 0.75765237, 0.07484798,
                0.37319298, 0.62900207, 0.26623339
            ],
            [
                0.66636035, 0.19227743, 0.48126272, 0.14611228, 0.6107612, 0.30056951, 0.77329224,
                0.93780084, 0.12710157, 0.96506847
            ],
            [
                0.76441608, 0.25583239, 0.14817458, 0.68389535, 0.85748418, 0.81745151, 0.71656758,
                0.11733889, 0.98476048, 0.26556185
            ]
        ];

        assert_eq!(
            chu_liu_edmonds(check2.view(), 0),
            [
                None,
                Some(0),
                Some(1),
                Some(4),
                Some(6),
                Some(4),
                Some(8),
                Some(8),
                Some(0),
                Some(8)
            ]
        );

        let check3 = array![
            [
                0.32226934, 0.03494655, 0.13943128, 0.77627796, 0.32289177, 0.20728151, 0.79354934,
                0.44277001, 0.70666543, 0.76361263
            ],
            [
                0.89787456, 0.19412729, 0.2769623, 0.42547065, 0.78306101, 0.99639906, 0.44910723,
                0.69166559, 0.5974235, 0.6019087
            ],
            [
                0.01936413, 0.77783413, 0.2635923, 0.24239049, 0.15320177, 0.58810727, 0.93770173,
                0.97238493, 0.40536974, 0.28189387
            ],
            [
                0.21176774, 0.90580752, 0.48167285, 0.17517493, 0.35126148, 0.09566258, 0.77651317,
                0.844114, 0.32902123, 0.93356815
            ],
            [
                0.68965019, 0.98577739, 0.06460552, 0.103729, 0.59807881, 0.82418659, 0.20288672,
                0.55119795, 0.01953631, 0.75208802
            ],
            [
                0.49706455, 0.52543525, 0.16288358, 0.72442708, 0.57151594, 0.68195141, 0.47521668,
                0.56127222, 0.6673682, 0.93037853
            ],
            [
                0.12841745, 0.89183647, 0.21585613, 0.73852511, 0.09812739, 0.06616884, 0.12730214,
                0.8322976, 0.93773286, 0.23950978
            ],
            [
                0.73496813, 0.52910843, 0.94925765, 0.77135859, 0.85716859, 0.47158383, 0.88753378,
                0.00141653, 0.47463287, 0.33777619
            ],
            [
                0.76116294, 0.77581507, 0.99508616, 0.24001213, 0.13688175, 0.57771731, 0.1435426,
                0.18420174, 0.07373099, 0.15492254
            ],
            [
                0.88146862, 0.27868822, 0.41427004, 0.989063, 0.08847578, 0.31721111, 0.13694788,
                0.99730908, 0.8523681, 0.81020978
            ]
        ];

        assert_eq!(
            chu_liu_edmonds(check3.view(), 0),
            [
                None,
                Some(4),
                Some(8),
                Some(9),
                Some(7),
                Some(1),
                Some(0),
                Some(2),
                Some(6),
                Some(5)
            ]
        );

        let check4 = array![
            [
                0.94146094, 0.08429249, 0.11658879, 0.7209569, 0.04588338, 0.41361274, 0.00335799,
                0.58725318, 0.37633847, 0.50978681
            ],
            [
                0.50163181, 0.96919669, 0.16614751, 0.15533209, 0.15054694, 0.08811524, 0.13978445,
                0.65591973, 0.95264964, 0.17669406
            ],
            [
                0.36864862, 0.95739286, 0.65356991, 0.71690581, 0.29263559, 0.98409776, 0.61308834,
                0.50921288, 0.49160935, 0.53610581
            ],
            [
                0.23275999, 0.60587704, 0.55893549, 0.69733286, 0.30008536, 0.13133368, 0.90196987,
                0.52283165, 0.96302483, 0.44467621
            ],
            [
                0.15057842, 0.58499236, 0.11330645, 0.57510935, 0.39645653, 0.53736407, 0.08391498,
                0.06004636, 0.88086527, 0.25429321
            ],
            [
                0.40042428, 0.08725659, 0.87216523, 0.18444633, 0.61547065, 0.8032823, 0.16163181,
                0.81884952, 0.51741822, 0.73005934
            ],
            [
                0.08460523, 0.01342742, 0.70127922, 0.45693109, 0.40153192, 0.07611445, 0.74831201,
                0.3385515, 0.24000027, 0.33290993
            ],
            [
                0.01990056, 0.28629396, 0.85476794, 0.68330081, 0.93204836, 0.14587584, 0.06681271,
                0.50342723, 0.30878763, 0.51632671
            ],
            [
                0.22297607, 0.99004514, 0.02590417, 0.61425698, 0.16932825, 0.06197453, 0.58227628,
                0.46317503, 0.21611736, 0.88426682
            ],
            [
                0.21695749, 0.52528143, 0.9569687, 0.70641648, 0.45516634, 0.59951297, 0.82591367,
                0.6038499, 0.14423517, 0.12984568
            ]
        ];

        assert_eq!(
            chu_liu_edmonds(check4.view(), 0),
            [
                None,
                Some(8),
                Some(9),
                Some(0),
                Some(7),
                Some(2),
                Some(3),
                Some(5),
                Some(3),
                Some(8)
            ]
        );

        let check5 = array![
            [
                0.19181828, 0.07215655, 0.49029481, 0.40338361, 0.77464947, 0.15287357, 0.33550702,
                0.9075557, 0.16816009, 0.12815985
            ],
            [
                0.39814249, 0.83951939, 0.6197687, 0.10285881, 0.35754604, 0.03372432, 0.26903616,
                0.39758852, 0.27831648, 0.8626124
            ],
            [
                0.32651809, 0.36621293, 0.55139869, 0.48841691, 0.86105511, 0.95220918, 0.99901665,
                0.43452191, 0.51957831, 0.12977951
            ],
            [
                0.24777433, 0.20835293, 0.35423981, 0.8647926, 0.54734269, 0.19705202, 0.20262791,
                0.29885766, 0.89558149, 0.48529723
            ],
            [
                0.99486246, 0.02998787, 0.94388915, 0.16682153, 0.04621821, 0.78283825, 0.32711021,
                0.11668783, 0.54230828, 0.01990573
            ],
            [
                0.81816179, 0.77223827, 0.3778254, 0.14590591, 0.53032985, 0.12751733, 0.80951733,
                0.94590486, 0.14917576, 0.0905699
            ],
            [
                0.56977204, 0.6759112, 0.86349563, 0.30270709, 0.03673155, 0.8814458, 0.52538187,
                0.97650872, 0.9278274, 0.73412665
            ],
            [
                0.96577082, 0.17352435, 0.71417166, 0.57713058, 0.99690502, 0.5856659, 0.87223811,
                0.8265802, 0.07539461, 0.28718492
            ],
            [
                0.64135636, 0.53712009, 0.98343642, 0.68861079, 0.33153221, 0.86677607, 0.65411023,
                0.97146557, 0.78007143, 0.24988737
            ],
            [
                0.52704545, 0.39384584, 0.99308, 0.03148114, 0.43305557, 0.11551732, 0.13331425,
                0.17881437, 0.05076005, 0.20889167
            ]
        ];

        assert_eq!(
            chu_liu_edmonds(check5.view(), 0),
            [
                None,
                Some(5),
                Some(4),
                Some(8),
                Some(7),
                Some(2),
                Some(2),
                Some(0),
                Some(6),
                Some(1)
            ]
        );
    }

    #[test]
    #[should_panic]
    fn panics_on_incorrect_shape_score_matrix() {
        let scores = Array::range(0f32, 16f32, 1f32).into_shape((2, 8)).unwrap();
        let _ = chu_liu_edmonds(scores.view(), 0);
    }
}
