use numpy::PyReadonlyArray2;
use pyo3::{exceptions::{PyIndexError, PyValueError}, prelude::*};

mod mst;

#[pyfunction]
fn chu_liu_edmonds(
    edge_weights: PyReadonlyArray2<f32>,
    root_vertex: usize,
) -> PyResult<Vec<Option<usize>>> {
    let shape = edge_weights.shape();

    if shape[0] != shape[1] {
        return Err(PyValueError::new_err(format!("Edge weight matrix with shape ({}, {}) is not a square matrix", shape[0], shape[1])))
    }

    if root_vertex >= shape[0] {
        return Err(PyIndexError::new_err(format!(
            "Head {} out of bounds for edge weight matrix with shape ({}, {})",
            root_vertex, shape[0], shape[1]
        )));
    }

    Ok(mst::chu_liu_edmonds(edge_weights.as_array(), root_vertex))
}

#[pymodule]
fn mst(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(chu_liu_edmonds, m)?)?;
    Ok(())
}
