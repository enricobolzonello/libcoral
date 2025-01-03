use ndarray::{prelude::*, Data};

use crate::metricdata::MetricData;

pub struct AngularData<S: Data<Elem=f32>> {
    data: ArrayBase<S, Ix2>,
    norms: Array1<f32>,
}

impl<S: Data<Elem = f32>> AngularData<S> {
    pub fn new(data: ArrayBase<S, Ix2>) -> Self {
        let norms = data.rows().into_iter().map(|row| row.dot(&row).sqrt()).collect();
        Self {
            data,
            norms
        }
    }
}

impl<S: Data<Elem = f32>> MetricData for AngularData<S> {
    fn distance(&self, i: usize, j: usize) -> f32 {
        1.0 - ( self.data.row(i).dot(&self.data.row(j)) / (self.norms[i] * self.norms[j]) )
    }

    fn all_distances(&self, j: usize, out: &mut [f32]){
        assert_eq!(out.len(), self.data.nrows());
        for (i, oo) in out.iter_mut().enumerate() {
            *oo = self.distance(i, j);
        }
    }

    fn num_points(&self) -> usize {
        self.data.nrows()
    }

    fn dimensions(&self) -> usize {
        self.data.ncols()
    }
}

// TODO: still left to implement Subset and NChunks traits