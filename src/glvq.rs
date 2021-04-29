use super::Prototype;
use super::GeneralLearningVectorQuantization;
use super::helpers::euclidean_distance;

use ndarray::Array1;
use rand::seq::SliceRandom;
use std::collections::HashMap;
use rand::prelude::thread_rng;

impl GeneralLearningVectorQuantization {

    /// TODO: Add comments.
    pub fn new ( num_prototypes: HashMap<String, usize>, 
                 learning_rate: f64,
                 max_epochs: u32, 
                 seed: Option<u32> ) -> GeneralLearningVectorQuantization {
        
        // Setup the model
        GeneralLearningVectorQuantization {
            num_prototypes,
            learning_rate,
            max_epochs, 
            seed, // TODO: Implement
            prototypes: Vec::<Prototype>::new(),
        }
    }

    /// TODO: Comment.
    fn find_closest_prototype (&self, sample : &Array1<f64>) -> usize {
        unimplemented!();
    }

    /// TODO: Comment.
    pub fn fit (&mut self, data : &Vec<Array1<f64>>, labels : &Vec<String>) {
        unimplemented!();
    }

    /// TODO: Comment.
    pub fn predict(&self, data : &Vec<Array1<f64>>) -> Vec<String> {
        unimplemented!();
    }

    /// Simple getter for the prototype clusters
    pub fn prototypes(&self) -> &Vec<Prototype> {
        &self.prototypes
    }

}