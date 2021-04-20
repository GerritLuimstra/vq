use super::Prototype;
use super::LearningVectorQuantization;
use super::helpers::euclidean_distance;

use ndarray::Array1;
use rand::prelude::thread_rng;
use rand::seq::SliceRandom;

impl LearningVectorQuantization {

    /// Constructs a new Learning Vector Quantization model
    /// 
    /// # Arguments
    /// 
    /// * `num_prototypes` The amount of prototypes to use per class (a vector, so this can be variable)
    /// * `learning_rate`  The learning rate for the update step of the prototypes
    /// * `max_epochs`     The amount of epochs to run
    /// * `prototypes`     A vector of the prototypes (initially empty)
    /// 
    pub fn new ( num_prototypes: Vec<usize>, 
                 learning_rate: f64,
                 max_epochs: u32, 
                 seed: Option<u32> ) -> LearningVectorQuantization {
        
        // Setup the model
        LearningVectorQuantization {
            num_prototypes,
            learning_rate,
            max_epochs, 
            seed, // TODO: Implement
            prototypes: Vec::<Prototype>::new(),
        }
    }

    /// Obtains the closest prototype index for a given sample
    /// 
    /// # Arguments
    /// 
    /// * `sample` The sample to find the closest prototype for
    /// 
    fn find_closest_prototype (&self, sample : &Array1<f64>) -> usize {

        // Initialize values
        let mut closest_prototype_index = 0 as usize;
        let mut smallest_distance       = f64::INFINITY;

        for (index, prototype) in self.prototypes.iter().enumerate() {

            // Obtain the difference between the sample and the current prototype
            let distance = euclidean_distance(&prototype.vector, sample);

            // Update the current closest, if we have found a one that is closer
            if distance < smallest_distance {
                closest_prototype_index = index;
                smallest_distance       = distance;
            }
        }

        closest_prototype_index
    }

    /// Fits the Vector Quantization model on the given data
    /// 
    /// # Arguments
    /// 
    /// * `data` The data to adapt the prototypes on
    /// 
    pub fn fit (&mut self, data : &Vec<Array1<f64>>) {
        assert!(false, "NOT YET IMPLEMENTED.");
    }

    /// Assign cluster labels (i.e. predict) to the data given data
    /// based on the learned prototype vectors
    /// 
    /// # Arguments
    /// 
    /// * `data` The data to obtain the cluster labels for
    /// 
    pub fn predict(&self, data : &Vec<Array1<f64>>) -> Vec<String> {

        // Check for valid input
        assert!(data.len() > 0, "There are no data samples given.");
        assert!(self.prototypes.len() > 0, "The model has not been fit yet.");
        assert!(self.prototypes[0].vector.len() == data[0].len(), 
                "Data must be the same sized as was used in fit!");

        let mut cluster_labels = Vec::<String>::new();

        for data_sample in data {

            // Obtain the closest prototype
            let closest_prototype_index = self.find_closest_prototype(&data_sample);
            let closest_prototype       = self.prototypes.get(closest_prototype_index).unwrap(); 

            // Add the cluster label to the list
            cluster_labels.push(closest_prototype.name.clone());
        }

        cluster_labels
    }
    
    /// Names the internal prototypes to the given names
    /// 
    /// NOTE: This affects the labels returned by the predict method.
    /// 
    /// # Arguments
    /// 
    /// * `names` The names (in order) to give to the prototypes
    /// 
    pub fn name_prototypes(&mut self, names : &Vec<String>) {

        // Check for valid input
        assert!(self.prototypes.len() > 0, "The model has not been fit yet.");
        assert!(names.len() == self.prototypes.len(), 
                "The size of the names vectors does not match the amount of the prototypes.");

        for (index, name) in names.iter().enumerate() {
            self.prototypes[index].name = name.clone();
        }
    }

    /// Simple getter for the prototype clusters
    pub fn prototypes(&self) -> &Vec<Prototype> {
        &self.prototypes
    }

}