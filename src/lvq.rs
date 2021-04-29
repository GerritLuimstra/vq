use super::Prototype;
use super::LearningVectorQuantization;
use super::helpers::euclidean_distance;

use ndarray::Array1;
use rand::seq::SliceRandom;
use std::collections::HashMap;
use rand::prelude::thread_rng;

impl LearningVectorQuantization {

    /// Constructs a new Learning Vector Quantization model
    /// 
    /// # Arguments
    /// 
    /// * `num_prototypes` The amount of prototypes to use per class (a hashmap, that maps the class name to the number of prototypes to use)
    ///                    This hashmap should be provided as a reference and the algorithm will panic if there are classes 
    ///                    in the data not present in this hashmap.
    /// * `learning_rate`  The learning rate for the update step of the prototypes
    /// * `max_epochs`     The amount of epochs to run
    /// * `prototypes`     A vector of the prototypes (initially empty)
    /// 
    pub fn new ( num_prototypes: HashMap<String, usize>, 
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
    /// * `data`   The data to adapt the prototypes on
    /// * `labels` The labels of the samples
    /// 
    pub fn fit (&mut self, data : &Vec<Array1<f64>>, labels : &Vec<String>) {

        // Compute the total amount of prototypes given
        let mut total_prototypes = 0;
        for num_prototypes in self.num_prototypes.values() {
            total_prototypes += num_prototypes;
        }

        // Assert that there is enough data
        assert!(data.len() > total_prototypes,
                "There are more prototypes than data samples. Consider lowering the amount of prototypes.");

        // Assert that the model has not been fit yet
        assert!(self.prototypes.len() == 0, "This model has already been fit.");

        // Setup the prototypes by grabbing the respective amount of vectors from the data
        // based on the amount of prototypes specified for that class
        for (class_name, num_prototypes) in self.num_prototypes.iter() {

            // Obtain all the data samples with the class 'class_name'
            let mut data_samples_by_class = vec![];
            for (index, sample_label) in labels.iter().enumerate() {
                if sample_label == class_name {
                    data_samples_by_class.push(data[index].clone());
                }
            }

            // Obtain a random prototype from the data samples by class and clone/own it
            let selected_prototype = data_samples_by_class.choose(&mut rand::thread_rng()).unwrap();
            let selected_prototype = selected_prototype.clone();
            let selected_prototype = Prototype::new(selected_prototype, class_name.clone());

            // Add the newly created prototypes to the prototype list
            self.prototypes.push(selected_prototype);
        }

        for _epoch in 1 .. self.max_epochs + 1 {

            // Shuffle the data to prevent artifacts during training
            // We should be careful to shuffle the labels and data in the same matter
            let mut shuffled_indices : Vec<usize> = (0 .. data.len()).collect();
            shuffled_indices.shuffle(&mut thread_rng());

            // Create shuffled (and cloned) data based on the shuffled indices
            let mut shuffled_data   = vec![];
            let mut shuffled_labels = vec![];
            for index in shuffled_indices {
                shuffled_data.push(data[index].clone());
                shuffled_labels.push(labels[index].clone());
            }

            // Iterate over the shuffled data and update the closest prototype
            for (index, data_sample) in shuffled_data.iter().enumerate() {

                // Find the closest prototype to the data point
                let closest_prototype_index = self.find_closest_prototype(&data_sample);
                let closest_prototype       = self.prototypes.get(closest_prototype_index).unwrap();

                // Compute the difference vector (the 'error')
                let difference = data_sample - closest_prototype.vector.clone();

                // Update the current prototype by either moving it closer to the data sample
                // if the classes of the data sample and the closest prototype match and move it
                // further away otherwise.
                let new_prototype;
                if shuffled_labels[index] == closest_prototype.name {
                    new_prototype = closest_prototype.vector.clone() + self.learning_rate * difference;
                } else {
                    new_prototype = closest_prototype.vector.clone() - self.learning_rate * difference;
                }

                // Replace the old prototype
                self.prototypes[closest_prototype_index].vector = new_prototype;
            }
        }
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

    /// Simple getter for the prototype clusters
    pub fn prototypes(&self) -> &Vec<Prototype> {
        &self.prototypes
    }

}