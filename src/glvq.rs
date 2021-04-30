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

    /// Obtains the closest matching prototype index for a given sample
    /// If the `find_closest_matching` is set to false, 
    /// obtain the closest prototype with a different class instead.
    /// 
    /// # Arguments
    /// 
    /// * `sample` The sample to find the closest prototype for
    /// * `label`  The label of the sample
    /// * `find_closest_matching` Determines whether the closest matching 
    /// or non-matching prototype is to be found.
    fn find_closest_prototype_matched (&self, 
                                        sample : &Array1<f64>, 
                                        label: &String,
                                        find_closest_matching: bool) -> usize {
       
        // Initialize values
        let mut closest_prototype_index = 0 as usize;
        let mut smallest_distance       = f64::INFINITY;

        for (index, prototype) in self.prototypes.iter().enumerate() {

            // Obtain the difference between the sample and the current prototype
            let distance = euclidean_distance(&prototype.vector, sample);

            // Find the closest prototype with the same class
            if find_closest_matching {
                
                // Update the current closest, if we have found a one that is closer
                if distance < smallest_distance && prototype.name == *label {
                    closest_prototype_index = index;
                    smallest_distance       = distance;
                }

            } else {
                // In this case, we want to find the closest prototype with a different class

                // Update the current closest, if we have found a one that is closer
                if distance < smallest_distance && prototype.name != *label {
                    closest_prototype_index = index;
                    smallest_distance       = distance;
                }

            }
        
        }

        closest_prototype_index
    }

    /// Fits the General Learning Vector Quantization model on the given data
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

                let label = &shuffled_labels[index];

                // Find the closest matching and non matching prototypes
                let closest_matching_prototype_index     = self.find_closest_prototype_matched(&data_sample, &label, true);
                let closest_non_matching_prototype_index = self.find_closest_prototype_matched(&data_sample, &label, false);
                let closest_matching_prototype     = self.prototypes.get(closest_matching_prototype_index).unwrap();
                let closest_non_matching_prototype = self.prototypes.get(closest_non_matching_prototype_index).unwrap();

                // Compute the distances d1 and d2 according to the paper of GLVQ
                // d1 = euclidean distance between sample and the closest matching prototype
                // d2 = euclidean distance between sample and the closest non-matching prototype
                let d1 = euclidean_distance(data_sample, &closest_matching_prototype.vector);
                let d2 = euclidean_distance(data_sample, &closest_non_matching_prototype.vector);

                // Compute the differences between the sample and the closest (non)-matching prototypes
                let match_difference     = data_sample - closest_matching_prototype.vector.clone();
                let non_match_difference = data_sample - closest_non_matching_prototype.vector.clone();

                // Compute the derivates with respect to the loss function S from the paper
                // TODO: Implement custom F functions, currently only the identity function is used
                let f_deriv = 1.0;
                let matching_derivative     =  - (f_deriv) * (4.0 * d2) / ((d1 + d2) * (d1 + d2)) * match_difference;
                let non_matching_derivative =    (f_deriv) * (4.0 * d1) / ((d1 + d2) * (d1 + d2)) * non_match_difference;

                // Perform the complete update rules
                let new_matching_prototype     = closest_matching_prototype.vector.clone() - self.learning_rate * matching_derivative;
                let new_non_matching_prototype = closest_non_matching_prototype.vector.clone() - self.learning_rate * non_matching_derivative;
                
                // Update the prototypes
                self.prototypes[closest_matching_prototype_index].vector     = new_matching_prototype;
                self.prototypes[closest_non_matching_prototype_index].vector = new_non_matching_prototype;
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