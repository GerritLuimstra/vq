use super::Prototype;
use super::GeneralLearningVectorQuantization;
use super::helpers::find_closest_prototype_matched;
use super::helpers::{euclidean_distance, find_closest_prototype};
use super::traits::Schedulable;

use rand::Rng;
use ndarray::Array1;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaChaRng;
use std::collections::BTreeMap;

impl GeneralLearningVectorQuantization {

    /// Constructs a new General Learning Vector Quantization model
    /// 
    /// # Arguments
    /// 
    /// * `num_prototypes` The amount of prototypes to use per class (a BTreeMap, that maps the class name to the number of prototypes to use)
    ///                    This BTreeMap should be provided as a reference and the algorithm will panic if there are classes
    ///                    in the data not present in this BTreeMap.
    ///                    A BTreeMap is used instead of a HashMap due to the ability of sorted keys, which is required for reproducability.
    /// * `initial_lr`     The initial learning rate to be used by the learning rate scheduler
    /// * `max_epochs`     The amount of epochs to run
    /// * `prototypes`     A vector of the prototypes (initially empty)
    /// * `seed`           The seed to be used by the internal ChaChaRng.
    /// 
    pub fn new ( num_prototypes: BTreeMap<String, usize>,
                 initial_lr: f64,
                 max_epochs: u32,
                 seed: Option<u64> ) -> GeneralLearningVectorQuantization {
        
        // Setup the model
        GeneralLearningVectorQuantization {
            num_prototypes,
            initial_lr,
            lr_scheduler : |initial_lr, _, _| -> f64 { initial_lr },
            max_epochs, 
            rng: {
                if seed != None {
                    ChaChaRng::seed_from_u64(seed.unwrap())
                } else {
                    ChaChaRng::seed_from_u64(rand::thread_rng().gen::<u64>())
                }
            },
            prototypes: Vec::<Prototype>::new(),
        }
    }

    ///
    /// Checks if the required constraints are met for the model fitting stage
    /// 
    fn check_fit_constraints(&mut self, data : &Vec<Array1<f64>>) {

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
    
    }

    ///
    /// Checks if the required constraints are met for the prediction stage
    /// 
    fn check_predict_constraints(&mut self, data : &Vec<Array1<f64>>) {
        assert!(data.len() > 0, "There are no data samples given.");
        assert!(self.prototypes.len() > 0, "The model has not been fit yet.");
        assert!(self.prototypes[0].vector.len() == data[0].len(), 
                "Data must be the same sized as was used in fit!");
    }

    ///
    /// Sets up the required data before the model is fit
    /// 
    fn setup(&mut self, data : &Vec<Array1<f64>>, labels : &Vec<String>) {

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

            // Grab 'num_prototypes' prototypes
            for _ in 0 .. *num_prototypes {

                // Obtain a random prototype from the data samples by class and clone/own it
                let selected_prototype = data_samples_by_class.choose(&mut self.rng).unwrap();
                let selected_prototype = selected_prototype.clone();
                let selected_prototype = Prototype::new(selected_prototype, class_name.clone());

                // Add the newly created prototypes to the prototype list
                self.prototypes.push(selected_prototype);
            }
        }

    }

    /// Fits the General Learning Vector Quantization model on the given data
    /// 
    /// # Arguments
    /// 
    /// * `data`   The data to adapt the prototypes on
    /// * `labels` The labels of the samples
    /// 
    pub fn fit (&mut self, data : &Vec<Array1<f64>>, labels : &Vec<String>) {

        // Check if the required constraints are present
        self.check_fit_constraints(&data);

        // Perform the required setup:
        // Initialize the prototypes
        self.setup(&data, &labels);

        for epoch in 1 .. self.max_epochs + 1 {

            // We should be careful to shuffle the labels and data in the same matter
            let mut shuffled_indices : Vec<usize> = (0 .. data.len()).collect();
            shuffled_indices.shuffle(&mut self.rng);

            // Iterate over the shuffled data and update the closest prototype
            for data_index in shuffled_indices.iter() {

                // Setup the required variables
                let data_index = *data_index;
                let data_label  = &labels[data_index];
                let data_sample = &data[data_index];

                // Find the closest matching and non matching prototypes
                let closest_matching_prototype_index     = find_closest_prototype_matched(&self.prototypes,
                                                                                          &data_sample, 
                                                                                          &data_label, true, None);
                let closest_non_matching_prototype_index = find_closest_prototype_matched(&self.prototypes,
                                                                                          &data_sample, 
                                                                                          &data_label, false, None);
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

                // Obtain the current learning rate
                let learning_rate = (self.lr_scheduler)(self.initial_lr, epoch, self.max_epochs);

                // Perform the complete update rules
                let new_matching_prototype     = closest_matching_prototype.vector.clone() - learning_rate * matching_derivative;
                let new_non_matching_prototype = closest_non_matching_prototype.vector.clone() - learning_rate * non_matching_derivative;
                
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
    pub fn predict(&mut self, data : &Vec<Array1<f64>>) -> Vec<String> {

        // Check predict constraints
        self.check_predict_constraints(&data);

        let mut cluster_labels = Vec::<String>::new();

        for data_sample in data {

            // TODO: Make this also work with the custom functions.

            // Obtain the closest prototype
            let closest_prototype_index = find_closest_prototype(&self.prototypes, &data_sample, None);
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

impl Schedulable for GeneralLearningVectorQuantization {

    /// Updates the learning rate scheduler
    /// 
    /// # Arguments
    /// 
    /// * `scheduler` A function that takes in three arguments of the form (initial_learning_rate, current epoch, max epochs)
    ///               The algorithm will insert these arguments when the learning rate is to be calculated.
    ///  
    fn set_learning_rate_scheduler (&mut self, scheduler : fn(f64, u32, u32) -> f64) {
        self.lr_scheduler = scheduler;
    }

}