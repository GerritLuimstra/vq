use super::Prototype;
use super::GeneralMatrixLearningVectorQuantization;
use super::helpers::find_closest_prototype_matched;
use super::helpers::{euclidean_distance, find_closest_prototype};

use rand::Rng;
use ndarray::Array1;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaChaRng;
use std::collections::BTreeMap;

impl GeneralMatrixLearningVectorQuantization {

    /// Constructs a new General Learning Vector Quantization model
    /// 
    /// TODO: Fix documentation.
    /// 
    /// # Arguments
    /// 
    /// * `num_prototypes` The amount of prototypes to use per class (a BTreeMap, that maps the class name to the number of prototypes to use)
    ///                    This BTreeMap should be provided as a reference and the algorithm will panic if there are classes
    ///                    in the data not present in this BTreeMap.
    ///                    A BTreeMap is used instead of a HashMap due to the ability of sorted keys, which is required for reproducability.
    /// * `learning_rate`  The learning rate for the update step of the prototypes
    /// * `max_epochs`     The amount of epochs to run
    /// * `prototypes`     A vector of the prototypes (initially empty)
    /// * `seed`           The seed to be used by the internal ChaChaRng.
    /// 
    pub fn new ( num_prototypes: BTreeMap<String, usize>,
                 learning_rate: f64,
                 max_epochs: u32,
                 seed: Option<u64> ) -> GeneralMatrixLearningVectorQuantization {
        
        // Setup the model
        GeneralMatrixLearningVectorQuantization {
            num_prototypes,
            omega: None,
            learning_rate,
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

    /// Fits the General Matrix Learning Vector Quantization model on the given data
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

        for _epoch in 1 .. self.max_epochs + 1 {
            // TODO: Implement.
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
            // TODO: Implement.
        }

        cluster_labels
    }

    /// Simple getter for the prototype clusters
    pub fn prototypes(&self) -> &Vec<Prototype> {
        &self.prototypes
    }

}