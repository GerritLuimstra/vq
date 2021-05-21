use super::Prototype;
use super::LVQ;
use super::helpers::find_closest_prototype;
use super::traits::Schedulable;

use rand::Rng;
use ndarray::Array1;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaChaRng;
use std::collections::BTreeMap;

impl LVQ {

    /// Constructs a new Learning Vector Quantization model
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
                 seed: Option<u64> ) -> LVQ {

        LVQ {
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
    fn check_fit_constraints(&mut self, data : &Vec<Array1<f64>>, labels : &Vec<String>) {

        // Check that the data and labels have the same length
        assert!(data.len() == labels.len(), "The data vector does not match the label vector in length.");

        // Compute the total amount of prototypes given
        let mut total_prototypes = 0;
        for num_prototypes in self.num_prototypes.values() {
            total_prototypes += num_prototypes;

            if num_prototypes <= &(0 as usize) {
                panic!("Each class needs to have at least one prototype!");
            }
        }

        // Check that each label that is provided, is found in the prototype mapping
        for label in labels.iter() {
            if !self.num_prototypes.contains_key(label) {
                panic!("Unknown label {}. Consider adding it to the prototype mapping.", label);
            }
        }

        // Assert that there is enough data
        assert!(data.len() >= total_prototypes,
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
                "Data must have the same dimensions as was used in fit!");
    }

    ///
    /// Sets up the required data before the model is fit
    /// 
    fn setup(&mut self, data : &Vec<Array1<f64>>, labels : &Vec<String>) {

        // Setup the prototypes by grabbing the respective amount of vectors from the data
        // based on the amount of prototypes specified for that class
        for (class_name, num_prototypes) in self.num_prototypes.iter() {

            // Obtain all the data samples with the class 'class_name'
            let mut data_class_indices = vec![];
            for (index, sample_label) in labels.iter().enumerate() {
                if sample_label == class_name {
                    data_class_indices.push(index);
                }
            }

            // Grab 'num_prototypes' prototypes
            for _ in 0 .. *num_prototypes {

                // Obtain a random prototype from the data samples by class and clone/own it
                let selected_prototype = *data_class_indices.choose(&mut self.rng).unwrap();
                let selected_prototype = data[selected_prototype].clone();
                let selected_prototype = Prototype::new(selected_prototype, class_name.clone());

                // Add the newly created prototypes to the prototype list
                self.prototypes.push(selected_prototype);
            }
        }

    }

    /// Fits the Learning Vector Quantization model on the given data
    /// 
    /// # Arguments
    /// 
    /// * `data`   The data to adapt the prototypes on
    /// * `labels` The labels of the samples
    /// 
    pub fn fit (&mut self, data : &Vec<Array1<f64>>, labels : &Vec<String>) {

        // Check if the required constraints are present
        self.check_fit_constraints(&data, &labels);

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

                // Find the closest prototype to the data point
                let closest_prototype_index = find_closest_prototype(&self.prototypes, &data_sample, None);
                let closest_prototype       = self.prototypes.get(closest_prototype_index).unwrap();

                // Compute the difference vector (the 'error')
                let difference = data_sample - closest_prototype.vector.clone();

                // Obtain the current learning rate
                let learning_rate = (self.lr_scheduler)(self.initial_lr, epoch, self.max_epochs);

                // Update the current prototype by either moving it closer to the data sample
                // if the classes of the data sample and the closest prototype match and move it
                // further away otherwise.
                let new_prototype;
                if *data_label == closest_prototype.name {
                    new_prototype = closest_prototype.vector.clone() + learning_rate * difference;
                } else {
                    new_prototype = closest_prototype.vector.clone() - learning_rate * difference;
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
    pub fn predict(&mut self, data : &Vec<Array1<f64>>) -> Vec<String> {

        // Check predict constraints
        self.check_predict_constraints(&data);

        let mut cluster_labels = Vec::<String>::new();

        for data_sample in data {

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

impl Schedulable for LVQ {

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