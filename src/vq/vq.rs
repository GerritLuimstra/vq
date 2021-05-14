use super::Prototype;
use super::VectorQuantization;
use super::helpers::find_closest_prototype;

use rand::Rng;
use ndarray::Array1;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaChaRng;

impl VectorQuantization {

    /// Constructs a new Vector Quantization model
    /// 
    /// # Arguments
    /// 
    /// * `num_prototypes` The number of prototypes to use
    /// * `learning_rate`  The learning rate for the update step of the prototypes
    /// * `max_epochs`     The amount of epochs to run
    /// * `prototypes`     A vector of the prototypes (initially empty)
    /// * `seed`           The seed to be used by the internal ChaChaRng.
    /// 
    pub fn new (num_prototypes: u32, 
                learning_rate: f64,
                max_epochs: u32, 
                seed: Option<u64> ) -> VectorQuantization {
        
        // Setup the model
        VectorQuantization {
            num_prototypes: num_prototypes,
            learning_rate: learning_rate,
            max_epochs: max_epochs, 
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

        // Assert that there is enough data
        assert!(self.num_prototypes >= 2, "The prototype amount needs to exceed 1.");

        // Assert that there is enough data
        assert!(data.len() as u32 > self.num_prototypes, 
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
    fn setup(&mut self, data : &Vec<Array1<f64>>) {

        // Setup the prototypes by grabbing `num_prototypes` vectors from the data
        for index in 0..self.num_prototypes {

            // Obtain a random prototype and clone/own it
            let selected_prototype = data.choose(&mut self.rng).unwrap();
            let selected_prototype = selected_prototype.clone();
            let selected_prototype = Prototype::new(selected_prototype, index.to_string());

            // Add the newly created prototypes to the prototype list
            self.prototypes.push(selected_prototype);
        }

    }

    /// Fits the Vector Quantization model on the given data
    /// 
    /// # Arguments
    /// 
    /// * `data` The data to adapt the prototypes on
    /// 
    pub fn fit (&mut self, data : &Vec<Array1<f64>>) {

        // Check if the required constraints are present
        self.check_fit_constraints(&data);

        // Perform the required setup:
        // Initialize the prototypes
        self.setup(&data);
        
        // Create a copy of the data, so we do not change the underlying data
        let mut cloned_data = data.clone();

        for _epoch in 1 .. self.max_epochs + 1 {

            // Shuffle the data to prevent artifacts during training
            cloned_data.shuffle(&mut self.rng);

            for data_sample in cloned_data.iter() {

                // Find the closest prototype to the data point
                let closest_prototype_index = find_closest_prototype(&self.prototypes, &data_sample, None);
                let closest_prototype       = self.prototypes.get(closest_prototype_index).unwrap(); 

                // Compute the new prototype
                let new_prototype = closest_prototype.vector.clone() 
                                    + self.learning_rate *
                                    (data_sample - closest_prototype.vector.clone());

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
