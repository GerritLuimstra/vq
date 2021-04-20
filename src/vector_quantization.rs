use super::Prototype;
use super::VectorQuantization;
use super::helpers::euclidean_distance;

use ndarray::Array1;
use rand::prelude::thread_rng;
use rand::seq::SliceRandom;

impl VectorQuantization {

    /// Constructs a new Vector Quantization model
    /// 
    /// # Arguments
    /// 
    /// * `vec1` The first vector
    /// * `vec2` The second vector
    /// 
    pub fn new ( num_prototypes: u32, 
                    learning_rate: f64,
                    max_epochs: u32, 
                    seed: Option<u32> ) -> VectorQuantization {
        
        // Setup the model
        VectorQuantization {
            num_prototypes: num_prototypes,
            learning_rate: learning_rate,
            max_epochs: max_epochs, 
            seed: seed,
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

        // Assert that there is enough data
        assert!(data.len() as u32 > self.num_prototypes, 
                "There are more prototypes than data samples. Consider lowering the amount of prototypes.");

        // Assert that the model has not been fit yet
        assert!(self.prototypes.len() == 0, "This model has already been fit.");

        // Setup the prototypes by grabbing `num_prototypes` vectors from the data
        for index in 0..self.num_prototypes {

            // Obtain a random prototype and clone/own it
            let selected_prototype = data.choose(&mut rand::thread_rng()).unwrap();
            let selected_prototype = selected_prototype.clone();
            let selected_prototype = Prototype::new(selected_prototype, index.to_string());

            // Add the newly created prototypes to the prototype list
            self.prototypes.push(selected_prototype);
        }
        
        // Create a copy of the data, so we do not change the underlying data
        let mut cloned_data = data.clone();

        for _epoch in 1 .. self.max_epochs + 1 {

            // Shuffle the data to prevent artifacts during training
            cloned_data.shuffle(&mut thread_rng());

            for data_sample in cloned_data.iter() {

                // Find the closest prototype to the data point
                let closest_prototype_index = self.find_closest_prototype(&data_sample);
                let closest_prototype       = self.prototypes.get(closest_prototype_index).unwrap(); 

                // Compute the new prototype
                let new_prototype = closest_prototype.vector.clone() + self.learning_rate * (data_sample - closest_prototype.vector.clone());

                // Replace the old prototype
                self.prototypes[closest_prototype_index].vector = new_prototype;
            }
        }
    }
    
    // TODO: Implement properly
    pub fn predict(&self, data : &Vec<Vec<f64>>) 
    {
        // Check for valid input
        assert!(data.len() > 0, "There are no data samples given.");
        assert!(self.prototypes.len() > 0, "The model has not been fit yet.");

        
    }
}