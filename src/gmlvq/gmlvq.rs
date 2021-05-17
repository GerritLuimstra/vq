use super::Prototype;
use super::GeneralMatrixLearningVectorQuantization;
use super::traits::TupledSchedulable;
use super::helpers::find_closest_prototype_matched;
use super::helpers::{generalized_distance, find_closest_prototype};

use rand::Rng;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::Array;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaChaRng;
use std::collections::BTreeMap;

impl GeneralMatrixLearningVectorQuantization {

    /// Constructs a new General Matrix Learning Vector Quantization model
    /// 
    /// # Arguments
    /// 
    /// * `num_prototypes` The amount of prototypes to use per class (a BTreeMap, that maps the class name to the number of prototypes to use)
    ///                    This BTreeMap should be provided as a reference and the algorithm will panic if there are classes
    ///                    in the data not present in this BTreeMap.
    ///                    A BTreeMap is used instead of a HashMap due to the ability of sorted keys, which is required for reproducability.
    /// * `initial_lr`     The initial learning rate to be used by the learning rate scheduler
    ///                    Note: This time, we require two learning rates (one for the prototypes and one for the matrix) as a tuple
    /// * `max_epochs`     The amount of epochs to run
    /// * `prototypes`     A vector of the prototypes (initially empty)
    /// * `seed`           The seed to be used by the internal ChaChaRng.
    /// 
    pub fn new ( num_prototypes: BTreeMap<String, usize>,
                 initial_lr : (f64, f64),
                 max_epochs: u32,
                 seed: Option<u64> ) -> GeneralMatrixLearningVectorQuantization {
        
        // Setup the model
        GeneralMatrixLearningVectorQuantization {
            num_prototypes,
            omega: None,
            initial_lr,
            lr_scheduler : |l_p, l_m, _, _| -> (f64, f64) { (l_p, l_m) },
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
    /// Normalizes the matrix omega such that omega^T omega contains to have a diagonal sum of 1.
    /// This prevents learning degeneration.
    /// 
    fn normalize_omega(&self, omega: &Array2<f64>) -> Array2<f64> {

        // Compute Omega^T Omega
        let combined = omega.t().dot(omega);

        // Compute the sum of the diagonal
        let diagonal_sum = combined.diag().sum();

        // Normalize omega by dividing the diagonal by sqrt(`diagonal_sum`)
        omega / diagonal_sum.sqrt()
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

        // Setup the n by n adaptive metric matrix Omega
        // and normalize omega such that Omega^T Omega has diagonal elements that sum to one
        let n = self.prototypes[0].vector.dim();
        self.omega = Some(self.normalize_omega(&Array::eye(n)));
    }

    /// Fits the General Matrix Learning Vector Quantization model on the given data
    /// 
    /// # Arguments
    /// 
    /// * `data`   The data to adapt the prototypes and the learned adaptive distance metric on
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

                // Compute Lambda = Omega^T Omega
                let omega : &Array2<f64> = self.omega.as_ref().unwrap();
                let lambda = omega.t().dot(&omega.to_owned());

                // Find the indices of w_J and w_K, which are the closest matching and closest non-matching prototype respectively
                let w_j_index = find_closest_prototype_matched(
                    &self.prototypes, &data_sample,
                    &data_label, true, Some(omega)
                );
                let w_k_index = find_closest_prototype_matched(
                    &self.prototypes, &data_sample, 
                    &data_label, false, Some(omega)
                );
                
                // From the indices, obtain the corresponding prototypes
                let w_j = self.prototypes.get(w_j_index).unwrap();
                let w_k = self.prototypes.get(w_k_index).unwrap();

                // Compute the distances to the closest correct and wrong prototype
                let d_j = generalized_distance(omega, data_sample, &w_j.vector);
                let d_k = generalized_distance(omega, data_sample, &w_k.vector);

                // Compute mu_plus and mu_minus (the derivative of mu with respect to the closest and furthers prototype distance)
                let norm = (d_k + d_j).powi(2);
                let mu_plus  = 2.0 * d_k / norm;
                let mu_minus = 2.0 * d_j / norm;

                // TODO: Replace the 1.0 with a general / sigmoid function
                let deriv_w_j = 2.0 * 1.0 * mu_plus  * lambda.dot(&(data_sample - w_j.vector.to_owned()));
                let deriv_w_k = 2.0 * 1.0 * mu_minus * lambda.dot(&(data_sample - w_k.vector.to_owned()));
                
                // Compute the differences with the samples and the corresponding vectors
                // and their versions dotted with Omega
                let diff_j = data_sample - w_j.vector.to_owned();
                let diff_k = data_sample - w_k.vector.to_owned();
                let omega_diff_j = omega.dot(&diff_j);
                let omega_diff_k = omega.dot(&diff_k);

                // Compute the gradient
                let n = omega.dim().0;
                let mut omega_gradient : Array2<f64> = Array::zeros((n, n));
                for l in 0 .. n {
                    for m in 0 .. n {
                        omega_gradient[[l, m]] = mu_plus * diff_j[m] * omega_diff_j[l] - mu_minus * diff_k[m] * omega_diff_k[l];
                    }
                }

                // TODO: Replace the 1.0 with a general / sigmoid function
                let omega_gradient = - 2.0 * 1.0 * omega_gradient;

                // Compute the learning rates
                let learning_rates = (self.lr_scheduler)(self.initial_lr.0, self.initial_lr.1, epoch, self.max_epochs);

                // Perform the complete update rules
                let new_w_j   = w_j.vector.clone() + learning_rates.0 * deriv_w_j;
                let new_w_k   = w_k.vector.clone() - learning_rates.0 * deriv_w_k;
                let new_omega = omega.clone()      + learning_rates.1 * omega_gradient;
                
                // Update the prototypes
                self.prototypes[w_j_index].vector = new_w_j;
                self.prototypes[w_k_index].vector = new_w_k;

                // Normalize omega such that Omega^T Omega has diagonal elements that sum to one
                // this is to prevent learning degeneration
                self.omega = Some(self.normalize_omega(&new_omega));
            }
        }
    }

    /// Assign cluster labels (i.e. predict) to the data given data
    /// based on the learned prototype vectors and the learned adaptive distance metric
    /// 
    /// # Arguments
    /// 
    /// * `data` The data to obtain the cluster labels for
    /// 
    pub fn predict(&mut self, data : &Vec<Array1<f64>>) -> Vec<String> {

        // Check predict constraints
        self.check_predict_constraints(&data);

        let mut cluster_labels = Vec::<String>::new();

        for data_sample in data.iter() {

            // TODO: Make this also work with the custom functions.

            // Obtain the closest prototype
            let closest_prototype_index = find_closest_prototype(&self.prototypes, &data_sample, Some(self.omega.as_ref().unwrap()));
            let closest_prototype       = self.prototypes.get(closest_prototype_index).unwrap();

            // Add the cluster label to the list
            cluster_labels.push(closest_prototype.name.clone());
        }

        cluster_labels
    }



    /// Simple getter for the prototype clusters
    /// 
    /// NOTE: This projects the prototypes using Lambda!
    /// 
    pub fn prototypes(&self) -> Vec<Prototype> {

        assert!(self.prototypes.len() > 0, 
        "The model has not been fit yet. \n
        There are no prototypes at this stage.");

        // Compute Lambda = Omega^T Omega
        let omega : &Array2<f64> = self.omega.as_ref().unwrap();
        let lambda = omega.t().dot(&omega.to_owned());

        // Setup the new project samples
        let mut projected_prototypes = Vec::<Prototype>::new();

        for prototype in self.prototypes.iter() {

            // Clone the prototype
            let mut prototype = prototype.clone();

            // Project the prototype using Lambda
            prototype.vector = lambda.dot(&prototype.vector);

            projected_prototypes.push(prototype);
        }

        projected_prototypes
    }

    /// Simple getter for the Omega matrix
    pub fn omega(&self) -> &Array2<f64> {

        assert!(self.prototypes.len() > 0, 
        "The model has not been fit yet. \n
        Omega is not available yet at this stage.");

        &self.omega.as_ref().unwrap()
    }

    /// Simple getter for the Lambda matrix
    pub fn lambda(&self) -> Array2<f64> {

        assert!(self.prototypes.len() > 0, 
        "The model has not been fit yet. \n
        Omega is not available yet at this stage.");

        // Clone omega, so that we can return a copy
        let omega = self.omega.clone().unwrap();

        omega.dot(&omega)
    }

    /// Projects the data based on learned Lambda = Omega^T Omega matrix
    /// 
    /// # Arguments
    /// 
    /// * `data` The data to project according to the learned matrix
    /// 
    pub fn project(&self, data : &Vec<Array1<f64>>) -> Vec::<Array1<f64>> {

        // Compute Lambda = Omega^T Omega
        let omega : &Array2<f64> = self.omega.as_ref().unwrap();
        let lambda = omega.t().dot(&omega.to_owned());

        // Setup the new project samples
        let mut projected_samples = Vec::<Array1<f64>>::new();

        for data_sample in data.iter() {

            // Project the data using Lambda
            let projected_sample : Array1<f64> = lambda.dot(&data_sample.clone());

            projected_samples.push(projected_sample);
        }

        projected_samples
    }

}

impl TupledSchedulable for GeneralMatrixLearningVectorQuantization {

    /// Updates the learning rate scheduler of the model this trait is implemented for
    /// 
    /// # Arguments
    /// 
    /// * `scheduler` A function that takes in four arguments of the form
    ///               (base_learning_rate_protototype, base_learning_rate_matrix, current_epoch, max_epochs) as parameters
    ///               The algorithm will insert these arguments when the learning rate is to be calculated.
    ///
    fn set_learning_rate_scheduler (&mut self, scheduler : fn(f64, f64, u32, u32) -> (f64, f64)) {
        self.lr_scheduler = scheduler;
    }

}