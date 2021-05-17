use ndarray::{Array1, Array2};
use super::Prototype;

/// Simple euclidean distance between two vectors
/// 
/// # Arguments
/// 
/// * `vec1` The first vector
/// * `vec2` The second vector
/// 
pub fn euclidean_distance (vec1 : &Array1<f64>, vec2 : &Array1<f64>) -> f64 {

    // Assert that the vectors are of equal size
    assert!(vec1.len() == vec2.len());

    // Compute the sum of the differences
    let mut sum = 0.0;
    for index in 0 .. vec1.len() {
        sum += (vec1[index] - vec2[index]).powi(2);
    }

    // Take the square root
    sum.sqrt()
}

/// Generalized distance between two vectors based on
/// the adaptive distance metric Lambda = Omega^T Omega
/// 
/// # Arguments
/// 
/// * `omega` The adaptive distance matix Omega
/// * `vec1`  The first vector
/// * `vec2`  The second vector
///
pub fn generalized_distance(omega: &Array2<f64>, vec1 : &Array1<f64>, vec2 : &Array1<f64>) -> f64 {

    // Obtain the Lambda = Omega^T Omega matrix
    let lambda = omega.t().dot(&omega.to_owned());

    // Obtain the vector difference
    let difference = vec1 - vec2;

    // Compute the distance
    difference.t().dot(&lambda).dot(&difference)
}


/// Obtains the closest prototype index for a given sample
/// 
/// # Arguments
/// 
/// * `prototypes` The prototypes
/// * `sample`     The sample to find the closest prototype for
/// 
pub fn find_closest_prototype (prototypes : &Vec<Prototype>, sample : &Array1<f64>, omega : Option<&Array2<f64>>) -> usize {

    // Initialize values
    let mut closest_prototype_index = 0 as usize;
    let mut smallest_distance       = f64::INFINITY;

    for (index, prototype) in prototypes.iter().enumerate() {

        // Compute the distance based on whether an adaptive matrix is given
        let distance = match omega {
            None        => euclidean_distance(&prototype.vector, sample),
            Some(omega) => generalized_distance(omega, sample, &prototype.vector)
        };

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
/// * `prototypes` The prototypes
/// * `sample` The sample to find the closest prototype for
/// * `label`  The label of the sample
/// * `find_closest_matching` Determines whether the closest matching 
/// or non-matching prototype is to be found.
/// 
pub fn find_closest_prototype_matched (prototypes: &Vec<Prototype>, 
                                    sample : &Array1<f64>, 
                                    label: &String,
                                    find_closest_matching: bool,
                                    omega : Option<&Array2<f64>>) -> usize {
    
    // Initialize values
    let mut closest_prototype_index = 0 as usize;
    let mut smallest_distance       = f64::INFINITY;

    for (index, prototype) in prototypes.iter().enumerate() {

        // Compute the distance based on whether an adaptive matrix is given
        let distance = match omega {
            None        => euclidean_distance(&prototype.vector, sample),
            Some(omega) => generalized_distance(omega, sample, &prototype.vector)
        };

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