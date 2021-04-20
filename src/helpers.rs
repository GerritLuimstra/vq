use ndarray::Array1;

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
        sum += (vec1[index] - vec2[index]).powf(2.0);
    }

    // Take the square root
    sum.sqrt()
}