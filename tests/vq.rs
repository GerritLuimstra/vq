use ndarray::Array1;
use vq::VectorQuantization;

#[test]
fn simple_clustering_vq() {

    // Setup
    let seed = Some(42);
    let data = vec![
        Array1::<f64>::from_vec(vec![ 1.0, 2.0]),
        Array1::<f64>::from_vec(vec![ 1.0, 3.0]),
        Array1::<f64>::from_vec(vec![-1.0, 2.0]),
        Array1::<f64>::from_vec(vec![-1.0, 3.0]),
    ];

    // Create the model
    let mut model = VectorQuantization::new(
        2,      // prototypes
        0.001,  // learning rate
        2,      // max epochs
        seed    // seed
    );

    // Perform the clustering
    model.fit(&data);
}