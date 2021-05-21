use ndarray::Array1;
use vq::VQ;
use approx::*;

fn vec_all_close(a : &Vec<f64>, b : &Vec<f64>, tolerance: f64) -> bool {
    if a.len() != b.len() { return false; }
    for (index, el_a) in a.iter().enumerate() {
        if !abs_diff_eq!(el_a.clone(), b[index].clone(), epsilon = tolerance) {
            return false;
        }
    }
    return true;
}

#[test]
#[should_panic(expected = "The prototype amount needs to exceed 1.")]
fn check_constraints_prototypes() {

    // Setup the data
    let seed = Some(42);
    let data = vec![
        Array1::<f64>::from_vec(vec![ 5.0, 2.0]),
        Array1::<f64>::from_vec(vec![ 5.0, 3.0])
    ];

    // Create the model (with just 1 prototype)
    // This should panic
    let mut model = VQ::new(
        1,      // prototypes
        0.1,    // learning rate
        1,      // max epochs
        seed    // seed
    );

    // Perform the clustering
    model.fit(&data);
}

#[test]
#[should_panic(expected = "There are more prototypes than data samples. Consider lowering the amount of prototypes.")]
fn check_constraints_enough_data() {

    // Setup the data (too little)
    let seed = Some(42);
    let data = vec![
        Array1::<f64>::from_vec(vec![ 5.0, 2.0])
    ];

    // Create the model
    let mut model = VQ::new(
        2,      // prototypes
        0.1,    // learning rate
        1,      // max epochs
        seed    // seed
    );

    // Perform the clustering
    // This panics, since there is not enough data
    model.fit(&data);
}

#[test]
#[should_panic(expected = "This model has already been fit.")]
fn check_constraints_fitting_again() {

    // Setup the data
    let seed = Some(42);
    let data = vec![
        Array1::<f64>::from_vec(vec![ 5.0, 2.0]),
        Array1::<f64>::from_vec(vec![ 5.0, 3.0])
    ];

    // Create the model
    let mut model = VQ::new(
        2,      // prototypes
        0.1,    // learning rate
        1,      // max epochs
        seed    // seed
    );

    // Perform the clustering
    model.fit(&data);

    // This should panic. The model is already fit.
    model.fit(&data);
}

#[test]
#[should_panic(expected = "There are no data samples given.")]
fn check_constraints_predict_no_data() {

    // Setup the data
    let seed = Some(42);
    let data = vec![
        Array1::<f64>::from_vec(vec![ 5.0, 2.0]),
        Array1::<f64>::from_vec(vec![ 5.0, 3.0])
    ];

    // Create the model
    let mut model = VQ::new(
        2,      // prototypes
        0.1,    // learning rate
        1,      // max epochs
        seed    // seed
    );

    // Perform the clustering
    model.fit(&data);

    // This should panic. There is no data available.
    model.predict(&vec![]);
}

#[test]
#[should_panic(expected = "The model has not been fit yet.")]
fn check_constraints_predict_not_fit() {

    // Setup the data
    let seed = Some(42);

    // Create the model
    let mut model = VQ::new(
        2,      // prototypes
        0.1,    // learning rate
        1,      // max epochs
        seed    // seed
    );

    // This should panic. The model is not fit yet!
    model.predict(&vec![
        Array1::<f64>::from_vec(vec![ 5.0, 2.0])
    ]);
}

#[test]
#[should_panic(expected = "Data must have the same dimensions as was used in fit!")]
fn check_constraints_predict_not_same_dim() {

    // Setup the data
    let seed = Some(42);
    let data = vec![
        Array1::<f64>::from_vec(vec![ 5.0, 2.0]),
        Array1::<f64>::from_vec(vec![ 5.0, 3.0])
    ];

    // Create the model
    let mut model = VQ::new(
        2,      // prototypes
        0.1,    // learning rate
        1,      // max epochs
        seed    // seed
    );

    // Perform the clustering
    model.fit(&data);

    // This should panic. A 3D vector is given this time
    model.predict(&vec![
        Array1::<f64>::from_vec(vec![ 5.0, 2.0, 5.0 ])
    ]);
}

#[test]
fn simple_clustering_vq() {

    // Setup the data
    let seed = Some(42);
    let data = vec![
        Array1::<f64>::from_vec(vec![ 5.0, 2.0]),
        Array1::<f64>::from_vec(vec![ 5.0, 3.0]),
        Array1::<f64>::from_vec(vec![ 5.0, 4.0]),
        Array1::<f64>::from_vec(vec![-5.0, 10.0]),
        Array1::<f64>::from_vec(vec![-5.0, 11.0]),
        Array1::<f64>::from_vec(vec![-5.0, 12.0]),
    ];

    // Create the model
    let mut model = VQ::new(
        2,      // prototypes
        0.1,    // learning rate
        100,    // max epochs
        seed    // seed
    );

    // Perform the clustering
    model.fit(&data);

    // Obtain the predictions
    let predictions = model.predict(&data);

    // Assert that the cluster predictions are correct
    assert_eq!(predictions, vec!["1", "1", "1", "0", "0", "0"]);

    // Obtain the prototype information
    let prototypes  = model.prototypes();
    let prototype_1 = prototypes[0].vector.to_vec(); 
    let prototype_2 = prototypes[1].vector.to_vec();

    // Assert that the prototypes are roughly at the cluster centers
    assert!(vec_all_close(&prototype_1, &vec![-5.0, 11.0], 1e-1));
    assert!(vec_all_close(&prototype_2, &vec![5.0, 3.0], 1e-1));
}

#[test]
fn renaming_prototypes() {

    // Setup the data
    let seed = Some(42);
    let data = vec![
        Array1::<f64>::from_vec(vec![ 5.0, 2.0]),
        Array1::<f64>::from_vec(vec![ 5.0, 3.0])
    ];

    // Create the model
    let mut model = VQ::new(
        2,      // prototypes
        0.1,    // learning rate
        100,    // max epochs
        seed    // seed
    );

    // Perform the clustering
    model.fit(&data);

    // Obtain the predictions
    let predictions = model.predict(&data);

    // Verify that the predictions are correct
    assert_eq!(predictions, vec!["0", "1"]);

    // Rename the prototypes
    model.name_prototypes(
        &vec![
            "class 0".to_string(), 
            "class 1".to_string()
        ]
    );

    // Verify that the cluster labels are now changed.
    assert_eq!(model.predict(&data), vec!["class 0", "class 1"]);
}