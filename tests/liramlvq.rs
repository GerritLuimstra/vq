use ndarray::Array1;
use vq::LiRaMLVQ;
use approx::*;
use std::collections::BTreeMap;

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
#[should_panic(expected = "The max rank needs to be bigger than 0!")]
fn check_constraints_max_rank() {

    // Create a class - prototype mapping for variable length prototypes
    let mut prototype_mapping = BTreeMap::new();
    prototype_mapping.insert(String::from("C0"), 1);
    prototype_mapping.insert(String::from("C1"), 1);

    // Setup the data
    let seed = Some(42);
    let data = vec![
        Array1::<f64>::from_vec(vec![ 5.0, 2.0]),
        Array1::<f64>::from_vec(vec![ 5.0, 3.0])
    ];
    let labels = vec![
        "C0".to_string(),
        "C1".to_string()
    ];

    // Create the model
    let mut model = LiRaMLVQ::new(
        0,                      // invalid max rank
        prototype_mapping,      // prototypes
        (0.1, 0.01),            // learning rate
        1,                      // max epochs
        seed                    // seed
    );

    // Perform the classification
    // Will panic since max rank is not positive
    model.fit(&data, &labels);
}

#[test]
#[should_panic(expected = "Each class needs to have at least one prototype!")]
fn check_constraints_prototypes() {

    // Create a class - prototype mapping for variable length prototypes
    let mut prototype_mapping = BTreeMap::new();
    prototype_mapping.insert(String::from("C0"), 0);
    prototype_mapping.insert(String::from("C1"), 0);

    // Setup the data
    let seed = Some(42);
    let data = vec![
        Array1::<f64>::from_vec(vec![ 5.0, 2.0]),
        Array1::<f64>::from_vec(vec![ 5.0, 3.0])
    ];
    let labels = vec![
        "C0".to_string(),
        "C1".to_string()
    ];

    // Create the model (with just 1 prototype)
    // This should panic
    let mut model = LiRaMLVQ::new(
        2,                 // max rank
        prototype_mapping, // prototypes
        (0.1, 0.01),       // learning rate
        1,                 // max epochs
        seed               // seed
    );

    // Perform the classification
    // This will panic, since there are not enough prototypes
    // and not each class has a prototype
    model.fit(&data, &labels);
}

#[test]
#[should_panic(expected = "There are more prototypes than data samples. Consider lowering the amount of prototypes.")]
fn check_constraints_enough_data() {

    // Create a class - prototype mapping for variable length prototypes
    let mut prototype_mapping = BTreeMap::new();
    prototype_mapping.insert(String::from("C0"), 1);
    prototype_mapping.insert(String::from("C1"), 1);

    // Setup the data (too little)
    let seed = Some(42);
    let data = vec![
        Array1::<f64>::from_vec(vec![ 5.0, 2.0])
    ];
    let labels = vec![
        "C0".to_string()
    ];

    // Create the model
    let mut model = LiRaMLVQ::new(
        2,                      // max rank
        prototype_mapping,      // prototypes
        (0.1, 0.01),            // learning rate
        1,                      // max epochs
        seed                    // seed
    );

    // This panics, since there is not enough data
    model.fit(&data, &labels);
}

#[test]
#[should_panic(expected = "Unknown label C2. Consider adding it to the prototype mapping.")]
fn check_constraints_unknown_label() {

    // Create a class - prototype mapping for variable length prototypes
    let mut prototype_mapping = BTreeMap::new();
    prototype_mapping.insert(String::from("C0"), 1);
    prototype_mapping.insert(String::from("C1"), 1);

    // Setup the data (too little)
    let seed = Some(42);
    let data = vec![
        Array1::<f64>::from_vec(vec![ 5.0, 2.0]),
        Array1::<f64>::from_vec(vec![ 5.0, 2.0])
    ];
    let labels = vec![
        "C0".to_string(),
        "C2".to_string() // unknown label
    ];

    // Create the model
    let mut model = LiRaMLVQ::new(
        2,                      // max rank
        prototype_mapping,      // prototypes
        (0.1, 0.01),            // learning rate
        1,                      // max epochs
        seed                    // seed
    );

    // This panics, since there is an unknown label C2,
    // that is not present in the prototype mapping
    model.fit(&data, &labels);
}

#[test]
#[should_panic(expected = "This model has already been fit.")]
fn check_constraints_fitting_again() {

    // Create a class - prototype mapping for variable length prototypes
    let mut prototype_mapping = BTreeMap::new();
    prototype_mapping.insert(String::from("C0"), 1);
    prototype_mapping.insert(String::from("C1"), 1);

    // Setup the data
    let seed = Some(42);
    let data = vec![
        Array1::<f64>::from_vec(vec![ 5.0, 2.0]),
        Array1::<f64>::from_vec(vec![ 5.0, 3.0])
    ];
    let labels = vec![
        "C0".to_string(),
        "C1".to_string()
    ];

    // Create the model
    let mut model = LiRaMLVQ::new(
        2,                      // max rank
        prototype_mapping,      // prototypes
        (0.1, 0.01),            // learning rate
        1,                      // max epochs
        seed                    // seed
    );

    // Perform the classification
    model.fit(&data, &labels);

    // This should panic. The model is already fit.
    model.fit(&data, &labels);
}

#[test]
#[should_panic(expected = "There are no data samples given.")]
fn check_constraints_predict_no_data() {

    // Create a class - prototype mapping for variable length prototypes
    let mut prototype_mapping = BTreeMap::new();
    prototype_mapping.insert(String::from("C0"), 1);
    prototype_mapping.insert(String::from("C1"), 1);

    // Setup the data
    let seed = Some(42);
    let data = vec![
        Array1::<f64>::from_vec(vec![ 5.0, 2.0]),
        Array1::<f64>::from_vec(vec![ 5.0, 3.0])
    ];
    let labels = vec![
        "C0".to_string(),
        "C1".to_string()
    ];

    // Create the model
    let mut model = LiRaMLVQ::new(
        2,                  // max rank
        prototype_mapping,  // prototypes
        (0.1, 0.01),        // learning rate
        1,                  // max epochs
        seed                // seed
    );

    // Perform the classification
    model.fit(&data, &labels);

    // This should panic. There is no data available.
    model.predict(&vec![]);
}

#[test]
#[should_panic(expected = "The data vector does not match the label vector in length.")]
fn check_constraints_len_data_not_eq_labels() {

    // Create a class - prototype mapping for variable length prototypes
    let mut prototype_mapping = BTreeMap::new();
    prototype_mapping.insert(String::from("C0"), 1);
    prototype_mapping.insert(String::from("C1"), 1);

    // Setup the data
    let seed = Some(42);
    let data = vec![
        Array1::<f64>::from_vec(vec![ 5.0, 2.0]),
        Array1::<f64>::from_vec(vec![ 5.0, 3.0])
    ];
    let labels = vec![
        "C0".to_string()
    ];

    // Create the model
    let mut model = LiRaMLVQ::new(
        2,                  // max rank
        prototype_mapping,  // prototypes
        (0.1, 0.01),        // learning rate
        1,                  // max epochs
        seed                // seed
    );

    // Perform the classification
    model.fit(&data, &labels);

    // This should panic.
    // The length of the data is not the same as the labels
    model.predict(&data);
}

#[test]
#[should_panic(expected = "The model has not been fit yet.")]
fn check_constraints_predict_not_fit() {

    // Create a class - prototype mapping for variable length prototypes
    let mut prototype_mapping = BTreeMap::new();
    prototype_mapping.insert(String::from("C0"), 1);
    prototype_mapping.insert(String::from("C1"), 1);

    // Setup the data
    let seed = Some(42);

    // Create the model
    let mut model = LiRaMLVQ::new(
        2,                  // max rank
        prototype_mapping,  // prototypes
        (0.1, 0.01),        // learning rate
        1,                  // max epochs
        seed                // seed
    );

    // This should panic. The model is not fit yet!
    model.predict(&vec![
        Array1::<f64>::from_vec(vec![ 5.0, 2.0])
    ]);
}

#[test]
#[should_panic(expected = "Data must have the same dimensions as was used in fit!")]
fn check_constraints_predict_not_same_dim() {

    // Create a class - prototype mapping for variable length prototypes
    let mut prototype_mapping = BTreeMap::new();
    prototype_mapping.insert(String::from("C0"), 1);
    prototype_mapping.insert(String::from("C1"), 1);

    // Setup the data
    let seed = Some(42);
    let data = vec![
        Array1::<f64>::from_vec(vec![ 5.0, 2.0]),
        Array1::<f64>::from_vec(vec![ 5.0, 3.0])
    ];
    let labels = vec![
        "C0".to_string(),
        "C1".to_string()
    ];

    // Create the model
    let mut model = LiRaMLVQ::new(
        2,                  // max rank
        prototype_mapping,  // prototypes
        (0.1, 0.01),        // learning rate
        1,                  // max epochs
        seed                // seed
    );

    // Perform the classification
    model.fit(&data, &labels);

    // This should panic. A 3D vector is given this time
    model.predict(&vec![
        Array1::<f64>::from_vec(vec![ 5.0, 2.0, 5.0 ])
    ]);
}

#[test]
fn simple_classification_liramlvq() {

    // Create a class - prototype mapping for variable length prototypes
    let mut prototype_mapping = BTreeMap::new();
    prototype_mapping.insert(String::from("C0"), 1);
    prototype_mapping.insert(String::from("C1"), 1);

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
    let labels = vec![
        "C0".to_string(),
        "C0".to_string(),
        "C0".to_string(),
        "C1".to_string(),
        "C1".to_string(),
        "C1".to_string()
    ];

    // Create the model
    let mut model = LiRaMLVQ::new(
        2,                  // max rank
        prototype_mapping,  // prototypes
        (0.1, 0.01),        // learning rate
        100,                // max epochs
        seed                // seed
    );

    // Perform the classification
    model.fit(&data, &labels);

    // Obtain the predictions
    let predictions = model.predict(&data);

    // Assert that the predictions are correct
    assert_eq!(predictions, vec!["C0", "C0", "C0", "C1", "C1", "C1"]);
}