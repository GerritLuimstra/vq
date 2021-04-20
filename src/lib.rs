use ndarray::Array1;

mod prototype;
mod vector_quantization;
mod helpers;

/// This Prototype struct is syntactic sugar that wraps a vector and a name
/// 
/// # Properties
/// `vector`    The vector data of the prototype
/// `name`      The name of the vector (for readability only)
#[derive(Debug)]
struct Prototype {
    vector: Array1<f64>,
    name: String
}

/// The Vector Quantization model
/// 
/// This struct and its methods allow the modeling of probability density functions of a given data set by the distribution of prototype vectors. 
/// It works by dividing a large set of points (vectors) into groups having approximately the same number of points closest to them. 
/// Each group is represented by its centroid point, in this case a prototype vector
/// 
/// # Properties
/// `num_prototypes` The amount of prototypes to use for the clustering
/// `learning_rate`  The learning rate for the update step of the prototypes
/// `max_epochs`     The amount of epochs to run
#[derive(Debug)]
pub struct VectorQuantization {
    num_prototypes : u32,
    learning_rate : f64,
    max_epochs : u32, 
    seed : Option<u32>, // TODO: Implement

    prototypes : Vec<Prototype>
}