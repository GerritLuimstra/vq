use ndarray::Array1;
use std::collections::BTreeMap;
use rand_chacha::ChaChaRng;

mod helpers;
mod prototype;

// Link the required modules
#[path = "vq/vq.rs"]
mod vq;
#[path = "lvq/lvq.rs"]
mod lvq;
#[path = "glvq/glvq.rs"]
mod glvq;

/// This Prototype struct is syntactic sugar that wraps a vector and a name
/// 
/// # Properties
/// * `vector`    The vector data of the prototype
/// * `name`      The name of the vector (for readability only)
/// 
#[derive(Debug)]
pub struct Prototype {
    pub vector: Array1<f64>,
    pub name: String
}

/// The Vector Quantization model
/// 
/// This struct and its methods allow the modeling of probability density functions of a given data set by the
/// distribution of prototype vectors using stochastic gradient descent.
/// It works by dividing a large set of points (vectors) into groups having approximately the same number of points closest to them. 
/// Each group is represented by its centroid point, in this case a prototype vector.
/// 
/// For more information on vector quantization:
/// [Wikipedia](https://en.wikipedia.org/wiki/Vector_quantization)
/// 
/// # Properties
/// * `num_prototypes` The amount of prototypes to use for the clustering
/// * `learning_rate`  The learning rate for the update step of the prototypes
/// * `max_epochs`     The amount of epochs to run
/// * `prototypes`     A vector of the prototypes (initially empty)
/// 
#[derive(Debug)]
pub struct VectorQuantization {
    num_prototypes : u32,
    learning_rate : f64,
    max_epochs : u32,
    seed : Option<u32>, // TODO: Implement

    prototypes : Vec<Prototype>
}

/// The Learning Vector Quantization model
/// 
/// This struct and its methods provide an implementation of the LVQ model using stochastic gradient descent.
/// 
/// An LVQ system is represented by prototypes which are defined in the feature space of observed data. 
/// In a Hebbian-like winner-take-all fashion the algorithm determines
/// the prototype which is closest to a data point according to a given distance measure.
/// The position of this so-called winner prototype is then adapted, i.e. the winner is moved closer if it correctly classifies the data point
/// or moved away if it classifies the data point incorrectly.
/// 
/// For more information on learning vector quantization:
/// [Wikipedia](https://en.wikipedia.org/wiki/Learning_vector_quantization)
/// 
/// This specific implementation allows for a variable number of prototypes per class.
/// 
/// # Properties
/// * `num_prototypes` The amount of prototypes to use per class (a BTreeMap, that maps the class name to the number of prototypes to use)
/// * `learning_rate`  The learning rate for the update step of the prototypes
/// * `max_epochs`     The amount of epochs to run
/// * `prototypes`     A vector of the prototypes (initially empty)
/// * `rng`            The internal ChaChaRng to be used for reproducability.
/// 
#[derive(Debug)]
pub struct LearningVectorQuantization {
    num_prototypes : BTreeMap<String, usize>,
    learning_rate : f64,
    max_epochs : u32, 
    rng : ChaChaRng,
    prototypes : Vec<Prototype>
}

/// The General Learning Vector Quantization (GLVQ) model
///
/// This struct and its methods provide an implementation of the generalization of the LVQ model using stochastic gradient descent,
/// in which reference vectors are updated based on the steepest descent method in order to minimize the cost function.
/// In the paper, the cost function is determined so that the obtained learning rule satisfies the convergence condition.
///
/// The implementation is heavily based on the following paper by Atsushi Sato & Keiji Yamada [[1]](https://papers.nips.cc/paper/1995/file/9c3b1830513cc3b8fc4b76635d32e692-Paper.pdf).
///
/// This specific implementation allows for a variable number of prototypes per class.
///
/// **NOTE**: Currently the distance metric is restricted to Euclidean distance only!
///
/// # Properties
/// * `num_prototypes` The amount of prototypes to use per class (a BTreeMap, that maps the class name to the number of prototypes to use)
/// * `learning_rate`  The learning rate for the update step of the prototypes
/// * `max_epochs`     The amount of epochs to run
/// * `prototypes`     A vector of the prototypes (initially empty)
/// * `rng`            The internal ChaChaRng to be used for reproducability.
///
#[derive(Debug)]
pub struct GeneralLearningVectorQuantization {
    num_prototypes : BTreeMap<String, usize>,
    learning_rate : f64,
    max_epochs : u32, 
    rng : ChaChaRng,
    prototypes : Vec<Prototype>
}