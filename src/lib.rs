// Link the required modules
#[path = "vq/vq.rs"]
mod vq;
#[path = "lvq/lvq.rs"]
mod lvq;
#[path = "glvq/glvq.rs"]
mod glvq;
#[path = "gmlvq/gmlvq.rs"]
mod gmlvq;
#[path = "liramlvq/liramlvq.rs"]
mod liramlvq;
#[path = "traits/traits.rs"]
pub mod traits;

use ndarray::{Array1, Array2};
use std::collections::BTreeMap;
use rand_chacha::ChaChaRng;

mod helpers;
mod prototype;

/// This Prototype struct is syntactic sugar that wraps a vector and a name
/// 
/// # Properties
/// * `vector`    The vector data of the prototype
/// * `name`      The name of the vector (for readability only)
/// 
#[derive(Debug, Clone)]
pub struct Prototype {
    pub vector: Array1<f64>,
    pub name: String
}

/// This struct allows a custom *monotonic* function to be supplied to certain LVQ algorithms.
/// As an example, for GLVQ this custom function can be supplied to change the learning behaviour.
/// The goal of the function is to give the ability to choose how the distances are weighted.
/// Popular function choices include the identity function and the sigmoid function.
/// 
/// # Properties
/// * `func`    The normal function to be used during the 'forward' stage. 
///             This is used to calculate the distances with the prototypes.
/// * `deriv`   The derivative of the function, to be used during the training stage.
/// 
#[derive(Debug, Clone)]
pub struct CustomMonotonicFunction {
    pub func: fn (distance : f64, epoch : u32) -> f64,
    pub deriv: fn (distance : f64, epoch : u32) -> f64
}

/// The Vector Quantization (VQ) model
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
/// * `prototypes`     A vector of the prototypes (initially empty)
/// * `initial_lr`     The initial learning rate to be used by the learning rate scheduler
/// * `lr_scheduler`   The learning rate scheduler for the update step of the prototypes
///                    This function can be custom and receives: (base_learning_rate, current_epoch, max_epochs) as parameters
///                    The default scheduler simply returns the initial learning rate every time
/// * `max_epochs`     The amount of epochs to run
/// * `rng`            The internal ChaChaRng to be used for reproducability.
/// 
#[derive(Debug)]
pub struct VQ {
    num_prototypes : u32,
    prototypes : Vec<Prototype>,
    initial_lr : f64,
    lr_scheduler : fn(f64, u32, u32) -> f64,
    max_epochs : u32,
    rng : ChaChaRng
}

/// The Learning Vector Quantization (LVQ) model
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
/// * `prototypes`     A vector of the prototypes (initially empty)
/// * `initial_lr`     The initial learning rate to be used by the learning rate scheduler
/// * `lr_scheduler`   The learning rate scheduler for the update step of the prototypes
///                    This function can be custom and receives: (base_learning_rate, current_epoch, max_epochs) as parameters
///                    The default scheduler simply returns the initial learning rate every time
/// * `max_epochs`     The amount of epochs to run
/// * `rng`            The internal ChaChaRng to be used for reproducability.
/// 
#[derive(Debug)]
pub struct LVQ {
    num_prototypes : BTreeMap<String, usize>,
    prototypes : Vec<Prototype>,
    initial_lr : f64,
    lr_scheduler : fn(f64, u32, u32) -> f64,
    max_epochs : u32,
    rng : ChaChaRng
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
/// * `prototypes`     A vector of the prototypes (initially empty)
/// * `initial_lr`     The initial learning rate to be used by the learning rate scheduler
/// * `lr_scheduler`   The learning rate scheduler for the update step of the prototypes
///                    This function can be custom and receives: (base_learning_rate, current_epoch, max_epochs) as parameters
///                    The default scheduler simply returns the initial learning rate every time
/// * `monotonic_func` The monotonic function to be used during the prediction and training.
///                    For more information about this function and its significance refer to the struct definition and the respective paper.
///                    Both the function and the derivative receive as parameters (distance, current epoch) in this order.
///                    This parameter defaults to the identity function.
/// * `max_epochs`     The amount of epochs to run
/// * `rng`            The internal ChaChaRng to be used for reproducability.
///
#[derive(Debug)]
pub struct GLVQ {
    num_prototypes : BTreeMap<String, usize>,
    prototypes : Vec<Prototype>,
    initial_lr : f64,
    lr_scheduler : fn(f64, u32, u32) -> f64,
    monotonic_func : CustomMonotonicFunction,
    max_epochs : u32, 
    rng : ChaChaRng
}

/// The General Matrix Learning Vector Quantization (GMLVQ) model
///
/// This struct and its methods provide an implementation of the GMLVQ algorithm using stochastic gradient descent.
/// By introducing a full matrix of relevance factors in the distance measure, correlations between different features,
/// and their importance for the classification scheme can be taken into account.
///
/// The implementation is entirely based on the following paper [[1]](http://www.cs.rug.nl/~biehl/Preprints/gmlvq.pdf).
///
/// This specific implementation allows for a variable number of prototypes per class.
///
/// # Properties
/// * `num_prototypes` The amount of prototypes to use per class (a BTreeMap, that maps the class name to the number of prototypes to use)
/// * `prototypes`     A vector of the prototypes (initially empty)
/// * `omega`          A matrix used to compute the adaptive relevance matrix Lambda = tranpose(Omega).dot(Omega)
/// * `initial_lr`     The initial learning rate to be used by the learning rate scheduler
///                    Note: This time, we require two learning rates (one for the prototypes and one for the matrix) as a tuple
/// * `lr_scheduler`   The learning rate scheduler for the update step of the prototypes
///                    This function can be custom and receives: 
///                    (base_learning_rate_protototype, base_learning_rate_matrix, current_epoch, max_epochs) as parameters
///                    The default scheduler simply returns the initial learning rates every time
///                    Note: This time, we require that the scheduler returns two learning rates (one for the prototypes and one for the matrix)
/// * `lr_scheduler`   The learning rate scheduler for the update step of the prototypes
/// * `monotonic_func` The monotonic function to be used during the prediction and training.
///                    For more information about this function and its significance refer to the struct definition and the respective paper.
///                    Both the function and the derivative receive as parameters (distance, current epoch) in this order.
///                    This parameter defaults to the identity function.
/// * `max_epochs`     The amount of epochs to run
/// * `rng`            The internal ChaChaRng to be used for reproducability.
///
#[derive(Debug)]
pub struct GMLVQ {
    num_prototypes : BTreeMap<String, usize>,
    prototypes : Vec<Prototype>,
    omega: Option<Array2<f64>>,
    initial_lr : (f64, f64),
    lr_scheduler : fn(f64, f64, u32, u32) -> (f64, f64),
    monotonic_func : CustomMonotonicFunction,
    max_epochs : u32,
    rng : ChaChaRng
}

/// The Limited Rank Matrix Learning Vector Quantization (LiRaMLVQ) model
///
/// This struct and its methods provide an implementation of the LiRamLVQ algorithm using stochastic gradient descent.
/// By introducing a matrix of relevance factors with limited rank in the distance measure, correlations between different features,
/// and their importance for the classification scheme can be taken into account.
/// 
/// This algorithm is very similar to GMLVQ, however this time one can limit the rank of the Lambda matrix.
///
/// The implementation is entirely based on the following paper [[1]](http://www.cs.rug.nl/biehl/Preprints/liram-preliminary.pdf).
///
/// This specific implementation allows for a variable number of prototypes per class.
///
/// # Properties
/// * `max_rank`       The maximum rank of the matrix Lambda = Omega^T Omega
/// * `num_prototypes` The amount of prototypes to use per class (a BTreeMap, that maps the class name to the number of prototypes to use)
/// * `prototypes`     A vector of the prototypes (initially empty)
/// * `omega`          A matrix used to compute the adaptive relevance matrix Lambda = tranpose(Omega).dot(Omega)
/// * `initial_lr`     The initial learning rate to be used by the learning rate scheduler
///                    Note: This time, we require two learning rates (one for the prototypes and one for the matrix) as a tuple
/// * `lr_scheduler`   The learning rate scheduler for the update step of the prototypes
///                    This function can be custom and receives: 
///                    (base_learning_rate_protototype, base_learning_rate_matrix, current_epoch, max_epochs) as parameters
///                    The default scheduler simply returns the initial learning rates every time
///                    Note: This time, we require that the scheduler returns two learning rates (one for the prototypes and one for the matrix)
/// * `lr_scheduler`   The learning rate scheduler for the update step of the prototypes
/// * `monotonic_func` The monotonic function to be used during the prediction and training.
///                    For more information about this function and its significance refer to the struct definition and the respective paper.
///                    Both the function and the derivative receive as parameters (distance, current epoch) in this order.
///                    This parameter defaults to the identity function.
/// * `max_epochs`     The amount of epochs to run
/// * `rng`            The internal ChaChaRng to be used for reproducability.
///
#[derive(Debug)]
pub struct LiRaMLVQ {
    max_rank : u32,
    num_prototypes : BTreeMap<String, usize>,
    prototypes : Vec<Prototype>,
    omega: Option<Array2<f64>>,
    initial_lr : (f64, f64),
    lr_scheduler : fn(f64, f64, u32, u32) -> (f64, f64),
    monotonic_func : CustomMonotonicFunction,
    max_epochs : u32,
    rng : ChaChaRng
}