use super::Prototype;

use ndarray::Array1;

impl Prototype {

    /// Constructs a new prototype that wraps a vector and a name
    /// 
    /// # Arguments
    /// 
    /// * `vector` The vector data of the prototype
    /// * `name`   The name of the vector (for readability only)
    /// 
    pub fn new ( vector: Array1<f64>, name : String ) -> Prototype {

        // Setup the prototype
        Prototype {
            vector,
            name
        }
    }

}