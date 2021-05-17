pub trait Schedulable {

    /// Updates the learning rate scheduler of the model this trait is implemented for
    /// 
    /// # Arguments
    /// 
    /// * `scheduler` A function that takes in three arguments of the form (initial_learning_rate, current epoch, max epochs)
    ///               The algorithm will insert these arguments when the learning rate is to be calculated.
    ///
    fn set_learning_rate_scheduler (&mut self, scheduler : fn(f64, u32, u32) -> f64);
}