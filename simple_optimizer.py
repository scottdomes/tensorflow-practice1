import tensorflow as tf

class SimpleOptimizer(tf.keras.optimizers.Optimizer):
  def __init__(self,
              learning_rate=0.001,
              beta_1=0.9,
              beta_2=0.999,
              epsilon=1e-7,
              amsgrad=False,
              name='Dummy',
              **kwargs):
    super().__init__(name, **kwargs)
    self._set_hyper("learning_rate", kwargs.get("lr", learning_rate)) # handle lr=learning_rate
    self._is_first = True

  def _create_slots(self, var_list):
    """For each model variable, create the optimizer variable associated with it.
    TensorFlow calls these optimizer variables "slots".
    """
    for var in var_list:
        self.add_slot(var, "previous_weight") #previous variable i.e. weight or bias
    for var in var_list:
        self.add_slot(var, "previous_gradient") #previous gradient

  def set_weights(self, weights):
    return weights

  def get_config(self):
    base_config = super().get_config()
    return {
      **base_config,
      "learning_rate": self._serialize_hyperparameter("learning_rate"),
    }

  @tf.function
  def _resource_apply_dense(self, gradient, model_variable):
    """Update the slots and perform one optimization step for one model variable
    """
    model_variable_dtype = model_variable.dtype.base_dtype
    # lr_t = self._decayed_lr(model_variable_dtype) # handle learning rate decay
    lr_t = .001
    new_weight = model_variable - gradient * lr_t
    previous_weight_var = self.get_slot(model_variable, "previous_weight")
    previous_gradient_var = self.get_slot(model_variable, "previous_gradient")
    
    if self._is_first:
        self._is_first = False
        new_weighted_variable = new_weight
    else:
        cond = gradient*previous_gradient_var >= 0
        print(cond)
        avg_weights = (previous_weight_var + model_variable)/2.0
        new_weighted_variable = tf.where(cond, new_weight, avg_weights)
    previous_weight_var.assign(model_variable)
    previous_gradient_var.assign(gradient)
    model_variable.assign(new_weighted_variable)