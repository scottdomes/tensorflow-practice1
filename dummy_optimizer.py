import tensorflow as tf

class DummyOptimizer(tf.keras.optimizers.Optimizer):
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

  def set_weights(self, weights):
    return weights

  def get_config(self):
    base_config = super().get_config()
    return {
      **base_config,
    }

  @tf.function
  def _resource_apply_dense(self, grad, var):
    return
    # var_dtype = var.dtype.base_dtype
    # lr_t = self._decayed_lr(var_dtype) # handle learning rate decay
    # var.assign_sub(grad * lr_t)
