"""AUC ROC one-vs-rest di EVA, calcolata durante il training su soglie discrete."""
import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package='eva')
class EvaAUC(tf.keras.metrics.AUC):
    def __init__(self,name='eva_auc',**kwargs):
        kwargs.setdefault('num_thresholds',1000)
        super().__init__(name=name,**kwargs)

    def update_state(self,y_true,y_pred,sample_weight=None):
        # EVA ha indice zero sia nel problema a tre classi sia in quello binario.
        target=tf.cast(tf.equal(tf.reshape(y_true,[-1]),0),tf.float32)
        return super().update_state(target,y_pred[:,0],sample_weight)
