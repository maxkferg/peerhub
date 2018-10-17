import keras
import keras.backend as K
import tensorflow as tf


def crossentropy_filtered_loss(y_true, y_pred):
    '''Just another crossentropy'''
    #y_true = K.print_tensor(y_true, message='y_true = ')
    #y_pred = K.print_tensor(y_pred, message='y_pred = ')
    
    # Mask out the tasks that are not included
    task_mask = tf.stop_gradient(tf.greater(y_true[:,0], -1), name="task_mask")
    task_mask = tf.cast(task_mask, tf.float32)

    #task_mask = K.print_tensor(task_mask, message='mask = ')
    
    #true_masked = tf.boolean_mask(y_true, task_mask, name="true_masked")
    #pred_masked = tf.boolean_mask(y_pred, task_mask, name="pred_masked")

    #true_masked = K.print_tensor(true_masked, message='true_masked = ')
    #pred_masked = K.print_tensor(pred_masked, message='pred_masked = ')

    #losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=true_masked, logits=pred_masked, name="loss")
    losses = keras.losses.categorical_crossentropy(y_true, y_pred)
    losses = losses*task_mask
    #losses = K.print_tensor(losses, message='losses = ')
    return losses
    #return tf.reduce_sum(losses)
