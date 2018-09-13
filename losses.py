import tensorflow as tf


def crossentropy_filtered_loss(y_true, y_pred):
    '''Just another crossentropy'''
    # Mask out the tasks that are not included
    task_mask = tf.stop_gradient(tf.greater(y_true[:,0], -1), name="task_mask")
    true_masked = tf.boolean_mask(y_true, task_mask, name="true_masked")
    pred_masked = tf.boolean_mask(y_pred, task_mask, name="pred_masked")
    losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=true_masked, logits=pred_masked, name="loss")
    return tf.reduce_sum(losses)
