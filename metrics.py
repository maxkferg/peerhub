import tensorflow as tf


def accuracy_fn(y_true, y_pred):
    '''Just another crossentropy'''
    # Compute the class prediction accuracy
    y_true_class = tf.argmax(y_true, 1)
    y_pred_class = tf.argmax(y_pred, 1)
    correct_predictions = tf.equal(y_pred_class, y_true_class)

    # Get a column of booleans that is true when this task is labelled
    task_mask = tf.stop_gradient(tf.greater(y_true[:,0], -1))

    # Only labelled task contribute to the error
    correct_predictions_masked = tf.boolean_mask(correct_predictions, task_mask)
    return tf.reduce_mean(tf.cast(correct_predictions_masked, "float"))
