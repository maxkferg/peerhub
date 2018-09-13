import tensorflow as tf


def accuracy_fn_factory(tasks,i):
    """
    Return the accuracy of task i

    @tasks: The delimeters between the tasks for example [2,4,6]
            where the first task occupies 0:2, the second occupies [2:4] and the last occupies [4:6]
    @i: The task that we are interested in
    """
    start = 0
    if i>0:
        start = tasks[i-1]
    end = tasks[i]

    print("Task %i: Start "%i, start, "End", end)

    def accuracy_fn(y_true, y_pred):
        '''Just another crossentropy'''
        # Index out the values relevant to this task
        y_true = y_true[:, start:end]
        y_pred = y_pred[:, start:end]

        #Compute the class prediction accuracy
        y_true_class = tf.argmax(y_true, 1)
        y_pred_class = tf.argmax(y_pred, 1)
        correct_predictions = tf.equal(y_pred_class, y_true_class)

        # Get a column of booleans that is true when this task is labelled
        task_mask = tf.stop_gradient(tf.greater(y_true[:,0], -1))

        # Only labelled task contribute to the error
        correct_predictions_masked = tf.boolean_mask(correct_predictions, task_mask)
        return tf.reduce_mean(tf.cast(correct_predictions_masked, "float"))
    return accuracy_fn