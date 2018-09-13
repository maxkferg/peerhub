import tensorflow as tf


def crossentropy_factory(tasks):
    """
    @tasks: The delimeters between the tasks for example [2,4,6]
    where the first task occupies 0:2, the second occupies [2:4] and the last occupies [4:6]
    """

    def crossentropy_loss_fn(y_true, y_pred):
        '''Just another crossentropy'''
        # Match up all the negative ones
        losses = []
        for i,end in enumerate(tasks):
            start = 0 if i==0 else tasks[i-1]
            true = y_true[:, start:end]
            pred = y_pred[:, start:end]
            task_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=true, logits=pred)
            # Get a column of booleans that is true for available tasks
            task_mask = tf.stop_gradient(tf.greater(y_true[:,0], -1))
            # Only these tasks contribute to the loss
            masked_loss = tf.boolean_mask(task_loss, task_mask)
            # Average loss is preferred so we don't favour tasks with many classes
            losses.append(tf.reduce_mean(masked_loss))
        return tf.add_n(losses)

    return crossentropy_loss_fn