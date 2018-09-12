import tensorflow as tf


def crossentropy_factory(batch_size, tasks):
    """
    @tasks: The delimeters between the tasks for example [2,4,6]
    where the first task occupies 0:2, the second occupies [2:4] and the last occupies [4:6]
    """
    zero_loss = tf.zeros((batch_size, tasks[-1]))

    def crossentropy_loss_fn(y_true, y_pred):
        '''Just another crossentropy'''
        # Match up all the negative ones
        losses = []
        for i,end in enumerate(tasks):
            start = 0 if i==0 else tasks[i-1]
            true = y_true[:, start:end]
            pred = y_pred[:, start:end]
            task_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=true, logits=pred)
            # Get a column vector which is zero for all missing tasks, and one for present tasks
            mask = tf.reduce_max(tf.cast(tf.greater_equal(true,0), tf.float32), -1)
            masked_loss = task_loss*mask
            losses.append(tf.reduce_sum(masked_loss))
        return tf.add_n(losses)

    return crossentropy_loss_fn