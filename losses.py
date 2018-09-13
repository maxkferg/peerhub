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
            with tf.name_scope("task_%i"%i) as scope:
                start = 0 if i==0 else tasks[i-1]
                true = y_true[:, start:end]
                pred = y_pred[:, start:end]

                # Mask out the tasks that are not included
                task_mask = tf.stop_gradient(tf.greater(true[:,0], -1), name="task_mask")
                true_masked = tf.boolean_mask(true, task_mask, name="true_masked")
                pred_masked = tf.boolean_mask(pred, task_mask, name="pred_masked")

                task_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=true_masked, logits=pred_masked, name="loss")

                # Using reduce_sum prevents nan when there are no contributing tasks
                losses.append(tf.reduce_sum(task_loss))
        return tf.add_n(losses)

    return crossentropy_loss_fn



def crossentropy_filtered_loss(y_true, y_pred):
    '''Just another crossentropy'''
    # Mask out the tasks that are not included
    task_mask = tf.stop_gradient(tf.greater(y_true[:,0], -1), name="task_mask")
    true_masked = tf.boolean_mask(y_true, task_mask, name="true_masked")
    pred_masked = tf.boolean_mask(y_pred, task_mask, name="pred_masked")
    losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=true_masked, logits=pred_masked, name="loss")
    return tf.reduce_sum(losses)
