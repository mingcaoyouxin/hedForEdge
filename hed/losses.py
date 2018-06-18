import tensorflow as tf


def sigmoid_cross_entropy_balanced(logits, label, name='cross_entropy_loss'):
    """
    howarddong 0617
    Implements Equation [2] in https://arxiv.org/pdf/1504.06375.pdf
    Compute edge pixels for each training sample and set as pos_weights to
    tf.nn.weighted_cross_entropy_with_logits

    y = tf.cast(label, tf.float32)
    count_neg = tf.reduce_sum(1. - y)
    count_pos = tf.reduce_sum(y)

    # Equation [2]
    beta = count_neg / (count_neg + count_pos)

    # Equation [2] divide by 1 - beta
    pos_weight = beta / (1 - beta)
    cost = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=y, pos_weight=pos_weight)

    # Multiply by 1 - beta
    cost = tf.reduce_mean(cost * (1 - beta))

    # check if image has no edge pixels return 0 else return complete error function
    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost, name=name)
    """
    with tf.name_scope('class_balanced_sigmoid_cross_entropy'):
        count_neg = tf.reduce_sum(1.0 - label) # 样本中0的数量
        count_pos = tf.reduce_sum(label) # 样本中1的数量(远小于count_neg)
        beta = count_neg / (count_neg + count_pos)  ## 负样本占得比例 e.g.  60000 / (60000 + 800) = 0.9868
        pos_weight = beta / (1.0 - beta)  ## 负正样本比例 0.9868 / (1.0 - 0.9868) = 0.9868 / 0.0132 = 74.75
        cost = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=label, pos_weight=pos_weight)
        cost = tf.reduce_mean(cost * (1 - beta))
        # 如果样本中1的数量等于0，那就直接让 cost 为 0，因为 beta == 1 时， 除法 pos_weight = beta / (1.0 - beta) 的结果是无穷大
        zero = tf.equal(count_pos, 0.0)
        final_cost = tf.where(zero, 0.0, cost) 
    return final_cost
