# coding: utf-8
# 测试tf的一些分布的类的用法
import tensorflow as tf
import numpy as np

a = tf.constant([1, 2, 30000], dtype=tf.float32)
b = tf.constant([10000, 10000, 10000], dtype=tf.float32)
Bernoulli = tf.distributions.Bernoulli
a_d = Bernoulli(logits=a)   # logits中的每一个值都确定一个伯努利分布，值为1的概率为sigmoid(a)
b_d = Bernoulli(logits=b)

with tf.Session() as sess:
    # print sess.run([a_d.entropy(), b_d.entropy(), a_d.probs, b_d.probs])
    # print sess.run(b_d.prob([1, 2, 3]))
    #  prob函数返回某一种取值的概率,  probs返回的是该分布对应的一些概率.
    print sess.run(a_d.cross_entropy(b_d))
    print sess.run(a_d.probs)
    print sess.run(a_d.log_prob([1.0, 0.2, 1.0]))
    # log_prob 差不多就是交叉熵的意思，只是参数提供的概率分布是第一项.
    # 用这个函数是说不需要将右侧的值转为相同的分布类型的对象，也可以直接计算交叉熵
    print sess.run(a_d.log_prob([1.0, 1.0, 1.0]))
    print sess.run(b_d.cross_entropy(a_d))
