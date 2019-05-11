# coding=utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


mnist_data = input_data.read_data_sets('./mnist_data/', one_hot=True)

is_train = False


def run_demo8():
    # [28, 28]
    with tf.variable_scope('original_data'):
        x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
        y_true = tf.placeholder(dtype=tf.float32, shape=[None, 10])

    # 定义网络结构
    with tf.variable_scope('model_param'):
        w = tf.Variable(initial_value=tf.random_normal([784, 10], mean=0.0, stddev=0.05), name='w')
        b = tf.Variable(initial_value=tf.random_normal([10], mean=0.0, stddev=0.05), name='b')

        y_predict = tf.matmul(x, w) + b

        # w2, b2, y_predicr2 = tf.matmul(y_predict, w2) + b2
    # 定义损失函数
    with tf.variable_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))

    # 定义训练操作
    with tf.variable_scope('optimizer'):
        train_op = tf.train.AdamOptimizer(0.01).minimize(loss)

    # 定义准确率 tf.argmax(), tf.equal()
    with tf.variable_scope('accuracy'):
        equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))
        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    #tf.summary.scalar('loss', loss)
    #tf.summary.scalar('accuracy', accuracy)
    #tf.summary.histogram('w', w)
    #tf.summary.histogram('b', b)

    init = tf.global_variables_initializer()

    #merge = tf.summary.merge_all()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        #file_writer = tf.summary.FileWriter(logdir='./demo8_summary/', graph=sess.graph)

        if os.path.exists('./demo8_save/checkpoint'):
            saver.restore(sess, './demo8_save/')

        if is_train:
            for epoch in range(500):
                x_epoch, y_epoch = mnist_data.train.next_batch(50)

                sess.run(train_op, feed_dict={x: x_epoch, y_true: y_epoch})

                loss_epoch, accuracy_epoch = sess.run([loss, accuracy], feed_dict={x: x_epoch, y_true: y_epoch})

                if (epoch + 1) % 20 == 0:
                    print('round=%d, loss=%f, accuracy=%f' % (epoch + 1, loss_epoch, accuracy_epoch))
                    saver.save(sess, './demo8_save/')

                #summary = sess.run(merge, feed_dict={x: x_epoch, y_true: y_epoch})
                #file_writer.add_summary(summary, epoch)
        else:
            for i in range(1000):
                x_i, y_i = mnist_data.test.next_batch(1)

                true_value = tf.argmax(sess.run(y_true, feed_dict={x: x_i, y_true: y_i}), 1).eval()
                predict_value = tf.argmax(sess.run(y_predict, feed_dict={x: x_i, y_true: y_i}), 1).eval()

                print('第%d个样本, 真实值=%d, 预测值=%d' % (i + 1, true_value, predict_value))


if __name__ == '__main__':
    run_demo8()