import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

is_train = True

mnist_data = input_data.read_data_sets('./mnist_data/', one_hot=True)


def cnn_model(x):
    # 设计第一层卷积网络
    with tf.variable_scope('conv_1'):
        # [None, 28, 28, 1]
        x_reshape = tf.reshape(x, [-1, 28, 28, 1])
        # 每一层都需要卷积核的定义
        conv1_w = tf.Variable(initial_value=tf.random_normal(shape=[5, 5, 1, 32]), name='conv1_w')
        conv1_b = tf.Variable(initial_value=tf.random_normal(shape=[32]), name='conv1_b')

        x_conv1 = tf.nn.conv2d(x_reshape, conv1_w, strides=[1, 1, 1, 1], padding='SAME') + conv1_b
        # [None, 28, 28, 32]
        x_relu1 = tf.nn.relu(x_conv1)
        # [None, 28, 28, 32]
        x_pool1 = tf.nn.max_pool(x_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # [None, 14, 14, 32]

    with tf.variable_scope('conv_2'):
        # 每一层都需要卷积核的定义
        conv2_w = tf.Variable(initial_value=tf.random_normal(shape=[5, 5, 32, 64]), name='conv2_w')
        conv2_b = tf.Variable(initial_value=tf.random_normal(shape=[64]), name='conv2_b')

        x_conv2 = tf.nn.conv2d(x_pool1, conv2_w, strides=[1, 1, 1, 1],padding='SAME') + conv2_b
        # [None, 14, 14, 64]
        x_relu2 = tf.nn.relu(x_conv2)
        # [None, 14, 14, 64]
        x_pool2 = tf.nn.max_pool(x_relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # [None, 7, 7, 64]
        # x_bn = tf.nn.batch_normalization(x_pool2)

    with tf.variable_scope('full_connection'):
        x_fc = tf.reshape(x_pool2, [-1, 7 * 7 * 64])
        fc_w = tf.Variable(initial_value=tf.random_normal(shape=[7 * 7 * 64, 10]))
        fc_b = tf.Variable(initial_value=tf.random_normal(shape=[10]))

        y_predict = tf.matmul(x_fc, fc_w) + fc_b

        y_output = tf.nn.dropout(y_predict, keep_prob=0.5)

    return y_predict


def mnist_demo():
    with tf.variable_scope('original_data'):
        x = tf.placeholder(tf.float32, [None, 784])
        y_true = tf.placeholder(tf.float32, [None, 10])

        y_predict = cnn_model(x)

    with tf.variable_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))

    with tf.variable_scope('optimizer'):
        # train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
        train_op = tf.train.AdamOptimizer(0.01).minimize(loss)

    with tf.variable_scope('accuracy'):
        equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))
        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    # tf.summary.scalar('loss', loss)
    # tf.summary.scalar('acc', accuracy)

    init = tf.global_variables_initializer()

    # merge = tf.summary.merge_all()

    # saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        # file_writer = tf.summary.FileWriter('./summary/', graph=sess.graph)

        # if os.path.exists('./model/checkpoint'):
        #     saver.restore(sess, './model/fc_model')

        if is_train:
            for epoch in range(1000):
                x_epoch, y_epoch = mnist_data.train.next_batch(50)

                sess.run(train_op, feed_dict={x: x_epoch, y_true: y_epoch})

                loss_epoch = sess.run(loss, feed_dict={x: x_epoch, y_true: y_epoch})

                accuracy_epoch = sess.run(accuracy, feed_dict={x: x_epoch, y_true: y_epoch})

                print('epoch=%d, loss=%f, accuracy=%f' % (epoch + 1, loss_epoch, accuracy_epoch))
                # saver.save(sess, './model/fc_model')

                # summary = sess.run(merge, feed_dict={x: x_epoch, y_true: y_epoch})
                # file_writer.add_summary(summary, epoch)
        else:
            for i in range(1000):
                x_i, y_i = mnist_data.train.next_batch(1)

                print('第%d个样本的真实值为: %d , 预测结果为: %d\n' % (
                    i + 1,
                    tf.argmax(sess.run(y_true, feed_dict={x: x_i, y_true: y_i}), 1),
                    tf.argmax(sess.run(y_predict, feed_dict={x: x_i, y_true: y_i}), 1)))

    return None


if __name__ == '__main__':
    mnist_demo()