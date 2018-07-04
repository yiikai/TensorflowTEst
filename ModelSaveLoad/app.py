import tensorflow as tf
import numpy as np

x = tf.placeholder(dtype=tf.float32, shape=[None], name='input_x')
y = tf.placeholder(dtype=tf.float32, shape=[None], name='output_y')

with tf.variable_scope("weight") as weight:
    w = tf.Variable(tf.truncated_normal([1,1], stddev=0.1), name='w1')

out = tf.multiply(x,w,name='out')
loss = tf.reduce_mean(tf.square(out - y))

train_op = tf.train.GradientDescentOptimizer(1e-3).minimize(loss)
saver = tf.train.Saver()

train_x = np.array([1,2,3,4,5])
train_x = train_x.reshape(5,)
train_y = np.array([3,6,9,12,15])
train_y = train_y.reshape(5,)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        cost,_ = sess.run([loss, train_op], feed_dict={x:train_x, y:train_y})
        print("LOSS =====>",cost)
        if i % 100 == 0:
            save_path = saver.save(sess, "./model/model",global_step=i)
            print("save path is ",save_path)
       

       