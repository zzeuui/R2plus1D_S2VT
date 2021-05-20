import os
import tensorflow as tf

import tensorflow as tf

sess = tf.Session()

a = tf.constant(1, name = "const1")
b = tf.constant(10, name = "const2")
c = a + b

asum = tf.summary.scalar("g1" , a)
bsum = tf.summary.scalar("g2",  b)
csum = tf.summary.scalar("gsum", c)

merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter('.\logs',sess.graph)
print("type  >>>>>> ", type(c))

for i in range(10):
    summary, _ = sess.run([merged, tf.convert_to_tensor(i, dtype=tf.float32)])
    print("type summary >>>", type(summary))
    train_writer.add_summary(summary, 0)
train_writer.close()