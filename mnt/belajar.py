import os
import tensorflow as tf

# Initialization 
x = tf.constant(4, shape=(1, 1))

# print(x)

x = tf.eye(2)
p = tf.ones((3, 3))
# print(x)
# print(p)

l = tf.random.normal((3, 3), mean=0, stddev=1)
m = tf.random.uniform((5, 3), minval=0, maxval=1)
ps = tf.range(start=100, limit=120, delta=2)
# print(l)
# print(m)
# print(ps)
casting = tf.cast(p, dtype=tf.int32)
# print(casting)

# Math
x = tf.constant([1, 2, 3])
y = tf.constant([9, 8, 9])
z = (x * y)
# print(z)
# print(tf.tensordot(x, y, axes=1))
x = tf.random.normal((2, 3))
y = tf.random.normal((3, 4))
z = tf.matmul(x, y)
# print(z)
z = x @ y
# print(z)

# Indexing
x = tf.constant([
    0, 1, 1, 2, 3, 1, 2, 3
])
# print(x[::2])
p = tf.constant([
    [1, 2],
    [3, 4],
    [5, 6]
])

# print(p)
# print(p[0, 0])
# Reshaping
mat = tf.range(9)
print(mat)

mat = tf.reshape(mat, (3, 3))
print(mat)

mat = tf.transpose(mat, perm=[1, 0])
print(mat)