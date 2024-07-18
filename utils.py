"""
GraphRanger: neural networks for heterogenous data
Copyright (C) 2024  Nathan Bronson

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import tensorflow as tf

@tf.function
def random_items(t, n):
    idx = tf.cast(tf.random.uniform((n,)) * tf.cast(tf.shape(t)[-1], tf.float32), tf.int32)
    return tf.gather(t, idx)

@tf.function
def random_pair(t):
    s = tf.shape(t)
    idx = tf.concat([tf.expand_dims(tf.range(s[0]), -1), tf.expand_dims(tf.cast(tf.random.uniform((s[0],)) * tf.cast(t.row_lengths(1), tf.float32), tf.int32), -1)], axis=-1)
    return tf.gather_nd(t, idx)