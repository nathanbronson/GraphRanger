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

class Predictor(tf.keras.Model):
    def __init__(self, output_classes=2, filters=64, flat_units=512, dropout=.1):
        super().__init__()
        self.l1 = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=2,
            padding="valid",
            activation="relu"
        )
        self.l2 = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=3,
            padding="valid",
            activation="relu"
        )
        self.l3 = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=4,
            padding="valid",
            activation="relu"
        )
        self.pool = tf.keras.layers.GlobalMaxPool1D()
        self.dense_1 = tf.keras.layers.Dense(units=flat_units, activation="relu")
        self.dropout = tf.keras.layers.Dropout(rate=dropout)
        self.proj = tf.keras.layers.Dense(output_classes)
    
    def call(self, x, training=False):
        x1 = self.pool(self.l1(x))
        x2 = self.pool(self.l2(x))
        x3 = self.pool(self.l3(x))
        x = tf.concat([x1, x2, x3], axis=-1)
        x = self.dense_1(x)
        x = self.dropout(x, training)
        return self.proj(x)
