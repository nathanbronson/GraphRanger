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
from t import Encoder, Decoder
from predictor import Predictor

class GraphModel(tf.keras.Model):
    def __init__(self, edge_vocab_size, node_vocab_size, num_layers=2, d_model=64, num_heads=8, dff=256, classes=2, filters=64, dropout=.1):
        super().__init__()
        self.encoder = Encoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            edge_vocab_size=edge_vocab_size,
            node_vocab_size=node_vocab_size,
            dropout_rate=dropout
        )
        self.reward_network = Decoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            edge_vocab_size=edge_vocab_size,
            node_vocab_size=node_vocab_size,
            dropout_rate=dropout
        )
        self.prediction_network = Predictor(output_classes=classes, filters=filters, flat_units=dff, dropout=dropout)
        self.mse = tf.keras.losses.MeanSquaredError()
        self.scc = tf.keras.losses.SparseCategoricalCrossentropy(reduction="none")

        self.encoder_optimizer = tf.keras.optimizers.legacy.Adam(1e-4)
        self.prediction_optimizer = tf.keras.optimizers.legacy.Adam(1e-4)
        self.reward_optimizer = tf.keras.optimizers.legacy.Adam(1e-4)

        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.predicition_loss_tracker = tf.keras.metrics.Mean(name="p_loss")
        self.reward_loss_tracker = tf.keras.metrics.Mean(name="r_loss")
        self.accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name="acc")
    
    def train_step(self, data):
        x, y = data
        x, marginal_node, current_mask, forward_looking_mask = x

        with tf.GradientTape(persistent=True) as tape:
            encoding = self.encoder(x, training=False)
            y_pred = self.prediction_network(encoding, training=False)

            element_wise_loss = self.scc(y, y_pred)
            marginal_loss = tf.gather(element_wise_loss, forward_looking_mask, axis=0) - tf.gather(element_wise_loss, current_mask, axis=0)

            predicted_marginal_loss = self.reward_network(tf.expand_dims(marginal_node, 1), tf.gather(encoding, current_mask, axis=0))
            
            prediction_loss = tf.reduce_mean(element_wise_loss)
            reward_loss = self.mse(marginal_loss, predicted_marginal_loss)
            total_loss = prediction_loss + reward_loss

        # Compute gradients
        encoder_vars = self.encoder.trainable_variables
        prediction_vars = self.prediction_network.trainable_variables
        reward_vars = self.reward_network.trainable_variables

        encoder_gradients = tape.gradient(total_loss, encoder_vars)
        prediction_gradients = tape.gradient(prediction_loss, prediction_vars)
        reward_gradients = tape.gradient(reward_loss, reward_vars)
        del tape

        # Update weights
        self.encoder_optimizer.apply_gradients(zip(encoder_gradients, encoder_vars))
        self.prediction_optimizer.apply_gradients(zip(prediction_gradients, prediction_vars))
        self.reward_optimizer.apply_gradients(zip(reward_gradients, reward_vars))

        # Compute our own metrics
        self.total_loss_tracker.update_state(total_loss)
        self.predicition_loss_tracker.update_state(prediction_loss)
        self.reward_loss_tracker.update_state(reward_loss)
        self.accuracy_tracker.update_state(y, y_pred)

        return {
            "loss": self.total_loss_tracker.result(),
            "p_loss": self.predicition_loss_tracker.result(),
            "r_loss": self.reward_loss_tracker.result(),
            "acc": self.accuracy_tracker.result()
        }
    
    def test_step(self, data):
        x, y = data
        x, marginal_node, current_mask, forward_looking_mask = x

        encoding = self.encoder(x, training=False)
        y_pred = self.prediction_network(encoding, training=False)

        element_wise_loss = self.scc(y, y_pred)
        marginal_loss = tf.gather(element_wise_loss, forward_looking_mask, axis=0) - tf.gather(element_wise_loss, current_mask, axis=0)

        predicted_marginal_loss = self.reward_network(tf.expand_dims(marginal_node, 1), tf.gather(encoding, current_mask, axis=0))
        
        prediction_loss = tf.reduce_mean(element_wise_loss)
        reward_loss = self.mse(marginal_loss, predicted_marginal_loss)
        total_loss = prediction_loss + reward_loss

        self.total_loss_tracker.update_state(total_loss)
        self.predicition_loss_tracker.update_state(prediction_loss)
        self.reward_loss_tracker.update_state(reward_loss)
        self.accuracy_tracker.update_state(y, y_pred)

        return {
            "loss": self.total_loss_tracker.result(),
            "p_loss": self.predicition_loss_tracker.result(),
            "r_loss": self.reward_loss_tracker.result(),
            "acc": self.accuracy_tracker.result()
        }