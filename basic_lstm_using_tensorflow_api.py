import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

class BasicLSTM(object):
    def __init__(self, state_size, batch_size, num_steps, num_layers,
                 num_classes, inlayer_dropout, learning_rate, summary_every=100):
        """Create a Basic RNN classfier with the given STATE_SIZE,
        NUM_STEPS, and NUM_CLASSES
        """
        self.state_size = state_size
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.inlayer_dropout = inlayer_dropout
        self.learning_rate = learning_rate

        # tensorflow machinery
        self.session = tf.Session()
        self.summary_writer = tf.train.SummaryWriter(os.path.join(os.getcwd(), "tensorboard/"))
        self.no_op = tf.no_op()

        # counters
        self.train_itr = 0

        # create and initialize variables
        self.create_graph()
        var_lists = tf.get_collection(tf.GraphKeys.VARIABLES)
        self.session.run(tf.initialize_variables(var_lists))

        # make sure all variables are initialized
        self.session.run(tf.assert_variables_initialized())

        # add the graph
        self.summary_writer.add_graph(self.session.graph)
        self.summary_every = summary_every

    def create_placeholders(self):
        self.input = tf.placeholder(tf.int32, shape=(None, self.num_steps), name="input")
        self.target = tf.placeholder(tf.int32, shape=(None, self.num_steps), name="target")
        self.keep_prob = tf.placeholder_with_default(1.0, ())
        self.input_one_hot = tf.one_hot(self.input, self.num_classes)  # one-hot encoding of the input
        self.rnn_inputs = tf.unpack(self.input_one_hot, axis=1)  # unpacking inuts to make a list

    def create_variables(self):
        """Create variables for one layer RNN and the softmax
        """
        with tf.variable_scope("softmax"):
            self.W_softmax = tf.get_variable("W_softmax", [self.state_size, self.num_classes])
            self.b_softmax = tf.get_variable("b_softmax", [self.num_classes],
                                       initializer=tf.constant_initializer(0))

    def rnn(self):
        """ multi step RNN using tensorflow api
        """
        with tf.name_scope("rnn"):
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.state_size, state_is_tuple=True)
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
            lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.num_layers, state_is_tuple=True)
            self.init_state = lstm_cell.zero_state(self.batch_size, tf.float32)
            self.outputs, self.final_state = tf.nn.rnn(lstm_cell, self.rnn_inputs, initial_state=self.init_state)

    def softmax_loss(self):
        """A softmax operations on the output of the RNN
        OUTPUTS: is a list of tensors representing the outut from each rnn step
        """
        logits = [tf.matmul(output, self.W_softmax) + self.b_softmax for output in self.outputs]
        self.probs = [tf.nn.softmax(logit) for logit in logits]
        losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logit, self.target[:, i]) for i, logit in
                    enumerate(logits)]
        self.loss = tf.reduce_mean(losses)

    def create_variables_for_optimizations(self):
        """create variables for optimizing
        """
        with tf.name_scope("optimization"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            self.gradients = self.optimizer.compute_gradients(self.loss, var_list=self.trainable_variables)
            self.train_op = self.optimizer.apply_gradients(self.gradients)

    def create_summaries(self):
        """create summary variables
        """
        self.loss_summary = tf.scalar_summary("loss", self.loss)
        self.gradient_summaries = []
        for grad, var in self.gradients:
            if grad is not None:
                gradient_summary = tf.histogram_summary(var.name + "/gradient", grad)
                self.gradient_summaries.append(gradient_summary)
        self.weight_summaries = []
        weights = tf.get_collection(tf.GraphKeys.VARIABLES)
        for w in weights:
            weight_summary = tf.histogram_summary(w.name, w)
            self.weight_summaries.append(weight_summary)

    def merge_summaries(self):
        """Merge all sumaries
        """
        self.summarize = tf.merge_summary([self.loss_summary]
                                            + self.weight_summaries
                                            + self.gradient_summaries)

    def create_graph(self):
        self.create_placeholders()
        self.create_variables()
        self.rnn()
        self.softmax_loss()
        self.create_variables_for_optimizations()
        self.create_summaries()
        self.merge_summaries()

    def update_params(self, batch):
        """Given a batch of data, update the network to minimize the loss
        """
        write_summay = self.train_itr % self.summary_every == 0
        _, self.init_state, loss, summary = self.session.run([self.train_op,
                                        self.final_state, self.loss,
                                        self.summarize if write_summay else self.no_op],
                                        feed_dict={self.input: batch[0],
                                                   self.target:batch[1],
                                                   self.keep_prob:self.inlayer_dropout})
        if write_summay:
            self.summary_writer.add_summary(summary, self.train_itr)

        self.train_itr += 1
        return loss
