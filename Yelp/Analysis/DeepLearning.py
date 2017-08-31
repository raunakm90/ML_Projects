import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from sklearn.model_selection import TimeSeriesSplit

# Function for batch training


def sample_batch(X_train, y_train, batch_size, num_steps):
	N, data_len = X_train.shape
	# Select rows from X_train
	ind_N = np.random.choice(N, batch_size, replace=False)
	# Select the time periods (Columns)
	ind_start = np.random.choice(data_len=num_steps, 1)
	X_batch = X_train[ind_N, ind_start:ind_start +
	    num_steps]  # Subset for batch training
	y_batch = y_train[ind_N]
	return (X_batch, y_batch)

# Function to reset tensorflow graph


def reset_graph():
	if 'sess' in globals() and sess:
		sess.close()
	tf.reset_default_graph()

# Check model on test data


def check_test(X_test, y_test, batch_size, num_steps):
  """ Function to check the test_accuracy on the entire test set"""
  N = X_test.shape[0]
  num_batch = np.floor(N / batch_size)
  test_acc = np.zeros(num_batch)
  for i in range(int(num_batch)):
    X_batch, y_batch = sample_batch(X_test, y_test, batch_size, num_steps)
    test_acc[i] = sess.run(accuracy, feed_dict={
                           input_data: X_batch, targets: y_batch, keep_prob: 1})
  return np.mean(test_acc)

# Read yelp data and split into train and test data


def read_data():
	yelp_data = pd.read_csv("./Data/final_df1.csv",index_col=0,parse_dates=[1])
	yelp_data.sort_values(by='Period_StartDate',inplace=True)

	X_data = yelp_df.drop(['Open_Flag'],axis = 1)  # Features
	y_data = yelp_df.Open_Flag #Target variables

	sub_ind = [X_data.Period_StartDate > '12/31/2014'].index
	X_test = X_data[sub_ind,]  # Subset for last 6 months
	y_test = y_data[sub_ind,]  # Subset for last 6 months

	X_data.drop(['Period_StartDate'], inplace=True)

	return (X_data, y_data, X_test, y_test)


def build_graph(cell_type='LSTM', num_layers=2, hidden_size=13, num_steps=60,
				 max_iter=2000, batch_size=30, build_with_dropout=True,
				 classifier='softmax',learning_rate=1e-3):
	'''Hyperparameters'''

	max_grad_norm = 5  # Clipping of the gradient before the update
	dropout = 0.8  # Keep probability of the dropout wrapper

	X_data, y_data, X_test, y_test = read_data()

	tscv = TimeSeriesSplit(n_splits=3)  # Number of time splits

	for train_index, val_index in tscv.split(X_data):
		X_train, X_val = X_data[train_index, ], X_data[val_index, ]
		y_train, y_val = y_data[train_index, ], y_data[val_index, ]

		# The following code should be in the for loop
		N = X_train.shape[0]  # Number of features
		num_classes = len(np.unique(y_train))  # Number of classes in the dataset

		epochs = np.floor(batch_size * max_iter / N)
		print ('Train with approx. %d epochs' % (epochs))
		# Collect performance numbers for every 100 iterations
		perf_collect = np.zeros((3, int(np.floor(max_iter / 100))))

		'''
		Placeholders
		tf.placeholder(dtype, shape = None, name = None)
		'''

		input_data = tf.placeholder(tf.float32, [None, num_steps], name= 'input_data')
		targets = tf.placeholder(tf.int32, [None], name= 'Targets')
		# Used for drop_out wrapper
		keep_prob = tf.placeholder(tf.float32, name='Drop_out_keep_prob')

		'''
		Setup LSTM

		tf.nn.rnn_cell.LSTMCell.__init__(num_units, input_size=None, use_peepholes=False, cell_clip=None, initializer=None, num_proj=None,
		                                 proj_clip=None, num_unit_shards=1, num_proj_shards=1, forget_bias=1.0, state_is_tuple=True, activation=tanh)
		Initialize the parameters for an LSTM cell.
		Args:
				num_units: int, The number of units in the LSTM cell
				input_size: Deprecated and unused.
				use_peepholes: bool, set True to enable diagonal/peephole connections.
				cell_clip: (optional) A float value, if provided the cell state is clipped by this value prior to the cell output activation.
				initializer: (optional) The initializer to use for the weight and projection matrices.
				num_proj: (optional) int, The output dimensionality for the projection matrices. If None, no projection is performed.
				proj_clip: (optional) A float value. If num_proj > 0 and proj_clip is provided, then the projected values are clipped elementwise to within [-proj_clip, proj_clip].
				num_unit_shards: How to split the weight matrix. If >1, the weight matrix is stored across num_unit_shards.
				num_proj_shards: How to split the projection matrix. If >1, the projection matrix is stored across num_proj_shards.
				forget_bias: Biases of the forget gate are initialized by default to 1 in order to reduce the scale of forgetting at the beginning of the training.
				state_is_tuple: If True, accepted and returned states are 2-tuples of the c_state and m_state. If False, they are concatenated along the column axis. This latter behavior will soon be deprecated.
				activation: Activation function of the inner states.

		tf.nn.rnn_cell.DropoutWrapper.__init__(
		    cell, input_keep_prob=1.0, output_keep_prob=1.0, seed=None)
				Create a cell with added input and/or output dropout.
				Dropout is never used on the state.
				Args:
				cell: an RNNCell, a projection to output_size is added to it.
				input_keep_prob: unit Tensor or float between 0 and 1, input keep probability; if it is float and 1, no input dropout will be added.
				output_keep_prob: unit Tensor or float between 0 and 1, output keep probability; if it is float and 1, no output dropout will be added.
				seed: (optional) integer, the randomness seed.
		'''

		with tf.name_scope('LSTM_Setup') as scope:
			# Define and initialize the LSTM cell
			if cell_type == 'LSTM':
				cell = tf.nn.rnn_cell.LSTMCell(hidden_size)
			elif cell_type == 'GRU':
				cell = tf.nn.rnn_cell.GRUCell(hidden_size)
			elif cell_type == 'LN_LSTM':
				cell = tf.nn.rnn_cell.LayerNormalizedLSTMCell(hidden_size)
			else:
				cell = tf.nn.rnn_celll.BasicRNNCell(hidden_size)

			if build_with_dropout:
				# Dropout wrapper for regularization
				cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)

			if cell_type == 'LSTM' or cell_type == 'LN_LSTM':
				cell = tf.nn.rnn_cell.MultiRNNCell(
				    [cell] * num_layers, state_is_tuple=True)
			else:
				cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)  # Stack the cells

			initial_state = cell.zero_state(batch_size, tf.float32)
			# Output - A Tensor. Has the same type as input. Contains the same data as
			# input, but its shape has an additional axisension of size 1 added.
			inputs = tf.expand_dims(input_data, 2)

		'''Define recurrent nature of LSTM'''
		with tf.name_scope('LSTM') as scope:
			outputs = []
			state = initial_state
			with tf.variable_scope('LSTM_state'):
				for time_step in range(num_steps):
					if time_step > 0:
						# Re-use variables only after the first time-step
						tf.get_variable_scope().reuse_variables()
					(cell_output, state) = cell(inputs[:, time_step, :], state)
					# Cell output is size [batch_size * hidden_size]
					outputs.append(cell_output)
			output = tf.reduce_mean(tf.pack(outputs), 0)

		# Generate classification from the last cell_output. This is specific for
		# time series classification

		if classifier == 'softmax':
			with tf.name_scope('SoftMax') as scope:
				with tf.variable_scope('SoftMax_params'):
					W = tf.get_variable('Weights', [hidden_size, num_classes])
					b = tf.get_variable('Bias', [num_classes])
				logits = tf.nn.xw_plus_b(output, W, b)
				loss = tf.nn.softmax_cross_entropy_with_logits(logits, targets, name='Cross_Entropy')
				cost = tf.reduce_sum(loss) / batch_size

			with tf.name_scope('Evaluating_Accuracy') as scope:
				correct_prediction = tf.equal(tf.argmax(logits, 1), targets)
				accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

		elif classifier == 'logistic_regression':
			with tf.name_scope('Logistic_Regression') as scope:
				prediction_prob, loss = learn.models.logistic_regression(output, targets)
				train_op = tf.contrib.layers.optimize_loss(loss, tf.contrib.framework.get_global_step(),
											optimizer= 'Adam', learning_rate=learning_rate)
				cost = tf.reduce_sum(loss) / batch_size
			with tf.name_scope("Evaluating_Accuracy") as scope:
				correct_prediction = tf.equall(tf.argmax(prediction_prob, 1), targets)
				accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

		'''
		Optimizer
		tf.trainable_variables()
		Returns all variables created with trainable=True.
		When passed trainable=True, the Variable() constructor automatically adds new variables to the
		graph collection GraphKeys.TRAINABLE_VARIABLES. This convenience function returns the contents of that collection.
		'''

		with tf.name_scope('Optimizer') as scope:
			tvars = tf.trainable_variables()
			# Clip the gradients to prevent explosion
			grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), max_grad_norm)
			optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
			gradients = zip(grads, tvars)
			train_op = optimizer.apply_gradients(gradients)

			# Add histograms for variables, gradients and gradient norms
			for gradient, variable in gradients:
				if isinstance(gradients, ops.IndexedSlices):
					grad_values = gradient.values
				else:
					grad_values = gradient
				h1 = tf.histogram_summary(variable.name, variable)
				h2 = tf.histogram_summary(variable.name + '/gradients', grad_values)
				h3 = tf.histogram_summary(variable.name + '/gradient_norm', tf.global_norm([grad_values])

		merged = tf.merge_all_summaries()  # Merge all summaries collected in the graph

		'''Session Time'''
		sess = tf.Session()
		writer = tf.train.SummaryWriter('./Analysis/log_tb')
		sess.run(tf.initialize_alll_variables())


		step = 0
		# Moving average of training cost
		cost_train_ma = -np.log(1 / float(num_classes) + 1e-9)
		for i in range(max_iter):
			N = X_train.shape[0]
			# Sample batch training samples
			X_batch, y_batch = sample_batch(X_train, y_train, batch_size, num_steps)

			# Training the network
			cost_train, _ = sess.run([cost, train_op], feed_dict={
				                        input_data: X_batch, targets: y_batch, keep_prob: dropout})
			cost_train_ma = (cost_train_ma * 0.99 + cost_train * 0.01)

			if i % 100 == 0:
				perf_collect[0, step]=cost_train

				# Evaluate validation performance at every 100th iteration
				X_batch, y_batch=sample_batch(X_val, y_val, batch_size, num_steps)
				result=sess.run([cost, merged, accuracy], feed_dict={
					                input_data: X_batch, targets: y_batch, keep_prob: 1})
				cost_val=results[0]
				perf_collect[1, step]=cost_val
				acc_val=result[2]
				perf_collect[2, step]=acc_val
				print('At %5.0f out of %5.0f: Cost is TRAIN %.3f(%.3f) VAL%.3f and val acc is %3f'
						(i, max_iterations, cost_train, cost_train_ma, cost_val, acc_val))

					# Write information to the TensorBoard
			summary_str=result[1]
			writer.add_summary(summary_str, i)
			writer.flush()
			step += 1
			acc_test=check_test(X_test, y_test, batch_size, num_steps)

			'''Additional Plots'''
			print(“The accuracy on the test data is 0.3 % f” % (acc_test))
			plt.plot(perf_collect[0], label= 'Train')
			plt.plot(perf_collect[1], label= 'Validation')
			plt.plt(perf_collect[2], label= 'Validation_Accuracy')
			plt.axis([0, step, 0, np.max(perf_collect)])
			plt.legend()
			plt.show()

# if __name__ == '__main__':
# 	build_graph(cell_type='LSTM', num_layers=2, hidden_size=13, num_steps=60,
# 				max_iter=200, batch_size=30, build_with_dropout=True,
# 				classifier='softmax',learning_rate=1e-3)


