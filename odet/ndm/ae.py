"""AutoEncoder

"""
# Authors: kun.bj@outlook.com
#
# License: GNU GENERAL PUBLIC LICENSE

import numpy as np
import torch
from pyod.models.base import BaseDetector
from pyod.utils.stat_models import pairwise_distances_no_broadcast
from pyod.utils.utility import check_parameter
from sklearn.utils import check_array
from torch import nn
from torch.utils.data import DataLoader
from loguru import logger as lg

class autoencoder(nn.Module):

	def __init__(self, in_dim=0, hid_dim=0, lat_dim=0, p=0.2):
		super(autoencoder, self).__init__()
		self.encoder = nn.Sequential(
			nn.Linear(in_dim, hid_dim),
			# nn_pdf.ReLU(True),
			nn.LeakyReLU(True),
			# nn_pdf.Dropout(p=p),
			# nn_pdf.Linear(hid_dim, hid_dim),
			# # nn_pdf.ReLU(True),
			# nn_pdf.LeakyReLU(True),
			# nn_pdf.Linear(hid_dim, hid_dim),
			# # nn_pdf.ReLU(True),
			# nn_pdf.LeakyReLU(True),
			nn.Linear(hid_dim, lat_dim),
			# nn_pdf.ReLU(True),
			nn.LeakyReLU(True),
			# nn_pdf.Dropout(p=p),
		)
		self.decoder = nn.Sequential(
			nn.Linear(lat_dim, hid_dim),
			# nn_pdf.ReLU(True),
			nn.LeakyReLU(True),
			# nn_pdf.Dropout(p=p),
			# nn_pdf.Linear(hid_dim, hid_dim*2),
			# # nn_pdf.ReLU(True),
			# nn_pdf.LeakyReLU(True),
			# nn_pdf.Linear(hid_dim*2, hid_dim),
			# # nn_pdf.ReLU(True),
			# nn_pdf.LeakyReLU(True),
			nn.Linear(hid_dim, in_dim),
			nn.LeakyReLU(True),
			# nn_pdf.Tanh()
			# nn_pdf.Sigmoid()
		)

	def forward(self, x):
		x = self.encoder(x)
		x = self.decoder(x)
		return x

	def predict(self, X=None):
		self.eval()  # Sets the module in evaluation mode.
		y = self.forward(torch.Tensor(X))

		return y.detach().numpy()


# noinspection PyUnresolvedReferences,PyPep8Naming,PyTypeChecker
class AE(BaseDetector):

	def __init__(self, hidden_neurons=None,
	             hidden_activation='leakyrelu', output_activation='leakyrelu',
	             loss=None, optimizer='adam', lr=1e-3,
	             epochs=20, batch_size=32, dropout_rate=0.2,
	             l2_regularizer=0.1, validation_size=0.1, preprocessing=False,
	             verbose=1, random_state=None, contamination=0.1):
		super(AE, self).__init__(contamination=contamination)
		self.hidden_neurons = hidden_neurons
		self.hidden_activation = hidden_activation
		self.output_activation = output_activation
		self.loss = loss
		self.optimizer = optimizer
		self.epochs = epochs
		self.batch_size = batch_size
		self.dropout_rate = dropout_rate
		self.l2_regularizer = l2_regularizer
		self.validation_size = validation_size
		self.preprocessing = preprocessing
		self.verbose = verbose
		self.random_state = random_state
		self.lr = lr

		self.hidden_neurons_ = self.hidden_neurons

		check_parameter(dropout_rate, 0, 1, param_name='dropout_rate',
		                include_left=True)

	def _build_model(self, X, y, hidden_neurons=''):

		self.model = autoencoder(in_dim=self.hidden_neurons[0], hid_dim=self.hidden_neurons[1],
		                         lat_dim=self.hidden_neurons[2], p=self.dropout_rate)
		criterion = nn.MSELoss()
		optimizer = torch.optim.Adam(
			self.model.parameters(), lr=self.lr, weight_decay=self.l2_regularizer)  # weight_decay=1e-5

		# decay the learning rate
		decayRate = 0.99
		lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

		# # re-seed to make DataLoader() will have the same result.
		# torch.backends.cudnn.deterministic = True
		# torch.backends.cudnn.benchmark = False
		# random.seed(42)
		# torch.manual_seed(42)
		# torch.cuda.manual_seed(42)
		# np.random.seed(42)
		val_size = int(self.validation_size * len(y))
		train_size = len(y) - val_size
		lg.debug(f'train_size: {train_size}, val_size: {val_size}')
		train_dataset, val_dataset = torch.utils.data.random_split(list(zip(X, y)), [train_size, val_size])

		dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
		val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
		for epoch in range(self.epochs):
			train_loss = 0
			for s, data in enumerate(dataloader):
				X_batch, y_batch = data
				# ===================forward=====================
				output = self.model(X_batch.float())
				loss = criterion(output, y_batch.float())
				# ===================backward====================
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				train_loss += loss.data
			# lg.debug(epoch, s, loss.data)
			# if epoch % 10 == 0:
			#     lr_scheduler.step()
			# ===================log========================
			with torch.no_grad():
				val_loss = 0
				for t, data in enumerate(val_dataloader):
					X_batch, y_batch = data
					output = self.model(X_batch.float())
					loss = criterion(output, y_batch.float())
					val_loss += loss.data
				lg.debug('epoch [{}/{}], loss:{:.4f}, eval: {:.4f}, lr: {}'
				         .format(epoch + 1, self.epochs, train_loss / (s + 1), val_loss / (t + 1),
				                 lr_scheduler.get_last_lr()))

		# if epoch % 10 == 0:
		#     pic = to_img(output_data.cpu().data)
		#     save_image(pic, './mlp_img/image_{}.png'.format(epoch))

		self.model.eval()  # Sets the module in evaluation mode.

		return self.model

	# noinspection PyUnresolvedReferences
	def fit(self, X, y=None):
		"""Fit detector. y is optional for unsupervised methods.

		Parameters
		----------
		X : numpy array of shape (n_samples, n_features)
			The input samples.

		y : numpy array of shape (n_samples,), optional (default=None)
			The ground truth of the input samples (labels).
		"""
		# validate inputs X and y (optional)
		X = check_array(X)
		self._set_n_classes(y)

		# Verify and construct the hidden units
		self.n_samples_, self.n_features_ = X.shape[0], X.shape[1]

		# # Standardize data for better performance
		# if self.preprocessing:
		#     self.scaler_ = StandardScaler()
		#     X_norm = self.scaler_.fit_transform(X)
		# else:
		#     X_norm = np.copy(X)

		# Shuffle the data for validation as Keras do not shuffling for
		# Validation Split
		np.random.shuffle(X)

		# Validate and complete the number of hidden neurons
		if np.min(self.hidden_neurons) > self.n_features_:
			raise ValueError("The number of neurons should not exceed "
			                 "the number of features")
		# self.hidden_neurons_.insert(0, self.n_features_)

		# Calculate the dimension of the encoding layer & compression rate
		self.encoding_dim_ = np.median(self.hidden_neurons)
		self.compression_rate_ = self.n_features_ // self.encoding_dim_

		# # Build AE ndm & fit with X
		self.model_ = self._build_model(X, X, hidden_neurons=self.hidden_neurons)
		# self.history_ = self.model_.fit(X_norm, X_norm,
		#                                 epochs=self.epochs,
		#                                 batch_size=self.batch_size,
		#                                 shuffle=True,
		#                                 validation_split=self.validation_size,
		#                                 verbose=self.verbose).history

		# # Reverse the operation for consistency
		# # self.hidden_neurons_.pop(0)
		# # Predict on X itself and calculate the reconstruction error as
		# # the outlier scores. Noted X_norm was shuffled has to recreate
		# if self.preprocessing:
		#     X_norm = self.scaler_.transform(X)
		# else:
		#     X_norm = np.copy(X)

		pred_scores = self.model_.predict(X)
		self.decision_scores_ = pairwise_distances_no_broadcast(X,
		                                                        pred_scores)
		self._process_decision_scores()
		return self

	def decision_function(self, X):
		"""Predict raw anomaly score of X using the fitted detector.

		The anomaly score of an input sample is computed based on different
		detector algorithms. For consistency, outliers are assigned with
		larger anomaly scores.

		Parameters
		----------
		X : numpy array of shape (n_samples, n_features)
			The training input samples. Sparse matrices are accepted only
			if they are supported by the base estimator.

		Returns
		-------
		anomaly_scores : numpy array of shape (n_samples,)
			The anomaly score of the input samples.
		"""
		# Predict on X and return the reconstruction errors
		pred_scores = self.model_.predict(X)
		return pairwise_distances_no_broadcast(X, pred_scores)
